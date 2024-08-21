from __future__ import annotations

import argparse
import functools
import json
import logging
import os
import pathlib
import sys
import time
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import cv2
import gymnasium as gym
import imageio
import numpy as np
import torch as th
import tqdm
from gymnasium.wrappers import RecordEpisodeStatistics
from matplotlib import pyplot as plt
from omegaconf import OmegaConf
from sem_objnav.obj_nav.dataset import TRAIN, VAL
from sem_objnav.obj_nav.env import MappingObjNavEnv
from sem_objnav.sb3.ppo import PPO_AUX
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import SubprocVecEnv

logger = logging.getLogger(__name__)


def main(args):
    logging.basicConfig(level=logging.INFO)

    if args.sp_policy:
        exp_dir = args.checkpoint
        current_dir = pathlib.Path(__file__).parent.absolute()
        cfg = OmegaConf.load(current_dir / "obj_nav" / "cfg" / "config.yaml")
        # cfg.env.lin_vel_range = [0, 2.0]
    else:
        exp_dir = args.checkpoint.parent
        cfg = OmegaConf.load(exp_dir / ".hydra" / "config.yaml")
    set_random_seed(cfg.exp.seed)
    data_dir = Path(".") / "data"
    assert data_dir.exists(), f"Data directory {data_dir} does not exist"
    if args.max_episode_steps > 0:
        cfg.env.max_episode_steps = args.max_episode_steps
    cfg.env.semantic_model.model_type = args.seg_model_type
    if args.seg_model_type:
        cfg.env.semantic_model.ckpt = args.seg_model
        cfg.env.semantic_model.config = args.seg_config
        cfg.env.semantic_model.prob_aggregate = args.seg_prob_aggregate
        cfg.env.semantic_model.temperature = args.temperature
        cfg.env.semantic_model.mode = args.mode

    split = "val"
    if args.scenes:
        assert not args.val_scenes, "Cannot specify both --scenes and --val-scenes"
        scenes = args.scenes
    elif args.val_scenes:
        scenes = VAL
    elif cfg.train.scenes:
        split = "train"
        scenes = cfg.train.scenes
    elif cfg.train.num_scenes > 0:
        split = "train"
        scenes = TRAIN[: cfg.train.num_scenes]
    else:
        raise ValueError("No scenes specified for training")

    if args.limit_scenes > 0:
        scenes = scenes[: args.limit_scenes]
    if len(scenes) < args.num_workers:
        scenes = scenes * (args.num_workers // len(scenes))
    # need this monitor file filename for save model callback
    checkpoint_path = args.checkpoint
    assert checkpoint_path.exists(), f"Checkpoint path {checkpoint_path} does not exist"
    # Create a videos directory if it does not exist
    eval_dir = exp_dir / args.out_prefix
    eval_dir.mkdir(parents=True, exist_ok=True)
    if args.collect_nb_data:
        (eval_dir / "nb_data").mkdir(exist_ok=True)

    def make_env(rank: int):
        nonlocal scenes
        if len(scenes) < args.num_workers:
            assert (
                args.num_workers % len(scenes) == 0
            ), "num_envs must be divisible by total scenes if total scenes < num_envs"
            scenes = list(scenes) * (args.num_workers // len(scenes))
        split_scenes = np.array_split(scenes, args.num_workers)
        env_scenes = list(split_scenes[rank])
        logger.info(f"Env {rank} will use scenes {env_scenes}")
        gpu = args.env_gpus[rank % len(args.env_gpus)]

        def _init() -> MappingObjNavEnv:
            set_random_seed(cfg.exp.seed + rank)
            env = MappingObjNavEnv(
                config=cfg.env,
                scenes=env_scenes,
                split=split,
                max_scene_repeat_episodes=1,
                use_aux_angle=args.use_aux_angle,
                shuffle=True,
                seed=cfg.exp.seed + rank,
                render_mode="rgb_array",
                render_options=args.render_options,
                gpu_device_id=gpu,
                exclude_goal_class_episodes=args.exclude_classes,
                shortest_path_policy=args.sp_policy,
            )
            if hasattr(env.mapping, "hv_ratio"):
                env.mapping.hv_ratio = args.hv_ratio
            if hasattr(env.mapping, "hv_min_views"):
                env.mapping.hv_min_views = args.hv_min_views
            if hasattr(env.mapping, "hv_limit_obj_view_dist"):
                env.mapping.hv_limit_obj_view_dist = args.hv_view_dist
            if hasattr(env.mapping, "goal_decay"):
                env.mapping.goal_decay = args.goal_decay
            if hasattr(env.mapping, "goal_mark"):
                env.mapping.goal_mark = args.goal_mark
            if hasattr(env.mapping, "entropy_threshold"):
                env.mapping.entropy_threshold = args.entropy_threshold
            if args.collect_nb_data:
                env.mapping.collect_nb_data = True
            if args.record_traj:
                env.mapping.record_traj = True
            env.mapping.filter_goal_prob = args.filter_goal_prob
            print("setting filter goal prob to", args.filter_goal_prob)
            env.mapping.found_dist_m = args.found_dist_m
            env.reset(cfg.exp.seed + rank)
            return RecordEpisodeStatistics(env)

        set_random_seed(cfg.exp.seed + rank)
        return _init

    train_env = SubprocVecEnv([make_env(i) for i in range(args.num_workers)])
    if not args.sp_policy:
        model = PPO_AUX.load(
            str(checkpoint_path), train_env, n_steps=1000, device="cuda"
        )
        model.policy.set_training_mode(False)
    obs = train_env.reset()
    if args.episodes == -1:
        scene_episodes = {}
        for item in train_env.reset_infos:
            scene_episodes = {**scene_episodes, **item["max_episodes"]}
    else:
        scene_episodes = {scene: args.episodes for scene in scenes}

    scene_metrics = defaultdict(lambda: defaultdict(list))
    if args.record:
        video_writers = {}
    total_eval_episodes = sum(scene_episodes.values())
    pbar = tqdm.tqdm(total=total_eval_episodes)
    ep_aux_acc = [[] for _ in range(args.num_workers)]
    start_time = time.time()
    env_steps = 0
    done_scenes = set()
    goal_classes = ["chair", "bed", "plant", "toilet", "tv_monitor", "sofa"]
    goal_counts = np.zeros(6)
    goal_success = np.zeros(6)
    # log_for_analysis = {
    #     "fn_found": [],
    #     "fp_found": [],
    #     "fp_detections": [],
    #     "fn_detections": [],
    #     "spl": [],
    # }

    while (
        sum([len(v["success"]) for v in scene_metrics.values()]) < total_eval_episodes
    ):
        if args.sp_policy:
            action = train_env.env_method("shortest_path_policy")
            action = np.array(action)
            aux_angle_pred = np.zeros(
                (args.num_workers, *train_env.action_space.spaces["aux_angle"].shape)
            )
            aux_angle_pred[:, 0] = 1.0
        else:
            action, _, aux_angle_pred = model.predict(obs)
        env_steps += len(action)
        obs, reward, dones, info = train_env.step(
            [
                {"action": action[i], "aux_angle": aux_angle_pred[i]}
                for i in range(len(action))
            ]
        )
        aux_angle_gt = np.array([[info_dict["true_aux"]] for info_dict in info])
        true_labels = th.from_numpy(aux_angle_gt).long().squeeze(1)
        aux_acc = (th.tensor(aux_angle_pred).argmax(1) == true_labels).float()
        for i, acc in enumerate(aux_acc):
            ep_aux_acc[i].append(acc)
        if args.record:
            frames = train_env.get_images()
        elif args.interactive:
            rgb_array = train_env.render()
            cv2.imshow("vecenv", rgb_array[:, :, ::-1])
            cv2.waitKey(0)
        for i, done in enumerate(dones):
            scene = info[i]["scene"]
            if len(scene_metrics[scene]["success"]) >= scene_episodes[scene]:
                ep_aux_acc[i] = []
                continue
            if args.record:
                key = f"{scene}_{i}"
                if key not in video_writers:
                    video_writers[key] = imageio.get_writer(
                        str(eval_dir / f"{key}.mp4"), fps=10
                    )
                video_writers[f"{scene}_{i}"].append_data(frames[i])
            if "traj" in info[i]:
                recorded = info[i]["traj"]
                global_map = recorded["global_map"][..., ::-1]
                gt_map = recorded["gt_global_map"][..., ::-1]
                step = recorded["step"]
                name = f"{recorded['goal_name']}_{info[i]['scene']}_{info[i]['episode_id']}_succ_{info[i]['success']}"
                cv2.imwrite(str(eval_dir / f"{name}_pred_map.png"), global_map)
                cv2.imwrite(str(eval_dir / f"{name}_gt_map.png"), gt_map)
                # Get the pixels that are different betweeen the global map and the ground truth map and color it in yellow and save it as name_diff.png
                diff = recorded["diff"]

                cv2.imwrite(str(eval_dir / f"{name}_diff.png"), diff[..., ::-1])

                # recorded_unc_map = recorded["recorded_unc_map"]
                # cmap = plt.get_cmap("viridis")
                # recorded_unc_map[recorded_unc_map < 0] = 0
                # rgba_img = cmap(recorded_unc_map)
                # rgb_img = np.delete(rgba_img.squeeze(), 3, 2) * 255
                # cv2.imwrite(str(eval_dir / f"{name}_unc.png"), rgb_img[..., ::-1])

                # current_unc_map = recorded["current_unc_map"]
                # current_unc_map[current_unc_map < 0] = 0
                # rgba_img = cmap(current_unc_map)
                # rgb_img = np.delete(rgba_img.squeeze(), 3, 2) * 255
                # cv2.imwrite(
                #     str(eval_dir / f"{name}_current_unc.png"), rgb_img[..., ::-1]
                # )
                cmap = plt.get_cmap("viridis")
                current_unc_map_full = recorded["current_unc_map_full"]
                current_unc_map_full[current_unc_map_full < 0] = 0
                rgba_img = cmap(current_unc_map_full)
                rgb_img = np.delete(rgba_img.squeeze(), 3, 2) * 255
                cv2.imwrite(
                    str(eval_dir / f"{name}_current_unc_full.png"), rgb_img[..., ::-1]
                )

            if done:
                success = int(info[i]["success"])
                fp_detections = info[i]["fp_detections"]
                fn_detections = info[i]["fn_detections"]
                fp_found = int(info[i]["fp_found"])
                fn_found = int(info[i]["fn_found"])
                goal_counts[info[i]["goal"]] += 1
                if info[i]["success"]:
                    goal_success[info[i]["goal"]] += 1

                # if fp_detections > 0:
                #     log_for_analysis["fp_detections"].append(
                #         [info[i]["episode_id"], info[i]["scene"], fp_detections]
                #     )
                # if fp_detections > 0:
                #     log_for_analysis["fn_detections"].append(
                #         [info[i]["episode_id"], info[i]["scene"], fn_detections]
                #     )
                # if fp_found > 0:
                #     log_for_analysis["fp_found"].append(
                #         [info[i]["episode_id"], info[i]["scene"], fp_found]
                #     )
                # if fn_found > 0:
                #     log_for_analysis["fn_found"].append(
                #         [info[i]["episode_id"], info[i]["scene"], fn_found]
                #     )

                # log_for_analysis["spl"].append(
                #     [info[i]["episode_id"], info[i]["scene"], info[i]["spl"]]
                # )
                ep_metrics = {
                    "success": success,
                    "return": info[i]["episode"]["r"][0],
                    "length": info[i]["episode"]["l"][0],
                    "spl": info[i]["spl"],
                    "collision": info[i]["collision_count"],
                    "aux_acc": np.mean(ep_aux_acc[i]),
                    "fp_found": fp_found,
                    "fn_found": fn_found,
                    "fp_detections": fp_detections,
                    "fn_detections": fn_detections,
                }
                ep_aux_acc[i] = []
                for k, v in ep_metrics.items():
                    scene_metrics[scene][k].append(float(v))
                if len(scene_metrics[scene]["success"]) == scene_episodes[scene]:
                    done_scenes.add(scene)
                    print(f"Done with scene {scene}")
                    remaining = set(scenes) - done_scenes
                    print(f"Remaining number of scenes: {len(remaining)}")
                    print(f"Remaining scenes: {remaining}")
                    print("Current scenes", [info_dict["scene"] for info_dict in info])

                if args.collect_nb_data:
                    nb_data = info[i]["nb_data"]
                    np.save(
                        str(
                            eval_dir
                            / "nb_data"
                            / f"{nb_data['goal_id']}_{info[i]['scene']}_{info[i]['episode_id']}.npy"
                        ),
                        nb_data,
                    )

                pbar.update(1)
    if args.record:
        for writer in video_writers.values():
            writer.close()

    # calculate average env steps per second across all episodes
    end_time = time.time()
    ave_env_steps_per_sec = env_steps / (end_time - start_time)

    logger.info(f"Average env steps per second: {ave_env_steps_per_sec}")
    # aggregate scene metrics
    # Calculate mean of aggregated metrics
    aggregated_metrics = defaultdict(list)
    log_scene_metrics = defaultdict(lambda: defaultdict(list))

    for scene, metrics in scene_metrics.items():
        for k, v in metrics.items():
            aggregated_metrics[k].extend(v)
            log_scene_metrics[scene][k] = v
            log_scene_metrics[scene][f"{k}_mean"] = np.mean(v)
        log_scene_metrics[scene]["n_episodes"] = len(v)
    log_aggregated_metrics = defaultdict(list)
    for k, v in aggregated_metrics.items():
        log_aggregated_metrics[k] = np.mean(v)
        log_aggregated_metrics["n_episodes"] = len(v)

    with open(eval_dir / "metrics.json", "w") as f:
        json.dump(
            {
                "aggregated": log_aggregated_metrics,
                "args": " ".join(sys.argv),
                "goal_classes": goal_classes,
                "goal_counts": goal_counts.tolist(),
                "goal_accuracy": (goal_success / goal_counts).tolist(),
                "scene": log_scene_metrics,
            },
            f,
            indent=4,
        )

    # with open(eval_dir / "log_for_analysis.json", "w") as f:
    #     json.dump(log_for_analysis, f, indent=4)
    return log_aggregated_metrics


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--out_prefix", type=str, default="eval")
    parser.add_argument("--scenes", type=str, nargs="+", default=[])
    parser.add_argument("--env_gpus", type=int, nargs="+", default=[0])
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--val_scenes", action="store_true")
    parser.add_argument("--limit_scenes", type=int, default=0)
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--episodes", type=int, default=-1)
    parser.add_argument("--use_aux_angle", type=str, default="pred")
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--seg_model", type=str, default=None)
    parser.add_argument("--seg_config", type=str, default=None)
    parser.add_argument("--seg_model_type", type=str, default=None)
    parser.add_argument("--seg_prob_aggregate", type=str, default="argmax")
    parser.add_argument("--filter_goal_prob", type=float, default=None)
    parser.add_argument("--max_episode_steps", type=int, default=-1)
    parser.add_argument("--mode", type=str, default="semantic")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--exclude_classes", nargs="+", default=[])
    parser.add_argument("--found_dist_m", type=float, default=0.9)
    parser.add_argument("--sp_policy", action="store_true")
    parser.add_argument("--entropy_threshold", type=float, default=0.0)
    parser.add_argument(
        "--render_options",
        nargs="+",
        default=["rgb", "map_large", "map_small", "map_global", "write_metrics"],
    )
    parser.add_argument("--hv_ratio", type=float, default=0.5)
    parser.add_argument("--hv_min_views", type=int, default=10)
    parser.add_argument("--hv_view_dist", type=float, default=0.0)
    parser.add_argument("--goal_decay", type=float, default=0.9)
    parser.add_argument("--goal_mark", type=float, default=2.0)
    parser.add_argument("--collect_nb_data", action="store_true")
    parser.add_argument("--record_traj", action="store_true")

    return parser


if __name__ == "__main__":
    os.environ["MAGNUM_LOG"] = "quiet"
    os.environ["HABITAT_SIM_LOG"] = "quiet"
    args = build_parser().parse_args()
    assert (
        (args.interactive and not args.record)
        or (not args.interactive and args.record)
        or (not args.interactive and not args.record)
    ), "Can only be interactive or record or neither, not both"
    main(args)

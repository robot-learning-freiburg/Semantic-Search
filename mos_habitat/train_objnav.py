from __future__ import annotations

import logging
import os
import pathlib
from pathlib import Path
from typing import Callable

import hydra
import numpy as np
import wandb
from omegaconf import DictConfig, OmegaConf
from sem_objnav.obj_nav.dataset import TRAIN
from sem_objnav.obj_nav.env import MappingObjNavEnv
from sem_objnav.sb3.callbacks import (
    MOSMetricsLoggerCheckpointerCallback,
    VideoRecorderCallback,
    linear_schedule,
)
from sem_objnav.sb3.encoder import EgocentricEncoders
from sem_objnav.sb3.ppo import PPO_AUX
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

logger = logging.getLogger(__name__)


@hydra.main(version_base="1.3", config_path="obj_nav/cfg", config_name="config.yaml")
def main(cfg: DictConfig) -> None:
    print(os.getcwd())
    set_random_seed(cfg.exp.seed)
    exp_dir = pathlib.Path(".")
    data_dir = exp_dir / "data"
    if not data_dir.exists():
        os.symlink(cfg.exp.data_dir, str(data_dir), target_is_directory=True)

    if cfg.wandb is not None:
        wandb_cfg = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
        run = wandb.init(**cfg.wandb, config=wandb_cfg)

    checkpoint_seed = 0
    ckpt_dir = Path.cwd()
    checkpoint_path = ckpt_dir / "last_model.zip"
    if checkpoint_path.exists():
        model = PPO_AUX.load(checkpoint_path)
        timesteps = model.num_timesteps
        # arbirtary prime to get a seed that is different from the last one
        checkpoint_seed = timesteps % 929

    def make_env(rank: int) -> Callable:
        if cfg.train.scenes:
            scenes = list(cfg.train.scenes)
        else:
            scenes = (
                TRAIN[: cfg.train.num_scenes] if cfg.train.num_scenes > 0 else TRAIN
            )
        if len(scenes) < cfg.train.num_envs:
            assert (
                cfg.train.num_envs % len(scenes) == 0
            ), "num_envs must be divisible by total scenes if total scenes < num_envs"
            scenes = list(scenes) * (cfg.train.num_envs // len(scenes))
        split_scenes = np.array_split(scenes, cfg.train.num_envs)
        env_scenes = split_scenes[rank]
        logger.info(f"Env {rank} will use scenes {env_scenes}")
        gpu_device = cfg.exp.env_gpu_pool[rank % len(cfg.exp.env_gpu_pool)]

        def _init() -> MappingObjNavEnv:
            env = MappingObjNavEnv(
                config=cfg.env,
                scenes=list(env_scenes),
                split="train",
                max_scene_repeat_episodes=cfg.train.max_scene_repeat_episodes,
                use_aux_angle=cfg.train.use_aux_angle,
                shuffle=True,
                seed=cfg.exp.seed + rank + checkpoint_seed,
                render_mode="rgb_array",
                gpu_device_id=gpu_device,
            )
            env.reset(cfg.exp.seed + rank + checkpoint_seed)
            return env

        set_random_seed(cfg.exp.seed)
        return _init

    train_env = SubprocVecEnv([make_env(i) for i in range(cfg.train.num_envs)])
    # need this monitor file filename for save model callback
    train_env = VecMonitor(
        train_env,
        filename=str(exp_dir / "train"),
    )
    aux_bin_number = 12
    task_obs = train_env.observation_space["task_obs"].shape[0] - aux_bin_number
    policy_kwargs = dict(
        features_extractor_class=EgocentricEncoders,
        aux_pred_dim=aux_bin_number,
        proprio_dim=task_obs,
        use_aux=True,
        deact_aux=False,
        cut_out_aux_head=aux_bin_number,
    )
    kwargs = {**cfg.ppo}
    kwargs["policy_kwargs"] = {**kwargs["policy_kwargs"], **policy_kwargs}
    kwargs["learning_rate"] = linear_schedule(**kwargs["learning_rate"])
    if checkpoint_path.exists():
        logger.info("Experiment already exists, will continue training")
        model = PPO_AUX.load(str(checkpoint_path), env=train_env, **kwargs)
        if model.num_timesteps >= cfg.train.steps:
            logger.info("Training already done")
            exit(0)
    else:
        model = PPO_AUX(
            "MultiInputPolicy",
            train_env,
            verbose=1,
            tensorboard_log="./logs",
            **kwargs,
        )
        if hasattr(cfg.train, "finetune_ckpt") and cfg.train.finetune_ckpt:
            from stable_baselines3.common.save_util import load_from_zip_file

            _, params, _ = load_from_zip_file(cfg.train.finetune_ckpt)
            model.policy.load_state_dict(params["policy"])
    if cfg.train.scenes:
        num_unique_scenes = len(cfg.train.scenes)
    elif cfg.train.num_scenes > 0:
        num_unique_scenes = cfg.train.num_scenes
    else:
        num_unique_scenes = len(TRAIN)
    model.learn(
        cfg.train.steps - model.num_timesteps,
        callback=[
            MOSMetricsLoggerCheckpointerCallback(
                total_train_scenes=num_unique_scenes,
                save_freq=cfg.train.ckpt_freq,
                save_model_dir=str(ckpt_dir),
            ),
            VideoRecorderCallback(
                video_length=1000,
                video_fps=int(20),
                video_freq=cfg.train.video_freq,
                num_envs_to_record=min(2, cfg.train.num_envs),
            ),
        ],
        tb_log_name="ppo_aux",
        reset_num_timesteps=False,
    )


if __name__ == "__main__":
    main()

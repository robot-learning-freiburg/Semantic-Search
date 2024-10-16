from __future__ import annotations

import argparse
import json
import logging
import os
import pathlib
from collections import Counter
from typing import Callable, List, Tuple

import cv2
import habitat_sim
import numpy as np
import tqdm
from habitat_sim import registry as registry
from habitat_sim.utils.common import quat_from_angle_axis, quat_from_two_vectors
from habitat_sim.utils.data import ImageExtractor
from nicr_mt_scene_analysis.metric.miou import MeanIntersectionOverUnion
from nicr_scene_analysis_datasets.utils.io import create_dir
from numpy import float32, ndarray
from omegaconf import OmegaConf
from quaternion import quaternion
from sem_objnav.obj_nav.dataset import TRAIN, VAL
from sem_objnav.obj_nav.env import MappingObjNavEnv
from sem_objnav.obj_nav.mapper import get_colormap
from sem_objnav.segment.hm3d.hm3d import Hm3dMeta
from sem_objnav.segment.pose_extractor import PoseExtractor, TopdownView
from stable_baselines3.common.utils import set_random_seed

logger = logging.getLogger(__name__)


def main(args) -> None:
    cfg_path = pathlib.Path(__file__).parent.absolute() / "cfg" / "config.yaml"
    cfg = OmegaConf.load(cfg_path)
    OmegaConf.set_readonly(cfg, False)
    set_random_seed(cfg.exp.seed)

    color_pallete = get_colormap(cfg.env.num_semantic_classes)
    rnd = np.random.RandomState(42)
    instance_min_relative_area = 0.25 / 100
    output_path = args.output_dir
    os.mkdir(output_path)
    with open(os.path.join(output_path, "creation_meta.json"), "w") as f:
        json.dump({}, f)

    for split, SCENES in [
        ("train", TRAIN),
        ("val", VAL),
    ]:
        split_dir = split
        rgb_base_path = os.path.join(output_path, split_dir, Hm3dMeta.RGB_DIR)
        instances_base_path = os.path.join(
            output_path, split_dir, Hm3dMeta.INSTANCES_DIR
        )
        depth_base_path = os.path.join(output_path, split_dir, Hm3dMeta.DEPTH_DIR)
        labels_base_path = os.path.join(
            output_path,
            split_dir,
            Hm3dMeta.SEMANTIC_DIR_FMT.format(cfg.env.num_semantic_classes - 1),
        )
        labels_colored_base_path = os.path.join(
            output_path,
            split_dir,
            Hm3dMeta.SEMANTIC_COLORED_DIR_FMT.format(cfg.env.num_semantic_classes - 1),
        )
        # debug_path = os.path.join(output_path, split_dir, "debug")

        create_dir(rgb_base_path)
        create_dir(instances_base_path)
        create_dir(depth_base_path)
        create_dir(labels_base_path)
        create_dir(labels_colored_base_path)
        # create_dir(debug_path)

        prev_height = None
        idx = 0

        def write_sample_at(env, idx, position, rotation, suffix=""):
            sim = env._env.habitat_env.sim
            obs = sim.get_observations_at(
                position=position,
                rotation=rotation,
                keep_agent_at_new_pose=True,
            )
            rgb = obs["rgb"]
            raw_instances_output = obs["semantic"][..., 0]
            depth = obs["depth"]
            raw_instances_output[
                raw_instances_output > env.instance2semantic.shape[0] - 1
            ] = 0
            semantic_seg = env.instance2semantic[raw_instances_output]
            known_labels = semantic_seg != 0
            if known_labels.sum() == 0:
                print("No known labels, .................................")
            instance_seg = (raw_instances_output * known_labels).astype(int)
            semantic_seg = semantic_seg.astype(int)
            # rgb image
            cv2.imwrite(
                os.path.join(rgb_base_path, f"{idx:06d}{suffix}.png"),
                cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
            )

            # Due to the conversion above, the instance ids are not continuous
            # anymore and sparse.
            # This can be a disadvantage for tasks such as PQ calculation.
            # Thats why the following code changes the instance ids to be
            # smaller.
            # Furthermore, we filter instances with quite small area as they are
            # most likely not valid
            current_instance = instance_seg
            min_instance_area = instance_min_relative_area * np.prod(
                current_instance.shape[-2:]
            )

            instances_write = np.zeros_like(current_instance)
            for new_id, c_id in enumerate(np.unique(current_instance)):
                mask = current_instance == c_id

                if mask.sum() < min_instance_area:
                    # instance is too small, skip it
                    continue

                # assign new id
                # note that new_id==0 always corresponds to the void label due
                # to the instance encoding above ((labels << 6) + instances),
                # thus, we do not assign new_id==0 to any instance of a thing
                # class except its area is smaller than the area threshold
                instances_write[mask] = new_id

            cv2.imwrite(
                os.path.join(instances_base_path, f"{idx:06d}{suffix}.png"),
                instances_write,
            )

            # depth image
            cv2.imwrite(
                os.path.join(depth_base_path, f"{idx:06d}{suffix}.png"),
                (depth * 1e3).astype("uint16"),
            )

            label = semantic_seg
            cv2.imwrite(os.path.join(labels_base_path, f"{idx:06d}{suffix}.png"), label)
            # colored label image
            # (indexed png8 with color palette)
            label_colored = color_pallete[label]
            cv2.imwrite(
                os.path.join(labels_colored_base_path, f"{idx:06d}{suffix}.png"),
                cv2.cvtColor(label_colored, cv2.COLOR_RGB2BGR),
            )

        for i in tqdm.tqdm(range(len(SCENES))):
            env = MappingObjNavEnv(
                config=cfg.env,
                split=split,
                scenes=[SCENES[i]],
                max_scene_repeat_episodes=1,
                use_aux_angle=cfg.train.use_aux_angle,
                gpu_device_id=0,
                shuffle=False,
                seed=cfg.exp.seed,
            )
            env.reset(cfg.exp.seed)
            sim = env._env.habitat_env.sim
            ref_point = get_pathfinder_reference_point(sim)
            if i == 0 or prev_height != ref_point[1]:
                # Panaroma extraction only when the height changes, meaning we are in a new floor
                prev_height = ref_point[1]
                meters_per_pixel = 0.1
                tdv = TopdownView(
                    sim, ref_point[1], meters_per_pixel=meters_per_pixel
                ).topdown_view
                height, width = tdv.shape
                dist = (
                    min(height, width) // 10
                )  # We can modify this to be user-defined later

                # Create a grid of camera positions
                n_gridpoints_width, n_gridpoints_height = (
                    width // dist - 1,
                    height // dist - 1,
                )

                # Exclude camera positions at invalid positions
                gridpoints = []
                for h in range(n_gridpoints_height):
                    for w in range(n_gridpoints_width):
                        point = (dist + h * dist, dist + w * dist)
                        if _valid_point(*point, tdv):
                            gridpoints.append(point)

                gridpt_poses = []
                for point in gridpoints:
                    point_label_pairs = _panorama_extraction(point, tdv, dist)
                    gridpt_poses.extend(
                        [(point, point_) for point_, label in point_label_pairs]
                    )

                poses = _convert_to_scene_coordinate_system(
                    gridpt_poses, ref_point, meters_per_pixel
                )
                for position, rotation in poses:
                    write_sample_at(env, idx, position, rotation)
                    idx += 1

            env.close()


def get_pathfinder_reference_point(sim):
    pf = sim.pathfinder
    bound1, bound2 = pf.get_bounds()
    startw = min(bound1[0], bound2[0])
    starth = min(bound1[2], bound2[2])
    starty = sim.get_agent_state().position[1]
    return (startw, starty, starth)  # width, y, height


def _convert_to_scene_coordinate_system(
    poses: List[Tuple[Tuple[int, int], Tuple[int, int], str]],
    ref_point: Tuple[float32, float32, float32],
    meters_per_pixel,
) -> List[Tuple[Tuple[int, int], quaternion, str]]:
    # Convert from topdown map coordinate system to that of the scene
    startw, starty, starth = ref_point
    for i, pose in enumerate(poses):
        pos, cpi = pose
        r1, c1 = pos
        r2, c2 = cpi
        new_pos = np.array(
            [
                startw + c1 * meters_per_pixel,
                starty,
                starth + r1 * meters_per_pixel,
            ]
        )
        new_cpi = np.array(
            [
                startw + c2 * meters_per_pixel,
                starty,
                starth + r2 * meters_per_pixel,
            ]
        )
        cam_normal = new_cpi - new_pos
        new_rot = quat_from_two_vectors(habitat_sim.geo.FRONT, cam_normal)
        new_pos_t: Tuple[int, int] = tuple(new_pos)  # type: ignore[assignment]
        poses[i] = (new_pos_t, new_rot)

    return poses


def _panorama_extraction(
    point: Tuple[int, int], view: ndarray, dist: int
) -> List[Tuple[Tuple[int, int], float]]:
    in_bounds_of_topdown_view = lambda row, col: 0 <= row < len(
        view
    ) and 0 <= col < len(view[0])
    point_label_pairs = []
    r, c = point
    neighbor_dist = dist // 2
    neighbors = [
        (r - neighbor_dist, c - neighbor_dist),
        (r - neighbor_dist, c),
        (r - neighbor_dist, c + neighbor_dist),
        (r, c - neighbor_dist),
        (r, c + neighbor_dist),
        (r + neighbor_dist, c - neighbor_dist),
        # (r + step, c), # Exclude the pose that is in the opposite direction of habitat_sim.geo.FRONT, causes the quaternion computation to mess up
        (r + neighbor_dist, c + neighbor_dist),
    ]

    for n in neighbors:
        # Only add the neighbor point if it is navigable. This prevents camera poses that
        # are just really close-up photos of some object
        if in_bounds_of_topdown_view(*n) and _valid_point(*n, view):
            point_label_pairs.append((n, 0.0))

    return point_label_pairs


def _valid_point(row: int, col: int, view: ndarray) -> bool:
    return view[row][col] == 1.0


if __name__ == "__main__":
    os.environ["MAGNUM_LOG"] = "quiet"
    os.environ["HABITAT_SIM_LOG"] = "quiet"
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()
    main(args)
    # from sem_objnav.segment.hm3d.dataset import Hm3d

    # dataset = Hm3d(
    #     dataset_path=args.output_dir,
    #     split="train",
    #     semantic_n_classes=args.num_semantic_classes,
    # )
    # print(dataset.depth_compute_stats())

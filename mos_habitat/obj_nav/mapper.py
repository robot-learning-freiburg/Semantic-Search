from __future__ import annotations

import math
import os
import pickle
from typing import Dict, List, Set, Tuple

import cv2
import numpy as np
import numpy_indexed as npi
import quaternion
import scipy
import scipy.stats
import torch
from habitat.tasks.utils import cartesian_to_polar


def pointcloud(depth, camera_hfov_deg):
    height, width = depth.shape

    hfov = np.deg2rad(camera_hfov_deg)
    fx = width / (2.0 * np.tan(hfov / 2.0))

    vfov = height / width * hfov
    fy = height / (2.0 * np.tan(vfov / 2.0))

    c, r = np.meshgrid(np.arange(width), np.arange(height), sparse=True)
    valid = depth > 0
    z = np.where(valid, depth, 0.0)
    x = np.where(valid, z * (c - (width / 2)) / fx, 0)
    y = np.where(valid, z * (r - (height / 2)) / fy, 0)
    return np.stack((x, y, z), axis=-1), valid


def compute_map_size(
    pixel2meter: float,
    bounds: np.ndarray,
) -> Tuple[int, int]:
    lower_bound_m, upper_bound_m = bounds[:3], bounds[3:]

    grid_rows, grid_cols = (
        math.ceil(abs(upper_bound_m[2] - lower_bound_m[2]) / pixel2meter),
        math.ceil(abs(upper_bound_m[0] - lower_bound_m[0]) / pixel2meter),
    )
    return grid_rows, grid_cols


class SemanticMapper:
    def __init__(
        self,
        num_semantic_classes: int,
        depth_hfov: float,
        depth_range: Tuple[float, float],
        egomap_small_side_m: float = 3.0,
        egomap_large_side_m: float = 10.0,
        egomap_size: Tuple[int, int] = (128, 128),
        pixel2meter: float = 0.03,
        aux_num_bins: int = 12,
        aux_radius_m: int = 0.4,
        aux_bin_m: int = 0.15,
        path_width_m: int = 0.1,
        height_cutoff_m: float = 0.1,
        agent_draw_radius_m: float = 0.1,
        entropy_threshold: float = 0.5,
        probablistic_mapping: bool = False,
        prob_aggregate: str = "height_map_argmax",
        success_dist_m: float = 1.0,
        seg_model_n_classes: int | None = None,
        filter_goal_prob: float | None = None,
        collect_nb_data: bool = False,
        model_name: str = None,
    ):
        self.pixel2meter = pixel2meter
        self.depth_range = depth_range
        self.depth_hfov = depth_hfov

        self.aux_bin_number = aux_num_bins
        self.bin_size = 360 / self.aux_bin_number
        self.angle_to_bins = np.arange(
            self.bin_size, 360 + self.bin_size, self.bin_size
        )
        deg_steps = np.arange(0, 360, self.bin_size)
        aux_radius_px = int(aux_radius_m / pixel2meter)
        aux_points = []
        for i, deg in enumerate(deg_steps):
            ax = self.pol2cart(aux_radius_px, np.deg2rad(deg))
            # The minus sign makes it pixel coordinates
            aux_points.append(-np.array(ax))
        self.aux_points = np.array(aux_points, dtype=np.int32)
        self.path_width_px = int(path_width_m / pixel2meter)
        ego_map_small_side_px = int(egomap_small_side_m / pixel2meter)
        ego_map_large_side_px = int(egomap_large_side_m / pixel2meter)

        self.egomap_small_crop = [ego_map_small_side_px, ego_map_small_side_px]
        self.egomap_large_crop = [ego_map_large_side_px, ego_map_large_side_px]
        self.offset_for_cut = 150
        self.agent_radius_px = int(agent_draw_radius_m / pixel2meter)
        self.egomap_size = egomap_size
        self.prev_pos_px = None
        self.aux_bin_px = int(aux_bin_m / pixel2meter)
        # Color pallete, it's bit convoluted, but it's to make sure that the colors are unique
        # self.color_pallete = get_colormap(num_semantic_classes + 3)
        # self.color_pallete = np.array(
        #     [
        #         [255, 245, 240] # unexplored
        #         [128, 64, 128], # unoccupied
        #         [244, 35, 232],  # occupied
        #         [70, 70, 70],  # chair
        #         [102, 102, 156],  # bed
        #         [190, 153, 153],  # plant
        #         [153, 153, 153],  # tv_monitor
        #         [250, 170, 30],  # toilet
        #         [220, 220, 0],  # sofa
        #         [255, 0, 0],  # goal
        #         [100, 251, 100],  # path
        #         [70, 130, 180],  # arrow
        #     ],
        #     dtype=np.uint8,
        # )
        self.color_pallete = np.array(
            [
                [255, 255, 255],  # unoccupied
                [230, 230, 230],  # unexplored
                [159, 159, 159],  # occupied
                [235, 189, 157],  # chair
                [234, 217, 156],  # bed
                [219, 235, 156],  # plant
                [189, 235, 156],  # tv_monitor
                [161, 236, 146],  # toilet
                [159, 229, 182],  # sofa
                [0, 255, 0],  # goal
                [0, 0, 255],  # path
                [70, 130, 180],  # arrow
            ],
            dtype=np.uint8,
        )
        self.active_goal_color = self.color_pallete[num_semantic_classes]
        # self.path_color = np.array([128, 0, 128], dtype=np.uint8)
        # self.color_pallete[num_semantic_classes + 1] = self.path_color
        self.path_color = self.color_pallete[num_semantic_classes + 1]
        self.arrow_color = tuple(
            int(c) for c in self.color_pallete[num_semantic_classes + 2]
        )
        self.aux_pred_color = np.array([255, 255, 255], dtype=np.uint8)

        self.path_id = num_semantic_classes + 1
        self.height_cutoff_m = height_cutoff_m
        self.probablistic_mapping = probablistic_mapping
        self.num_semantic_classes = num_semantic_classes
        self.prob_aggregate = prob_aggregate

        self.found_dist_m = success_dist_m
        self.log_data = {}
        if self.probablistic_mapping and self.prob_aggregate in [
            "height_map_hv",
            "height_map_prob_hv",
        ]:
            self.hv_limit_obj_view_dist = 0.0
            self.hv_min_views = 10
            self.hv_ratio = 0.5
        elif self.probablistic_mapping and self.prob_aggregate in [
            "avg_prob",
            "height_map_avg_prob",
            "w_avg_prob",
            "height_map_w_avg_prob",
            "w2_avg_prob",
            "height_map_w2_avg_prob",
            "log_odds",
            "height_map_log_odds",
        ]:
            self.entropy_threshold = entropy_threshold
        elif (
            self.probablistic_mapping
            and self.prob_aggregate == "goal_decay"
            or self.prob_aggregate == "height_map_goal_decay"
        ):
            self.goal_decay = 0.9
            self.goal_mark = 2.0
        # {'chair': 0, 'bed': 1, 'plant': 2, 'toilet': 3, 'tv_monitor': 4, 'sofa': 5}
        if num_semantic_classes == 3 + 6:
            self.goal2sem_id = np.array([3, 4, 5, 6, 7, 8])
        elif num_semantic_classes == 41 or num_semantic_classes == 38:
            # {0: 3, 1: 11, 2: 14, 3: 18, 4: 22, 5: 10}
            self.goal2sem_id = np.array([3, 11, 14, 18, 22, 10])
        self.map_seg_to_schema = None
        self.objnav_to_sem_pred = None
        self.sem_model_n_classes = seg_model_n_classes
        self.filter_goal_prob = filter_goal_prob
        self.collect_nb_data = collect_nb_data
        self.record_traj = False

        if self.prob_aggregate in ["height_map_stubborn", "stubborn"]:
            with open(
                os.path.join(
                    os.path.dirname(__file__), "nb_models", f"{model_name}.pkl"
                ),
                "rb",
            ) as f:
                self.nb_models = pickle.load(f)

    def reset(self, map_bounds: np.ndarray):
        # map_bounds = np.array([-10, 0, -10, 10, 0, 10])
        self.map_bounds = map_bounds
        lower_bound_m, upper_bound_m = map_bounds[:3], map_bounds[3:]
        grid_rows, grid_cols = (
            np.ceil(
                np.abs(upper_bound_m[2] - lower_bound_m[2]) / self.pixel2meter,
            ).astype(np.int32),
            np.ceil(
                np.abs(upper_bound_m[0] - lower_bound_m[0]) / self.pixel2meter,
            ).astype(np.int32),
        )
        self.semantic_map = np.zeros((grid_rows, grid_cols), dtype=np.uint32)
        self.semantic_map_color = np.zeros((grid_rows, grid_cols, 3), dtype=np.uint8)
        self.height_map = np.ones((grid_rows, grid_cols), dtype=np.float32) * -np.inf
        self.unc_map = None
        if self.probablistic_mapping:
            self.gt_map = np.zeros((grid_rows, grid_cols), dtype=np.uint32)
            if self.prob_aggregate in ["goal_decay", "height_map_goal_decay"]:
                self.goal_map = np.zeros(
                    (grid_rows, grid_cols),
                    dtype=np.float32,
                )
            elif self.prob_aggregate in ["log_odds", "height_map_log_odds"]:
                self.log_odds_seg_map = np.zeros(
                    (grid_rows, grid_cols, self.sem_model_n_classes),
                    dtype=np.float32,
                )
            elif self.prob_aggregate in [
                "height_map_avg_prob",
                "avg_prob",
                "w_avg_prob",
                "height_map_w_avg_prob",
                "w2_avg_prob",
                "height_map_w2_avg_prob",
            ]:
                self.avg_prob_seg_map = np.zeros(
                    (grid_rows, grid_cols, self.sem_model_n_classes),
                    dtype=np.float32,
                )
                self.avg_prob_weight_map = np.zeros(
                    (grid_rows, grid_cols, 1),
                    dtype=np.float32,
                )
            elif self.prob_aggregate in [
                "height_map_hv",
                "height_map_prob_hv",
            ]:
                self.background_map = np.zeros((grid_rows, grid_cols), dtype=np.uint32)
                self.hits = np.zeros((grid_rows, grid_cols), dtype=np.float32)
                self.views = np.zeros((grid_rows, grid_cols), dtype=np.float32)
            elif self.prob_aggregate in ["height_map_stubborn", "stubborn"]:
                self.total_view = np.zeros((grid_rows, grid_cols), dtype=np.uint32)
                self.cumulative_conf = np.zeros(
                    (grid_rows, grid_cols), dtype=np.float32
                )
                self.max_conf = np.zeros((grid_rows, grid_cols), dtype=np.float32)
                self.max_other_conf = np.zeros((grid_rows, grid_cols), dtype=np.float32)
            elif self.prob_aggregate in ["argmax", "height_map_argmax"]:
                pass
            else:
                raise RuntimeError("Invalid prob_aggregate")
            self.unc_map = np.ones((grid_rows, grid_cols, 1), dtype=np.float32) * -0.5
        else:
            self.gt_map = self.semantic_map
            self.gt_map_color = self.semantic_map_color
        if self.collect_nb_data:
            self.total_view = np.zeros((grid_rows, grid_cols), dtype=np.uint32)
            self.cumulative_conf = np.zeros((grid_rows, grid_cols), dtype=np.float32)
            self.max_conf = np.zeros((grid_rows, grid_cols), dtype=np.float32)
            self.max_other_conf = np.zeros((grid_rows, grid_cols), dtype=np.float32)
            self.eps_nb_data = {"goal": [], "non_goal": [], "goal_id": None}
        self.path_mask = np.zeros((grid_rows, grid_cols), dtype=np.uint8)
        self.prev_pos_px = None
        self.found_idx = None
        self._found_decision_idx = None
        self._mask_around_robot = None

    def pol2cart(self, rho, phi):
        x = rho * np.cos(phi)
        y = rho * np.sin(phi)
        return (x, y)

    def update_map(
        self, obs: Dict[str, np.ndarray], auxillary_angle: np.ndarray
    ) -> np.ndarray:
        depth_map = obs["depth"]

        # Set invalid depth to 0.
        depth_map[obs["depth"] == self.depth_range[0]] = 0.0
        depth_map[obs["depth"] == self.depth_range[1]] = 0.0
        # rescale to actual depth
        # depth_map = (
        #     depth_map * (self.depth_range[1] - self.depth_range[0])
        #     + self.depth_range[0]
        # )
        # self.semantic_map *= 0
        agent_state = obs["agent_position_and_rotation"]
        agent_position, agent_rotation = agent_state[:3], quaternion.from_float_array(
            agent_state[3:]
        ) * quaternion.from_rotation_vector(np.array([0, np.pi, 0]))
        camera_state = obs["camera_position_and_rotation"]
        camera_position, camera_rotation = camera_state[
            :3
        ], quaternion.from_float_array(
            camera_state[3:]
        ) * quaternion.from_rotation_vector(
            np.array([np.pi, 0, 0])
        )

        point_cloud_camera, valid_depth_mask = pointcloud(depth_map, self.depth_hfov)
        # path_mask = self.semantic_map == self.path_id
        point_cloud_camera_points = point_cloud_camera.reshape(-1, 3)
        camera_rotated = quaternion.rotate_vectors(
            camera_rotation,
            point_cloud_camera_points,
        )
        # flip the z axis to match the map
        point_cloud_world_flat = camera_rotated + camera_position
        task_goal_ids = obs["objectgoal"]
        goal_id = self.goal2sem_id[task_goal_ids][0]
        bounds = self.map_bounds
        lower_bound_m = bounds[:3]
        point_cloud_map_flat = self.world2map(point_cloud_world_flat, lower_bound_m)

        heights_z = point_cloud_world_flat[:, 1]
        trunc_mask = (
            heights_z < (camera_position[1] + self.height_cutoff_m)
        ) & valid_depth_mask.flatten()
        trunc_mask = np.argwhere(trunc_mask)

        map_height, map_width = self.semantic_map.shape
        map_bounds_mask = (point_cloud_map_flat[:, 0] < map_height) & (
            point_cloud_map_flat[:, 1] < map_width
        )
        map_bounds_idx = np.argwhere(map_bounds_mask)

        _, tallest_point_idx = npi.group_by(point_cloud_map_flat).argmax(heights_z)

        valid_flat_idx = np.intersect1d(
            np.intersect1d(tallest_point_idx, trunc_mask), map_bounds_idx
        )
        valid_map_points = point_cloud_map_flat[valid_flat_idx]
        update_cell_idx = tuple(valid_map_points.T)
        self.height_map[update_cell_idx] = np.maximum(
            self.height_map[update_cell_idx],
            heights_z[valid_flat_idx] - agent_position[1],
        )
        # Direct mapping
        if not self.probablistic_mapping:
            semantic_seg_flat = obs["semantic"].reshape(-1)
            updated_agg_seg_ids = semantic_seg_flat[valid_flat_idx]
            if self.prob_aggregate == "height_map_argmax":
                heights = self.height_map[update_cell_idx]
                update_height_map_mask = (
                    (updated_agg_seg_ids == 0)
                    | (updated_agg_seg_ids == 1)
                    | (updated_agg_seg_ids == 2)
                )
                occupied_mask = (heights > 0.1) & update_height_map_mask
                unoccupied_mask = (heights != -np.inf) * (
                    heights < 0.1
                ) & update_height_map_mask
                if self.num_semantic_classes == 3 + 6:
                    updated_agg_seg_ids[occupied_mask] = 2
                    updated_agg_seg_ids[unoccupied_mask] = 1
        else:
            updated_agg_gt_seg_ids = obs["semantic_gt"].reshape(-1)[valid_flat_idx]
            if "height_map" in self.prob_aggregate and self.record_traj:
                heights = self.height_map[update_cell_idx]
                update_height_map_mask = (
                    (updated_agg_gt_seg_ids == 0)
                    | (updated_agg_gt_seg_ids == 1)
                    | (updated_agg_gt_seg_ids == 2)
                )
                occupied_mask = (heights > 0.1) & update_height_map_mask
                unoccupied_mask = (heights != -np.inf) * (
                    heights < 0.1
                ) & update_height_map_mask
                if self.num_semantic_classes == 3 + 6:
                    updated_agg_gt_seg_ids[occupied_mask] = 2
                    updated_agg_gt_seg_ids[unoccupied_mask] = 1
            self.gt_map[update_cell_idx] = updated_agg_gt_seg_ids
            if (
                self.prob_aggregate == "goal_decay"
                or self.prob_aggregate == "height_map_goal_decay"
            ):
                updated_agg_seg_ids = obs["semantic"].reshape(-1)[valid_flat_idx]
                if self.filter_goal_prob is not None:
                    semantic_prob_flat = (
                        1 - obs["semantic_unc"].reshape(-1)[valid_flat_idx]
                    )
                    filter_mask = (semantic_prob_flat < self.filter_goal_prob) & (
                        updated_agg_seg_ids == goal_id
                    )
                    updated_agg_seg_ids[filter_mask] = 0

                (
                    update_cell_idx,
                    updated_agg_seg_ids,
                ) = self._erode_goal_2d(update_cell_idx, updated_agg_seg_ids, goal_id)
                is_goal_mask = updated_agg_seg_ids == goal_id
                goal_cell_idx = (
                    update_cell_idx[0][is_goal_mask],
                    update_cell_idx[1][is_goal_mask],
                )
                non_goal_cell_idx = (
                    update_cell_idx[0][~is_goal_mask],
                    update_cell_idx[1][~is_goal_mask],
                )
                self.goal_map[goal_cell_idx] += 1
                self.goal_map[non_goal_cell_idx] *= self.goal_decay

                can_show_goal_mask = self.goal_map[update_cell_idx] > self.goal_mark
                updated_agg_seg_ids[is_goal_mask] = 0
                updated_agg_seg_ids[can_show_goal_mask] = goal_id

                if "height_map" in self.prob_aggregate:
                    heights = self.height_map[update_cell_idx]
                    update_height_map_mask = (
                        (updated_agg_seg_ids == 0)
                        | (updated_agg_seg_ids == 1)
                        | (updated_agg_seg_ids == 2)
                    )
                    occupied_mask = (heights > 0.1) & update_height_map_mask
                    unoccupied_mask = (heights != -np.inf) * (
                        heights < 0.1
                    ) & update_height_map_mask
                    if self.num_semantic_classes == 3 + 6:
                        updated_agg_seg_ids[occupied_mask] = 2
                        updated_agg_seg_ids[unoccupied_mask] = 1
            elif self.prob_aggregate in [
                "argmax",
                "height_map_argmax",
                "stubborn",
                "height_map_stubborn",
            ]:
                updated_agg_seg_ids = obs["semantic"].reshape(-1)[valid_flat_idx]

                if "semantic_unc" in obs:
                    uncertainty = obs["semantic_unc"][..., np.newaxis]  # (h, w, c)
                    uncertainty_flat = uncertainty.reshape(-1, 1)
                    self.unc_map[update_cell_idx] = uncertainty_flat[valid_flat_idx]

                if self.filter_goal_prob is not None:
                    semantic_prob_flat = (
                        1 - obs["semantic_unc"].reshape(-1)[valid_flat_idx]
                    )
                    filter_mask = (semantic_prob_flat < self.filter_goal_prob) & (
                        updated_agg_seg_ids == goal_id
                    )
                    updated_agg_seg_ids[filter_mask] = 0

                if "height_map" in self.prob_aggregate:
                    heights = self.height_map[update_cell_idx]
                    update_height_map_mask = (
                        (updated_agg_seg_ids == 0)
                        | (updated_agg_seg_ids == 1)
                        | (updated_agg_seg_ids == 2)
                    )
                    occupied_mask = (heights > 0.1) & update_height_map_mask
                    unoccupied_mask = (heights != -np.inf) * (
                        heights < 0.1
                    ) & update_height_map_mask
                    if self.num_semantic_classes == 3 + 6:
                        updated_agg_seg_ids[occupied_mask] = 2
                        updated_agg_seg_ids[unoccupied_mask] = 1
                if self.collect_nb_data or "stubborn" in self.prob_aggregate:
                    self.total_view[update_cell_idx] += 1
                    goal_sem_ids = self.objnav_to_sem_pred[goal_id]
                    prob_seg = obs["semantic_prob"].transpose(1, 2, 0)
                    seg_n_classes = prob_seg.shape[-1]
                    non_goal_sem_ids = np.delete(np.arange(seg_n_classes), goal_sem_ids)

                    prob_seg_flat = prob_seg.reshape(-1, seg_n_classes)
                    goal_prob_mass = np.sum(prob_seg_flat[:, goal_sem_ids], axis=-1)
                    self.cumulative_conf[update_cell_idx] += goal_prob_mass[
                        valid_flat_idx
                    ]
                    self.max_conf[update_cell_idx] = np.maximum(
                        self.max_conf[update_cell_idx], goal_prob_mass[valid_flat_idx]
                    )
                    max_other_prob_mass = np.max(
                        prob_seg_flat[:, non_goal_sem_ids], axis=-1
                    )
                    self.max_other_conf[update_cell_idx] = np.maximum(
                        self.max_other_conf[update_cell_idx],
                        max_other_prob_mass[valid_flat_idx],
                    )
                    total_update_view = self.total_view[update_cell_idx]
                    cumulative_conf = self.cumulative_conf[update_cell_idx]
                    max_conf = self.max_conf[update_cell_idx]
                    max_other_conf = self.max_other_conf[update_cell_idx]
                    if self.collect_nb_data:
                        gt_goal_idx = self.gt_map[update_cell_idx] == goal_id
                        if gt_goal_idx.any():
                            self.eps_nb_data["goal"].append(
                                np.stack(
                                    [
                                        total_update_view[gt_goal_idx],
                                        cumulative_conf[gt_goal_idx],
                                        max_conf[gt_goal_idx],
                                        max_other_conf[gt_goal_idx],
                                    ],
                                    axis=-1,
                                )
                            )
                        if ~gt_goal_idx.any():
                            self.eps_nb_data["non_goal"].append(
                                np.stack(
                                    [
                                        total_update_view[~gt_goal_idx],
                                        cumulative_conf[~gt_goal_idx],
                                        max_conf[~gt_goal_idx],
                                        max_other_conf[~gt_goal_idx],
                                    ],
                                    axis=-1,
                                )
                            )
                    else:
                        features = np.stack(
                            [
                                total_update_view,
                                cumulative_conf,
                                max_conf,
                                max_other_conf,
                            ],
                            axis=-1,
                        )
                        if len(features) > 0:
                            pred = self.nb_models[goal_id].predict(features)
                            updated_agg_seg_ids[
                                (pred == 0) & (updated_agg_seg_ids == goal_id)
                            ] = 2

            elif self.prob_aggregate in [
                "avg_prob",
                "height_map_avg_prob",
                "w_avg_prob",
                "height_map_w_avg_prob",
                "w2_avg_prob",
                "height_map_w2_avg_prob",
            ]:
                prob_seg = obs["semantic_prob"].transpose(1, 2, 0)
                seg_n_classes = prob_seg.shape[-1]
                prob_seg_flat = prob_seg.reshape(-1, seg_n_classes)
                if "w_avg_prob" in self.prob_aggregate:
                    entropy = scipy.stats.entropy(
                        prob_seg_flat[valid_flat_idx], axis=-1, base=seg_n_classes
                    )
                    weights = 1 - entropy[..., np.newaxis]
                elif "w2_avg_prob" in self.prob_aggregate:
                    entropy = scipy.stats.entropy(
                        prob_seg_flat[valid_flat_idx], axis=-1, base=seg_n_classes
                    )
                    weights = 1 / entropy[..., np.newaxis]
                else:
                    weights = 1
                self.avg_prob_weight_map[update_cell_idx] += weights
                self.avg_prob_seg_map[update_cell_idx] += (
                    (1 / self.avg_prob_weight_map[update_cell_idx])
                    * (
                        prob_seg_flat[valid_flat_idx]
                        - self.avg_prob_seg_map[update_cell_idx]
                    )
                    * weights
                )
                self.unc_map[update_cell_idx] = scipy.stats.entropy(
                    self.avg_prob_seg_map[update_cell_idx], axis=-1, base=seg_n_classes
                )[..., np.newaxis]
                updated_agg_seg_ids = np.argmax(
                    self.avg_prob_seg_map[update_cell_idx], axis=-1
                )
                updated_agg_seg_ids = self.map_seg_to_schema(updated_agg_seg_ids)
                if "height_map" in self.prob_aggregate:
                    heights = self.height_map[update_cell_idx]
                    update_height_map_mask = (
                        (updated_agg_seg_ids == 0)
                        | (updated_agg_seg_ids == 1)
                        | (updated_agg_seg_ids == 2)
                    )
                    occupied_mask = (heights > 0.1) & update_height_map_mask
                    unoccupied_mask = (heights != -np.inf) * (
                        heights < 0.1
                    ) & update_height_map_mask
                    if self.num_semantic_classes == 3 + 6:
                        updated_agg_seg_ids[occupied_mask] = 2
                        updated_agg_seg_ids[unoccupied_mask] = 1
            elif (
                self.prob_aggregate == "log_odds"
                or self.prob_aggregate == "height_map_log_odds"
            ):
                prob_seg = obs["semantic_prob"].transpose(1, 2, 0)  # (h, w, c)
                seg_n_classes = prob_seg.shape[-1]
                log_odds_seg = np.log(prob_seg / prob_seg[..., 0:1])
                log_odds_seg_flat = log_odds_seg.reshape(-1, log_odds_seg.shape[-1])
                self.log_odds_seg_map[update_cell_idx] += log_odds_seg_flat[
                    valid_flat_idx
                ]  # prior log odds is zero so we don't have to add
                # Calculate the argmax after updating the log odds
                updated_agg_seg_ids = np.argmax(
                    self.log_odds_seg_map[update_cell_idx], axis=-1
                )
                updated_agg_seg_ids = self.map_seg_to_schema(updated_agg_seg_ids)
                if "height_map" in self.prob_aggregate:
                    heights = self.height_map[update_cell_idx]
                    update_height_map_mask = (
                        (updated_agg_seg_ids == 0)
                        | (updated_agg_seg_ids == 1)
                        | (updated_agg_seg_ids == 2)
                    )
                    occupied_mask = (heights > 0.1) & update_height_map_mask
                    unoccupied_mask = (heights != -np.inf) * (
                        heights < 0.1
                    ) & update_height_map_mask
                    if self.num_semantic_classes == 3 + 6:
                        updated_agg_seg_ids[occupied_mask] = 2
                        updated_agg_seg_ids[unoccupied_mask] = 1
                    else:
                        raise RuntimeError(
                            "Invalid num_semantic_classes for height_map_log_odds aggregation"
                        )
            elif self.prob_aggregate == "height_map_hv":
                updated_agg_seg_ids = obs["semantic"].reshape(-1)[valid_flat_idx]
                # we are using argmax values so no need to map to schema as it's implicitly done
                # before in the model
                if self.filter_goal_prob is not None:
                    semantic_prob_flat = obs["semantic_max_prob"].reshape(-1)[
                        valid_flat_idx
                    ]
                    filter_mask = (semantic_prob_flat < self.filter_goal_prob) & (
                        updated_agg_seg_ids == goal_id
                    )
                    updated_agg_seg_ids[filter_mask] = 0
                heights = self.height_map[update_cell_idx]
                update_height_map_mask = (
                    (updated_agg_seg_ids == 0)
                    | (updated_agg_seg_ids == 1)
                    | (updated_agg_seg_ids == 2)
                )
                occupied_mask = (heights > 0.1) & update_height_map_mask
                unoccupied_mask = (heights != -np.inf) * (
                    heights < 0.1
                ) & update_height_map_mask
                if self.num_semantic_classes == 3 + 6:
                    updated_agg_seg_ids[occupied_mask] = 2
                    updated_agg_seg_ids[unoccupied_mask] = 1

                background_mask = updated_agg_seg_ids != goal_id
                self.background_map[
                    update_cell_idx[0][background_mask],
                    update_cell_idx[1][background_mask],
                ] = updated_agg_seg_ids[background_mask]

                update_hit_view_idx = update_cell_idx
                flat_hit_view_idx = valid_flat_idx
                if self.hv_limit_obj_view_dist > 0.0:
                    valid_pos2d = self.map2world(valid_map_points, lower_bound_m)
                    diff = valid_pos2d - np.array(
                        [agent_position[0], agent_position[2]]
                    )
                    grid_distances = np.linalg.norm(diff, axis=-1)
                    view_dist_mask = grid_distances < self.hv_limit_obj_view_dist
                    update_hit_view_idx = (
                        update_hit_view_idx[0][view_dist_mask],
                        update_hit_view_idx[1][view_dist_mask],
                    )
                    flat_hit_view_idx = flat_hit_view_idx[view_dist_mask]

                self.views[update_cell_idx] += 1
                updated_argmax_seg_ids = obs["semantic"].reshape(-1)[flat_hit_view_idx]
                argmax_goal_mask = updated_argmax_seg_ids == goal_id
                current_pred_goal_idx = (
                    update_hit_view_idx[0][argmax_goal_mask],
                    update_hit_view_idx[1][argmax_goal_mask],
                )
                self.hits[current_pred_goal_idx] += 1
                show_goal_mask = (self.hits[update_hit_view_idx] > 0) & (
                    self.views[update_hit_view_idx] != np.inf
                )
                self.semantic_map = self.background_map.copy()
                self.semantic_map[
                    update_hit_view_idx[0][show_goal_mask],
                    update_hit_view_idx[1][show_goal_mask],
                ] = goal_id
            elif self.prob_aggregate == "height_map_prob_hv":
                prob_seg = obs["semantic_prob"].transpose(1, 2, 0)
                self.unc_map[update_cell_idx] = obs["semantic_unc"].reshape(-1, 1)[
                    valid_flat_idx
                ]
                updated_agg_seg_ids = obs["semantic"].reshape(-1)[valid_flat_idx]
                # we are using argmax values so no need to map to schema as it's implicitly done
                # before in the model
                heights = self.height_map[update_cell_idx]
                update_height_map_mask = (
                    (updated_agg_seg_ids == 0)
                    | (updated_agg_seg_ids == 1)
                    | (updated_agg_seg_ids == 2)
                )
                occupied_mask = (heights > 0.1) & update_height_map_mask
                unoccupied_mask = (heights != -np.inf) * (
                    heights < 0.1
                ) & update_height_map_mask
                if self.num_semantic_classes == 3 + 6:
                    updated_agg_seg_ids[occupied_mask] = 2
                    updated_agg_seg_ids[unoccupied_mask] = 1

                background_mask = updated_agg_seg_ids != goal_id
                self.background_map[
                    update_cell_idx[0][background_mask],
                    update_cell_idx[1][background_mask],
                ] = updated_agg_seg_ids[background_mask]

                update_hit_view_idx = update_cell_idx
                flat_hit_view_idx = valid_flat_idx
                if self.hv_limit_obj_view_dist > 0.0:
                    valid_pos2d = self.map2world(valid_map_points, lower_bound_m)
                    diff = valid_pos2d - np.array(
                        [agent_position[0], agent_position[2]]
                    )
                    grid_distances = np.linalg.norm(diff, axis=-1)
                    view_dist_mask = grid_distances < self.hv_limit_obj_view_dist
                    update_hit_view_idx = (
                        update_hit_view_idx[0][view_dist_mask],
                        update_hit_view_idx[1][view_dist_mask],
                    )
                    flat_hit_view_idx = flat_hit_view_idx[view_dist_mask]
                semantic_prob_flat = obs["semantic_max_prob"].reshape(-1)
                self.views[update_hit_view_idx] += semantic_prob_flat[flat_hit_view_idx]
                updated_argmax_seg_ids = obs["semantic"].reshape(-1)[flat_hit_view_idx]
                argmax_goal_mask = updated_argmax_seg_ids == goal_id
                current_pred_goal_idx = (
                    update_hit_view_idx[0][argmax_goal_mask],
                    update_hit_view_idx[1][argmax_goal_mask],
                )
                self.hits[current_pred_goal_idx] += semantic_prob_flat[
                    flat_hit_view_idx[argmax_goal_mask]
                ]
                show_goal_mask = (self.hits[update_hit_view_idx] > 0) & (
                    self.views[update_hit_view_idx] != np.inf
                )
                self.semantic_map = self.background_map.copy()
                self.semantic_map[
                    update_hit_view_idx[0][show_goal_mask],
                    update_hit_view_idx[1][show_goal_mask],
                ] = goal_id
        is_hit_view_agg = self.probablistic_mapping and "_hv" in self.prob_aggregate
        if not is_hit_view_agg:
            self.semantic_map[update_cell_idx] = updated_agg_seg_ids
        self.semantic_map_color[update_cell_idx] = self.color_pallete[
            self.semantic_map[update_cell_idx]
        ]

        robot_pos_px = self.world2map(agent_position, lower_bound_m)
        found_radius_px = int(self.found_dist_m / self.pixel2meter)
        mask_around_robot = cv2.circle(
            np.zeros_like(self.semantic_map, dtype=np.uint8),
            tuple(robot_pos_px[::-1]),
            found_radius_px,
            1,
            -1,
        ).astype(bool)
        self._mask_around_robot = mask_around_robot

        # TODO: found obj logic
        is_found = None
        if is_hit_view_agg:
            mask_around_robot_idx = tuple(
                np.argwhere(
                    mask_around_robot
                    & (self.views > self.hv_min_views)
                    & (self.hits > 0)
                ).T
            )
            views_of_hits_around_robot = self.views[mask_around_robot_idx]
            hit_vals_around_robot = self.hits[mask_around_robot_idx]
            if len(mask_around_robot_idx) > 0:
                is_goal = views_of_hits_around_robot != np.inf
                found_decision = (
                    hit_vals_around_robot / views_of_hits_around_robot
                ) > self.hv_ratio
                if np.any(is_goal):
                    row_idx = mask_around_robot_idx[0][is_goal & ~found_decision]
                    col_idx = mask_around_robot_idx[1][is_goal & ~found_decision]
                    self.views[row_idx, col_idx] = np.inf
                    # cv2.imshow("views", np.uint8(self.views == np.inf) * 255)
                    self.semantic_map[row_idx, col_idx] = self.background_map[
                        row_idx, col_idx
                    ]
                    self.semantic_map_color[row_idx, col_idx] = self.color_pallete[
                        self.background_map[row_idx, col_idx]
                    ]
                if is_found is None:
                    goal_mask = is_goal & found_decision
                    if np.any(goal_mask):
                        is_found = True
                        self.found_idx = mask_around_robot_idx
                        self._found_decision_idx = (
                            mask_around_robot_idx[0][goal_mask],
                            mask_around_robot_idx[1][goal_mask],
                        )
        elif "avg_prob" in self.prob_aggregate:
            seg_n_classes = self.avg_prob_seg_map.shape[-1]
            mask_around_robot_idx = tuple(np.argwhere(mask_around_robot).T)
            goal_mask = self.semantic_map[mask_around_robot_idx] == goal_id
            entropy = scipy.stats.entropy(
                self.avg_prob_seg_map[mask_around_robot_idx][goal_mask],
                axis=-1,
                base=seg_n_classes,
            )
            entropy_mask = entropy < self.entropy_threshold
            if np.any(entropy_mask):
                is_found = True
                self.found_idx = mask_around_robot_idx
                self._found_decision_idx = (
                    mask_around_robot_idx[0][goal_mask][entropy_mask],
                    mask_around_robot_idx[1][goal_mask][entropy_mask],
                )
        elif "log_odds" in self.prob_aggregate:
            seg_n_classes = self.log_odds_seg_map.shape[-1]
            mask_around_robot_idx = tuple(np.argwhere(mask_around_robot).T)
            goal_mask = self.semantic_map[mask_around_robot_idx] == goal_id
            log_odds = self.log_odds_seg_map[mask_around_robot_idx][goal_mask]
            # softmax with scipy
            probs = scipy.special.softmax(log_odds, axis=-1)
            entropy = scipy.stats.entropy(
                probs,
                axis=-1,
                base=seg_n_classes,
            )
            entropy_mask = entropy < self.entropy_threshold
            if np.any(entropy < self.entropy_threshold):
                is_found = True
                self.found_idx = mask_around_robot_idx
                self._found_decision_idx = (
                    mask_around_robot_idx[0][goal_mask][entropy_mask],
                    mask_around_robot_idx[1][goal_mask][entropy_mask],
                )
        else:
            mask_around_robot_idx = tuple(np.argwhere(mask_around_robot).T)
            goal_mask = self.semantic_map[mask_around_robot_idx] == goal_id
            if np.any(goal_mask):
                is_found = True
                self.found_idx = mask_around_robot_idx
                self._found_decision_idx = (
                    mask_around_robot_idx[0][goal_mask],
                    mask_around_robot_idx[1][goal_mask],
                )
        if self.prev_pos_px is not None:
            self.path_mask = cv2.line(
                self.path_mask,
                # Cv2 line takes (y,x) as input
                (self.prev_pos_px[1], self.prev_pos_px[0]),
                (robot_pos_px[1], robot_pos_px[0]),
                1,
                self.path_width_px,
            )
            path_mask = self.path_mask.astype(np.bool)
            self.semantic_map[path_mask] = self.path_id
            self.semantic_map_color[path_mask] = self.path_color
            self.gt_map[path_mask] = self.path_id
        # Plotting agent path

        self.prev_pos_px = robot_pos_px
        ego_maps = self.ego_map(
            lower_bound_m,
            agent_position,
            agent_rotation,
            camera_rotation,
            auxillary_angle,
            goal_id,
        )

        return ego_maps, is_found

    def world2map(
        self, world_coords_m: np.ndarray, world_lower_bound_m: np.ndarray
    ) -> np.array:
        # make the lower bound the origin
        origin_adjusted_coords = world_coords_m - world_lower_bound_m
        # Select only the x and z coordinates and transpose them to match grid coordinates
        uv_world = np.stack(
            [origin_adjusted_coords[..., 2], origin_adjusted_coords[..., 0]], axis=-1
        )

        grid_coord = np.floor(uv_world / self.pixel2meter).astype(np.int64)

        return grid_coord

    def map2world(
        self, map_coords: np.ndarray, world_lower_bound_m: np.ndarray
    ) -> np.array:
        # Select only the x and z coordinates and transpose them to match grid coordinates
        uv_world = (
            np.stack([map_coords[..., 1], map_coords[..., 0]], axis=-1)
            * self.pixel2meter
        )
        world_coords_2d_m = uv_world + world_lower_bound_m[[0, 2]]
        return world_coords_2d_m

    def ego_map(
        self,
        lower_bound_m,
        agent_position,
        agent_rotation,
        camera_rotation,
        aux_angle_onehot,
        goal_id,
    ):
        direction_vector = np.array([0, 0, 1])
        heading_vector = quaternion.rotate_vectors(
            agent_rotation,
            direction_vector,
        )
        phi = cartesian_to_polar(heading_vector[0], heading_vector[2])[1]
        pos = self.world2map(agent_position, lower_bound_m)

        semantic_map_global = self.semantic_map
        semantic_map_color_global = self.semantic_map_color

        # Uncomment for visualizing waypoints
        # semantic_map_color_global = semantic_map_color_global.copy()
        # waypoint_pos = self.world2map(self.waypoint, lower_bound_m)
        # semantic_map_color_global[
        #     waypoint_pos[0] - 10 : waypoint_pos[0] + 10,
        #     waypoint_pos[1] - 10 : waypoint_pos[1] + 10,
        # ] = np.array([255, 0, 0])

        cropped_map = self.crop_fn(
            semantic_map_global,
            center=pos,
            output_size=(
                self.egomap_large_crop[0] + self.offset_for_cut,
                self.egomap_large_crop[1] + self.offset_for_cut,
            ),
        )
        cropped_map_color = self.crop_fn(
            semantic_map_color_global,
            center=pos,
            output_size=(
                self.egomap_large_crop[0] + self.offset_for_cut,
                self.egomap_large_crop[1] + self.offset_for_cut,
            ),
        ).copy()
        cropped_map_color[cropped_map == goal_id] = self.active_goal_color
        w, h, _ = cropped_map_color.shape
        center = (h / 2, w / 2)
        M = cv2.getRotationMatrix2D(center, np.rad2deg(phi) + 90, 1.0)
        ego_map = cv2.warpAffine(
            cropped_map_color,
            M,
            (h, w),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )
        ego_map = self.draw_agent(ego_map)

        if aux_angle_onehot is not None:
            ego_map = self.draw_aux_angles(ego_map, aux_angle_onehot)

        ego_map_small = self.crop_fn(
            ego_map,
            center=(ego_map.shape[0] / 2, ego_map.shape[1] / 2),
            output_size=(self.egomap_small_crop[0], self.egomap_small_crop[1]),
        )
        ego_map_large = self.crop_fn(
            ego_map,
            center=(ego_map.shape[0] / 2, ego_map.shape[1] / 2),
            output_size=(self.egomap_large_crop[0], self.egomap_large_crop[1]),
        )
        ego_map_small = cv2.resize(
            ego_map_small,
            self.egomap_size,
            interpolation=cv2.INTER_AREA,
        )
        ego_map_large = cv2.resize(
            ego_map_large,
            self.egomap_size,
            interpolation=cv2.INTER_AREA,
        )

        if self.probablistic_mapping:
            cropped_unc = self.crop_fn(
                self.unc_map,
                center=pos,
                output_size=(
                    self.egomap_large_crop[0] + self.offset_for_cut,
                    self.egomap_large_crop[1] + self.offset_for_cut,
                ),
            )
            ego_unc = cv2.warpAffine(
                cropped_unc,
                M,
                (h, w),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=-0.5,
            )
            unc_small = self.crop_fn(
                ego_unc,
                center=(ego_unc.shape[0] / 2, ego_unc.shape[1] / 2),
                output_size=(self.egomap_small_crop[0], self.egomap_small_crop[1]),
            )
            unc_large = self.crop_fn(
                ego_unc,
                center=(ego_unc.shape[0] / 2, ego_unc.shape[1] / 2),
                output_size=(self.egomap_large_crop[0], self.egomap_large_crop[1]),
            )
            unc_small = cv2.resize(
                unc_small,
                self.egomap_size,
                interpolation=cv2.INTER_NEAREST,
            )
            unc_large = cv2.resize(
                unc_large,
                self.egomap_size,
                interpolation=cv2.INTER_NEAREST,
            )
        else:
            unc_small = None
            unc_large = None
        return ego_map_small, ego_map_large, unc_small, unc_large

    def crop_fn(self, img: np.ndarray, center, output_size):
        h, w = np.array(output_size, dtype=int)
        x = int(center[0] - h / 2)
        y = int(center[1] - w / 2)
        x1, x2 = x, x + h
        y1, y2 = y, y + w
        if x1 < 0 or y1 < 0 or x2 > img.shape[0] or y2 > img.shape[1]:
            img, x1, x2, y1, y2 = self.pad_img_to_fit_bbox(img, x1, x2, y1, y2)
        return img[x1:x2, y1:y2]

    def pad_img_to_fit_bbox(self, img, x1, x2, y1, y2):
        left = np.abs(np.minimum(0, x1))
        right = np.maximum(x2 - img.shape[0], 0)
        top = np.abs(np.minimum(0, y1))
        bottom = np.maximum(y2 - img.shape[1], 0)
        if len(img.shape) == 2:
            img = np.pad(img, ((left, right), (top, bottom)), mode="constant")
        elif len(img.shape) == 3:
            img = np.pad(img, ((left, right), (top, bottom), (0, 0)), mode="constant")
        else:
            raise ValueError("img.shape must be 2 or 3")
        x1 += left
        x2 += left
        y1 += top
        y2 += top
        return img, x1, x2, y1, y2

    def draw_aux_angles(self, ego_map_color, aux_angle_onehot):
        origin = np.array(
            [ego_map_color.shape[0] / 2, ego_map_color.shape[1] / 2], dtype=np.int32
        )
        bin_side = self.aux_bin_px // 2
        for j, p in enumerate(self.aux_points):
            p = p + origin
            ego_map_color[
                int(p[0]) - bin_side : int(p[0]) + bin_side,
                int(p[1]) - bin_side : int(p[1]) + bin_side,
                :,
            ] = (self.aux_pred_color * aux_angle_onehot[j]).astype(np.uint8)
        return ego_map_color

    def draw_agent(self, ego_map_color):
        origin = np.array(
            [ego_map_color.shape[0] / 2, ego_map_color.shape[1] / 2], dtype=np.float32
        )
        return cv2.circle(
            ego_map_color,
            tuple(origin.astype(np.int32)),
            self.agent_radius_px,
            self.arrow_color,
            -1,
        )

    def compute_fp_fn_detections(self, task_goal_ids, bboxes):
        is_hit_view_agg = self.probablistic_mapping and ("hv" in self.prob_aggregate)
        goal_sem_id = self.goal2sem_id[task_goal_ids][0]
        fn = 0
        goal_obj_bboxes_grid_mask = np.zeros_like(self.semantic_map, dtype=bool)
        is_fn_bbox = []
        goal_obj_bbox_grid_mask = np.zeros_like(self.semantic_map, dtype=bool)
        goal_ob_pred_mask = self.semantic_map == goal_sem_id
        for bbox in bboxes:
            goal_obj_bbox_grid_mask.fill(False)
            pos_1 = self.world2map(bbox.center - bbox.sizes, self.map_bounds[:3])
            pos_2 = self.world2map(bbox.center + bbox.sizes, self.map_bounds[:3])
            pos_1 = (
                np.clip(pos_1[0], 0, self.semantic_map.shape[0] - 1),
                np.clip(pos_1[1], 0, self.semantic_map.shape[1] - 1),
            )
            pos_2 = (
                np.clip(pos_2[0], 0, self.semantic_map.shape[0] - 1),
                np.clip(pos_2[1], 0, self.semantic_map.shape[1] - 1),
            )
            if np.any(
                self.gt_map[pos_1[0] : pos_2[0], pos_1[1] : pos_2[1]] == goal_sem_id
            ):
                goal_obj_bbox_grid_mask[pos_1[0] : pos_2[0], pos_1[1] : pos_2[1]] = True
                goal_obj_bboxes_grid_mask[pos_1[0] : pos_2[0], pos_1[1] : pos_2[1]] = (
                    True
                )
            else:
                # This bounding box wasn't sufficently explored, goal is not mapped to any grid
                # cell
                is_fn_bbox.append(False)
                continue
            if is_hit_view_agg:
                if np.any(
                    (self.hits > 0) & (self.views != np.inf) & goal_obj_bbox_grid_mask
                ):
                    is_fn_bbox.append(False)
                    continue
                else:
                    fn += 1
                    is_fn_bbox.append(True)
            else:
                if np.any(goal_ob_pred_mask & goal_obj_bbox_grid_mask):
                    is_fn_bbox.append(False)
                    continue
                else:
                    fn += 1
                    is_fn_bbox.append(True)
        fp_mask = ~goal_obj_bboxes_grid_mask & goal_ob_pred_mask
        # aggregate the gridcells into connected components
        # dilate the mask to connect the grid cells
        dilate_size = int(0.1 / self.pixel2meter)
        fp_mask = cv2.dilate(
            fp_mask.astype(np.uint8),
            cv2.getStructuringElement(cv2.MORPH_RECT, (dilate_size, dilate_size)),
        )
        connected_comp, fp = scipy.ndimage.label(
            fp_mask, scipy.ndimage.generate_binary_structure(2, 2)
        )
        return fp, fn, is_fn_bbox

    def _erode_goal_2d(self, update_cell_idx, updated_agg_seg_ids, goal_id):
        if np.any(updated_agg_seg_ids == goal_id):
            # take the 2d tuple update_cell_idx and updated_agg_seg_ids, create a local map
            min_x, max_x = min(update_cell_idx[0]), max(update_cell_idx[0])
            min_y, max_y = min(update_cell_idx[1]), max(update_cell_idx[1])

            local_map = np.zeros(
                (max_x - min_x + 1, max_y - min_y + 1), dtype=np.uint32
            )
            # Erode the goal to remove noise in the localmap
            local_map[update_cell_idx[0] - min_x, update_cell_idx[1] - min_y] = (
                updated_agg_seg_ids
            )
            # local_map_rgb = self.color_pallete[local_map]
            # cv2.imshow("local_map", local_map_rgb[:, :, ::-1])

            local_goal_mask = local_map == goal_id
            # cv2.imshow("local_goal_mask", local_goal_mask.astype(np.uint8) * 255)
            erosion_size = int(math.ceil(0.1 / self.pixel2meter))

            objgoal_map = cv2.erode(
                local_goal_mask.astype(np.uint8),
                np.ones(
                    ((erosion_size, erosion_size)),
                    dtype=np.uint8,
                ),
                iterations=1,
            )
            dilate_size = int(math.ceil(0.2 / self.pixel2meter))
            kernel = np.ones((dilate_size, dilate_size), dtype=np.uint8)
            kernel_tensor = torch.from_numpy(
                np.expand_dims(np.expand_dims(kernel, 0), 0)
            )
            objgoal_map = torch.nn.functional.conv2d(
                torch.from_numpy(np.expand_dims(objgoal_map, 0)),
                kernel_tensor,
                padding="same",
            )
            objgoal_map = torch.clamp(objgoal_map, 0, 1)[0].numpy().astype(bool)

            # cv2.imshow(
            #     "local_eroded_goal_mask", local_eroded_goal_mask.astype(np.uint8) * 255
            # )
            valid_mask = objgoal_map | (local_map != goal_id) & (local_map != 0)
            local_idx = np.argwhere(valid_mask)
            updated_agg_seg_ids = local_map[valid_mask]
            update_cell_idx = (local_idx[:, 0] + min_x, local_idx[:, 1] + min_y)
        return update_cell_idx, updated_agg_seg_ids


def softmax(x, axis=0):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / e_x.sum(axis=axis, keepdims=True)  # only difference


def group_max_mask(groups, data):
    order = np.lexsort((data, groups))
    groups = groups[order]  # this is only needed if groups is unsorted
    data = data[order]
    index = np.empty(len(groups), "bool")
    index[-1] = True
    index[:-1] = groups[1:] != groups[:-1]
    reverse_order = np.argsort(order)
    return index[reverse_order]


def draw_arrow(image, pixel_coords, color, size=10):
    # points: [center, forward, left, right]
    norm = lambda p: p / np.linalg.norm(p)
    c = pixel_coords[0]
    f = norm(pixel_coords[1] - pixel_coords[0]) * (size * 2) + pixel_coords[0]
    l = norm(pixel_coords[2] - pixel_coords[0]) * (size * 2) + pixel_coords[0]
    r = norm(pixel_coords[3] - pixel_coords[0]) * (size * 2) + pixel_coords[0]
    pts = np.asarray(
        [[f[1], f[0]], [l[1], l[0]], [c[1], c[0]], [r[1], r[0]]], dtype=np.int32
    )
    return cv2.fillPoly(
        image, [pts], color=(int(color[0]), int(color[1]), int(color[2]))
    )


# Function to find cluster centers of boolean mask and draw circles with value 1
def set_circles_on_clusters(mask, radius):
    int_mask = mask.astype(np.uint8)
    contours, _ = cv2.findContours(int_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        # Simplify the contour
        # epsilon = 0.01 * cv2.arcLength(cnt, True)
        # approx = cv2.approxPolyDP(cnt, epsilon, True)

        M = cv2.moments(cnt)

        if M["m00"] == 0:
            continue

        center_x = int(M["m10"] / M["m00"])
        center_y = int(M["m01"] / M["m00"])
        cv2.circle(int_mask, (center_x, center_y), radius, 1, -1)

    return int_mask.astype(np.bool)


def get_colormap(n):
    def bitget(byteval, idx):
        return (byteval & (1 << idx)) != 0

    cmap = np.zeros((n, 3), dtype="uint8")
    for i in range(n):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7 - j)
            g = g | (bitget(c, 1) << 7 - j)
            b = b | (bitget(c, 2) << 7 - j)
            c = c >> 3
        cmap[i] = np.array([r, g, b])
    return cmap

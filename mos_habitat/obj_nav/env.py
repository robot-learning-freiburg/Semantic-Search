from __future__ import annotations

import csv
import logging
import math
import os
from collections import OrderedDict, deque
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import gymnasium as gym
import habitat
import habitat_sim
import numpy as np
from gymnasium import spaces
from habitat.core.environments import RLTaskEnv
from habitat.datasets.registration import make_dataset
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
from habitat_sim.utils.common import quat_rotate_vector
from matplotlib import pyplot as plt
from PIL import Image
from scipy.special import softmax
from scipy.stats import circvar
from sem_objnav.obj_nav.mapper import SemanticMapper
from sem_objnav.segment.infer import (
    EmsanetWrapper,
    MaskRCNNWrapper,
    OneformerWrapper,
    SegformerWrapper,
)

logger = logging.getLogger(__name__)

os.environ["MAGNUM_LOG"] = "quiet"
os.environ["HABITAT_SIM_LOG"] = "quiet"


class RenderOptions(Enum):
    rgb = "rgb"
    map_large = "map_large"
    map_small = "map_small"
    map_global = "map_global"
    write_metrics = "write_metrics"
    depth = "depth"
    semantic = "semantic"


class MappingObjNavEnv(gym.Env):
    metadata: dict[str, Any] = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        config,
        scenes: List[str],
        split: str,
        seed: int = 42,
        shuffle: bool = True,
        max_scene_repeat_episodes: int = -1,
        use_aux_angle: str = "curriculum",
        render_mode: str = "rgb_array",
        render_options: Tuple[str] = (
            "rgb",
            "map_large",
            "map_small",
            "write_metrics",
        ),
        gpu_device_id: int = 0,
        raise_found_action: bool = True,
        exclude_goal_class_episodes: List[str] | None = None,
        shortest_path_policy: bool = False,
    ):
        super().__init__()
        h_config_path = Path(__file__).parent.absolute() / "cfg" / "h_config.yaml"
        h_config = habitat.get_config(str(h_config_path))
        assert use_aux_angle in [
            "curriculum",
            "pred",
            "gt_debug",
            "gt",
        ], 'use_aux_angle must be one of ["curriculum", "pred", "gt_debug", "gt"]'

        with habitat.config.read_write(h_config):
            h_config.habitat.seed = seed
            h_config.habitat.environment.iterator_options.shuffle = shuffle
            h_config.habitat.environment.iterator_options.max_scene_repeat_episodes = (
                max_scene_repeat_episodes
            )
            h_config.habitat.dataset.split = split
            h_config.habitat.dataset.content_scenes = scenes
            h_config.habitat.task.actions.velocity_control_fixed.lin_vel_range = (
                config.lin_vel_range
            )
            h_config.habitat.simulator.habitat_sim_v0.gpu_device_id = gpu_device_id
            # This might be essential for the shortest path planner step size (which is planned in discrete action space) to match our continuous action space step size and rotation
            h_config.habitat.simulator.forward_step_size = (
                config.lin_vel_range[1]
                * h_config.habitat.task.actions.velocity_control_fixed.time_step
            )
            h_config.habitat.simulator.turn_angle = int(
                np.rad2deg(
                    h_config.habitat.task.actions.velocity_control_fixed.ang_vel_range[
                        1
                    ]
                    * h_config.habitat.task.actions.velocity_control_fixed.time_step
                )
            )
        self.h_config = h_config

        dataset = make_dataset(
            id_dataset=h_config.habitat.dataset.type, config=h_config.habitat.dataset
        )
        if exclude_goal_class_episodes is not None and exclude_goal_class_episodes:
            dataset = dataset.filter_episodes(
                lambda ep: all(
                    [
                        goal_class not in ep.goals_key
                        for goal_class in exclude_goal_class_episodes
                    ]
                )
            )
        self._env = RLTaskEnv(config=h_config, dataset=dataset)
        depth_sensor_config = (
            h_config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor
        )
        self.segmentation_model = None
        if config.semantic_model.model_type is not None:
            if config.semantic_model.model_type == "emsanet":
                self.segmentation_model = EmsanetWrapper(
                    ckpt_path=config.semantic_model.ckpt,
                    config_path=config.semantic_model.config,
                    map_schema=config.label_schema_col,
                    mode=config.semantic_model.mode,
                    temperature=config.semantic_model.temperature,
                    device=f"cuda:{gpu_device_id}",
                )
            elif config.semantic_model.model_type == "oneformer":
                self.segmentation_model = OneformerWrapper(
                    config.label_schema_col,
                    device=f"cuda:{gpu_device_id}",
                    mode=config.semantic_model.mode,
                )
            elif config.semantic_model.model_type == "segformer":
                self.segmentation_model = SegformerWrapper(
                    config.label_schema_col,
                    device=f"cuda:{gpu_device_id}",
                    temperature=config.semantic_model.temperature,
                )
            elif config.semantic_model.model_type == "maskrcnn":
                self.segmentation_model = MaskRCNNWrapper(
                    config.label_schema_col, f"cuda:{gpu_device_id}"
                )
            else:
                raise NotImplementedError(
                    f"Semantic model {config.semantic_model.model_type} not implemented"
                )
        self.mapping = SemanticMapper(
            num_semantic_classes=config.num_semantic_classes,
            depth_range=[depth_sensor_config.min_depth, depth_sensor_config.max_depth],
            depth_hfov=depth_sensor_config.hfov,
            agent_draw_radius_m=h_config.habitat.simulator.agents.main_agent.radius
            + config.mapping.pixel2meter,
            probablistic_mapping=self.segmentation_model is not None,
            success_dist_m=0.9,  # hardcoded, this overcomes mapping
            # discretization mismatches
            prob_aggregate=config.semantic_model.prob_aggregate,
            seg_model_n_classes=(
                self.segmentation_model.num_model_classes
                if self.segmentation_model
                else None
            ),
            model_name=config.semantic_model.model_type,
            **config.mapping,
        )
        if self.segmentation_model:
            self.mapping.map_seg_to_schema = self.segmentation_model.map_seg_to_schema
            self.mapping.objnav_to_sem_pred = self.segmentation_model.objnav2pred
        observation_space = OrderedDict()

        self.current_aux_episodic_prob = config.curriculum.start_aux_prob
        self.max_aux_episodic_prob = config.curriculum.max_aux_prob

        self.history_length_aux = config.history_length_aux
        auxillary_history_size = (
            self.history_length_aux + 1  # +1 for the current prediction/ground truth
        ) * self.mapping.aux_bin_number
        self.original_observation_space = self._env.observation_space
        self.original_action_space = self._env.action_space
        assert "velocity_control" in self.original_action_space.spaces
        self.num_actions = 2  # For velocity control
        self.auto_found_action = (
            config.auto_found_action if hasattr(config, "auto_found_action") else True
        )
        if not self.auto_found_action:
            self.num_actions += 1
        observation_space["task_obs"] = spaces.Box(
            shape=(
                6  # num goal classes
                + 1  # collision boolean
                + 1  # aux angle circular variance
                + 1  # Number of collisions
                + 2  # x, z variance
                + auxillary_history_size
                + self.num_actions,
            ),
            low=-np.inf,
            high=np.inf,
        )
        observation_space["map_small"] = spaces.Box(
            low=0,
            high=255,
            shape=(3, *self.mapping.egomap_size),
            dtype=np.uint8,
        )
        observation_space["map_large"] = spaces.Box(
            low=0,
            high=255,
            shape=(3, *self.mapping.egomap_size),
            dtype=np.uint8,
        )
        self.use_uncertainty_obs = False
        if config.semantic_model is not None and config.semantic_model.use_uncertainty:
            self.use_uncertainty_obs = True
            observation_space["unc_map_small"] = spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(1, *self.mapping.egomap_size),
                dtype=np.float32,
            )
            observation_space["unc_map_large"] = spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(1, *self.mapping.egomap_size),
                dtype=np.float32,
            )

        self.observation_space = spaces.Dict(observation_space)
        action_space = OrderedDict(
            {
                "action": spaces.Box(
                    low=-1.0, high=1.0, shape=(self.num_actions,), dtype=np.float32
                ),
                "aux_angle": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(self.mapping.aux_bin_number,),
                    dtype=np.float32,
                ),
            }
        )
        self.action_space = spaces.Dict(action_space)
        self.aux_prob_counter = 0
        self.curr_success_tracker = deque(
            maxlen=config.curriculum.success_history_length
        )
        self.curr_success_rate = config.curriculum.success_rate
        self.curr_min_episodes = config.curriculum.min_episodes
        self.curr_min_episodes_for_change = config.curriculum.min_episodes_for_change

        self._cached_render_data = {}
        self.episode_counter = 0
        self.use_aux_angle = use_aux_angle
        self.render_mode = render_mode
        self.render_options = [RenderOptions(v) for v in render_options]

        current_dir = Path(__file__).parent
        with (
            current_dir.parent / "segment/hm3d/matterport_category_mappings.tsv"
        ).open() as f:
            rows = csv.DictReader(f, delimiter="\t")
            label_name2id = {}
            for row in rows:
                try:
                    idx = int(row[config.label_schema_col])
                except ValueError:
                    continue
                label_name2id[row["raw_category"]] = idx
                label_name2id[row["category"]] = idx
        self.schema_label_name2id = label_name2id
        self.raise_found_action = raise_found_action
        if shortest_path_policy:
            self._sp_follower = ShortestPathFollower(
                self._env.habitat_env.sim, 0.0, False
            )
            self._sp_viewpoint_pos = None
            self._sp_goal_pos = None
        else:
            self._sp_follower = None
            self._sp_viewpoint_pos = None
            self._sp_goal_pos = None

    def reset_variables(self):
        self.aux_prob_counter += 1
        self.current_step = 0
        self.episode_counter += 1
        self.aux_pred_reset = np.zeros(self.mapping.aux_bin_number) + (
            1 / self.mapping.aux_bin_number
        )
        hist_len = max(self.history_length_aux, 16)
        self.aux_angle_track = deque([], maxlen=hist_len)
        self.prev_locations = deque([], maxlen=hist_len)
        self.collision_tracker = deque([0] * hist_len, maxlen=hist_len)
        self.prev_aux_predictions = deque(
            [np.zeros(self.mapping.aux_bin_number)] * self.history_length_aux,
            maxlen=self.history_length_aux,
        )
        self._sp_viewpoint_pos = None
        self._sp_goal_pos = None
        self._sp_fp_found = False
        self._sp_fn_found = False
        # no need to reset sp_follower as it resets by itself if scene changes

    def get_running_metrics(self):
        return {
            "aux_episodic_prob": self.current_aux_episodic_prob,
            "aux_prob_counter": self.aux_prob_counter,
            "episode_counter": self.episode_counter,
            "success_tracker": self.curr_success_tracker,
        }

    def set_running_metrics(self, params):
        self.current_aux_episodic_prob = params["aux_episodic_prob"]
        self.aux_prob_counter = params["aux_prob_counter"]
        self.episode_counter = params["episode_counter"]
        self.curr_success_tracker = params["success_tracker"]

    def reset(
        self, seed: int | None = None, options: dict[str:Any] | None = None
    ) -> Dict[str, np.ndarray]:
        super().reset(seed=seed, options=options)
        if seed is not None:
            self._env.seed(seed)
        self.reset_variables()
        self._cached_render_data = {}
        obs = self._env.reset()

        self.reachable_goals_bbox = []
        self.is_fn_bbox = []
        sim = self._env.habitat_env.sim
        agent_pos = sim.get_agent_state().position
        agent_island = sim.pathfinder.get_island(agent_pos)
        map_semantic_annotations = self._env.habitat_env.sim.semantic_annotations()
        id2bbox = {obj.id: obj.aabb for obj in map_semantic_annotations.objects}
        for g in self._env.habitat_env.current_episode.goals:
            for vp in g.view_points:
                if sim.pathfinder.get_island(vp.agent_state.position) == agent_island:
                    self.reachable_goals_bbox.append(id2bbox[g.object_name])
                    self.is_fn_bbox.append(False)
                    break

        instance_id_to_semantic_id = {
            int(obj.id.split("_")[-1]): self.schema_label_name2id.get(
                obj.category.name().strip().lower(), 0
            )
            for obj in map_semantic_annotations.objects
        }
        self.instance2semantic = np.array(
            [
                instance_id_to_semantic_id[i]
                for i in range(max(instance_id_to_semantic_id.keys()) + 1)
            ]
        )

        obs = self._process_obs(obs)

        lower_bound_m, upper_bound_m = self._env.habitat_env.sim.pathfinder.get_bounds()
        map_bounds = np.concatenate([lower_bound_m, upper_bound_m])

        self.mapping.reset(map_bounds)
        # curicullum learning
        if self.use_aux_angle == "pred":
            self.episodic_use_aux_prediction = True
        elif self.use_aux_angle == "gt_debug" or self.use_aux_angle == "gt":
            self.episodic_use_aux_prediction = False
        elif self.use_aux_angle == "curriculum":
            self.episodic_use_aux_prediction = (
                self.np_random.uniform() < self.current_aux_episodic_prob
            )
            if (
                self.aux_prob_counter > self.curr_min_episodes_for_change
                and self.episode_counter > self.curr_min_episodes
                and np.mean(np.array(self.curr_success_tracker))
                > self.curr_success_rate
            ):
                self.current_aux_episodic_prob = min(
                    self.max_aux_episodic_prob, self.current_aux_episodic_prob + 0.02
                )
                self.aux_prob_counter = 0
        # self.mapping.waypoint = self._env._env.task.waypoint
        (
            ego_map_local,
            ego_map_global,
            unc_map_local,
            unc_map_global,
        ), self._is_found = self.mapping.update_map(obs, None)

        aux_uniform_rand_bin_vec = np.zeros(self.mapping.aux_bin_number) + (
            1 / self.mapping.aux_bin_number
        )
        self._cached_render_data.update(
            {
                "depth": obs["depth"],
                "map_small": ego_map_local,
                "map_large": ego_map_global,
                "goal": obs["objectgoal"][0],
                "reward": 0,
                "is_collision": False,
                "aux_prob": self.current_aux_episodic_prob,
                "aux_episode": self.episodic_use_aux_prediction,
                "scene": self._env.habitat_env.sim.curr_scene_name,
                "episode_id": self._env.habitat_env.current_episode.episode_id,
                "action": [0.0] * self.num_actions,
                "unc_map_small": unc_map_local,
                "unc_map_large": unc_map_global,
            }
        )
        if "semantic" in obs:
            self._cached_render_data["semantic"] = obs["semantic"]
        if "rgb" in obs:
            self._cached_render_data["rgb"] = obs["rgb"]
        one_hot_goal = np.zeros(6)
        one_hot_goal[obs["objectgoal"]] = 1.0
        task_obs = np.concatenate(
            [
                aux_uniform_rand_bin_vec,
                np.array(self.prev_aux_predictions).flatten(),
                [0.0, 0.0, 0.0],
                np.array([0.0]),
                np.array([0.0]),
                np.zeros((self.num_actions,)),
                one_hot_goal,
            ]
        )
        obs = {
            "map_small": ego_map_local.transpose(2, 0, 1),
            "map_large": ego_map_global.transpose(2, 0, 1),
            "task_obs": task_obs,
        }
        if self.use_uncertainty_obs:
            obs["unc_map_small"] = unc_map_local[np.newaxis, ...]
            obs["unc_map_large"] = unc_map_global[np.newaxis, ...]

        # easier to compute it here
        max_episodes = {}
        for ep in self._env.habitat_env.episodes:
            scene_id = ep.scene_id.split("/")[-1].split(".")[0]
            if scene_id not in max_episodes:
                max_episodes[scene_id] = 0
            max_episodes[scene_id] += 1

        return obs, {"max_episodes": max_episodes}

    def _process_obs(self, obs):
        obs["depth"] = obs["depth"].squeeze()
        obs["semantic"][obs["semantic"] > self.instance2semantic.shape[0] - 1] = 0
        obs["semantic"][obs["semantic"] < 0] = 0
        obs["semantic"] = self.instance2semantic[obs["semantic"]].squeeze()
        obs["rgb"] = obs["rgb"].squeeze()
        model_out = {}
        if self.segmentation_model is not None:
            model_out = self.segmentation_model.predict(
                obs["rgb"].astype(np.uint8), obs["depth"] * 1e3  # 1e3 to convert to mm
            )
            semantic_gt = obs["semantic"]
            obs["semantic"] = model_out["semantic"]
            if "semantic_prob" in model_out:
                obs["semantic_prob"] = model_out["semantic_prob"]
                obs["semantic_unc"] = model_out["semantic_unc"]
                obs["logits"] = model_out["logits"]
                self._cached_render_data["semantic_prob"] = model_out["semantic_prob"]
                self._cached_render_data["semantic_unc"] = model_out["semantic_unc"]
            if "semantic_max_prob" in model_out:
                obs["semantic_max_prob"] = model_out["semantic_max_prob"]
            obs["semantic_gt"] = semantic_gt

            self._cached_render_data["semantic_gt"] = semantic_gt

        self._cached_render_data["semantic"] = obs["semantic"]
        self._cached_render_data["rgb"] = obs["rgb"]
        self._cached_render_data["depth"] = obs["depth"]
        if self.mapping.record_traj:
            self.recorded_unc_map = None
            self.found_idxes = []
        return obs

    def step(self, action):
        self._cached_render_data = {}
        self.current_step += 1
        task_action = {
            # always have velocity_control_fixed at the end,
            # or the collision data gets overwritten
            "action": ("velocity_stop", "velocity_control_fixed"),
            "action_args": {
                "linear_velocity": action["action"][0],
                "angular_velocity": action["action"][1],
                # "camera_pitch_angular_velocity": np.array([-1.0]),
            },
        }
        if self.auto_found_action:
            task_action["action_args"]["velocity_stop"] = np.array(
                [1.0 if self._is_found and self.raise_found_action else -1.0]
            )
        else:
            task_action["action_args"]["velocity_stop"] = np.array(
                [action["action"][2]]
            )

        obs, reward, done, info = self._env.step(task_action)
        if np.isnan(reward) or np.isinf(reward):
            reward = 0.0
            done = True
        obs = self._process_obs(obs)
        aux_pred_action = softmax(action["aux_angle"])

        angle_2_nearest_waypoint_gt_deg = obs["angle_to_nearest_object_waypoint"]
        # To map angle +- bin_size/2 we add bin_size/2 to the bin representing the angle
        gt_bin_deg = (
            angle_2_nearest_waypoint_gt_deg + self.mapping.bin_size // 2
        ) % 360

        gt_bin = np.digitize(
            [gt_bin_deg],
            self.mapping.angle_to_bins,
            right=True,
        )
        # mimic a soft distribution when providing ground truth angle
        gt_aux_bin_vec = np.zeros(self.mapping.aux_bin_number)
        gt_aux_bin_vec[gt_bin] = np.random.uniform(1.0, 5.0)
        gt_aux_bin_vec = softmax(gt_aux_bin_vec)

        if self.episodic_use_aux_prediction:
            angle_2_nearest_waypoint_pred_deg = (
                np.argmax(aux_pred_action) * 360 / len(aux_pred_action)
            )
            self.aux_angle_track.append(np.deg2rad(angle_2_nearest_waypoint_pred_deg))
        else:
            self.aux_angle_track.append(np.deg2rad(angle_2_nearest_waypoint_gt_deg))
        agent_pos = obs["agent_position_and_rotation"][:3]
        self.prev_locations.append(agent_pos)
        angle_variance = circvar(self.aux_angle_track)
        var0 = np.var(np.array(self.prev_locations)[:, 2])
        var1 = np.var(np.array(self.prev_locations)[:, 0])
        is_collision = info["collisions"]["is_collision"]
        # gt_debug is only used for debugging, in training and evaluation we always only
        # imprint the aux prediction on the map
        (
            ego_map_local,
            ego_map_global,
            unc_map_local,
            unc_map_global,
        ), self._is_found = self.mapping.update_map(
            obs, gt_aux_bin_vec if self.use_aux_angle == "gt_debug" else aux_pred_action
        )

        if self._sp_follower is not None:
            # raise found only when we reach the viewpoint and see the object
            # in the
            # override the distance based found decision
            distance_to_vp = self._env.habitat_env.sim.geodesic_distance(
                self._sp_viewpoint_pos, agent_pos
            )
            success_distance = self._env.habitat_env.task.measurements.measures[
                "success"
            ]._success_distance
            true_found_decision = self._is_found
            self._is_found = False
            task_goal_ids = obs["objectgoal"]
            goal_id = self.mapping.goal2sem_id[task_goal_ids][0]
            fp_found = False
            if true_found_decision:
                lower_bound_m = self.mapping.map_bounds[:3]
                robot_pos_px = self.mapping.world2map(agent_pos, lower_bound_m)
                found_radius_px = int(
                    self.mapping.found_dist_m / self.mapping.pixel2meter
                )
                mask_around_robot = cv2.circle(
                    np.zeros_like(self.mapping.semantic_map, dtype=np.uint8),
                    tuple(robot_pos_px[::-1]),
                    found_radius_px,
                    1,
                    -1,
                ).astype(bool)
                if not np.any(self.mapping.gt_map[mask_around_robot] == goal_id):
                    self._sp_fp_found = True
                    fp_found = True

            if distance_to_vp < success_distance:
                self._sp_fn_found = (
                    np.any(self.mapping.gt_map == goal_id)
                    and not true_found_decision
                    and not self._sp_fp_found
                )
                if np.any(self.mapping.gt_map == goal_id):
                    self._is_found = True

        self.collision_tracker.append(int(is_collision))
        one_hot_goal = np.zeros(6)
        one_hot_goal[obs["objectgoal"]] = 1.0
        task_obs = np.concatenate(
            [
                aux_pred_action if self.episodic_use_aux_prediction else gt_aux_bin_vec,
                np.array(self.prev_aux_predictions).flatten(),
                [angle_variance, var0, var1],
                np.array([int(is_collision)]),
                np.array([sum(self.collision_tracker) / len(self.collision_tracker)]),
                action["action"],
                one_hot_goal,
            ]
        )
        self.prev_aux_predictions.append(aux_pred_action)

        info = {
            "goal": obs["objectgoal"][0],
            "aux_episode": self.episodic_use_aux_prediction,
            "aux_prob": self.current_aux_episodic_prob,
            "true_aux": gt_bin.squeeze(),
            "collision_count": info["collisions"]["count"],
            "success": info["success"],
            "spl": info["spl"],
            "scene": self._env.habitat_env.sim.curr_scene_name,
            "episode_id": self._env.habitat_env.current_episode.episode_id,
        }

        if self.mapping.record_traj and true_found_decision and done:
            if self.recorded_unc_map is None:
                self.recorded_unc_map = np.zeros_like(self.mapping.unc_map)
            # Hacky code to get the trajectory visualization
            goal_sem_id = self.mapping.goal2sem_id[obs["objectgoal"]][0]
            goal_name = ["chair", "bed", "plant", "toilet", "tv", "sofa"][
                obs["objectgoal"][0]
            ]
            global_map = self.mapping.semantic_map_color.copy()
            global_map[self.mapping.semantic_map == goal_sem_id] = [0, 255, 0]
            global_map[self.mapping.semantic_map == 0] = [255, 255, 255]
            found_idx = self.mapping._found_decision_idx

            gt_global_map = self.mapping.color_pallete[self.mapping.gt_map]
            gt_global_map[self.mapping.gt_map == goal_sem_id] = [0, 255, 0]
            gt_global_map[self.mapping.gt_map == 0] = [255, 255, 255]

            diff = cv2.absdiff(global_map, gt_global_map)
            diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            _, diff = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY)
            # make the diff such hat differences are in red and the background is in cream
            diff = cv2.cvtColor(diff, cv2.COLOR_GRAY2BGR)
            diff[diff[..., 0] == 255] = [255, 0, 0]
            diff[diff[..., 0] == 0] = [255, 255, 255]

            new_unc_map = np.zeros_like(self.mapping.unc_map)

            new_unc_map[self.mapping.semantic_map == goal_sem_id] = (
                self.mapping.unc_map[self.mapping.semantic_map == goal_sem_id]
            )
            if found_idx is not None:
                new_unc_map[found_idx[0], found_idx[1]] = self.mapping.unc_map[
                    found_idx[0], found_idx[1]
                ]

                self.recorded_unc_map[found_idx[0], found_idx[1]] = (
                    self.mapping.unc_map[found_idx[0], found_idx[1]]
                )
                # if self._sp_fp_found:
                #     self.found_idxes.append(found_idx)
                #     for f_idx in self.found_idxes:
                #         min_x = min(f_idx[0])
                #         max_x = max(f_idx[0])
                #         min_y = min(f_idx[1])
                #         max_y = max(f_idx[1])
                #         # draw a rectangle around the found object using cv2
                #         global_map = cv2.rectangle(
                #             global_map,
                #             (min_y, min_x),
                #             (max_y, max_x),
                #             (0, 0, 255),
                #             2,
                #         )
                # else:
                #     # global_map[found_idx[0], found_idx[1]] = [102, 255, 102]
                #     min_x = min(found_idx[0])
                #     max_x = max(found_idx[0])
                #     min_y = min(found_idx[1])
                #     max_y = max(found_idx[1])
                #     # draw a rectangle around the found object using cv2
                #     global_map = cv2.rectangle(
                #         global_map, (min_y, min_x), (max_y, max_x), (102, 255, 102), 2
                #     )

            # gt_global_map[
            #     self.mapping._mask_around_robot & (self.mapping.gt_map == goal_sem_id)
            # ] = [204, 255, 153]
            goal_sem_id = self.mapping.goal2sem_id[obs["objectgoal"]][0]
            goal_name = ["chair", "bed", "plant", "toilet", "tv", "sofa"][
                obs["objectgoal"][0]
            ]
            info["traj"] = {
                "goal_name": goal_name,
                "global_map": global_map,
                "gt_global_map": gt_global_map,
                "step": self.current_step,
                "recorded_unc_map": self.recorded_unc_map,
                "current_unc_map": new_unc_map,
                "current_unc_map_full": self.mapping.unc_map,
                "diff": diff,
            }

        if done:
            info["fp_detections"], info["fn_detections"], self.is_fn_bbox = (
                self.mapping.compute_fp_fn_detections(
                    obs["objectgoal"], self.reachable_goals_bbox
                )
            )
            info["fn_found"] = (
                self._sp_fn_found
                if self._sp_follower
                else (not info["success"] and self.current_step == 1000)
            )
            info["fp_found"] = (
                self._sp_fp_found
                if self._sp_follower
                else (not info["success"] and self.current_step < 1000)
            )
            info["success"] = not info["fn_found"] and not info["fp_found"]
            # if self.mapping.record_traj:
            #     # Hacky code to get the trajectory visualization
            #     goal_sem_id = self.mapping.goal2sem_id[obs["objectgoal"]][0]
            #     goal_name = ["chair", "bed", "plant", "toilet", "tv", "sofa"][
            #         obs["objectgoal"][0]
            #     ]
            #     global_map = self.mapping.semantic_map_color.copy()
            #     global_map[self.mapping.semantic_map == goal_sem_id] = [255, 0, 0]
            #     global_map[self.mapping.semantic_map == 0] = [255, 255, 255]
            #     found_idx = self.mapping._found_decision_idx

            #     gt_global_map = self.mapping.color_pallete[self.mapping.gt_map]
            #     gt_global_map[self.mapping.gt_map == goal_sem_id] = [255, 0, 0]
            #     gt_global_map[self.mapping.gt_map == 0] = [255, 255, 255]

            #     if found_idx is not None:
            #         global_map[found_idx[0], found_idx[1]] = [0, 255, 255]
            #     gt_global_map[
            #         self.mapping._mask_around_robot
            #         & (self.mapping.gt_map == goal_sem_id)
            #     ] = [0, 255, 255]

            #     info["traj"] = {
            #         "goal_name": goal_name,
            #         "global_map": global_map,
            #         "gt_global_map": gt_global_map,
            #     }
            if self.mapping.collect_nb_data:
                info["nb_data"] = {
                    "goal": (
                        np.concatenate(self.mapping.eps_nb_data["goal"], axis=0)
                        if len(self.mapping.eps_nb_data["goal"]) > 0
                        else self.mapping.eps_nb_data["goal"]
                    ),
                    "non_goal": (
                        np.concatenate(self.mapping.eps_nb_data["non_goal"], axis=0)
                        if len(self.mapping.eps_nb_data["non_goal"]) > 0
                        else self.mapping.eps_nb_data["non_goal"]
                    ),
                    "goal_id": self.mapping.eps_nb_data["goal_id"],
                }

        self._cached_render_data.update(
            {
                "map_small": ego_map_local,
                "map_large": ego_map_global,
                "reward": reward,
                "is_collision": is_collision,
                "action": action["action"],
                "unc_map_small": unc_map_local,
                "unc_map_large": unc_map_global,
                **info,
            }
        )
        if done:
            print("Success", info["success"])
            self.curr_success_tracker.append(int(info["success"]))

        obs = {
            "map_small": ego_map_local.transpose(2, 0, 1),
            "map_large": ego_map_global.transpose(2, 0, 1),
            "task_obs": task_obs,
        }
        if self.use_uncertainty_obs:
            obs["unc_map_small"] = unc_map_local[np.newaxis, ...]
            obs["unc_map_large"] = unc_map_global[np.newaxis, ...]

        return (
            obs,
            reward,
            done,
            False,
            info,
        )

    def close(self):
        self._env.close()

    def render(self):
        # shallow copy to avoid modifying the original
        render_data = self._cached_render_data
        img_to_render = []
        if RenderOptions.rgb in self.render_options:
            img = np.array(Image.fromarray(render_data["rgb"], mode="RGB"))
            img_to_render.append(img)
        if RenderOptions.semantic in self.render_options:
            semantic = render_data["semantic"]
            semantic_color_obs = self.mapping.color_pallete[semantic]

            img = np.array(Image.fromarray(semantic_color_obs, mode="RGB"))
            img_to_render.append(img)

            if "semantic_prob" in self._cached_render_data:
                semantic_unc = self._cached_render_data["semantic_prob"]
                prob_img = np.array(
                    Image.fromarray(
                        (semantic_unc.max(0) * 255).astype(np.uint8), mode="L"
                    ).convert("RGB")
                )
                img_to_render.append(prob_img)
            if "semantic_unc" in self._cached_render_data:
                uncertainty = self._cached_render_data["semantic_unc"]
                uncertainty = (uncertainty * 255).astype(np.uint8)
                unc_img = np.array(
                    Image.fromarray(uncertainty, mode="L").convert("RGB")
                )
                img_to_render.append(unc_img)
            if "semantic_gt" in self._cached_render_data:
                semantic_gt = self._cached_render_data["semantic_gt"]
                semantic_gt_color_obs = self.mapping.color_pallete[semantic_gt]
                img = np.array(Image.fromarray(semantic_gt_color_obs, mode="RGB"))
                img_to_render.append(img)

        if RenderOptions.depth in self.render_options:
            depth_obs = render_data["depth"]
            depth_range = self.mapping.depth_range
            # Normalizing for visualization
            depth_obs = (depth_obs - depth_range[0]) / (depth_range[1] - depth_range[0])

            # Normalizing for visualization
            depth_img = Image.fromarray((depth_obs * 255).astype(np.uint8), mode="L")
            img = np.array(depth_img.convert("RGB"))
            img_to_render.append(img)
        if RenderOptions.map_global in self.render_options:
            global_map = self.mapping.color_pallete[self.mapping.semantic_map]
            if self._sp_follower is not None:
                lower_bound_m, _ = self._env.habitat_env.sim.pathfinder.get_bounds()
                # goal_vps = self.mapping.world2map(self.goal_vps, lower_bound_m)
                # circle_px = math.ceil(0.1 / self.mapping.pixel2meter)
                # for x, y in goal_vps:
                #     cv2.circle(global_map, (y, x), circle_px, (255, 0, 0), -1)

                agent_position = self._env.habitat_env.sim.get_agent_state().position
                agent_position = self.mapping.world2map(agent_position, lower_bound_m)
                circle_radius_px = int(
                    self.mapping.found_dist_m / self.mapping.pixel2meter
                )
                cv2.circle(
                    global_map,
                    (agent_position[1], agent_position[0]),
                    circle_radius_px,
                    (0, 255, 0),
                    1,
                )
                for bbox, is_fn in zip(self.reachable_goals_bbox, self.is_fn_bbox):
                    pos_1 = self.mapping.world2map(
                        bbox.center - bbox.sizes, lower_bound_m
                    )
                    pos_2 = self.mapping.world2map(
                        bbox.center + bbox.sizes, lower_bound_m
                    )
                    global_map = cv2.rectangle(
                        global_map,
                        tuple(pos_1[::-1]),
                        tuple(pos_2[::-1]),
                        (0, 0, 255) if is_fn else (255, 0, 0),
                        2,
                    )

            # img = np.array(Image.fromarray(global_map, mode="RGB"))
            # cv2.imshow("global", resize_and_pad(global_map[..., ::-1], (768, 1024)))
            cv2.imshow(
                "global",
                resize_and_pad(
                    global_map[..., ::-1],
                    (global_map.shape[1] // 2, global_map.shape[0] // 2),
                ),
            )

            if self.mapping.unc_map is not None:
                cmap = plt.get_cmap("viridis")
                unc_map = self.mapping.unc_map.copy()
                unc_map[unc_map < 0] = 0
                rgba_img = cmap(unc_map)
                rgb_img = np.delete(rgba_img.squeeze(), 3, 2) * 255
                cv2.imshow(
                    "Unc",
                    resize_and_pad(rgb_img.astype(np.uint8)[..., ::-1], (768, 1024)),
                )

        if RenderOptions.map_small in self.render_options:
            img = np.array(Image.fromarray(render_data["map_small"], mode="RGB"))
            img_to_render.append(img)

        if RenderOptions.map_large in self.render_options:
            img = np.array(Image.fromarray(render_data["map_large"], mode="RGB"))
            img_to_render.append(img)

        padded_images = [resize_and_pad(img, (256, 256)) for img in img_to_render]
        render_rgb_array = np.concatenate(padded_images, axis=1)
        if RenderOptions.write_metrics in self.render_options:
            self._write_metrics_on_image(render_rgb_array, render_data)

        if self.render_mode == "human":
            cv2.imshow("Observation", render_rgb_array[:, :, [2, 1, 0]])
            cv2.waitKey(1)
        else:
            return render_rgb_array

    def _write_metrics_on_image(self, rgb_array, render_data):
        goal_name = ["chair", "bed", "plant", "toilet", "tv", "sofa"][
            render_data["goal"]
        ]
        texts = [
            (f"reward: {render_data['reward']:.4f}", (255, 255, 255)),
            (f"aux_prob: {render_data['aux_prob']:.2f}", (255, 255, 255)),
            (f"aux_episode: {render_data['aux_episode']}", (255, 255, 255)),
            (f"scene {render_data['scene']}", (255, 255, 255)),
            (f"action {list(render_data['action'])}", (255, 255, 255)),
            (f"goal {goal_name}", (255, 255, 255)),
        ]
        if render_data["is_collision"]:
            texts.append(("collision", (255, 0, 0)))

        for row, (text, color) in enumerate(texts):
            cv2.putText(
                rgb_array,
                text,
                (10, 20 + row * 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )
        return rgb_array

    def shortest_path_policy(self):
        sim = self._env.habitat_env.sim
        if self._sp_viewpoint_pos is None:
            episode = self._env.habitat_env.current_episode
            # self._sp_goal_pos = episode._shortest_path_cache.points[-1]

            episode_viewpoints = []
            vp_goals = []
            agent_pos = sim.get_agent_state().position
            for goal in episode.goals:
                # take viewpoints that are navigable from the current agent position
                vps = np.array(
                    [
                        vp.agent_state.position
                        for vp in goal.view_points
                        if sim.pathfinder.get_island(vp.agent_state.position)
                        == sim.pathfinder.get_island(agent_pos)
                    ]
                )
                if len(vps) > 0:
                    # pick viewpoint closest to the goal
                    min_dist_idx = np.linalg.norm(vps - goal.position, axis=1).argmin()
                    episode_viewpoints.append(vps[min_dist_idx].astype(np.float32))
                    vp_goals.append(goal.position)
            agent_state = sim.get_agent_state()
            path = habitat_sim.MultiGoalShortestPath()
            path.requested_start = agent_state.position
            path.requested_ends = episode_viewpoints
            found_path = sim.pathfinder.find_path(path)
            self._sp_viewpoint_pos = path.points[-1]
            self._sp_goal_pos = vp_goals[path.closest_end_point_index]
            # print(path)

            # for vp, g in zip(episode_viewpoints, vp_goals):
            #     if np.all(vp == self._sp_viewpoint_pos):
            #         self._sp_goal_pos = g
            #         break
            # if self._sp_goal_pos is None:
            #     print("||| vp", self._sp_viewpoint_pos, episode_viewpoints)

        sp_act = self._sp_follower.get_next_action(self._sp_viewpoint_pos)

        action = np.array([-1.0, 0.0])

        if sp_act == HabitatSimActions.move_forward:
            action[0] = 1.0
        elif sp_act == HabitatSimActions.turn_left:
            action[1] = 1.0
        elif sp_act == HabitatSimActions.turn_right:
            action[1] = -1.0
        else:
            # and rotate towards goal
            ags = sim.get_agent_state()
            agent_pos = ags.position
            agent_rot = ags.rotation
            goal_dir = self._sp_goal_pos - agent_pos
            goal_dir = goal_dir / np.linalg.norm(goal_dir)
            agent_dir = quat_rotate_vector(agent_rot, np.array([0, 0, -1]))
            cross = np.cross(agent_dir, goal_dir)
            if cross[2] > 0:
                action[1] = 1.0
            else:
                action[1] = -1.0
        return action


def resize_and_pad(image, shape_out):
    """
    Resizes an image to the specified size,
    adding padding to preserve the aspect ratio.
    """
    if image.ndim == 3 and len(shape_out) == 2:
        shape_out = [*shape_out, 3]
    hw_out, hw_image = [np.array(x[:2]) for x in (shape_out, image.shape)]
    resize_ratio = np.min(hw_out / hw_image)
    hw_wk = (hw_image * resize_ratio + 1e-5).astype(int)

    # Resize the image
    resized_image = cv2.resize(
        image, tuple(hw_wk[::-1]), interpolation=cv2.INTER_NEAREST
    )
    if np.all(hw_out == hw_wk):
        return resized_image

    # Create a black image with the target size
    padded_image = np.zeros(shape_out, dtype=np.uint8)

    # Calculate the number of rows/columns to add as padding
    dh, dw = (hw_out - hw_wk) // 2
    # Add the resized image to the padded image, with padding on the left and right sides
    padded_image[dh : hw_wk[0] + dh, dw : hw_wk[1] + dw] = resized_image

    return padded_image

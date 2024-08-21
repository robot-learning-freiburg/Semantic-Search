import logging
import os
import queue
import random
import warnings
from collections import Counter, defaultdict, deque
from typing import Any, Callable, Dict, Optional, Union

import gymnasium as gym
import numpy as np
import torch
from sem_objnav.sb3.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.logger import Video
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.vec_env import VecEnv, sync_envs_normalization
from stable_baselines3.common.vec_env.base_vec_env import tile_images

logger = logging.getLogger(__name__)


def linear_schedule(initial: float, final: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """

    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        return progress_remaining * initial + (1 - progress_remaining) * final

    return func


class VideoRecorderCallback(BaseCallback):
    """
    Callback for recording videos of the trained agents.
    Uses numpy style docstrings.

    Args:
        video_length (int): The length of each recorded video in environment steps.
        video_freq (int): The frequency of recordings in environment steps.
        video_fps (int): The frames per second rate of the recorded videos.
        num_envs_to_record (int): The number of environments to record videos from.
        verbose (int, optional): Verbosity level. 0 = silent, 1 = info. Defaults to 0.
    """

    def __init__(
        self,
        video_length: int,
        video_freq: int,
        video_fps: int,
        num_envs_to_record: int,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose)
        self.video_length = video_length
        self.video_freq = video_freq
        self.video_fps = video_fps
        self.num_envs_to_record = num_envs_to_record
        self.video_frames = []
        self.recording_envs = []

    def _init_callback(self) -> None:
        self.video_freq = (
            self.video_freq // self.training_env.num_envs
        ) * self.training_env.num_envs

    def _on_step(self) -> bool:
        """
        Records videos at specified intervals.

        Returns:
            bool: Always returns True.
        """
        # Use self.num_timesteps always instead of n_calls because num_timesteps is set to right value after loading a checkpoint
        if self.num_timesteps != 0 and self.num_timesteps % self.video_freq == 0:
            self.video_frames = []
            self.recording_envs = random.sample(
                range(self.training_env.num_envs), self.num_envs_to_record
            )
        if self.recording_envs and len(self.video_frames) < self.video_length:
            screens = self.training_env.env_method(
                "render", indices=self.recording_envs
            )
            frame = tile_images(screens)
            self.video_frames.append(frame.transpose(2, 0, 1))
            if len(self.video_frames) == self.video_length:
                self.logger.record(
                    "video/train",
                    Video(
                        torch.ByteTensor(np.array([self.video_frames])),
                        fps=self.video_fps,
                    ),
                    exclude=("stdout", "log", "json", "csv"),
                )
                self.logger.dump(step=self.num_timesteps)
                self.video_frames = []
                self.recording_envs = []
        return True


class MOSMetricsLoggerCheckpointerCallback(BaseCallback):
    def __init__(
        self,
        total_train_scenes: int,
        save_freq: int,
        save_model_dir: str = "./",
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.save_path = os.path.join(save_model_dir, "model_{}")
        self.save_path_last = os.path.join(save_model_dir, "last_model")
        self.save_freq = save_freq
        self.total_train_scenes = total_train_scenes
        self._log_metrics = {
            "episodes": Counter(),
            "aux_pred_episodes": Counter(),
            "aux_gt_episodes": Counter(),
            "all_scene_success_gt": {},
            "all_scene_success_pred": {},
        }
        self._aux_accuracy_ep_tracker = defaultdict(list)

    def _init_callback(
        self,
    ) -> None:
        if self.model.env_params:  # checkpoint parameters
            self._log_metrics = self.model.env_params["logger_metrics"]
            for i in range(self.training_env.num_envs):
                self.training_env.env_method(
                    "set_running_metrics",
                    self.model.env_params["env_metrics"][i],
                    indices=i,
                )

    def _on_step(self) -> bool:
        """
        :return: If the callback returns False, training is aborted early.
        """
        dones = self.locals["dones"]
        infos = self.locals["infos"]
        aux_angle = np.argmax(self.locals["aux_angle"], axis=1)
        aux_angle_gt = self.locals[
            "aux_angle_gt"
        ].squeeze()  # (bs, 1) -> (bs,) Really important for correct accuracy calculation
        aux_eq = (aux_angle == aux_angle_gt).astype(np.float32)
        for i, inf in enumerate(infos):
            scene_name = inf["scene"]
            self._aux_accuracy_ep_tracker[scene_name].append(aux_eq[i])
            if dones[i]:
                self._log_metrics["episodes"][scene_name] += 1
                self.logger.record("rollout/aux_prob", inf["aux_prob"])
                self.logger.record(
                    f"scene_metrics/aux_prob/{scene_name}", inf["aux_prob"]
                )
                self.logger.record(
                    "rollout/episodes", sum(self._log_metrics["episodes"].values())
                )
                self.logger.record(
                    f"scene_metrics/episodes/{scene_name}",
                    self._log_metrics["episodes"][scene_name],
                )
                if inf["aux_episode"]:
                    self._log_metrics["aux_pred_episodes"][scene_name] += 1
                    self._log_metrics["all_scene_success_pred"][scene_name] = int(
                        inf["success"]
                    )
                    self.logger.record(
                        "rollout/aux_pred_episodes",
                        sum(self._log_metrics["aux_pred_episodes"].values()),
                    )
                    self.logger.record(
                        f"scene_metrics/aux_pred_episodes/{scene_name}",
                        self._log_metrics["aux_pred_episodes"][scene_name],
                    )
                    if (
                        len(self._log_metrics["all_scene_success_pred"])
                        == self.total_train_scenes
                    ):
                        all_scene_success = np.mean(
                            list(self._log_metrics["all_scene_success_pred"].values())
                        )
                        self.logger.record(
                            "rollout/all_scene_success_pred",
                            all_scene_success,
                        )
                        self._log_metrics["all_scene_success_pred"] = {}

                else:
                    self._log_metrics["aux_gt_episodes"][scene_name] += 1
                    self._log_metrics["all_scene_success_gt"][scene_name] = int(
                        inf["success"]
                    )

                    self.logger.record(
                        "rollout/aux_gt_episodes",
                        sum(self._log_metrics["aux_gt_episodes"].values()),
                    )
                    self.logger.record(
                        f"scene_metrics/aux_gt_episodes/{scene_name}",
                        self._log_metrics["aux_gt_episodes"][scene_name],
                    )
                    if (
                        len(self._log_metrics["all_scene_success_gt"])
                        == self.total_train_scenes
                    ):
                        all_scene_success = np.mean(
                            list(self._log_metrics["all_scene_success_gt"].values())
                        )
                        self.logger.record(
                            "rollout/all_scene_success_gt",
                            all_scene_success,
                        )
                        self._log_metrics["all_scene_success_gt"] = {}

                metrics = {
                    "ep_return": inf["episode"]["r"],
                    "ep_length": inf["episode"]["l"],
                    "ep_collision_cnt": inf["collision_count"],
                    "spl": inf["spl"],
                    "ep_success": 1.0 if inf["success"] else 0.0,
                    "angle_acc": np.mean(self._aux_accuracy_ep_tracker[scene_name]),
                }
                self._aux_accuracy_ep_tracker[scene_name] = []
                for root_key in ["rollout", "scene_metrics"]:
                    if root_key == "scene_metrics":
                        suffix = f"/{scene_name}"
                    else:
                        suffix = ""
                    for metric_key, metric_value in metrics.items():
                        self.logger.record(
                            f"{root_key}/{metric_key}{suffix}", metric_value
                        )
                        if inf["aux_episode"]:
                            self.logger.record(
                                f"{root_key}/aux_pred_{metric_key}{suffix}",
                                metric_value,
                            )
                        else:
                            self.logger.record(
                                f"{root_key}/aux_gt_{metric_key}{suffix}", metric_value
                            )
                # we want to log each episode separately
                self.logger.dump(step=self.num_timesteps - len(dones) + i + 1)

        if (self.n_calls + 1) % (self.save_freq // self.training_env.num_envs) == 0:
            self.model.save(self.save_path.format(self.num_timesteps))
            self.model.save(self.save_path_last)
        return True

    def _on_rollout_end(self) -> None:
        self.model.env_params = {
            "env_metrics": self.training_env.env_method("get_running_metrics"),
            "logger_metrics": self._log_metrics,
        }

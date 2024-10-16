#!/usr/bin/env python3

import csv
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import attr
import habitat_sim
import magnum as mn
import numpy as np
import quaternion
from gym import spaces
from habitat.config import DictConfig
from habitat.core.dataset import Dataset, Episode
from habitat.core.embodied_task import EmbodiedTask, Measure, SimulatorTaskAction
from habitat.core.logging import logger
from habitat.core.registry import registry
from habitat.core.simulator import (
    AgentState,
    SemanticSensor,
    Sensor,
    SensorTypes,
    Simulator,
    VisualObservation,
)
from habitat.core.spaces import ActionSpace
from habitat.tasks.nav.nav import Collisions, DistanceToGoal, Success, VelocityAction
from habitat_sim import RigidState
from habitat_sim.physics import VelocityControl
from sem_objnav.obj_nav.utils import suppress_stdout_stderr


@registry.register_sensor
class AngleToNearestObjectWaypointSensor(Sensor):
    r"""This sensor gives the ground truth value of the angle to the nearest
    waypoint in the geodesic to the nearest unvisited goal. The angle is discretized
    into the number of bins specified by the config. The angle is computed in the agent's
    frame of reference.
    Args:
        sim: a reference to the simulator for calculating task observations.
        config: a config for the AngleToNearestObjectWaypointMeasure measure.
        dataset: a Object Goal navigation dataset that contains dictionaries
        of categories id to text mapping.
    """

    cls_uuid: str = "angle_to_nearest_object_waypoint"

    def __init__(
        self,
        sim: Simulator,
        config: "DictConfig",
        dataset: Dataset,
        *args: Any,
        **kwargs: Any,
    ):
        self._sim = sim
        self._dataset = dataset
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return self.cls_uuid

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.TENSOR

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=np.array([0.0], dtype=np.float32),
            high=np.array([359.0], dtype=np.float32),
            shape=(1,),
            dtype=np.float32,
        )

    def get_observation(
        self,
        observations,
        *args: Any,
        task,
        episode,
        **kwargs: Any,
    ) -> Optional[int]:

        agent_state = self._sim.get_agent_state()
        if episode._shortest_path_cache is None:
            episode_view_points = [
                view_point.agent_state.position
                for goal in episode.goals
                for view_point in goal.view_points
            ]

            path = habitat_sim.MultiGoalShortestPath()
            path.requested_start = agent_state.position
            path.requested_ends = episode_view_points
            found_path = self._sim.pathfinder.find_path(path)
        else:
            # Use the cached path computed by the
            # distance reward measure
            path = episode._shortest_path_cache
            found_path = True
        if not found_path:
            logger.warning(
                f"No path found to goal scene: {episode.scene_id} episode: {episode.episode_id} pos: {agent_state.position}, returning random angle"
            )
            return np.random.rand() * 359.0
        else:
            if len(path.points) <= 1:
                # when the agent is close to the goal, we return angle to the goal instead of the next waypoint
                nearest_goal_idx = np.linalg.norm(
                    np.array([g.position for g in episode.goals])
                    - (
                        path.points[-1]
                        if len(path.points) > 1
                        else agent_state.position
                    ),
                    2,
                    axis=-1,
                ).argmin()
                waypoint = episode.goals[nearest_goal_idx].position
            else:
                waypoint = path.points[1]
            agent_position = agent_state.position
            agent_rotation = agent_state.rotation * quaternion.from_rotation_vector(
                np.array([0, np.pi, 0])
            )
            waypoint_local = quaternion.rotate_vectors(
                agent_rotation.inverse(), waypoint - agent_position
            )
            x, y = waypoint_local[0], waypoint_local[2]
            phi = np.arctan2(x, y)
            if phi < 0:
                phi += 2 * np.pi
            angle = np.rad2deg(phi)
            task.waypoint = waypoint  # Used for crude debugging
            return angle


@registry.register_sensor
class AgentPositionAndRotationSensor(Sensor):
    r"""This sensor gives the ground truth value of the robot's absolute position rotation.
    Args:
        sim: a reference to the simulator for calculating task observations.
        config: a config for the AbsolutePositionAndRotationSensor measure.
        dataset: a Object Goal navigation dataset that contains dictionaries
        of categories id to text mapping.
    """

    cls_uuid: str = "agent_position_and_rotation"

    def __init__(
        self,
        sim: Simulator,
        config: "DictConfig",
        dataset: Dataset,
        *args: Any,
        **kwargs: Any,
    ):
        self._sim = sim
        self._dataset = dataset
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return self.cls_uuid

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.TENSOR

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=np.array(
                [-np.inf, -np.inf, -np.inf, -1, -1, -1, -1],
                dtype=np.float32,
            ),
            high=np.array(
                [np.inf, np.inf, np.inf, 1.0, 1.0, 1.0, 1.0],
                dtype=np.float32,
            ),
            shape=(7,),
            dtype=np.float32,
        )

    def get_observation(
        self,
        observations,
        *args: Any,
        task,
        episode,
        **kwargs: Any,
    ) -> Optional[int]:
        agent_state = self._sim.get_agent_state()
        position = agent_state.position
        rotation = quaternion.as_float_array(agent_state.rotation)
        return np.concatenate([position, rotation])


@registry.register_sensor
class CameraPositionAndRotationSensor(Sensor):
    r"""This sensor gives the ground truth value of the robot's absolute position rotation.
    Args:
        sim: a reference to the simulator for calculating task observations.
        config: a config for the AbsolutePositionAndRotationSensor measure.
        dataset: a Object Goal navigation dataset that contains dictionaries
        of categories id to text mapping.
    """

    cls_uuid: str = "camera_position_and_rotation"

    def __init__(
        self,
        sim: Simulator,
        config: "DictConfig",
        dataset: Dataset,
        *args: Any,
        **kwargs: Any,
    ):
        self._sim = sim
        self._dataset = dataset
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return self.cls_uuid

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.TENSOR

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=np.array(
                [-np.inf, -np.inf, -np.inf, -1, -1, -1, -1],
                dtype=np.float32,
            ),
            high=np.array(
                [np.inf, np.inf, np.inf, 1.0, 1.0, 1.0, 1.0],
                dtype=np.float32,
            ),
            shape=(7,),
            dtype=np.float32,
        )

    def get_observation(
        self,
        observations,
        *args: Any,
        task,
        episode,
        **kwargs: Any,
    ) -> Optional[int]:
        agent_state = self._sim.get_agent_state()
        camera_state = agent_state.sensor_states["depth"]
        position = camera_state.position
        rotation = quaternion.as_float_array(camera_state.rotation)
        return np.concatenate([position, rotation])


@registry.register_measure
class ObjNavReward(Measure):
    """Measure for computing the per-step reward of the MultiObjectSearch task"""

    cls_uuid: str = "objnav_reward"

    def __init__(self, *args: Any, sim: Simulator, config: "DictConfig", **kwargs: Any):
        self._sim = sim
        self._config = config

        super().__init__()

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return self.cls_uuid

    def reset_metric(
        self, *args: Any, episode, task, **kwargs: Any
    ):  ##Called only when episode begins
        task.measurements.check_measure_dependencies(
            self.uuid,
            [Success.cls_uuid, "collisions", DistanceToGoal.cls_uuid],
        )
        self._previous_distance = task.measurements.measures[
            DistanceToGoal.cls_uuid
        ].get_metric()
        self._metric = 0
        self.update_metric(*args, episode=episode, task=task, **kwargs)

    def update_metric(self, *args: Any, episode, task: EmbodiedTask, **kwargs: Any):
        success = task.measurements.measures[Success.cls_uuid].get_metric()
        collisions = task.measurements.measures["collisions"].get_metric()
        distance_to_target = task.measurements.measures[
            DistanceToGoal.cls_uuid
        ].get_metric()
        delta = -(distance_to_target - self._previous_distance)
        if np.isinf(distance_to_target):
            print(
                f"Warning distance to target is large in reward: episode {episode.episode_id}, scene {episode.scene_id}"
            )
        self._previous_distance = distance_to_target
        is_collision = collisions["is_collision"]
        reward = 0.0
        reward += delta * self._config.delta_distance_reward_weight
        if is_collision:
            reward += self._config.collision_penalty
        if success:
            reward += self._config.task_success_reward
        reward += self._config.step_penalty
        self._metric = reward


@registry.register_task_action
class VelocityControlFixedAction(VelocityAction):
    name: str = "velocity_control_fixed"

    def _apply_velocity_action(
        self,
        linear_velocity: float,
        angular_velocity: float,
        time_step: Optional[float] = None,
    ):
        """
        Apply velocity command to simulation, step simulation, and return agent observation
        """
        # Parse inputs
        if time_step is None:
            time_step = self._time_step

        allow_sliding = self._sim.config.sim_cfg.allow_sliding

        self.vel_control.linear_velocity = np.array([0.0, 0.0, -linear_velocity])
        self.vel_control.angular_velocity = np.array([0.0, angular_velocity, 0.0])
        agent_state = self._sim.get_agent_state()

        # Convert from np.quaternion (quaternion.quaternion) to mn.Quaternion
        normalized_quaternion = agent_state.rotation
        agent_mn_quat = mn.Quaternion(
            normalized_quaternion.imag, normalized_quaternion.real
        )
        current_rigid_state = RigidState(
            agent_mn_quat,
            agent_state.position,
        )
        # snap rigid state to navmesh and set state to object/agent
        if allow_sliding:
            # uses the simulator's pathfinder which doesn't inflate the navmesh
            step_fn = self._sim.pathfinder.try_step  # type: ignore
        else:
            step_fn = self._sim.pathfinder.try_step_no_sliding  # type: ignore

        # manually integrate the rigid state
        physics_time_step = 0.01
        while time_step > 0:
            time_to_integrate = min(time_step, physics_time_step)

            goal_rigid_state = self.vel_control.integrate_transform(
                time_to_integrate, current_rigid_state
            )
            time_step -= time_to_integrate

            final_position = step_fn(
                current_rigid_state.translation, goal_rigid_state.translation
            )

            # Check if a collision occured
            dist_moved_before_filter = (
                goal_rigid_state.translation - current_rigid_state.translation
            ).dot()
            dist_moved_after_filter = (
                final_position - current_rigid_state.translation
            ).dot()

            # NB: There are some cases where ||filter_end - end_pos|| > 0 when a
            # collision _didn't_ happen. One such case is going up stairs.  Instead,
            # we check to see if the the amount moved after the application of the
            # filter is _less_ than the amount moved before the application of the
            # filter.
            EPS = 5e-6
            collided = (dist_moved_after_filter + EPS) < dist_moved_before_filter
            if collided:
                break
            current_rigid_state = goal_rigid_state
        final_rotation = [
            *goal_rigid_state.rotation.vector,
            goal_rigid_state.rotation.scalar,
        ]

        # TODO: Make a better way to flag collisions
        self.collided = collided  # type: ignore

        # Update the state of the agent
        self._sim.set_agent_state(  # type: ignore
            final_position, final_rotation, reset_sensors=False
        )

        final_agent_state = self._sim.get_agent_state()
        final_agent_state.position = final_position
        final_agent_state.rotation = goal_rigid_state.rotation

        return final_agent_state

    def _get_agent_observation(self, agent_state=None):
        position = None
        rotation = None

        if agent_state is not None:
            position = agent_state.position
            rotation = [
                *agent_state.rotation.vector,
                agent_state.rotation.scalar,
            ]

        obs = self._sim.get_observations_at(
            position=position,
            rotation=rotation,
            keep_agent_at_new_pose=True,
        )
        self._sim._prev_sim_obs["collided"] = self.collided
        return obs


@registry.register_measure
class VelocityMeasure(Measure):
    r"""Calculates the episode length"""

    cls_uuid: str = "velocity"

    def __init__(self, *args: Any, sim: Simulator, config: "DictConfig", **kwargs: Any):
        self._episode_length = None
        self._sim = sim
        self._config = config

        super().__init__()
        self.previous_position = None
        self.previous_rotation = None

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return self.cls_uuid

    def reset_metric(self, *args: Any, episode, task, **kwargs: Any):
        self._episode_length = 0
        self._metric = self._episode_length
        agent_state = self._sim.get_agent_state()
        self.previous_position = agent_state.position
        self.previous_rotation = agent_state.rotation
        self.previous_time = self._sim.get_world_time()
        self.update_metric(*args, episode=episode, task=task, **kwargs)

    def update_metric(
        self,
        *args: Any,
        episode,
        task: EmbodiedTask,
        **kwargs: Any,
    ):
        self._episode_length += 1
        self._metric = self._episode_length

        agent_state = self._sim.get_agent_state()

        current_time = self._sim.get_world_time()
        current_position = agent_state.position
        current_rotation = agent_state.rotation
        delta_time = current_time - self.previous_time
        if delta_time == 0.0:
            linear_velocity_x = 0.0
            angular_velocity_yaw = 0.0
        else:
            # calculate linear velocity along the forward direction
            linear_velocity = (current_position - self.previous_position) / delta_time
            # We have xzy order, and the forward direction is -y
            linear_velocity_x = -quaternion.rotate_vectors(
                current_rotation.conjugate(), linear_velocity
            )[2]
            # roll, yaw, pitch
            angular_velocity = quaternion.angular_velocity(
                [self.previous_rotation, current_rotation], [0, delta_time]
            )[-1]
            angular_velocity_yaw = quaternion.rotate_vectors(
                current_rotation.conjugate(), angular_velocity
            )[1]

        self.previous_position = current_position
        self.previous_rotation = current_rotation
        self.previous_time = current_time
        self._metric = {
            "linear": linear_velocity_x,
            "angular": angular_velocity_yaw,
        }

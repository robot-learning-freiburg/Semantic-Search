from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import attr
from habitat.config.default_structured_configs import (
    ActionConfig,
    LabSensorConfig,
    MeasurementConfig,
    TaskConfig,
    VelocityControlActionConfig,
)
from hydra.core.config_store import ConfigStore


@dataclass
class VelocityMeasurementConfig(MeasurementConfig):
    type: str = "VelocityMeasure"


@attr.s(auto_attribs=True, slots=True)
class VelocityControlFixedActionConfig(ActionConfig):
    type: str = "VelocityControlFixedAction"
    lin_vel_range: List[float] = [0.0, 0.3]  # meters/sec
    ang_vel_range: List[float] = [-0.45, 0.45]  # rad/sec
    ang_vel_range_camera_pitch: List[float] = [-0.45, 0.45]  # rad/sec
    ang_range_camera_pitch: List[float] = [-1.57, 0.43]  # rad
    time_step: float = 0.1  # seconds
    enable_scale_convert: bool = True


@dataclass
class AngleToNearestObjectWaypointSensorConfig(LabSensorConfig):
    type: str = "AngleToNearestObjectWaypointSensor"


@dataclass
class AgentPositionAndRotationSensorConfig(LabSensorConfig):
    type: str = "AgentPositionAndRotationSensor"


@dataclass
class CameraPositionAndRotationSensorConfig(LabSensorConfig):
    type: str = "CameraPositionAndRotationSensor"


@dataclass
class ObjNavRewardMeasureConfig(MeasurementConfig):
    type: str = "ObjNavReward"
    delta_distance_reward_weight: float = 1.0
    collision_penalty: float = -0.1
    task_success_reward: float = 10.0
    step_penalty: float = -0.0025


cs = ConfigStore.instance()

cs.store(
    package="habitat.task.actions.velocity_control_fixed",
    group="habitat/task/actions",
    name="velocity_control_fixed",
    node=VelocityControlFixedActionConfig,
)

cs.store(
    package="habitat.task.lab_sensors.agent_position_and_rotation_sensor",
    group="habitat/task/lab_sensors",
    name="agent_position_and_rotation_sensor",
    node=AgentPositionAndRotationSensorConfig,
)

cs.store(
    package="habitat.task.lab_sensors.camera_position_and_rotation_sensor",
    group="habitat/task/lab_sensors",
    name="camera_position_and_rotation_sensor",
    node=CameraPositionAndRotationSensorConfig,
)
cs.store(
    package="habitat.task.lab_sensors.angle_to_nearest_object_waypoint_sensor",
    group="habitat/task/lab_sensors",
    name="angle_to_nearest_object_waypoint_sensor",
    node=AngleToNearestObjectWaypointSensorConfig,
)

cs.store(
    package="habitat.task.measurements.objnav_reward",
    group="habitat/task/measurements",
    name="objnav_reward",
    node=ObjNavRewardMeasureConfig,
)
cs.store(
    package="habitat.task.measurements.velocity_measure",
    group="habitat/task/measurements",
    name="velocity_measure",
    node=VelocityMeasurementConfig,
)

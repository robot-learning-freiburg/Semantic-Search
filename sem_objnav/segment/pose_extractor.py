import collections
from typing import List, Optional, Tuple, Union

import numpy as np
import scipy.stats
from habitat_sim import registry as registry
from habitat_sim.utils.common import quat_from_angle_axis
from habitat_sim.utils.data.pose_extractor import (
    PanoramaExtractor,
    PoseExtractor,
    TopdownView,
)
from numpy import float32, float64, ndarray


@registry.register_pose_extractor(name="robot_extractor")
class RobotPanaromaExtractor(PanoramaExtractor):
    def __init__(
        self,
        topdown_views: List[Tuple[TopdownView, str, Tuple[float32, float32, float32]]],
        meters_per_pixel: float = 0.1,
        camera_tilt_bounds_deg: Tuple[float, float] = (-45, 20.0),
        camera_height_m_bounds: Tuple[float, float] = (1.096, 1.496),
    ) -> None:
        super().__init__(topdown_views, meters_per_pixel)
        self.camera_tilt_bounds_deg = camera_tilt_bounds_deg
        self.camera_tilt_mean_deg = (
            camera_tilt_bounds_deg[1] + camera_tilt_bounds_deg[0]
        ) / 2
        self.camera_tilt_std_deg = (
            camera_tilt_bounds_deg[1] - camera_tilt_bounds_deg[0]
        ) / 4
        self.camera_height_m_bounds = camera_height_m_bounds
        self.rnd = np.random.RandomState(42)

    def extract_all_poses(self) -> np.ndarray:
        r"""Returns a numpy array of camera poses. For each scene, this method extends the list of poses according to the extraction rule defined in extract_poses."""
        poses = super().extract_all_poses()
        lower = self.camera_tilt_bounds_deg[0]
        upper = self.camera_tilt_bounds_deg[1]
        mu = self.camera_tilt_mean_deg
        sigma = self.camera_tilt_std_deg
        sample_angles = scipy.stats.truncnorm.rvs(
            (lower - mu) / sigma,
            (upper - mu) / sigma,
            loc=mu,
            scale=sigma,
            size=len(poses),
        )
        new_poses = []
        for pose, sample_angle in zip(poses, sample_angles):
            (pos_t, rot_t, filepath) = pose
            new_rot = rot_t * quat_from_angle_axis(
                np.deg2rad(sample_angle), np.array([1, 0, 0])
            )
            new_pos = pos_t + np.array(
                [0, self.rnd.uniform(*self.camera_height_m_bounds), 0]
            )
            new_poses.append((new_pos, new_rot, filepath))
        return np.array(new_poses)

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable, List, Optional, Tuple, Union

import habitat_sim
import magnum as mn
import numpy as np
from habitat_sim import bindings as hsim
from habitat_sim import registry as registry
from habitat_sim.agent.agent import AgentConfiguration, AgentState
from sem_objnav.segment.pose_extractor import PoseExtractor, TopdownView


def make_pose_extractor(name: str) -> Callable[..., PoseExtractor]:
    r"""Constructs a pose_extractor using the given name and keyword arguments

    :param name: The name of the pose_extractor in the `habitat_sim.registry`
    :param kwargs: The keyword arguments to be passed to the constructor of the pose extractor
    """

    model = registry.get_pose_extractor(name)
    assert model is not None, "Could not find a pose extractor for name '{}'".format(
        name
    )

    return model


class ImageExtractor:
    r"""Main class that extracts data by creating a simulator and generating a topdown map from which to
    iteratively generate image data.

    :property scene_filepath: The filepath to the .glb file for the scene
    :property scene_dataset_config_file: Dataset config file (e.g. hm3d_annotated_basis.scene_dataset_config.json)
    :property cur_fp: The current scene filepath. This is only relevant when the extractor is operating on multiple scenes
    :property labels: class labels of things to gather images of (currently not used)
    :property img_size: Tuple of image output dimensions (height, width)
    :property cfg: configuration for simulator of type SimulatorConfiguration
    :property sim: Simulator object
    :property meters_per_pixel: Resolution of topdown map. 0.1 means each pixel in the topdown map
        represents 0.1 x 0.1 meters in the coordinate system of the pathfinder

    :property tdv_fp_ref_triples: List of tuples containing (TopdownView Object, scene_filepath, reference point)
        information for each scene. Each scene requires:
            TopdownView: To extract poses
            scene_filepath: The file path to the mesh file. Necessary for scene switches.
            reference point: A reference point from the coordinate system of the scene. Necessary for specifying poses
                in the scene's coordinate system.

    :property pose_extractor_name: name of the pose_extractor in habitat.registry
    :property poses: list of camera poses gathered from pose_extractor
    :property train: A subset of poses used for training
    :property test: A subset of poses used for testing
    :property mode: Mode that determines which poses to use, train poses, test poses, or all (full) poses
    :property instance_id_to_name: Maps instance_ids in the scene to their english names. E.g. 31 -> 'wall'
    :property cache: Dictionary to cache samples. Change capacity with the cache_capacity param.

    :property output: list of output names that the user wants e.g. ['rgba', 'depth']
    """

    def __init__(
        self,
        scene_filepath: Union[List[str], str],
        scene_dataset_config_file: str = None,
        labels: List[float] = None,
        img_size: tuple = (512, 512),
        output: List[str] = None,
        pose_extractor_name: str = "panorama_extractor",
        meters_per_pixel: float = 0.1,
        camera_hfov_deg: float = 54,
        agent_radius: float = 0.2795,
        agent_height: float = 1.496,
    ):
        if labels is None:
            labels = [0.0]
        if output is None:
            output = ["rgba"]
        self.scene_filepaths = None
        self.scene_dataset_config_file = scene_dataset_config_file
        self.cur_fp = None
        if isinstance(scene_filepath, list):
            self.scene_filepaths = scene_filepath
        else:
            self.scene_filepaths = [scene_filepath]
            self.cur_fp = scene_filepath
        self.agent_radius = agent_radius
        self.agent_height = agent_height
        self.labels = set(labels)
        self.img_size = img_size
        self.camera_hfov_deg = camera_hfov_deg
        self.meters_per_pixel = meters_per_pixel
        self.out_name_to_sensor_name = {
            "rgba": "color_sensor",
            "depth": "depth_sensor",
            "semantic": "semantic_sensor",
        }
        self.output = output
        self.random_state = np.random.RandomState(42)
        self.instance_id_to_name = None
        self.pose_extractor_name = pose_extractor_name

    def __iter__(self):

        for filepath in self.scene_filepaths:
            cfg = self._config_sim(
                filepath,
                self.img_size,
                self.camera_hfov_deg,
                self.agent_radius,
                self.agent_height,
            )
            sim = habitat_sim.Simulator(cfg)
            ref_point = self._get_pathfinder_reference_point(sim.pathfinder)
            tdv = TopdownView(sim, ref_point[1], meters_per_pixel=self.meters_per_pixel)
            args = ([(tdv, filepath, ref_point)], self.meters_per_pixel)
            pose_extractor = make_pose_extractor(self.pose_extractor_name)(*args)
            self.instance_id_to_name = self._generate_label_map(sim.semantic_scene)
            poses = pose_extractor.extract_all_poses()
            for pose in poses:
                pos, rot, fp = pose
                new_state = AgentState()
                new_state.position = pos
                new_state.rotation = rot
                sim.agents[0].set_state(new_state)
                obs = sim.get_sensor_observations()
                sample = {
                    out_name: obs[self.out_name_to_sensor_name[out_name]]
                    for out_name in self.output
                }
                yield sample
            sim.close()

    def close(self) -> None:
        r"""Deletes the instance of the simulator. Necessary for instantiating a different ImageExtractor."""
        if self.sim is not None:
            self.sim.close()
            del self.sim
            self.sim = None

    def set_mode(self, mode: str) -> None:
        r"""Sets the mode of the simulator. This controls which poses to use; train, test, or all (full)"""
        mymode = mode.lower()
        if mymode not in ["full", "train", "test"]:
            raise Exception(
                f'Mode {mode} is not a valid mode for ImageExtractor. Please enter "full, train, or test"'
            )

        self.mode = mymode

    def get_semantic_class_names(self) -> List[str]:
        r"""Returns a list of english class names in the scene(s). E.g. ['wall', 'ceiling', 'chair']"""
        class_names = list(set(self.instance_id_to_name.values()))
        return class_names

    def _get_pathfinder_reference_point(self, pf):
        bound1, bound2 = pf.get_bounds()
        startw = min(bound1[0], bound2[0])
        starth = min(bound1[2], bound2[2])
        starty = pf.get_random_navigable_point()[
            1
        ]  # Can't think of a better way to get a valid y-axis value
        return (startw, starty, starth)  # width, y, height

    def _generate_label_map(self, scene, verbose=True):
        if verbose:
            print(
                f"House has {len(scene.levels)} levels, {len(scene.regions)} regions and {len(scene.objects)} objects"
            )
            print(f"House center:{scene.aabb.center} dims:{scene.aabb.sizes}")

        instance_id_to_name = {}
        for obj in scene.objects:
            if obj and obj.category:
                obj_id = int(obj.id.split("_")[-1])
                instance_id_to_name[obj_id] = obj.category.name()

        return instance_id_to_name

    def _config_sim(
        self, scene_filepath, img_size, camera_hfov_deg, agent_radius, agent_height
    ):
        settings = {
            "width": img_size[1],  # Spatial resolution of the observations
            "height": img_size[0],
            "scene": scene_filepath,  # Scene path
            "default_agent": 0,
            "sensor_height": 0.03,  # Height of sensors in meters
            "color_sensor": True,  # RGBA sensor
            "semantic_sensor": True,  # Semantic sensor
            "depth_sensor": True,  # Depth sensor
            "audio_sensor": False,  # Audio sensor
            "silent": True,
            "seed": 1337,
        }

        sim_cfg = hsim.SimulatorConfiguration()
        sim_cfg.enable_physics = False
        sim_cfg.gpu_device_id = 0
        sim_cfg.scene_id = settings["scene"]
        sim_cfg.random_seed = settings["seed"]
        if self.scene_dataset_config_file is not None:
            sim_cfg.scene_dataset_config_file = self.scene_dataset_config_file

        # define default sensor parameters (see src/esp/Sensor/Sensor.h)
        sensor_specs = []
        if settings["color_sensor"]:
            color_sensor_spec = habitat_sim.CameraSensorSpec()
            color_sensor_spec.uuid = "color_sensor"
            color_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
            color_sensor_spec.resolution = [settings["height"], settings["width"]]
            color_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
            color_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
            color_sensor_spec.hfov = mn.Deg(camera_hfov_deg)
            sensor_specs.append(color_sensor_spec)

        if settings["depth_sensor"]:
            depth_sensor_spec = habitat_sim.CameraSensorSpec()
            depth_sensor_spec.uuid = "depth_sensor"
            depth_sensor_spec.sensor_type = habitat_sim.SensorType.DEPTH
            depth_sensor_spec.resolution = [settings["height"], settings["width"]]
            depth_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
            depth_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
            depth_sensor_spec.hfov = mn.Deg(camera_hfov_deg)
            depth_sensor_spec.near = 0.05
            depth_sensor_spec.far = 5.6
            sensor_specs.append(depth_sensor_spec)

        if settings["semantic_sensor"]:
            semantic_sensor_spec = habitat_sim.CameraSensorSpec()
            semantic_sensor_spec.uuid = "semantic_sensor"
            semantic_sensor_spec.sensor_type = habitat_sim.SensorType.SEMANTIC
            semantic_sensor_spec.resolution = [settings["height"], settings["width"]]
            semantic_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
            semantic_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
            semantic_sensor_spec.hfov = mn.Deg(camera_hfov_deg)
            sensor_specs.append(semantic_sensor_spec)

        if settings["audio_sensor"]:
            audio_sensor_spec = habitat_sim.AudioSensorSpec()
            audio_sensor_spec.uuid = "audio_sensor"
            audio_sensor_spec.sensor_type = habitat_sim.SensorType.AUDIO
            audio_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
            sensor_specs.append(audio_sensor_spec)

        # create agent specifications
        agent_cfg = AgentConfiguration()
        agent_cfg.sensor_specifications = sensor_specs
        agent_cfg.height = agent_height
        agent_cfg.radius = agent_radius
        return habitat_sim.Configuration(sim_cfg, [agent_cfg])

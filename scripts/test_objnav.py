import random
import sys
import time
from pathlib import Path

import cv2
import habitat_sim
import numpy as np
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
from omegaconf import OmegaConf

from sem_objnav.obj_nav.dataset import TRAIN, VAL
from sem_objnav.obj_nav.env import MappingObjNavEnv

if __name__ == "__main__":
    import sem_objnav

    random.seed(42)

    VELOCITY_CONTROL_LINEAR_FORWARD_KEY = "w"
    VELOCITY_CONTROL_LINEAR_BACKWARD_KEY = "s"
    VELOCITY_CONTROL_ANGLE_LEFT_KEY = "a"
    VELOCITY_CONTROL_ANGLE_RIGHT_KEY = "d"

    config = OmegaConf.load("./sem_objnav/obj_nav/cfg/config.yaml")
    config.env.label_schema_col = "objnav_occ_id"
    config.env.num_semantic_classes = 9
    # config.env.mapping.egomap_small_crop = [256, 256]
    # config.env.mapping.egomap_large_crop = [1024, 1024]
    # # config.env.mapping.egomap_small_crop = [77, 77]
    # # config.env.mapping.egomap_large_crop = [341, 341]
    config.env.mapping.egomap_size = [128, 128]
    config.env.mapping.pixel2meter = 0.03
    config.env.semantic_model.prob_aggregate = "height_map_argmax"

    config.env.semantic_model.ckpt = None
    config.env.semantic_model.config = None
    config.env.semantic_model.model_type = "segformer"

    # carpet scene "00741-w8GiikYuFRk"
    env = MappingObjNavEnv(
        config=config.env,
        scenes=VAL[1:5],
        split="val",
        use_aux_angle="gt_debug",
        render_mode="human",
        render_options=[
            "rgb",
            "semantic",
            # "depth",
            "write_metrics",
            "map_small",
            "map_large",
        ],
        exclude_goal_class_episodes=["plant"],
        shortest_path_policy=False,
    )
    env.segmentation_model.threshold = 0.75

    env.mapping.collect_nb_data = False
    # env._env.habitat_env.episodes = [
    #     ep for ep in env._env.habitat_env.episodes if ep.episode_id == "17"
    # ]
    aux_angle = np.zeros(env.action_space.spaces["aux_angle"].shape)
    aux_angle[0] = 1.0
    while True:
        env.reset()
        done = False
        while not done:
            key = cv2.waitKey(0)

            # action = env.shortest_path_policy()

            action = np.array([-1.0, 0.0])
            if key == ord(VELOCITY_CONTROL_LINEAR_FORWARD_KEY):
                action[0] = 1.0
            if key == ord(VELOCITY_CONTROL_LINEAR_BACKWARD_KEY):
                action[0] = -1.0
            if key == ord(VELOCITY_CONTROL_ANGLE_LEFT_KEY):
                action[1] = 1.0
            if key == ord(VELOCITY_CONTROL_ANGLE_RIGHT_KEY):
                action[1] = -1.0

            if key == ord("q"):
                exit()
            action = {
                "action": action,
                "aux_angle": aux_angle,
            }  # "sp_action": sp_action}
            start_time = time.time()
            observations, reward, done, _, info = env.step(action)
            end_time = time.time()
            print(f"FPS: {1.0 / (end_time - start_time)}")
            env.render()

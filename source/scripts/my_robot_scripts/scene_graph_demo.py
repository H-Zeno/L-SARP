from __future__ import annotations
import logging
import time
import numpy as np
import os
import pandas as pd
import copy
import sys
import re
import textwrap
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple, Union

from bosdyn.client import Sdk

from robot_utils.base import ControlFunction, take_control_with_function
from robot_utils.basic_movements import carry_arm, stow_arm, move_body, gaze, move_arm
from robot_utils.advanced_movement import push_light_switch, turn_light_switch
from robot_utils.frame_transformer import FrameTransformerSingleton
from robot_utils.video import localize_from_images, get_camera_rgbd, set_gripper_camera_params

from utils.coordinates import Pose3D, Pose2D
from utils.recursive_config import Config
from utils.singletons import (
    GraphNavClientSingleton,
    ImageClientSingleton,
    RobotCommandClientSingleton,
    RobotSingleton,
    RobotStateClientSingleton,
    WorldObjectClientSingleton,
)

from utils.light_switch_interaction import LightSwitchDetection, LightSwitchInteraction

# Import LostFound modules
from LostFound.src import (
    SceneGraph,
    preprocess_scan,
)

# The drawer_integration is only in the Spotlight repo!!
from LostFound.src.utils import parse_txt
from LostFound.src.scene_graph import get_scene_graph

########################################################

AFFORDANCE_DICT_LIGHT_SWITCHES = {"button type": ["push button switch", "rotating switch", "none"],
                   "button count": ["single", "double", "none"],
                   "button position (wrt. other button!)": ["buttons stacked vertically", "buttons side-by-side", "none"],
                   "interaction inference from symbols": ["top/bot push", "left/right push", "center push", "no symbols present"]}

config = Config()
API_KEY = config["gpt_api_key"]
STAND_DISTANCE = 0.75
GRIPPER_WIDTH = 0.03
GRIPPER_HEIGHT = 0.03
ADVANCED_AFFORDANCE = True
FORCES = [10, 0, 0, 0, 0, 0]#8


## TESTING PARAMETERS
TEST_NUMBER = 5
RUN =10
LEVEL = "lower"

POSE_CENTER = Pose2D(coordinates=(1.5, -1))
POSE_CENTER.set_rot_from_angle(180, degrees=True)

DETECTION_DISTANCE = 0.75

NUM_REFINEMENT_POSES = 3
NUM_REFINEMENTS_MAX_TRIES = 1
SHUFFLE = False

base_path = config.get_subpath("prescans")
ending = config["pre_scanned_graphs"]["high_res"]
SCAN_DIR = os.path.join(base_path, ending)

########################################################

frame_transformer = FrameTransformerSingleton()
graph_nav_client = GraphNavClientSingleton()
image_client = ImageClientSingleton()
robot_command_client = RobotCommandClientSingleton()
robot = RobotSingleton()
robot_state_client = RobotStateClientSingleton()
world_object_client = WorldObjectClientSingleton()

light_switch_detection = LightSwitchDetection()
light_switch_interaction = LightSwitchInteraction(frame_transformer, config)


class _Push_Light_Switch(ControlFunction):
    def __call__(
        self,
        config: Config,
        sdk: Sdk,
        *args,
        **kwargs,
    ) -> str:

        self.vis_block = True
        #################################
        # get scene graph and visualize it
        #################################
        scene_graph = get_scene_graph(SCAN_DIR)
        scene_graph.visualize(labels=True, connections=True, centroids=True)

        #################################
        # get light switch and lamp nodes
        #################################
        sem_label_switch = next((k for k, v in scene_graph.label_mapping.items() if v == "light switch"), None)
        sem_label_lamp = next((k for k, v in scene_graph.label_mapping.items() if v == "lamp"), None)
        lamp_nodes = [node for node in scene_graph.nodes.values() if node.sem_label == sem_label_lamp]

        switch_nodes = []
        for node in scene_graph.nodes.values():
            if node.sem_label == sem_label_switch:
                # compute face normal of light switch
                # the normal is set to -x direction (HARDCODED)
                node.set_normal(np.array([-1, 0, 0])) 
                switch_nodes.append(node)

        POSES_LAMPS = [Pose3D(node.centroid) for node in lamp_nodes]
        IDS_LAMPS = [node.object_id for node in lamp_nodes]

        #################################
        # localization of spot based on camera images and depth scans
        #################################
        start_time = time.time()
        set_gripper_camera_params('640x480')

        frame_name = localize_from_images(config, vis_block=False)

        set_gripper_camera_params('1280x720')
        end_time_localization = time.time()
        logging.info(f"Localization time: {end_time_localization - start_time}")

        #################################
        # Move spot to the center of the scene
        #################################
        move_body(POSE_CENTER, frame_name)
    
        #################################
        # Check lamp states pre interaction
        #################################
        lamp_images_pre = light_switch_interaction.check_lamps(POSES_LAMPS, frame_name)

        # Reverse the order of the switch_nodes list -> why? 
        switch_nodes.reverse()

        #################################
        # Interact with each switch
        #################################
        for idx, switch in enumerate(switch_nodes):

            #################################
            # Move body to switch
            #################################
            body_to_switch_start_time = time.time()

            pose = Pose3D(switch.centroid)
            pose.set_rot_from_direction(switch.normal)

            body_add_pose_refinement_right = Pose3D((-STAND_DISTANCE, -0.00, -0.00))
            body_add_pose_refinement_right.set_rot_from_rpy((0, 0, 0), degrees=True)
            p_body = pose.copy() @ body_add_pose_refinement_right.copy()

            move_body(p_body.to_dimension(2), frame_name)
            logging.info(f"Moved body to switch {idx+1} of {len(switch_nodes)}")
            
            body_to_switch_end_time = time.time()
            logging.info(f"Time to move body to switch {idx+1}: {body_to_switch_end_time - body_to_switch_start_time}")

            #################################
            # Extend the arm to a neutral carrying position
            #################################
            carry_arm() 

            #################################
            # Push light switch
            #################################
            push_light_switch(pose, frame_name, z_offset=True, forces=FORCES)

            #################################
            # refine handle position
            #################################
            x_offset = -0.2
            refined_pose, refined_box, color_response = light_switch_interaction.get_average_refined_switch_pose(
                pose, 
                frame_name, 
                x_offset,
                num_refinement_poses=NUM_REFINEMENT_POSES,
                num_refinement_max_tries=NUM_REFINEMENTS_MAX_TRIES,
                bounding_box_optimization=True
            )

            if refined_pose is not None:
                push_light_switch(refined_pose, frame_name, z_offset=True, forces=FORCES)
            else:
                logging.warning(f"Refined pose is None for switch {idx+1} of {len(switch_nodes)}")
                logging.warning(f"Pushing light switch without refinement")
                push_light_switch(pose, frame_name, z_offset=True, forces=FORCES)

            stow_arm()
            
            #################################
            # affordance detection
            #################################
            logging.info("affordance detection starting...")
            affordance_dict = light_switch_detection.light_switch_affordance_detection(refined_box, color_response, 
                                                 AFFORDANCE_DICT_LIGHT_SWITCHES, config["gpt_api_key"])

            #################################
            #  light switch interaction based on affordance
            #################################
            switch_interaction_start_time = time.time()

            offsets, switch_type = light_switch_interaction.determine_switch_offsets_and_type(affordance_dict, GRIPPER_HEIGHT, GRIPPER_WIDTH)
            light_switch_interaction.switch_interaction(switch_type, refined_pose, offsets, frame_name, FORCES)
            stow_arm()
            logging.info(f"Tried interaction with switch {idx + 1} of {len(switch_nodes)}")
            
            #################################
            # check lamp states post interaction
            #################################
            move_body(POSE_CENTER, frame_name)
            lamp_images_post = light_switch_interaction.check_lamps(POSES_LAMPS, frame_name)

            lamp_state_changes = light_switch_interaction.get_lamp_state_changes(lamp_images_pre, lamp_images_post, vis_block=self.vis_block)

            #################################
            # add lamps to the scene graph
            #################################
            for idx, state_change in enumerate(lamp_state_changes):
                if state_change == 1 or state_change == -1:
                    # add lamp to switch, here the scene graph gets updated
                    switch.add_lamp(IDS_LAMPS[idx])
                elif state_change == 0:
                    pass
            
            if self.vis_block:
                scene_graph.visualize(labels=True, connections=True, centroids=True)

            # Copy lamp images for future use
            lamp_images_pre = lamp_images_post.copy()

            # Logging
            logging.info(f"Interaction with switch {idx+1} of {len(switch_nodes)} finished")
            switch_interaction_end_time = time.time()
            logging.info(f"Switch interaction time: {switch_interaction_end_time - switch_interaction_start_time}")
            end_time_total = time.time()
            logging.info(f"Total time per switch: {end_time_total - body_to_switch_start_time}")

        stow_arm()
        return frame_name


def main():
    config = Config()
    take_control_with_function(config, function=_Push_Light_Switch(), body_assist=True)


if __name__ == "__main__":
    main()

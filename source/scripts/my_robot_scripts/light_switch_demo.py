from __future__ import annotations

# =============================================================================
# Standard Library Imports
import logging
import random
import time
import os
import copy

# =============================================================================
# Third-party Imports
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
from typing import List, Optional, Tuple

# =============================================================================
# Bosdyn and Robot Utilities
from bosdyn.client import Sdk
from robot_utils.basic_movements import (
    carry_arm, stow_arm, move_body, gaze, carry, move_arm
)
from robot_utils.advanced_movement import push_light_switch, turn_light_switch, move_body_distanced, push
from robot_utils.frame_transformer import FrameTransformerSingleton
from robot_utils.video import (
    localize_from_images, get_camera_rgbd, set_gripper_camera_params, set_gripper, relocalize,
    frame_coordinate_from_depth_image, select_points_from_bounding_box
)
from robot_utils.base import ControlFunction, take_control_with_function

# =============================================================================
# Custom Utilities
from utils.coordinates import Pose3D, Pose2D, pose_distanced, average_pose3Ds
from utils.recursive_config import Config
from utils.singletons import (
    GraphNavClientSingleton, ImageClientSingleton, RobotCommandClientSingleton,
    RobotSingleton, RobotStateClientSingleton, WorldObjectClientSingleton
)
from utils.light_switch_interaction import LightSwitchDetection, LightSwitchInteraction
from utils.affordance_detection_light_switch import compute_affordance_VLM_GPT4, compute_advanced_affordance_VLM_GPT4
from bosdyn.api.image_pb2 import ImageResponse
from utils.object_detetion import BBox, Detection, Match
from utils.pose_utils import calculate_light_switch_poses
from utils.bounding_box_refinement import refine_bounding_box
from random import uniform

# Load configuration
config = Config("light_switch_configs")

# Get workspace root for absolute paths
WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
LOGGING_PATH = os.path.join(WORKSPACE_ROOT, "logging_lightswitch_experiments")

# Load configuration values
STAND_DISTANCE = config["STAND_DISTANCE"]
GRIPPER_WIDTH = config["GRIPPER_WIDTH"]
GRIPPER_HEIGHT = config["GRIPPER_HEIGHT"]

X_BODY = config["X_BODY"]
Y_BODY = config["Y_BODY"]
ANGLE_BODY = config["ANGLE_BODY"]

ADVANCED_AFFORDANCE = config["ADVANCED_AFFORDANCE"]
FORCES = config["FORCES"]

AFFORDANCE_CLASSES = config["AFFORDANCE_CLASSES"]
AFFORDANCE_DICT_LIGHT_SWITCHES = config["AFFORDANCE_DICT_LIGHT_SWITCHES"]

# Load refinement parameters
NUM_REFINEMENT_POSES = config["NUM_REFINEMENT_POSES"]
NUM_REFINEMENTS_MAX_TRIES = config["NUM_REFINEMENTS_MAX_TRIES"]
BOUNDING_BOX_OPTIMIZATION = config["BOUNDING_BOX_OPTIMIZATION"]
REFINEMENT_X_OFFSET = config["REFINEMENT_X_OFFSET"]
SHUFFLE = config["SHUFFLE"]

# =============================================================================
# Global Singletons and Initializations
frame_transformer = FrameTransformerSingleton()
graph_nav_client = GraphNavClientSingleton()
image_client = ImageClientSingleton()
robot_command_client = RobotCommandClientSingleton()
robot = RobotSingleton()
robot_state_client = RobotStateClientSingleton()
world_object_client = WorldObjectClientSingleton()

# Initialize Light Switch Detection
light_switch_detection = LightSwitchDetection()
light_switch_interaction = LightSwitchInteraction(frame_transformer, config)

# Create logging directory if it doesn't exist
os.makedirs(LOGGING_PATH, exist_ok=True)

# =============================================================================
# Experiment Configuration
# =============================================================================
LEVEL = "upper"
RUN = 1

if LEVEL == "upper":
    X_CABINET = 0.15
    Y_CABINET = 0.85  # 0.9
    Z_CABINET = 0.5  # 0.42
elif LEVEL == "lower":
    X_CABINET = 0.15
    Y_CABINET = 0.85
    Z_CABINET = 0.0
elif LEVEL == "both":
    X_CABINET = 0.15
    Y_CABINET = 0.9
    Z_CABINET = 0.25
    # Adjust body pose for 'both'
    X_BODY = 2.3
    Y_BODY = 0.6

# =============================================================================
# Logging Configuration
WORKSPACE_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../..")
LOGGING_PATH = os.path.join(WORKSPACE_ROOT, "logging_lightswitch_experiments")
os.makedirs(LOGGING_PATH, exist_ok=True)
log_file_path = os.path.join(LOGGING_PATH, f"switches_experiment_LEVEL_{LEVEL}_RUN_{RUN}.log")
if os.path.exists(log_file_path):
    raise FileExistsError(f"The file '{log_file_path}' already exists.")
logging.basicConfig(filename=log_file_path, filemode='w', format='%(name)s - %(levelname)s - %(message)s')
logging.getLogger("understanding-spot.192.168.50.3").disabled = True

class _Push_Light_Switch(ControlFunction):
    def __call__(
        self,
        config: Config,
        sdk: Sdk,
        *args,
        **kwargs,
    ) -> str:

        logging.info(f"ANGLE_BODY: {ANGLE_BODY}")
        logging.info(f"X_BODY: {X_BODY}")
        logging.info(f"Y_BODY: {Y_BODY}")
        logging.info(f"X_CABINET: {X_CABINET}")
        logging.info(f"Y_CABINET: {Y_CABINET}")
        logging.info(f"Z_CABINET: {Z_CABINET}")
        logging.info(f"STAND_DISTANCE: {STAND_DISTANCE}")
        logging.info(f"GRIPPER_WIDTH: {GRIPPER_WIDTH}")
        logging.info(f"GRIPPER_HEIGHT: {GRIPPER_HEIGHT}")
        logging.info(f"ADVANCED_AFFORDANCE: {ADVANCED_AFFORDANCE}")
        logging.info(f"FORCES: {FORCES}")
        logging.info(f"LOGGING_PATH: {LOGGING_PATH}")
        logging.info(f"RUN: {RUN}")
        logging.info(f"LEVEL: {LEVEL}")
        # logging.info(f"DETECTION_DISTANCE: {DETECTION_DISTANCE}")
        logging.info(f"SHUFFLE: {SHUFFLE}")
        logging.info(f"NUM_REFINEMENT_POSES: {NUM_REFINEMENT_POSES}")
        logging.info(f"NUM_REFINEMENTS_MAX_TRIES: {NUM_REFINEMENTS_MAX_TRIES}")

        #################################
        # localization of spot based on camera images and depth scans
        #################################
        start_time = time.time()
        set_gripper_camera_params('640x480')

        frame_name = localize_from_images(config, vis_block=False)

        end_time_localization = time.time()
        logging.info(f"Localization time: {end_time_localization - start_time}")

        #################################
        # Move spot to the required pose
        #################################
        pose = Pose2D(np.array([X_BODY, Y_BODY]))
        pose.set_rot_from_angle(ANGLE_BODY, degrees=True)
        move_body(
            pose=pose,
            frame_name=frame_name,
        )
        # Set the rotation of the cabinet
        cabinet_pose = Pose3D((X_CABINET, Y_CABINET, Z_CABINET))
        cabinet_pose.set_rot_from_rpy((0,0,ANGLE_BODY), degrees=True)

        # Put the robotic arm in the carry mode
        carry()

        #################################
        # Gaze at the cabinet and get the depth and color images
        # (The pose of the cabinet is necessary for the gaze at the cabinet)
        #################################
        set_gripper_camera_params('1920x1080')
        time.sleep(1)
        gaze(cabinet_pose, frame_name, gripper_open=True)

        depth_image_response, color_response = get_camera_rgbd(
            in_frame="image",
            vis_block=False,
            cut_to_size=False,
        )
        set_gripper_camera_params('1280x720')
        stow_arm()

        #################################
        # Detect the light switch bounding boxes and poses in the scene
        #################################
        boxes = light_switch_detection.predict_light_switches(color_response[0], vis_block=True)
        logging.info(f"INITIAL LIGHT SWITCH DETECTION")
        logging.info(f"Number of detected switches: {len(boxes)}")
        end_time_detection = time.time()
        logging.info(f"Detection time: {end_time_detection - end_time_localization}")

        if SHUFFLE:
            random.shuffle(boxes)

        poses = calculate_light_switch_poses(boxes, depth_image_response, frame_name, frame_transformer)
        logging.info(f"Number of calculated poses: {len(poses)}")
        end_time_pose_calculation = time.time()
        logging.info(f"Pose calculation time: {end_time_pose_calculation - end_time_detection}")

        #################################
        # Interact with each light switch
        #################################
        for idx, pose in enumerate(poses):
            pose_start_time = time.time()
            body_add_pose_refinement_right = Pose3D((-STAND_DISTANCE, -0.00, -0.00))
            body_add_pose_refinement_right.set_rot_from_rpy((0, 0, 0), degrees=True)
            p_body = pose.copy() @ body_add_pose_refinement_right.copy()
            
            move_body(p_body.to_dimension(2), frame_name)
            logging.info(f"Moved body to switch {idx+1} of {len(poses)}")
            
            end_time_move_body = time.time()
            logging.info(f"Move body time: {end_time_move_body - pose_start_time}")

            #################################
            # Extend the arm to a neutral carrying position
            #################################
            carry_arm() 

            #################################
            # refine handle position
            #################################
            x_offset = -0.3 # -0.2
            refined_pose, refined_box, color_response = light_switch_interaction.get_average_refined_switch_pose(
                pose, 
                frame_name, 
                x_offset,
                num_refinement_poses=NUM_REFINEMENT_POSES,
                num_refinement_max_tries=NUM_REFINEMENTS_MAX_TRIES,
                bounding_box_optimization=True
            )

            #################################
            # affordance detection
            #################################
            logging.info("affordance detection starting...")
            affordance_dict = light_switch_detection.light_switch_affordance_detection(refined_box, color_response, 
                                                 AFFORDANCE_DICT_LIGHT_SWITCHES, config["gpt_api_key"])

            #################################
            # interaction based on affordance
            #################################
            switch_interaction_start_time = time.time()
            result = light_switch_interaction.determine_switch_offsets_and_type(affordance_dict, GRIPPER_HEIGHT, GRIPPER_WIDTH)
            if result is not None:
                offsets, switch_type = result
                light_switch_interaction.switch_interaction(switch_type, refined_pose, offsets, frame_name, FORCES)
            else:
                logging.warning(f"Could not determine switch offsets and type for switch {idx+1}")
            stow_arm()
            logging.info(f"Interaction with switch {idx+1} of {len(poses)} finished")

            # Logging
            end_time_switch = time.time()
            logging.info(f"Switch interaction time: {end_time_switch - switch_interaction_start_time}")
            end_time_total = time.time()
            logging.info(f"total time per switch: {end_time_total - pose_start_time}")

        stow_arm()
        return frame_name
        

def main():
    config = Config()
    take_control_with_function(config, function=_Push_Light_Switch(), body_assist=True)


if __name__ == "__main__":
    main()

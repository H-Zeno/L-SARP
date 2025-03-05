from __future__ import annotations

# =============================================================================
# Standard Library Imports
import logging
import random
import time
import os
import copy
import numpy as np
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

# =============================================================================
# Semantic Kernel
from semantic_kernel.functions.kernel_function_decorator import kernel_function

from planner_core.interfaces import AbstractLlmChat

import logging
from pathlib import Path
from typing import Annotated, Optional, Set

logger = logging.getLogger("plugins")

class NavigationPlugin:
    def __init__(self): 
        self.config = Config()

    def _move_to_pose(self, X_BODY_GOAL: float, Y_BODY_GOAL: float, ANGLE_BODY_GOAL: float):
        #################################
        # localization of spot based on camera images and depth scans
        #################################
        start_time = time.time()
        set_gripper_camera_params('640x480')

        frame_name = localize_from_images(self.config, vis_block=False)

        end_time_localization = time.time()
        logging.info(f"Localization time: {end_time_localization - start_time}")

        #################################
        # Move spot to the required pose
        #################################
        pose = Pose2D(np.array([X_BODY_GOAL, Y_BODY_GOAL]))
        pose.set_rot_from_angle(ANGLE_BODY_GOAL, degrees=True)
        move_body(
            pose=pose,
            frame_name=frame_name,
        )

    @kernel_function(description="function to call when the robot needs to navigate from place A (coordinates) to place B (coordinates)", name="RobotNavigation")
    def move_to_pose(self, X_BODY_GOAL: float, Y_BODY_GOAL: float, ANGLE_BODY_GOAL: float):
        config = Config()
        take_control_with_function(config, function=self._move_to_pose, X_BODY_GOAL=X_BODY_GOAL, Y_BODY_GOAL=Y_BODY_GOAL, ANGLE_BODY_GOAL=ANGLE_BODY_GOAL, body_assist=True)
        logging.info(f"Robot moved to pose {X_BODY_GOAL}, {Y_BODY_GOAL}, {ANGLE_BODY_GOAL}")
    
    


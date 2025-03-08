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

from robot_utils.base import ControlFunction, take_control_with_function

# =============================================================================
# Custom Utilities
from utils.coordinates import Pose3D, Pose2D, pose_distanced, average_pose3Ds
from utils.recursive_config import Config

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

from source.planner_core.robot_state import RobotStateSingleton
robot_state = RobotStateSingleton()

import logging
from pathlib import Path
from typing import Annotated, Optional, Set

logger = logging.getLogger("plugins")

class NavigationPlugin:
    general_config = Config()
    object_interaction_config = Config("light_switch_configs")
    stand_distance = object_interaction_config["STAND_DISTANCE"]

    def _move_to_object(self, object_centroid_pose, interaction_normal_of_object):

        body_to_object_start_time = time.time()

        #################################
        # Calculate the required position spot should move to to interact with the object
        #################################
        pose = Pose3D(object_centroid_pose)

        pose.set_rot_from_direction(interaction_normal_of_object)
        body_add_pose_refinement_right = Pose3D((-self.stand_distance, -0.00, -0.00))
        body_add_pose_refinement_right.set_rot_from_rpy((0, 0, 0), degrees=True)
        p_body = pose.copy() @ body_add_pose_refinement_right.copy()

        #################################
        # Move spot to the required pose
        #################################
        move_body(
            pose=p_body.to_dimension(2),
            frame_name=robot_state.frame_name,
        )

        body_to_object_end_time = time.time()
        logging.info(f"Moved spot succesfully to the object. Time to move body to object: {body_to_object_end_time - body_to_object_start_time}")

    @kernel_function(description="function to call when the robot needs to navigate from place A (coordinates) to place B (coordinates)", name="RobotNavigation")
    def move_to_object(self, object_id: int):

        object_centroid_pose = robot_state.scene_graph.nodes[object_id].centroid

        # A function to calculate the normal is necessary! For now we hardcode it with [-1, 0, 0]
        # HARDCODED
        interaction_normal_of_object = np.array([-1, 0, 0])

        take_control_with_function(general_robot_state=robot_state, config=self.general_config, function=self._move_to_object, object_centroid_pose=object_centroid_pose, interaction_normal_of_object=interaction_normal_of_object, body_assist=True)

        logging.info(f"Robot moved to the object with id {object_id}")
    
    


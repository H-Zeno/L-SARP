from __future__ import annotations

# =============================================================================
# Standard Library Imports
import logging
import random
import time
import os
import copy
import numpy as np
from typing import List, Optional, Annotated

# =============================================================================
# Bosdyn and Robot Utilities
from bosdyn.api.image_pb2 import ImageResponse
from robot_utils.basic_movements import (
    carry_arm, stow_arm, move_body, gaze, carry, move_arm
)
from robot_utils.advanced_movement import push_light_switch, turn_light_switch, move_body_distanced, push
from robot_utils.frame_transformer import FrameTransformerSingleton
frame_transformer = FrameTransformerSingleton()
from robot_utils.base_LSARP import ControlFunction , take_control_with_function
from robot_utils.object_interaction_utils import get_best_poses_in_front_of_object, get_pose_in_front_of_furniture

from utils.light_switch_interaction import LightSwitchDetection, LightSwitchInteraction
from utils.pose_utils import calculate_light_switch_poses
# =============================================================================

from planner_core.robot_state import RobotStateSingleton
robot_state = RobotStateSingleton()


# =============================================================================
# Custom Utilities
from utils.coordinates import Pose3D, Pose2D, pose_distanced, average_pose3Ds
from utils.recursive_config import Config
from utils.affordance_detection_light_switch import compute_affordance_VLM_GPT4, compute_advanced_affordance_VLM_GPT4
from utils.object_detetion import BBox, Detection, Match
from utils.bounding_box_refinement import refine_bounding_box
from random import uniform

# =============================================================================
# Semantic Kernel
from semantic_kernel.functions.kernel_function_decorator import kernel_function


from robot_plugins.communication import CommunicationPlugin
communication = CommunicationPlugin()

import logging

logger = logging.getLogger("plugins")

class NavigationPlugin:
    general_config = Config()
    global object_interaction_config
    object_interaction_config = Config("light_switch_configs")
    stand_distance = object_interaction_config["STAND_DISTANCE"]
    
    class _Move_To_Object(ControlFunction):
        def __call__(
            self,
            config: Config,
            object_interaction_pose: Pose3D,
            *args,
            **kwargs,
        ) -> None:
            
            body_to_object_start_time = time.time()

            #################################
            # Calculate the required position spot should move to to interact with the object
            #################################

            # # Rotate the pose to the interaction normal of the object
            # pose.set_rot_from_direction(interaction_normal_of_object)

            # # Create a body offset without rotation
            # body_offset = Pose3D((-object_interaction_config["STAND_DISTANCE"], -0.00, -0.00))
            # body_offset.set_rot_from_rpy((0, 0, 0), degrees=True)
            # p_body = pose.copy() @ body_offset.copy()

            #################################
            # Move spot to the required pose
            #################################
            move_body(
                pose=object_interaction_pose.to_dimension(2),
                frame_name=robot_state.frame_name,
            )

            body_to_object_end_time = time.time()
            logging.info(f"Moved spot succesfully to the object. Time to move body to object: {body_to_object_end_time - body_to_object_start_time}")

    @kernel_function(description="function to call when the robot needs to navigate from place A (coordinates) to place B (coordinates)", name="RobotNavigation")
    async def move_to_object(self, object_id: Annotated[int, "ID of the object in the scene graph"], object_description: Annotated[str, "A clear (3-5 words) description of the object."]) -> None:

        # If the object is a shelf:
        object_centroid_pose = robot_state.scene_graph.nodes[object_id].centroid
        sem_label = robot_state.scene_graph.label_mapping.get(robot_state.scene_graph.nodes[object_id].sem_label)
        logging.info(f"Object with id {object_id} has label {sem_label}.")
        if sem_label in ["shelf", "cabinet", "coffee table", "tv stand"]:
            object_center_openmask, object_interaction_pose = get_pose_in_front_of_furniture(index=object_id, object_description=sem_label)
        else:
            object_targets = get_best_poses_in_front_of_object(index=object_id, object_description=object_description)
            object_interaction_pose = object_targets[0][0]

        await communication.inform_user(f"Moving to object with id {object_id}, label {robot_state.scene_graph.nodes[object_id].sem_label} and centroid {object_centroid_pose}."
                                        f"The object interaction pose is: {object_interaction_pose}"
                                        f"The current position of the robot is {frame_transformer.get_current_body_position_in_frame(robot_state.frame_name)}")
        
        response = await communication.ask_user("Do you want me to move to the object? Please enter exactly 'yes' if you want me to move to the object.")   
        if response == "yes":
            take_control_with_function(config=self.general_config, function=self._Move_To_Object(), object_interaction_pose=object_interaction_pose, body_assist=True)
            logging.info(f"Robot moved to the object with id {object_id}")
        else:
            await communication.inform_user("I will not move to the object.")

        
    
    


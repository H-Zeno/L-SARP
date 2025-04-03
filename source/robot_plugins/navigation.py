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
import asyncio

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
from robot_utils.object_interaction_utils import get_best_pose_in_front_of_object, get_pose_in_front_of_furniture

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

from robot_plugins.user_communication import CommunicationPlugin
communication = CommunicationPlugin()

logger = logging.getLogger("plugins")

class NavigationPlugin:
    general_config = Config()
    object_interaction_config = Config("object_interaction_configs")
    inspection_distance = object_interaction_config["INSPECTION_DISTANCE"]
    
    class _Move_To_Object(ControlFunction):
        def __call__(
            self,
            config: Config,
            object_interaction_pose: Pose2D,
            *args,
            **kwargs,
        ) -> None:
            
            logger.info(f"Starting _Move_To_Object with target pose: {object_interaction_pose}")
            logger.info(f"Current robot position: {frame_transformer.get_current_body_position_in_frame(robot_state.frame_name)}")
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
            try:
                logger.info(f"Moving robot to pose {object_interaction_pose} in frame {robot_state.frame_name}")
                success = move_body(
                    pose=object_interaction_pose,
                    frame_name=robot_state.frame_name,
                )
                if success:
                    logger.info("move_body function returned success")
                else:
                    logger.error("move_body function returned failure")
            except Exception as e:
                logger.error(f"Exception during move_body: {e}")
                logger.exception("Traceback:")

            body_to_object_end_time = time.time()
            logger.info(f"Moved spot to the object. Time to move: {body_to_object_end_time - body_to_object_start_time:.2f} seconds")
            logger.info(f"New robot position: {frame_transformer.get_current_body_position_in_frame(robot_state.frame_name)}")

    @kernel_function(description="function to call when the robot needs to navigate from place A (coordinates) to place B (coordinates)", name="RobotNavigation")
    async def move_to_object(self, object_id: Annotated[int, "ID of the object in the scene graph"], object_description: Annotated[str, "A clear (3-5 words) description of the object."]) -> None:
        try:
            # Safety check: verify the object exists in the scene graph
            if object_id not in robot_state.scene_graph.nodes:
                error_msg = f"Object with ID {object_id} not found in scene graph."
                logger.error(error_msg)
                await communication.inform_user(error_msg)
                return None

            object_centroid_pose = robot_state.scene_graph.nodes[object_id].centroid
            sem_label = robot_state.scene_graph.label_mapping.get(robot_state.scene_graph.nodes[object_id].sem_label, "Unknown object")
            logger.info(f"Object with id {object_id} has label {sem_label}.")

            if self.general_config["robot_planner_settings"]["use_with_robot"] is not True:
                logger.info(f"Moving to object with id {object_id} and centroid {object_centroid_pose} in simulation (without robot).")
                logger.info(f"Setting virtual robot pose to {object_centroid_pose}")
                robot_state.virtual_robot_pose = object_centroid_pose
                return None
            
            # Determine appropriate interaction pose based on object type
            furniture_labels = self.general_config["semantic_labels"]["furniture"]
            object_interaction_pose = None
            
            try:
                if sem_label in furniture_labels:
                    object_interaction_pose = get_pose_in_front_of_furniture(
                        index=object_id, 
                        min_distance=self.inspection_distance,
                        object_description=sem_label
                    )
                else:
                    object_interaction_pose = get_best_pose_in_front_of_object(
                        index=object_id, 
                        min_interaction_distance=self.inspection_distance,
                        object_description=object_description, 
                    )
            except Exception as e:
                logger.error(f"Error calculating interaction pose: {e}")
                await communication.inform_user(f"Could not calculate interaction pose: {str(e)}")
                return None
                
            if object_interaction_pose is None:
                logger.error("Could not determine interaction pose for object")
                await communication.inform_user("Could not determine how to approach this object")
                return None
            
            # Inform user and get confirmation
            status_message = f"Moving to object with id {object_id}, label {sem_label} and centroid {object_centroid_pose}. "
            status_message += f"The object interaction pose is: {object_interaction_pose}. "
            status_message += f"The current position of the robot is {frame_transformer.get_current_body_position_in_frame(robot_state.frame_name)}"
            
            await communication.inform_user(status_message)
            
            response = await communication.ask_user("Do you want me to move to the object? Please enter exactly 'yes' if you want me to move to the object.")
            if response == "yes":
                logger.info(f"User confirmed. Moving to object with pose: {object_interaction_pose}")
                
                try:
                    take_control_with_function(
                        config=self.general_config, 
                        function=self._Move_To_Object(), 
                        object_interaction_pose=object_interaction_pose.to_dimension(2), 
                        body_assist=True
                    )
                    logger.info(f"Robot moved to the object with id {object_id}")
                    
                    # Add a small delay to ensure message logging is complete
                    await asyncio.sleep(0.2)
                    
                    # This is where the segmentation fault might be happening - nothing should
                    # execute after this point if we can't identify the issue
                    return None
                    
                except Exception as e:
                    error_msg = f"Error during robot movement: {str(e)}"
                    logger.error(error_msg)
                    await communication.inform_user(f"An error occurred while moving to the object: {str(e)}")
                    return None
            else:
                await communication.inform_user("I will not move to the object.")

            return None
            
        except Exception as e:
            logger.error(f"Unhandled exception in move_to_object: {str(e)}")
            await communication.inform_user(f"An unexpected error occurred: {str(e)}")
            return None

        




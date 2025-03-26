from __future__ import annotations

# =============================================================================
# Standard Library Imports
import logging
import time
import os
import copy
from dotenv import dotenv_values
import numpy as np
from pathlib import Path
from typing import Annotated, Optional, Set

# =============================================================================
# Bosdyn and Robot Utilities
from bosdyn.client import Sdk
from bosdyn.api.image_pb2 import ImageResponse
from robot_utils.basic_movements import (
    carry_arm, gaze, stow_arm, move_body)
from robot_utils.advanced_movement import push_light_switch, turn_light_switch, move_body_distanced, push
from robot_utils.frame_transformer import FrameTransformerSingleton
from robot_utils.video import (
    localize_from_images, get_camera_rgbd, set_gripper_camera_params, set_gripper, relocalize,
    frame_coordinate_from_depth_image, select_points_from_bounding_box
)
from robot_utils.base_LSARP import ControlFunction, take_control_with_function
from robot_utils.object_interaction_utils import get_best_pose_in_front_of_object
from robot_plugins.communication import CommunicationPlugin
from utils.light_switch_interaction import LightSwitchDetection, LightSwitchInteraction
from utils.pose_utils import calculate_light_switch_poses
light_switch_detection = LightSwitchDetection()

# =============================================================================
# Custom Utilities
from utils.coordinates import Pose3D, Pose2D, pose_distanced, average_pose3Ds
from utils.recursive_config import Config

# =============================================================================
# Semantic Kernel
from semantic_kernel.functions.kernel_function_decorator import kernel_function
from planner_core.interfaces import AbstractLlmChat

# =============================================================================
# Scene Graph
from LostFound.src.graph_nodes import LightSwitchNode, DrawerNode, ObjectNode
from LostFound.src.scene_graph import SceneGraph

from planner_core.robot_state import RobotStateSingleton

robot_state = RobotStateSingleton()
frame_transformer = FrameTransformerSingleton()
communication = CommunicationPlugin()

logger = logging.getLogger("plugins")
config = Config("object_interaction_configs")

class ItemInteractionsPlugin:
    
    class _Push_Light_Switch(ControlFunction):
        light_switch_detection = LightSwitchDetection()
        light_switch_interaction = LightSwitchInteraction(frame_transformer, config)
        planner_settings = dotenv_values(".env_core_planner")

        def __call__(
            self,
            config: Config,
            light_switch_object_id: int,
            *args,
            **kwargs,
        ) -> None:
            
            light_switch_node = robot_state.scene_graph.nodes[light_switch_object_id]

            # #################################
            # # Check lamp states pre interaction
            # #################################
            # lamp_images_pre = self.light_switch_interaction.check_lamps(POSES_LAMPS, frame_name)

            #################################
            # Extend the arm to a neutral carrying position
            #################################
            carry_arm() 
            switch_interaction_start_time = time.time()

            #################################
            # Gaze at the cabinet and get the depth and color images
            # (The pose of the cabinet is necessary for the gaze at the cabinet)
            #################################
            set_gripper_camera_params('1920x1080')
            time.sleep(1)
            logger.info("The robot now will gaze at the light switch with the pose {light_switch_pose}.")
            
            gaze(Pose3D(light_switch_node.centroid), robot_state.frame_name, gripper_open=True)

            depth_image_response, color_response = get_camera_rgbd(
                in_frame="image",
                vis_block=False,
                cut_to_size=False,
            )
            set_gripper_camera_params('1280x720')

            #################################
            # Detect the light switch bounding boxes and poses in the scene
            #################################
            boxes = light_switch_detection.predict_light_switches(color_response[0], vis_block=True)
            logger.info(f"INITIAL LIGHT SWITCH DETECTION")
            logger.info(f"Number of detected switches: {len(boxes)}")
            end_time_detection = time.time()

            poses = calculate_light_switch_poses(boxes, depth_image_response, robot_state.frame_name, frame_transformer)
            logger.info(f"Number of calculated poses: {len(poses)}")
            end_time_pose_calculation = time.time()
            logger.info(f"Pose calculation time: {end_time_pose_calculation - end_time_detection}")
            light_switch_pose = poses[0]
            
            #################################
            # refine handle position
            #################################
            refined_pose, refined_box, color_response = self.light_switch_interaction.get_average_refined_switch_pose(
                light_switch_pose, 
                robot_state.frame_name, 
                config["REFINEMENT_X_OFFSET"],
                num_refinement_poses=config["NUM_REFINEMENT_POSES"],
                num_refinement_max_tries=config["NUM_REFINEMENTS_MAX_TRIES"],
                bounding_box_optimization=True
            )
            
            #################################
            # affordance detection
            #################################
            logger.info("affordance detection starting...")
            affordance_dict = light_switch_detection.light_switch_affordance_detection(
                refined_box, color_response, 
                config["AFFORDANCE_DICT_LIGHT_SWITCHES"], self.planner_settings.get("OPENAI_API_KEY")
            )

            #################################
            #  light switch interaction based on affordance
            #################################
            offsets, switch_type = self.light_switch_interaction.determine_switch_offsets_and_type(
                affordance_dict, config["GRIPPER_HEIGHT"], config["GRIPPER_WIDTH"]
            )
            self.light_switch_interaction.switch_interaction(
                switch_type, refined_pose, offsets, robot_state.frame_name, config["FORCES"]
            )
            stow_arm()
            logger.info(f"Tried interaction with switch")
            
            # #################################
            # # check lamp states post interaction
            # #################################
            # move_body(POSE_CENTER, robot_state.frame_name)
            # lamp_images_post = self.light_switch_interaction.check_lamps(POSES_LAMPS, robot_state.frame_name)

            # lamp_state_changes = self.light_switch_interaction.get_lamp_state_changes(lamp_images_pre, lamp_images_post, vis_block=self.vis_block)

            # #################################
            # # add lamps to the scene graph
            # #################################
            # for idx, state_change in enumerate(lamp_state_changes):
            #     if state_change == 1 or state_change == -1:
            #         # add lamp to switch, here the scene graph gets updated
            #         light_switch_node.add_lamp(IDS_LAMPS[idx])
            #     elif state_change == 0:
            #         pass
            
            # if self.vis_block:
            #     scene_graph.visualize(labels=True, connections=True, centroids=True)

            # # Copy lamp images for future use
            # lamp_images_pre = lamp_images_post.copy()

            # Logging
            logger.info(f"Interaction with switch finished")
            switch_interaction_end_time = time.time()
            logger.info(f"Switch interaction time: {switch_interaction_end_time - switch_interaction_start_time}")

            stow_arm()


    @kernel_function(description="function to call to push a certain light switch present in the scene graph", name="push_light_switch")
    async def push_light_switch(self, light_switch_object_id: Annotated[int, "The ID of the light switch object"], 
                                object_description: Annotated[str, "A clear (3-5 words) description of the object."]) -> None:
        
        if config["robot_planner_settings"]["use_with_robot"] is not True:
            logger.info("Pushed light switch in simulation (without robot).")
            return None

        # Get object information from the scene graph
        light_switch_node = robot_state.scene_graph.nodes[light_switch_object_id]
        light_switch_centroid = light_switch_node.centroid
        sem_label = robot_state.scene_graph.label_mapping.get(light_switch_node.sem_label, "light switch")
        
        logger.info(f"Light switch with ID {light_switch_object_id} has label {sem_label} and is at position {light_switch_centroid}")
        
        # Use the furniture labels from config when needed
        light_switch_interaction_pose = get_best_pose_in_front_of_object(
            light_switch_object_id, 
            object_description=object_description, 
            min_interaction_distance=config["LIGHT_SWITCH_DISTANCE"] 
        )

        light_switch_interaction_pose_2d = light_switch_interaction_pose.to_dimension(2)

        await communication.inform_user(f"The robot is about to move to the light switch interaction position at {light_switch_interaction_pose}")
        response = await communication.ask_user("The robot would like to move to the light switch interaction position? Please enter exactly 'yes' if you want me to proceed.")

        if response == "yes":
            body_to_object_start_time = time.time()

            move_body(
                pose=light_switch_interaction_pose_2d,
                frame_name=robot_state.frame_name,
            )

            body_to_object_end_time = time.time()
            logger.info(f"Moved spot succesfully to the light switch interaction pose. Time to move body to object: {body_to_object_end_time - body_to_object_start_time}")

        else:
            await communication.inform_user("The robot will not move to the light switch interaction position and will not continue the light switch interaction.")
            return None

        await communication.inform_user(f"The robot is now about to push the light switch at position: {light_switch_centroid}")
        response = await communication.ask_user("Do you want me to push the light switch? Please enter exactly 'yes' if you want me to proceed.")
        
        if response == "yes":
            take_control_with_function(
                config=config,  # Changed from self.config to global config 
                function=self._Push_Light_Switch(), 
                light_switch_object_id=light_switch_object_id
            )
            logger.info(f"Light switch with ID {light_switch_object_id} pushed successfully")
            await communication.inform_user("Light switch interaction completed.")

        else:
            await communication.inform_user("The robot will not push the light switch.")
            
        return None

    # @kernel_function(description="function to call to pick up a certain object", name="pick_up_object")
    # async def pick_up_object(self, object_node: ObjectNode) -> None:
    #     pass

    # @kernel_function(description="function to call to place a certain object", name="place_object")
    # async def place_object(self, object_node: ObjectNode) -> None:
    #     pass

    # @kernel_function(description="function to call to open a certain drawer present in the scene graph", name="open_drawer")
    # async def open_drawer(self, drawer_node: DrawerNode) -> None:
    #     pass

    # @kernel_function(description="function to call to close a certain drawer present in the scene graph", name="close_drawer")
    # async def close_drawer(self, drawer_node: DrawerNode) -> None:
    #     pass

    # @kernel_function(description="function to call to turn a certain light switch present in the scene graph", name="turn_light_switch")
    # async def turn_light_switch(self, light_switch_node: LightSwitchNode) -> None:
    #     pass


from __future__ import annotations

# =============================================================================
# Standard Library Imports
import logging
import time
import os
import copy
import numpy as np
from pathlib import Path
from typing import Annotated, Optional, Set

# =============================================================================
# # Bosdyn and Robot Utilities
# from bosdyn.client import Sdk
# from bosdyn.api.image_pb2 import ImageResponse
# from robot_utils.basic_movements import (
#     carry_arm, stow_arm, move_body)
# from robot_utils.advanced_movement import push_light_switch, turn_light_switch, move_body_distanced, push
# from robot_utils.frame_transformer import FrameTransformerSingleton
# from robot_utils.video import (
#     localize_from_images, get_camera_rgbd, set_gripper_camera_params, set_gripper, relocalize,
#     frame_coordinate_from_depth_image, select_points_from_bounding_box
# )
# from robot_utils.base_LSARP import ControlFunction, take_control_with_function
# from utils.light_switch_interaction import LightSwitchDetection, LightSwitchInteraction
# from utils.pose_utils import calculate_light_switch_poses
# light_switch_detection = LightSwitchDetection()

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

from robot_plugins.communication import CommunicationPlugin
communication = CommunicationPlugin()


logger = logging.getLogger("plugins")
POSE_CENTER = Pose2D(coordinates=(1.5, -1))
POSE_CENTER.set_rot_from_angle(180, degrees=True)


class ItemInteractionsPlugin:
    # def __init__(self): 
    #     self.config = Config("light_switch_configs")
    #     self.light_switch_detection = LightSwitchDetection()
    #     self.frame_transformer = FrameTransformerSingleton()
    #     self.light_switch_interaction = LightSwitchInteraction(self.frame_transformer, self.config)
    #     self.vis_block = True

    # def _push_light_switch(self, light_switch_node: LightSwitchNode, scene_graph: SceneGraph) -> str:
        
    #     # # Now hard code the normal of the light switch (has to be done in the scene graph)

    #     # #################################
    #     # # Check lamp states pre interaction
    #     # #################################
    #     # lamp_images_pre = self.light_switch_interaction.check_lamps(POSES_LAMPS, frame_name)

    #     # #################################
    #     # # Move body to switch
    #     # #################################
    #     # body_to_switch_start_time = time.time()

    #     # pose = Pose3D(light_switch_node.centroid)
    #     # pose.set_rot_from_direction(light_switch_node.normal)

    #     # body_add_pose_refinement_right = Pose3D((-self.config["STAND_DISTANCE"], -0.00, -0.00))
    #     # body_add_pose_refinement_right.set_rot_from_rpy((0, 0, 0), degrees=True)
    #     # p_body = pose.copy() @ body_add_pose_refinement_right.copy()

    #     # move_body(p_body.to_dimension(2), frame_name)
    #     # logging.info(f"Moved body to switch")
        
    #     # body_to_switch_end_time = time.time()
    #     # logging.info(f"Time to move body to switch: {body_to_switch_end_time - body_to_switch_start_time}")

    #     # Set the pose of the light switch that we have to push
    #     pose = Pose3D(light_switch_node.centroid)
    #     pose.set_rot_from_direction(light_switch_node.normal)

    #     #################################
    #     # Extend the arm to a neutral carrying position
    #     #################################
    #     carry_arm() 

    #     #################################
    #     # refine handle position
    #     #################################
    #     refined_pose, refined_box, color_response = self.light_switch_interaction.get_average_refined_switch_pose(
    #         pose, 
    #         robot_state.frame_name, 
    #         self.config["REFINEMENT_X_OFFSET"],
    #         num_refinement_poses=self.config["NUM_REFINEMENT_POSES"],
    #         num_refinement_max_tries=self.config["NUM_REFINEMENTS_MAX_TRIES"],
    #         bounding_box_optimization=True
    #     )

    #     # #################################
    #     # # Push light switch (without affordance detection)
    #     # #################################
    #     # if refined_pose is not None:
    #     #     push_light_switch(refined_pose, robot_state.frame_name, z_offset=True, forces=self.config["FORCES"])
    #     # else:
    #     #     logging.warning(f"Refined pose is None for switch")
    #     #     logging.warning(f"Pushing light switch without refinement")
    #     #     push_light_switch(pose, robot_state.frame_name, z_offset=True, forces=self.config["FORCES"])

    #     # stow_arm()
        
    #     #################################
    #     # affordance detection
    #     #################################
    #     logging.info("affordance detection starting...")
    #     affordance_dict = self.light_switch_detection.light_switch_affordance_detection(refined_box, color_response, 
    #                                             self.config["AFFORDANCE_DICT_LIGHT_SWITCHES"], self.config["gpt_api_key"])

    #     #################################
    #     #  light switch interaction based on affordance
    #     #################################
    #     switch_interaction_start_time = time.time()

    #     offsets, switch_type = self.light_switch_interaction.determine_switch_offsets_and_type(affordance_dict, self.config["GRIPPER_HEIGHT"], self.config["GRIPPER_WIDTH"])
    #     self.light_switch_interaction.switch_interaction(switch_type, refined_pose, offsets, robot_state.frame_name, self.config["FORCES"])
    #     stow_arm()
    #     logging.info(f"Tried interaction with switch")
        
    #     # #################################
    #     # # check lamp states post interaction
    #     # #################################
    #     # move_body(POSE_CENTER, robot_state.frame_name)
    #     # lamp_images_post = self.light_switch_interaction.check_lamps(POSES_LAMPS, robot_state.frame_name)

    #     # lamp_state_changes = self.light_switch_interaction.get_lamp_state_changes(lamp_images_pre, lamp_images_post, vis_block=self.vis_block)

    #     # #################################
    #     # # add lamps to the scene graph
    #     # #################################
    #     # for idx, state_change in enumerate(lamp_state_changes):
    #     #     if state_change == 1 or state_change == -1:
    #     #         # add lamp to switch, here the scene graph gets updated
    #     #         light_switch_node.add_lamp(IDS_LAMPS[idx])
    #     #     elif state_change == 0:
    #     #         pass
        
    #     # if self.vis_block:
    #     #     scene_graph.visualize(labels=True, connections=True, centroids=True)

    #     # # Copy lamp images for future use
    #     # lamp_images_pre = lamp_images_post.copy()

    #     # # Logging
    #     # logging.info(f"Interaction with switch finished")
    #     # switch_interaction_end_time = time.time()
    #     # logging.info(f"Switch interaction time: {switch_interaction_end_time - switch_interaction_start_time}")
    #     # end_time_total = time.time()
    #     # logging.info(f"Total time per switch: {end_time_total - body_to_switch_start_time}")

    #     # stow_arm()

    #     #################################
    #     # Move back to the center of the scene  
    #     #################################
    #     move_body(POSE_CENTER, robot_state.frame_name)

    #     return robot_state.frame_name

    @kernel_function(description="function to call to pick up a certain object", name="pick_up_object")
    async def pick_up_object(self, object_node: ObjectNode, scene_graph: SceneGraph) -> None:
        pass

    @kernel_function(description="function to call to place a certain object", name="place_object")
    async def place_object(self, object_node: ObjectNode, scene_graph: SceneGraph) -> None:
        pass

    @kernel_function(description="function to call to open a certain drawer present in the scene graph", name="open_drawer")
    async def open_drawer(self, drawer_node: DrawerNode, scene_graph: SceneGraph) -> None:
        pass

    @kernel_function(description="function to call to close a certain drawer present in the scene graph", name="close_drawer")
    async def close_drawer(self, drawer_node: DrawerNode, scene_graph: SceneGraph) -> None:
        pass

    @kernel_function(description="function to call to turn a certain light switch present in the scene graph", name="turn_light_switch")
    async def turn_light_switch(self, light_switch_node: LightSwitchNode, scene_graph: SceneGraph) -> None:
        pass


    # @kernel_function(description="function to call to push a certain light switch present in the scene graph", name="push_light_switch")
    # async def push_light_switch(self, light_switch_node: LightSwitchNode, scene_graph: SceneGraph) -> None:
    #     pass
    #     config = Config()

    #     await communication.inform_user("The robot is about to push the light switch as position: " + str(light_switch_node.centroid))

    #     response = await communication.ask_user("Do you want to proceed with the interaction? Please reply with 'yes' or 'no'.")

    #     if response == "yes":
    #         take_control_with_function(config, function=self._push_light_switch, light_switch_node=light_switch_node, scene_graph=scene_graph, body_assist=True)
    #     else:
    #         await communication.inform_user("The robot will not push the light switch.")

    #     logging.info("Light switch pushed")

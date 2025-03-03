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
from bosdyn.api.image_pb2 import ImageResponse
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
from utils.light_switch_interaction import LightSwitchDetection, LightSwitchInteraction
from utils.pose_utils import calculate_light_switch_poses

# =============================================================================
# Semantic Kernel
from semantic_kernel.functions.kernel_function_decorator import kernel_function
from planner_core.interfaces import AbstractLlmChat

# =============================================================================
# Scene Graph
from LostFound.src.graph_nodes import LightSwitchNode, DrawerNode, ObjectNode



import logging
from pathlib import Path
from typing import Annotated, Optional, Set

logger = logging.getLogger("plugins")
POSE_CENTER = Pose2D(coordinates=(1.5, -1))
POSE_CENTER.set_rot_from_angle(180, degrees=True)


class ItemInteractionsPlugin:
    def __init__(self): 
        self.config = Config("light_switch_configs")
        self.light_switch_detection = LightSwitchDetection()
        self.frame_transformer = FrameTransformerSingleton()
        self.light_switch_interaction = LightSwitchInteraction(self.frame_transformer, self.config)

    @kernel_function(description="function to call to push a certain light switch", name="RobotNavigation")
    def push_light_switch(self, light_switch_node: LightSwitchNode):
        
        # Now hard code the normal of the light switch (has to be done in the scene graph)
        light_switch_node.set_normal(np.array([-1, 0, 0]))

        sem_label_lamp = next((k for k, v in scene_graph.label_mapping.items() if v == "lamp"), None)
        lamp_nodes = [node for node in scene_graph.nodes.values() if node.sem_label == sem_label_lamp]
        POSES_LAMPS = [Pose3D(node.centroid) for node in lamp_nodes]
        IDS_LAMPS = [node.object_id for node in lamp_nodes]

        #################################
        # localization of spot based on camera images and depth scans
        #################################
        start_time = time.time()
        set_gripper_camera_params('640x480')

        frame_name = localize_from_images(self.config, vis_block=False)

        end_time_localization = time.time()
        logging.info(f"Localization time: {end_time_localization - start_time}")

        #################################
        # Move spot to the center of the scene
        #################################
        move_body(POSE_CENTER, frame_name)
    
        #################################
        # Move body to switch
        #################################
        body_to_switch_start_time = time.time()

        pose = Pose3D(light_switch_node.centroid)
        pose.set_rot_from_direction(light_switch_node.normal)

        body_add_pose_refinement_right = Pose3D((-self.config["STAND_DISTANCE"], -0.00, -0.00))
        body_add_pose_refinement_right.set_rot_from_rpy((0, 0, 0), degrees=True)
        p_body = pose.copy() @ body_add_pose_refinement_right.copy()

        move_body(p_body.to_dimension(2), frame_name)
        logging.info(f"Moved body to switch")
        
        body_to_switch_end_time = time.time()
        logging.info(f"Time to move body to switch: {body_to_switch_end_time - body_to_switch_start_time}")

        #################################
        # Extend the arm to a neutral carrying position
        #################################
        carry_arm() 

        #################################
        # Push light switch
        #################################
        push_light_switch(pose, frame_name, z_offset=True, forces=self.config["FORCES"])

        #################################
        # refine handle position
        #################################
        refined_pose, refined_box, color_response = self.light_switch_interaction.get_average_refined_switch_pose(
            pose, 
            frame_name, 
            self.config["REFINEMENT_X_OFFSET"],
            num_refinement_poses=self.config["NUM_REFINEMENT_POSES"],
            num_refinement_max_tries=self.config["NUM_REFINEMENTS_MAX_TRIES"],
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

        # Copy lamp images fo
        lamp_images_pre = lamp_images_post.copy()

        # Logging
        logging.info(f"Interaction with switch {idx+1} of {len(switch_nodes)} finished")
        switch_interaction_end_time = time.time()
        logging.info(f"Switch interaction time: {switch_interaction_end_time - switch_interaction_start_time}")
        end_time_total = time.time()
        logging.info(f"Total time per switch: {end_time_total - body_to_switch_start_time}")

        stow_arm()
        return frame_name
from __future__ import annotations

# =============================================================================
# Standard Library Imports
import logging
import time
import os
import copy
import ast
from dotenv import dotenv_values
import numpy as np
from pathlib import Path
from typing import Annotated, Optional, Set, List
from scipy.spatial.transform import Rotation

# =============================================================================
# Bosdyn and Robot Utilities
from bosdyn.client import Sdk
from bosdyn.api.image_pb2 import ImageResponse
from robot_utils.basic_movements import (
    carry_arm, gaze, stow_arm, move_body)

from robot_utils.frame_transformer import FrameTransformerSingleton
from robot_utils.video import get_camera_rgbd, set_gripper_camera_params
from robot_utils.base_LSARP import ControlFunction, take_control_with_function
from robot_utils.object_interaction_utils import get_best_pose_in_front_of_object
from robot_plugins.user_communication import CommunicationPlugin
from utils.light_switch_interaction import LightSwitchDetection, LightSwitchInteraction
from utils.pose_utils import calculate_light_switch_poses
from utils.mask3D_interface import get_coordinates_from_item
from utils.point_clouds import body_planning, get_radius_env_cloud
from utils import graspnet_interface

# =============================================================================
# Custom Utilities
from utils.coordinates import Pose3D, Pose2D, pose_distanced, average_pose3Ds
from utils.recursive_config import Config

# =============================================================================
# Semantic Kernel
from semantic_kernel.functions.kernel_function_decorator import kernel_function

from planner_core.robot_state import RobotStateSingleton
from robot_utils.advanced_movement import positional_grab

robot_state = RobotStateSingleton()
frame_transformer = FrameTransformerSingleton()
communication = CommunicationPlugin()

logger = logging.getLogger("plugins")
general_config = Config()
object_interaction_config = Config("object_interaction_configs")
planner_settings = dotenv_values(".env_core_planner")
use_robot = general_config["robot_planner_settings"]["use_with_robot"]

class ItemInteractionsPlugin:
    """Plugin for interacting with objects in the scene."""
    
    class _Push_Light_Switch(ControlFunction):
        light_switch_detection = LightSwitchDetection()
        light_switch_interaction = LightSwitchInteraction(frame_transformer, object_interaction_config)
        
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
            boxes = self.light_switch_detection.predict_light_switches(color_response[0], vis_block=True)
            logger.info("INITIAL LIGHT SWITCH DETECTION")
            logger.info("Number of detected switches: %d", len(boxes))
            end_time_detection = time.time()

            poses = calculate_light_switch_poses(boxes, depth_image_response, robot_state.frame_name, frame_transformer)
            logger.info("Number of calculated poses: %d", len(poses))
            end_time_pose_calculation = time.time()
            logger.info("Pose calculation time: %.2f", end_time_pose_calculation - end_time_detection)
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
            if light_switch_node.affordance_dict is not None:
                affordance_dict = light_switch_node.affordance_dict
            else:
                #################################
                # affordance detection
                #################################
                logger.info("affordance detection starting...")
                affordance_dict = self.light_switch_interaction.light_switch_affordance_detection(
                    refined_box, color_response, 
                    config["AFFORDANCE_DICT_LIGHT_SWITCHES"], planner_settings.get("OPENAI_API_KEY")
                )

            #################################
            #  light switch interaction based on affordance
            #################################
            logger.info("Determining switch offsets and type...")
            offsets, switch_type = self.light_switch_interaction.determine_switch_offsets_and_type(
                affordance_dict, config["GRIPPER_HEIGHT"], config["GRIPPER_WIDTH"]
            )

            logger.info("Actually pressing the light switch now...")
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
    async def push_light_switch(self, light_switch_object_id: Annotated[int, "The ID of the light switch object"], object_description: Annotated[str, "A clear (3-5 words) description of the object."]) -> str:
        
        if not use_robot:
            logger.info("Pushed light switch in simulation (without robot).")
            # Add a way in the scene graph to confirm that the light switch has been pushed (or other things changed)
            light_switch_node = robot_state.scene_graph.nodes[light_switch_object_id]
            light_switch_node.interactions_with_object.append("pressed") # Log interaction
            feedback = f"Light switch with ID {light_switch_object_id} pushed successfully"
            logger.info(feedback)
            return feedback

        # Get object information from the scene graph
        light_switch_node = robot_state.scene_graph.nodes[light_switch_object_id]
        light_switch_centroid = light_switch_node.centroid
        sem_label = robot_state.scene_graph.label_mapping.get(light_switch_node.sem_label, "light switch")
        
        logger.info(f"Light switch with ID {light_switch_object_id} hfas label {sem_label} and is at position {light_switch_centroid}")
        
        # # Check if the light switch is closer than the light switch interaction distance
        # distance_to_light_switch = np.linalg.norm(light_switch_centroid - frame_transformer.get_current_body_position_in_frame(robot_state.frame_name))
        
        # if distance_to_light_switch < config["LIGHT_SWITCH_DISTANCE"]:
        #     logger.info("Great! The light switch is already in the range of motion of the robot.")
        #     return None

        # Use the furniture labels from config when needed
        light_switch_interaction_pose = get_best_pose_in_front_of_object(
            light_switch_object_id, 
            object_description=object_description, 
            min_interaction_distance=object_interaction_config["LIGHT_SWITCH_DISTANCE"] 
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
                config=object_interaction_config,  # Changed from self.config to global config 
                function=self._Push_Light_Switch(), 
                light_switch_object_id=light_switch_object_id
            )
            light_switch_node.interactions_with_object.append("pressed") # Log interaction
            feedback = f"Light switch with ID {light_switch_object_id} pushed successfully"
            logger.info(feedback)
            return feedback

        else:
            feedback = "The robot will not push the light switch since the user did not confirm the action."
            logger.info(feedback)
            return feedback

    # @kernel_function(description="function to call to pick up a certain object", name="pick_up_object")
    # async def pick_up_object(self, object_id: Annotated[int, "The ID of the object to pick up"]) -> None:
        

        
        
    @kernel_function(description="function to call to grasp a certain object", name="grasp_object")
    async def grasp_object(self, object_id: Annotated[int, "The ID of the object to grasp"]) -> str:
        if not use_robot:
            if robot_state.object_in_gripper is not None:
                sem_label = robot_state.scene_graph.label_mapping.get(robot_state.object_in_gripper.sem_label)
                feedback = f"The robot is already holding an object ({sem_label} with ID {robot_state.object_in_gripper.object_id}). Please place the other object down first."
                logger.info(feedback)
                return feedback
            
            # Check that the robot is close enought to the object
            if not np.linalg.norm(robot_state.virtual_robot_pose - robot_state.scene_graph.nodes[object_id].centroid) < object_interaction_config["OBJECT_GRASP_DISTANCE"]:
                feedback = f"The robot is not close enough to the object with ID {object_id} to grasp it. Please move the robot closer to the object first."
                logger.info(feedback)
                return feedback
            
            object_node = robot_state.scene_graph.nodes[object_id]
            robot_state.object_in_gripper = object_node
            object_node.interactions_with_object.append("grasped object") # Log interaction
            feedback = f"Grasped object with ID {object_id}."
            logger.info(feedback)
            return feedback
        
        # TODO: Implement actual robot logic for grasping
        object_node = robot_state.scene_graph.nodes[object_id]
        object_coordinates = Pose3D(object_node.centroid)
        sem_label = robot_state.scene_graph.label_mapping.get(object_node.sem_label, "object")
        
        mask_path = general_config.get_subpath("prescans")
        ending = general_config["pre_scanned_graphs"]["high_res"]
        mask_path = os.path.join(mask_path, ending)

        pc_path = general_config.get_subpath("aligned_point_clouds")
        ending = general_config["pre_scanned_graphs"]["high_res"]
        pc_path = os.path.join(str(pc_path), ending, "scene.ply")

        instance_index = 0 
        item_cloud, environment_cloud = get_coordinates_from_item(
            sem_label, mask_path, pc_path, instance_index
        )
        lim_env_cloud = get_radius_env_cloud(item_cloud, environment_cloud, 0.5)

        logger.info("Starting graspnet request.")
        tf_matrix, widths, scores = graspnet_interface.predict_full_grasp(
            item_cloud,
            lim_env_cloud,
            general_config,
            logger,
            rotation_resolution=16,
            top_n=2,
            vis_block=False,
        )

        # best_grasp_idx = np.argmax(scores)
        # tf_matrix = tf_matrix[best_grasp_idx]
        # width = widths[best_grasp_idx]

        ###############################################################################
        ################################ ARM COMMANDS #################################
        ###############################################################################

        grasp_pose = Pose3D.from_matrix(tf_matrix)
        carry_arm(True)
        # unstow_arm(robot, robot_command_client, True)

        # correct tf_matrix, we need to rotate by 90 degrees
        correct_roll_matrix = Rotation.from_euler(
            "xyz", (-90, 0, 0), degrees=True
        ).as_matrix()
        roll = Pose3D(rot_matrix=correct_roll_matrix)
        grasp_pose = grasp_pose @ roll

        positional_grab(
            grasp_pose,
            0.1,
            -0.05,
            robot_state.frame_name,
            already_gripping=False,
        )

        carry_arm()


    @kernel_function(description="Function to call to place a certain object somewhere. Before placing an object at a location, the robot should have navigated to the location first.", name="place_object")
    async def place_object(self,
                           object_id: Annotated[int, "The ID of the object to place"],
                           placing_3d_coordinates: Annotated[str, "A string representation of a list containing the 3D coordinates (e.g., '[-0.1, 0.2, 0.9]') for the location to place the object."]) -> str:
        if not use_robot:
            
            # Check that the robot is holding an object
            if robot_state.object_in_gripper is None:
                feedback = "The robot is not holding any object. Please grasp an object first."
                logger.info(feedback)
                return feedback
            
            sem_label = robot_state.scene_graph.label_mapping.get(robot_state.scene_graph.nodes[object_id].sem_label, "object")
            
            # convert the string to a numpy array
            try:
                placing_3d_coordinates_list = ast.literal_eval(placing_3d_coordinates)
                placing_3d_coordinates_np = np.array(placing_3d_coordinates_list)  # Convert list to numpy array
            except (ValueError, SyntaxError) as e:
                feedback = f"Error parsing placing_3d_coordinates: {e}. Please provide coordinates as a string representation of a list, e.g., '[-0.1, 0.2, 0.9]'."
                logger.error(feedback)
                return feedback
            
            if not np.linalg.norm(placing_3d_coordinates_np - robot_state.virtual_robot_pose) < object_interaction_config["OBJECT_GRASP_DISTANCE"]:
                # first make the robot navigate to the placing location
                robot_state.virtual_robot_pose = placing_3d_coordinates_np
                feedback = f"Successfully navigated the robot to the placing location and placed object with ID {object_id}, semantic label {sem_label} at the location {placing_3d_coordinates}."
                
                # feedback = f"To place an object somewhere, you have to first navigate to the location of where you want to place the object. Currently you are at {robot_state.virtual_robot_pose} and you wanted to place the object at {placing_3d_coordinates}."
                logger.info(feedback)
                return feedback
            # This can be simply solved be adding automatic navigation to the location first, when the robot is not close to the placing location
            
            # Place the object at the new location (for now the same location as the robot just navigated to)
            object_node = robot_state.scene_graph.nodes[object_id]
            robot_state.object_in_gripper = None
            object_node.centroid = placing_3d_coordinates_np
            
            # Log the interaction
            if not hasattr(object_node, 'interactions_with_object'):
                object_node.interactions_with_object = []
            object_node.interactions_with_object.append("placed object") # Log interaction
            
            feedback = f"Placed object with ID {object_id} at location {placing_3d_coordinates}"
            logger.info(feedback)
            logger.info("Object interaction logged in the scene graph.")
            return feedback
        
        # TODO: Implement actual robot logic for placing
        pass

    @kernel_function(description="function to call to open a certain drawer present in the scene graph", name="open_drawer")
    async def open_drawer(self, drawer_id: Annotated[int, "The ID of the drawer to open"]) -> str:
        if not use_robot:
            # Get drawer node
            drawer_node = robot_state.scene_graph.nodes[drawer_id]
            
            # Check if the drawer is already open
            if hasattr(drawer_node, 'is_open') and drawer_node.is_open:
                feedback = f"Drawer with ID {drawer_id} is already open."
                logger.info(feedback)
                return feedback
            
            # Open the drawer
            logger.info(f"Opening drawer with ID {drawer_id}.")
            drawer_node.interactions_with_object.append("opened")  # Log interaction
            
            # Set the is_open attribute to True
            drawer_node.is_open = True
            
            feedback = f"Opened drawer with ID {drawer_id}."
            logger.info(feedback)
            logger.info("Drawer interaction logged in the scene graph.")
            return feedback

        # TODO: Implement actual robot logic for opening drawer
        pass

    @kernel_function(description="function to call to close a certain drawer present in the scene graph", name="close_drawer")
    async def close_drawer(self, drawer_id: Annotated[int, "The ID of the drawer to close"]) -> str:
        if not use_robot:
            # Get drawer node
            drawer_node = robot_state.scene_graph.nodes[drawer_id]
            
            # Check if the drawer is already closed
            if hasattr(drawer_node, 'is_open') and not drawer_node.is_open:
                feedback = f"Drawer with ID {drawer_id} is already closed."
                logger.info(feedback)
                return feedback
            
            # Close the drawer
            logger.info(f"Closing drawer with ID {drawer_id}.")
            drawer_node.interactions_with_object.append("closed")  # Log interaction
            
            # Set the is_open attribute to False
            drawer_node.is_open = False
            
            feedback = f"Closed drawer with ID {drawer_id}."
            logger.info(feedback)
            logger.info("Drawer interaction logged in the scene graph.")
            return feedback
            
        # TODO: Implement actual robot logic for closing drawer
        pass
    
    @kernel_function(description="function to call to use a certain object", name="use_object")
    async def use_object(self, object_id: Annotated[int, "The ID of the object to use"], description_of_use: Annotated[str, "A clear (3-5 words) description on how to use the object"]) -> str:
        if not use_robot:
            logger.info(f"Used object with ID {object_id} ({description_of_use}) in simulation (without robot).")
            object_node = robot_state.scene_graph.nodes[object_id]
            
            # Check if interactions_with_object exists
            if not hasattr(object_node, 'interactions_with_object'):
                object_node.interactions_with_object = []
            object_node.interactions_with_object.append(description_of_use)
            
            feedback = f"Used object with ID {object_id} ({description_of_use}) in simulation (without robot)."
            logger.info(feedback)
            logger.info("Object interaction logged in the scene graph.")
            return feedback

        # TODO: Implement actual robot logic for using object
        pass


from __future__ import annotations

# =============================================================================
# Standard Library Imports
import logging
import random
import time
import os
import copy
import numpy as np
from typing import Annotated, List, Optional
# =============================================================================
# Robot Utilities
from robot_utils.basic_movements import (
    carry_arm, stow_arm, move_body, gaze, carry, move_arm
)
from robot_utils.advanced_movement import push_light_switch, turn_light_switch, move_body_distanced, push
from robot_utils.video import (
    localize_from_images, get_camera_rgbd, set_gripper_camera_params, set_gripper, relocalize,
    frame_coordinate_from_depth_image, select_points_from_bounding_box
)
from robot_utils.base_LSARP import ControlFunction , take_control_with_function

from utils.pose_utils import calculate_light_switch_poses
from utils.light_switch_interaction import LightSwitchDetection
light_switch_detection = LightSwitchDetection()

# =============================================================================
# Custom Utilities
from utils.recursive_config import Config
from utils.coordinates import Pose2D, Pose3D, average_pose3Ds, pose_distanced

from source.LostFound.src.graph_nodes import LightSwitchNode

# =============================================================================
# Singletons
from robot_utils.frame_transformer import FrameTransformerSingleton
from planner_core.robot_state import RobotStateSingleton
frame_transformer = FrameTransformerSingleton()
robot_state = RobotStateSingleton()

# =============================================================================
# Semantic Kernel
from semantic_kernel.functions.kernel_function_decorator import kernel_function

# =============================================================================
# Plugins
from robot_plugins.communication import CommunicationPlugin
communication = CommunicationPlugin()


logger = logging.getLogger("plugins")

class InspectionPlugin:
    """This plugin contains functions to inspect certain objects in the scene."""
    general_config = Config()
    global object_interaction_config
    object_interaction_config = Config("object_interaction_configs")
    
    class _Inspect_Object_With_Gaze(ControlFunction):
        def __init__(self):
            super().__init__()
            # self.color_image = None
            # self.depth_image = None

        def __call__(
            self,
            config: Config, 
            object_id: int,
            object_centroid_pose: Pose3D,
            *args,
            **kwargs,
        ) -> bool:  # Return success/failure flag instead of the actual images
            try:
                logging.info("Starting object inspection with gaze")
                
                set_gripper_camera_params('1920x1080')
                carry()
                gaze(object_centroid_pose, robot_state.frame_name, gripper_open=True)
                
                # Get images
                rgbd_response = get_camera_rgbd(
                    in_frame="image",
                    vis_block=False, 
                    cut_to_size=False,
                )
                
                # Validate response
                if not rgbd_response or len(rgbd_response) < 2:
                    logging.error(f"Invalid RGBD response: got {len(rgbd_response) if rgbd_response else 0} items")
                    return False
                
                # Unpack the response
                depth_tuple = rgbd_response[0]
                color_tuple = rgbd_response[1]
                
                # Validate tuples
                if len(depth_tuple) != 2 or len(color_tuple) != 2:
                    logging.error("Invalid response format from camera")
                    return False
                    
                depth_image, depth_response = depth_tuple
                color_image, color_response = color_tuple
                
                # Validate images
                if depth_image is None or color_image is None:
                    logging.error("Received None for depth or color image")
                    return False
                
                stow_arm()
                logging.info("Successfully captured images")
                logging.info(f"Captured images: color_shape={color_image.shape}, depth_shape={depth_image.shape}")
                
                # Store in robot state
                robot_state.set_image_state(color_image)
                robot_state.set_depth_image_state(depth_image)

                if robot_state.image_state is None or robot_state.depth_image_state is None:
                    logging.error("Failed to save images to robot state")
                    return False
                
                robot_state.save_image_state(f"inspection_object_{object_id}")
                return True  # Return success flag
                
            except Exception as e:
                logging.error(f"Error in _Inspect_Object_With_Gaze: {str(e)}")
                stow_arm()  # Always try to return to a safe position
                return False


    class _Calculate_LightSwitch_Poses(ControlFunction):
        def __call__(
            self,
            *args,
            **kwargs,
        ) -> List[Pose3D]:

            #################################
            # Detect the light switch bounding boxes and poses in the scene
            #################################
            boxes = light_switch_detection.predict_light_switches(robot_state.image_state, vis_block=True)
            logging.info(f"INITIAL LIGHT SWITCH DETECTION")
            logging.info(f"Number of detected switches: {len(boxes)}")

            poses = calculate_light_switch_poses(boxes, robot_state.depth_image_state, robot_state.frame_name, frame_transformer)
            logging.info(f"Number of calculated poses: {len(poses)}")

            #################################
            # Add the light switches to the scene graph (not fully implemented)
            #################################
            for pose in poses:
                light_switch_node = LightSwitchNode(pose)
                robot_state.scene_graph.add_node(light_switch_node)

            return poses

    @kernel_function(description="After having navigated to an object/furniture, you can call this function to inspect the object with gaze and save the image to your memory.")
    async def inspect_object_with_gaze(self, object_id: Annotated[int, "ID of the object in the scene graph"]) -> None:
        # Check if object exists in scene graph
        if object_id not in robot_state.scene_graph.nodes:
            await communication.inform_user(f"Object with ID {object_id} not found in scene graph.")
            return None
            
        centroid_pose = Pose3D(robot_state.scene_graph.nodes[object_id].centroid)
        response = await communication.ask_user(f"The robot would like to inspect object with id {object_id} and centroid {centroid_pose} with a gaze. Do you want to proceed? Please enter exactly 'yes' if you want to proceed.")   
        
        if response == "yes":
            # Create an instance we can reference after execution
            inspection_func = self._Inspect_Object_With_Gaze()
            
            # Call function and get success/failure flag
            logging.info("Calling take_control_with_function")
            take_control_with_function(
                function=inspection_func, 
                config=self.general_config, 
                object_id=object_id,
                object_centroid_pose=centroid_pose
            )
            logging.info(f"Completed inspecting of object with id {object_id} and centroid {centroid_pose} successfully (including saving to robot state).")
        else:
            await communication.inform_user("I will not inspect the object.")
            
        return None






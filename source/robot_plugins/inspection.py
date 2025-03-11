from __future__ import annotations

# =============================================================================
# Standard Library Imports
import logging
import random
import time
import os
import copy
import numpy as np
from typing import List
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

# =============================================================================
# Custom Utilities
from utils.recursive_config import Config
from utils.coordinates import Pose2D, Pose3D, average_pose3Ds, pose_distanced
from utils.pose_utils import calculate_light_switch_poses
from utils.light_switch_interaction import LightSwitchDetection
from ..LostFound.src.graph_nodes import LightSwitchNode
light_switch_detection = LightSwitchDetection()


# =============================================================================
# Singletons
from robot_utils.frame_transformer import FrameTransformerSingleton
from source.planner_core.robot_state import RobotStateSingleton

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
    general_config = Config()
    global object_interaction_config
    object_interaction_config = Config("light_switch_configs")
    stand_distance = object_interaction_config["STAND_DISTANCE"]
    
    class _Inspect_Object_With_Gaze(ControlFunction):
        def __call__(
            self,
            object_centroid_pose: Pose3D,
            *args,
            **kwargs,
        ) -> None:

            set_gripper_camera_params('1920x1080')
            carry()
            gaze(object_centroid_pose, robot_state.frame_name, gripper_open=True)

            depth_image_response, color_response = get_camera_rgbd(
                in_frame="image",
                vis_block=False,
                cut_to_size=False,
            )
            stow_arm()

            robot_state.image_state = color_response[0]
            robot_state.depth_image_state = depth_image_response[0]

    class _Calculate_LightSwitch_Poses(ControlFunction):
        def __call__(
            self,
            *args,
            **kwargs,
        ) -> List[Pose3D]:

            #################################
            # Detect the light switch bounding boxes and poses in the scene
            #################################
            boxes = light_switch_detection.predict_light_switches(robot_state.image_state[0], vis_block=True)
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

    @kernel_function(description="After having navigated to an object, you can call this function to inspect the object with gaze and save the image to your memory.")
    async def inspect_object_with_gaze(self, object_id: int):
        pass
        # object_centroid_pose = robot_state.scene_graph.nodes[object_id].centroid

        # # A function to calculate the normal is necessary! For now we hardcode it with [-1, 0, 0]
        # # HARDCODED
        # interaction_normal_of_object = np.array([-1, 0, 0])

        # await communication.inform_user( f"Moving to object with id {object_id}, label {robot_state.scene_graph.nodes[object_id].sem_label} and centroid {object_centroid_pose}." 
        #                                 f"The current position of the robot is {frame_transformer.get_current_body_position_in_frame(robot_state.frame_name)}")
        
        # response = await communication.ask_user("Do you want me to move to the object? Please enter exactly 'yes' if you want me to move to the object.")   
        # if response == "yes":
        #     take_control_with_function(config=self.general_config, function=self._Move_To_Object(), object_centroid_pose=object_centroid_pose, interaction_normal_of_object=interaction_normal_of_object, body_assist=True)
        #     logging.info(f"Robot moved to the object with id {object_id}")
        # else:
        #     await communication.inform_user("I will not move to the object.")

        
    
    


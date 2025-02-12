from __future__ import annotations
import logging
import time
import numpy as np
import copy
import matplotlib.pyplot as plt
import sys
import os
import pandas as pd
from typing import List, Optional
from bosdyn.client import Sdk
from robot_utils.base import ControlFunction, take_control_with_function
from robot_utils.basic_movements import carry_arm, stow_arm, move_body, gaze, move_arm
from robot_utils.advanced_movement import push_light_switch, turn_light_switch
from robot_utils.frame_transformer import FrameTransformerSingleton
from robot_utils.video import localize_from_images, get_camera_rgbd, set_gripper_camera_params
from utils.coordinates import Pose3D, Pose2D, average_pose3Ds
from utils.recursive_config import Config
from utils.singletons import (
    GraphNavClientSingleton,
    ImageClientSingleton,
    RobotCommandClientSingleton,
    RobotSingleton,
    RobotStateClientSingleton,
    WorldObjectClientSingleton,
)
from utils.pose_utils import calculate_light_switch_poses
from utils.bounding_box_refinement import refine_bounding_box
import re
import textwrap
from utils.light_switch_detection import predict_light_switches
from utils.affordance_detection_light_switch import compute_advanced_affordance_VLM_GPT4, check_lamp_state
from utils.object_detetion import BBox, Detection, Match

# Import LostFound modules
from LostFound.src import (
    SceneGraph,
    preprocess_scan,
)

# The drawer_integration is only in the Spotlight repo!!
from scenegraph.drawer_integration import parse_txt

frame_transformer = FrameTransformerSingleton()
graph_nav_client = GraphNavClientSingleton()
image_client = ImageClientSingleton()
robot_command_client = RobotCommandClientSingleton()
robot = RobotSingleton()
robot_state_client = RobotStateClientSingleton()
world_object_client = WorldObjectClientSingleton()


AFFORDANCE_DICT = {"button type": ["push button switch", "rotating switch", "none"],
                   "button count": ["single", "double", "none"],
                   "button position (wrt. other button!)": ["buttons stacked vertically", "buttons side-by-side", "none"],
                   "interaction inference from symbols": ["top/bot push", "left/right push", "center push", "no symbols present"]}

config = Config()
API_KEY = config["gpt_api_key"]
STAND_DISTANCE = 0.75
GRIPPER_WIDTH = 0.03
GRIPPER_HEIGHT = 0.03
ADVANCED_AFFORDANCE = True
FORCES = [10, 0, 0, 0, 0, 0]#8


## TESTING PARAMETERS
TEST_NUMBER = 5
RUN =10
LEVEL = "lower"

POSE_CENTER = Pose2D(coordinates=(1.5, -1))
POSE_CENTER.set_rot_from_angle(180, degrees=True)


DETECTION_DISTANCE = 0.75

NUM_REFINEMENT_POSES = 3
NUM_REFINEMENTS_MAX_TRIES = 1
BOUNDING_BOX_OPTIMIZATION = True
SHUFFLE = False

base_path = config.get_subpath("prescans")
ending = config["pre_scanned_graphs"]["high_res"]
SCAN_DIR = os.path.join(base_path, ending)


def get_scene_graph(SCAN_DIR: str, categories_to_remove: Optional[List[str]] = ["curtain", "door"], transform_to_spot_frame: bool = True) -> SceneGraph:
    """
    This function builds a semantic 3D scene graph based on the instance segmentated 3D point clouds by Mask3D
    
    Args:
        SCAN_DIR (str): The directory of the prescan data (prescans, defined in config.yaml).
        categories_to_remove (Optional[List[str]]): The object categories to remove from the scene graph, default is "curtain" and "door".
        transform_to_spot_frame (bool): Whether to transform the scene graph to the coordinate system of the Spot robot.

    Returns:
        SceneGraph: The scene graph object.
    """
    # instantiate the label mapping for Mask3D object classes (would change if using different 3D instance segmentation model)
    label_map = pd.read_csv(SCAN_DIR + '/mask3d_label_mapping.csv', usecols=['id', 'category'])
    mask3d_label_mapping = pd.Series(label_map['category'].values, index=label_map['id']).to_dict()
    preprocess_scan(SCAN_DIR, drawer_detection=True, light_switch_detection=True)
    T_ipad = np.load(SCAN_DIR + "/aruco_pose.npy")
    immovable=["armchair", "bookshelf", "end table", "shelf", "coffee table", "dresser"]
    scene_graph = SceneGraph(label_mapping=mask3d_label_mapping, min_confidence=0.2, immovable=immovable, pose=T_ipad)
    scene_graph.build(SCAN_DIR, drawers=False, light_switches=False)

    # potentially remove a category
    for category in categories_to_remove:
        scene_graph.remove_category(category)

    scene_graph.color_with_ibm_palette()
    
    if transform_to_spot_frame:
        # to transform to Spot coordinate system:
        T_spot = parse_txt(os.path.join(SCAN_DIR, "icp_tform_ground.txt"))
        scene_graph.change_coordinate_system(T_spot)  # where T_spot is a 4x4 transformation matrix of the aruco marker in Spot coordinate system

    return scene_graph

def refinement(pose: Pose3D, frame_name: str, bb_optimization: bool = True):

    depth_image_response, color_response = get_camera_rgbd(
        in_frame="image", vis_block=False, cut_to_size=False
    )
    ref_boxes = predict_light_switches(color_response[0], vis_block=False)

    #################################
    # test for refined boxes
    #################################
    a = 2
    if bb_optimization:
        boxes = []
        for ref_box in ref_boxes:
            bb = np.array([ref_box.xmin, ref_box.ymin, ref_box.xmax, ref_box.ymax])
            bb_refined = refine_bounding_box(color_response[0], bb, vis_block=False)
            bb_refined = BBox(bb_refined[0], bb_refined[1], bb_refined[2], bb_refined[3])
            boxes.append(bb_refined)
        ref_boxes = boxes
    ###############################
    # test for refined boxes
    ###############################

    refined_posess = calculate_light_switch_poses(ref_boxes, depth_image_response, frame_name, frame_transformer)
    # filter refined poses
    distances = np.linalg.norm(
        np.array([refined_pose.coordinates for refined_pose in refined_posess]) - pose.coordinates, axis=1)

    # handle not finding the correct bounding box
    if distances.min() > 0.1:  # 0.05
        return None, None, None
    else:
        idx = np.argmin(distances)
        refined_pose = refined_posess[idx]
        # refined_poses.append(refined_pose)
        refined_box = ref_boxes[idx]
        # refined_boxes.append(bbox)
        # color_responses.append(color_response[0])

        return refined_pose, refined_box, color_response[0]

def determine_offsets(affordance_dict: dict):

    offsets = []
    if affordance_dict["button type"] == "rotating switch":
        offsets.append([0.0, 0.0, 0.0])
        return offsets

    elif affordance_dict["button type"] == "push button switch":
        if affordance_dict["button count"] == "single":
            if affordance_dict["interaction inference from symbols"] == "top/bot push":
                offsets.append([0.0, 0.0, GRIPPER_HEIGHT / 2])
                offsets.append([0.0, 0.0, -GRIPPER_HEIGHT / 2])
            elif affordance_dict["interaction inference from symbols"] == "left/right push":
                offsets.append([0.0, GRIPPER_WIDTH / 2, 0.0])
                offsets.append([0.0, -GRIPPER_WIDTH / 2, 0.0])
            elif affordance_dict["interaction inference from symbols"] == "no symbols present" or affordance_dict[
                "interaction inference from symbols"] == "center push":
                offsets.append([0.0, 0.0, 0.0])
            else:
                logging.warning(f"AFFORDANCE ERROR: {affordance_dict} NOT EXPECTED")
                return None
        elif affordance_dict["button count"] == "double":
            if affordance_dict["button position (wrt. other button!)"] == "buttons side-by-side":
                if affordance_dict["interaction inference from symbols"] == "top/bot push":
                    offsets.append([0.0, GRIPPER_WIDTH / 2, GRIPPER_HEIGHT / 2])
                    offsets.append([0.0, GRIPPER_WIDTH / 2, -GRIPPER_HEIGHT / 2])
                    offsets.append([0.0, -GRIPPER_WIDTH / 2, GRIPPER_HEIGHT / 2])
                    offsets.append([0.0, -GRIPPER_WIDTH / 2, -GRIPPER_HEIGHT / 2])
                elif affordance_dict["interaction inference from symbols"] == "left/right push":
                    logging.warning(f"AFFORDANCE ERROR: {affordance_dict} NOT EXPECTED")
                    return None
                elif affordance_dict["interaction inference from symbols"] == "no symbols present" or affordance_dict[
                    "interaction inference from symbols"] == "center push":
                    offsets.append([0.0, GRIPPER_WIDTH / 2, 0.0])
                    offsets.append([0.0, -GRIPPER_WIDTH / 2, 0.0])
                else:
                    logging.warning(f"AFFORDANCE ERROR: {affordance_dict} NOT EXPECTED")
                    return None
            elif affordance_dict["button position (wrt. other button!)"] == "buttons stacked vertically":
                if affordance_dict["interaction inference from symbols"] == "no symbols present" or affordance_dict[
                    "interaction inference from symbols"] == "center push":
                    offsets.append([0.0, 0.0, GRIPPER_HEIGHT / 2])
                    offsets.append([0.0, 0.0, -GRIPPER_HEIGHT / 2])
                else:
                    logging.warning(f"AFFORDANCE ERROR: {affordance_dict} NOT EXPECTED")
                    return None
            elif affordance_dict["button position (wrt. other button!)"] == "none":
                logging.warning(f"AFFORDANCE ERROR: {affordance_dict} NOT EXPECTED")
                return None
            else:
                logging.warning(f"AFFORDANCE ERROR: {affordance_dict} NOT EXPECTED")
                return None
        return offsets
    else:
        print("THATS NOT A LIGHT SWITCH!")
        return None

def interaction(affordance_dict: dict, refined_pose: Pose3D, offsets: List[List], frame_name: str):

    if affordance_dict["button type"] == "rotating switch":
        turn_light_switch(refined_pose, frame_name)
        return None

    if affordance_dict["button type"] == "push button switch":
        for offset_coords in offsets:
            pose_offset = copy.deepcopy(refined_pose)
            pose_offset.coordinates += np.array(offset_coords)
            push_light_switch(pose_offset, frame_name, z_offset=True, forces=FORCES)
        return None


def check_lamps(lamp_poses: List[Pose3D], frame_name: str):

    carry_arm(body_assist=True)
    lamp_images = []
    for lamp_pose in lamp_poses:
        gaze(lamp_pose, frame_name)
        depth_image_response, color_response = get_camera_rgbd(
            in_frame="image",
            vis_block=False,
            cut_to_size=False,
        )
        lamp_images.append(color_response[0])

    stow_arm()

    return lamp_images


def get_lamp_state_changes(lamp_images_1: List[np.ndarray], lamp_images_2: List[np.ndarray], vis_block: bool = True):

    option_1_patterns = [
        r"lamp goes from off to on",  # Exact match
        r"1",  # Match number
    ]
    option_2_patterns = [
        r"lamp goes from on to off",
        r"2",
    ]
    option_3_patterns = [
        r"lamp state does not change",
        r"3",
    ]

    lamp_states = []
    for img_1, img_2 in zip(lamp_images_1, lamp_images_2):

        resp = check_lamp_state(img_1, img_2, API_KEY)
        lamp_state = resp
        resp = resp.strip().lower()

        for pattern in option_1_patterns:
            if re.search(pattern, resp):
                change = 1
                lamp_states.append(change)
                break
        for pattern in option_2_patterns:
            if re.search(pattern, resp):
                change = -1
                lamp_states.append(change)
                break
        for pattern in option_3_patterns:
            if re.search(pattern, resp):
                change = 0
                lamp_states.append(change)
                break

        if vis_block:
            figure, ax = plt.subplots(1, 2)
            ax[0].imshow(img_1)
            ax[1].imshow(img_2)
            wrapped_title = "\n".join(textwrap.wrap(resp, width=60))
            figure.suptitle(f"Lamp state change: {wrapped_title}")
            plt.show()

    return lamp_states


class _Push_Light_Switch(ControlFunction):
    def __call__(
        self,
        config: Config,
        sdk: Sdk,
        *args,
        **kwargs,
    ) -> str:

        # get scene graph and visualize it
        scene_graph = get_scene_graph(SCAN_DIR)
        scene_graph.visualize(labels=True, connections=True, centroids=True)


        start_time = time.time()
        set_gripper_camera_params('640x480')
        
        # get light switch and lamp nodes
        sem_label_switch = next((k for k, v in scene_graph.label_mapping.items() if v == "light switch"), None)
        sem_label_lamp = next((k for k, v in scene_graph.label_mapping.items() if v == "lamp"), None)
        lamp_nodes = [node for node in scene_graph.nodes.values() if node.sem_label == sem_label_lamp]

        switch_nodes = []
        for node in scene_graph.nodes.values():
            if node.sem_label == sem_label_switch:
                # compute face normal of light switch
                # the normal is set to -x direction (HARDCODED)
                node.set_normal(np.array([-1, 0, 0])) 
                switch_nodes.append(node)

        POSES_LAMPS = [Pose3D(node.centroid) for node in lamp_nodes]
        IDS_LAMPS = [node.object_id for node in lamp_nodes]

        frame_name = localize_from_images(config, vis_block=False)

        end_time_localization = time.time()
        logging.info(f"Localization time: {end_time_localization - start_time}")

        
        #################################
        # check lamp states pre interaction
        #################################

        set_gripper_camera_params('1280x720')
        move_body(POSE_CENTER, frame_name)
        lamp_images_pre = check_lamps(POSES_LAMPS, frame_name)

        switch_nodes.reverse()

        for idx, switch in enumerate(switch_nodes):
            pose_start_time = time.time()

            pose = Pose3D(switch.centroid)
            pose.set_rot_from_direction(switch.normal)

            body_add_pose_refinement_right = Pose3D((-STAND_DISTANCE, -0.00, -0.00))
            body_add_pose_refinement_right.set_rot_from_rpy((0, 0, 0), degrees=True)
            p_body = pose.copy() @ body_add_pose_refinement_right.copy()
            move_body(p_body.to_dimension(2), frame_name)
            logging.info(f"Moved body to switch {idx+1} of {len(switch_nodes)}")
            end_time_move_body = time.time()
            logging.info(f"Move body time: {end_time_move_body - pose_start_time}")

            carry_arm()

            push_light_switch(pose, frame_name, z_offset=True, forces=FORCES)

            #################################
            # refine handle position
            #################################

            x_offset = -0.2 # -0.2

            camera_add_pose_refinement_right = Pose3D((x_offset, -0.05, -0.04))
            camera_add_pose_refinement_right.set_rot_from_rpy((0, 0, 0), degrees=True)
            camera_add_pose_refinement_left = Pose3D((x_offset, 0.05, -0.04))
            camera_add_pose_refinement_left.set_rot_from_rpy((0, 0, 0), degrees=True)
            camera_add_pose_refinement_bot = Pose3D((x_offset, -0.0, -0.1))
            camera_add_pose_refinement_bot.set_rot_from_rpy((0, 0, 0), degrees=True)
            camera_add_pose_refinement_top = Pose3D((x_offset, -0.0, -0.02))
            camera_add_pose_refinement_top.set_rot_from_rpy((0, 0, 0), degrees=True)


            if NUM_REFINEMENT_POSES == 4:
                ref_add_poses = [camera_add_pose_refinement_right, camera_add_pose_refinement_left,
                                 camera_add_pose_refinement_bot, camera_add_pose_refinement_top]
            if NUM_REFINEMENT_POSES == 3:
                ref_add_poses = [camera_add_pose_refinement_right, camera_add_pose_refinement_left,
                                 camera_add_pose_refinement_bot]
            elif NUM_REFINEMENT_POSES == 1:
                ref_add_poses = [camera_add_pose_refinement_right]

            elif NUM_REFINEMENT_POSES == 2:
                ref_add_poses = [camera_add_pose_refinement_right, camera_add_pose_refinement_left]

            refined_poses = []
            refined_boxes = []
            color_responses = []
            count = 0
            while count < NUM_REFINEMENTS_MAX_TRIES:
                if len(refined_poses) == 0:
                    logging.info(f"Refinement try {count+1} of {NUM_REFINEMENTS_MAX_TRIES}")
                    for idx_ref_pose, ref_pose in enumerate(ref_add_poses):
                        p = pose.copy() @ ref_pose.copy()
                        move_arm(p, frame_name, body_assist=True)
                        try:
                            refined_pose, refined_box, color_response = refinement(pose, frame_name, BOUNDING_BOX_OPTIMIZATION)
                            if refined_pose is not None:
                                refined_poses.append(refined_pose)
                                refined_boxes.append(refined_box)
                                color_responses.append(color_response)
                        except:
                            logging.warning(f"Refinement try {count+1} failed at refinement pose {idx_ref_pose+1} of {len(ref_add_poses)}")
                            continue
                else:
                    logging.info(f"Refinement exited or finished at try {count} of {NUM_REFINEMENTS_MAX_TRIES}")
                    push_light_switch(pose, frame_name, z_offset=True, forces=FORCES)
                    break
                count += 1
                time.sleep(1)

            try:
                refined_pose = average_pose3Ds(refined_poses)
            except:
                logging.warning(f"Refinement failed, no average pose could be calculated")
                push_light_switch(pose, frame_name, z_offset=True, forces=FORCES)
                stow_arm()
                continue

            logging.info(f"Refinement finished, average pose calculated")
            logging.info(f"Number of refined poses: {len(refined_poses)}")
            end_time_refinement = time.time()
            logging.info(f"Refinement time: {end_time_refinement - end_time_move_body}")
            logging.info("affordance detection starting...")
            #################################
            # affordance detection
            #################################

            refined_box = refined_boxes[-1]
            cropped_image = color_responses[-1][int(refined_box.ymin):int(refined_box.ymax), int(refined_box.xmin):int(refined_box.xmax)]
            plt.imshow(cropped_image)
            plt.show()
            affordance_dict = compute_advanced_affordance_VLM_GPT4(cropped_image, AFFORDANCE_DICT, API_KEY)
            print(affordance_dict)
            logging.info(f"Affordance detection finished")
            end_time_affordance = time.time()
            logging.info(f"Affordance time: {end_time_affordance - end_time_refinement}")

            #################################
            #  calc interaction based on affordance
            #################################
            offsets = determine_offsets(affordance_dict)

            #################################
            #  interaction
            ################################
            interaction(affordance_dict, refined_pose, offsets, frame_name)
            stow_arm()
            logging.info(f"Tried interaction with switch {idx + 1} of {len(switch_nodes)}")

            #################################
            # check lamp states post interaction
            #################################
            move_body(POSE_CENTER, frame_name)
            lamp_images_post = check_lamps(POSES_LAMPS, frame_name)

            lamp_state_changes = get_lamp_state_changes(lamp_images_pre, lamp_images_post)

            for idx, state_change in enumerate(lamp_state_changes):
                if state_change == 1 or state_change == -1:
                    # add lamp to switch
                    switch.add_lamp(IDS_LAMPS[idx])
                elif state_change == 0:
                    pass

            scene_graph.visualize(labels=True, connections=True, centroids=True)

            lamp_images_pre = lamp_images_post.copy()

            logging.info(f"Interaction with switch {idx+1} of {len(switch_nodes)} finished")
            end_time_switch = time.time()
            logging.info(f"Switch interaction time: {end_time_switch - end_time_affordance}")
            end_time_total = time.time()
            logging.info(f"total time per switch: {end_time_total - pose_start_time}")
            a = 2

        stow_arm()
        return frame_name


def main():
    config = Config()
    take_control_with_function(config, function=_Push_Light_Switch(), body_assist=True)


if __name__ == "__main__":
    main()

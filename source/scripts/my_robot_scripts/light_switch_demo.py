from __future__ import annotations

import logging
import random
import time

import cv2
import numpy as np
import os
import copy
import matplotlib.pyplot as plt
from typing import List

from bosdyn.client import Sdk
from robot_utils.advanced_movement import move_body_distanced, push
from robot_utils.base import ControlFunction, take_control_with_function
from robot_utils.basic_movements import carry_arm, stow_arm, move_body, gaze, carry, move_arm_distanced, move_arm
from robot_utils.advanced_movement import push_light_switch, turn_light_switch
from robot_utils.frame_transformer import FrameTransformerSingleton
from robot_utils.video import localize_from_images, get_camera_rgbd, set_gripper_camera_params, set_gripper, relocalize
from scipy.spatial.transform import Rotation
from utils.coordinates import Pose3D, Pose2D, pose_distanced, average_pose3Ds
from utils.recursive_config import Config
from utils.singletons import (
    GraphNavClientSingleton,
    ImageClientSingleton,
    RobotCommandClientSingleton,
    RobotSingleton,
    RobotStateClientSingleton,
    WorldObjectClientSingleton,
)
from utils.light_switch_interaction import LightSwitchDetection
from utils.affordance_detection_light_switch import compute_affordance_VLM_GPT4, compute_advanced_affordance_VLM_GPT4
from bosdyn.api.image_pb2 import ImageResponse
from utils.object_detetion import BBox, Detection, Match
from robot_utils.video import frame_coordinate_from_depth_image, select_points_from_bounding_box

frame_transformer = FrameTransformerSingleton()
graph_nav_client = GraphNavClientSingleton()
image_client = ImageClientSingleton()
robot_command_client = RobotCommandClientSingleton()
robot = RobotSingleton()
robot_state_client = RobotStateClientSingleton()
world_object_client = WorldObjectClientSingleton()

from utils.pose_utils import calculate_light_switch_poses
from utils.bounding_box_refinement import refine_bounding_box
from random import uniform

from utils.pose_utils import (
    determine_handle_center,
    find_plane_normal_pose,
    calculate_handle_poses,
    cluster_handle_poses,
    filter_handle_poses,
    refine_handle_position,

)

AFFORDANCE_CLASSES = {0: "SINGLE PUSH",
                          1: "DOUBLE PUSH",
                          2: "ROTATING",
                          3: "something else"}

AFFORDANCE_DICT = {"button type": ["push button switch", "rotating switch", "none"],
                   "button count": ["single", "double", "none"],
                   "button position (wrt. other button!)": ["buttons stacked vertically", "buttons side-by-side", "none"],
                   "interaction inference from symbols": ["top/bot push", "left/right push", "center push", "no symbols present"]}


config = Config()
API_KEY = config["gpt_api_key"]
STAND_DISTANCE = 0.9 #1.0
GRIPPER_WIDTH = 0.03
GRIPPER_HEIGHT = 0.03
ADVANCED_AFFORDANCE = True
FORCES = [8, 0, 0, 0, 0, 0]
WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
LOGGING_PATH = os.path.join(WORKSPACE_ROOT, "logging_lightswitch_experiments")

# Create logging directory if it doesn't exist
os.makedirs(LOGGING_PATH, exist_ok=True)

## TESTING PARAMETERS
TEST_NUMBER = 8
RUN =1
LEVEL = "upper"


DETECTION_DISTANCE = 0.75
X_BODY = 1.6
Y_BODY = 0.6
ANGLE_BODY = 180
SHUFFLE = True
NUM_REFINEMENT_POSES = 4
NUM_REFINEMENTS_MAX_TRIES = 5
BOUNDING_BOX_OPTIMIZATION = True

if LEVEL == "upper":
    X_CABINET = 0.15
    Y_CABINET = 0.85  # 0.9
    Z_CABINET = 0.5  # 0.42
elif LEVEL == "lower":
    X_CABINET = 0.15
    Y_CABINET = 0.85
    Z_CABINET = 0.0
elif LEVEL == "both":
    X_CABINET = 0.15
    Y_CABINET = 0.9
    Z_CABINET = 0.25
    X_BODY = 2.3
    Y_BODY = 0.6

if TEST_NUMBER == 1:
    # 1. STAND AT FIXED DISTANCE AND ANGLE, PUSH ALL SWITCHES IN RANDOMIZED ORDER
    pass
elif TEST_NUMBER == 2:
    # 2. STAND AT FIXED DISTANCE AND VARIABLE ANGLE, PUSH ALL SWITCHES IN RANDOMIZED ORDER
    angle = uniform(-22.5,22.5)
    print(angle)
    X_BODY = X_BODY * np.cos(np.deg2rad(angle))
    Y_BODY = Y_BODY + X_BODY * np.sin(np.deg2rad(angle))
elif TEST_NUMBER == 3:
    # 3. STAND FURTHER AWAY AT FIXED DISTANCE AND VARIABLE ANGLE, PUSH ALL SWITCHES IN RANDOMIZED ORDER
    X_BODY = 2.1
    angle = uniform(-22.5, 22.5)
    print(angle)
    X_BODY = X_BODY * np.cos(np.deg2rad(angle))
    Y_BODY = Y_BODY + X_BODY * np.sin(np.deg2rad(angle))
    if LEVEL == "upper":
        Z_CABINET = 0.6
    elif LEVEL == "lower":
        Z_CABINET = -0.1
elif TEST_NUMBER == 4:
    # 4. STAND AT FIXED DISTANCE AND ANGLE, SINGLE REFINEMENT POSE, PUSH ALL SWITCHES IN RANDOMIZED ORDER
    NUM_REFINEMENT_POSES = 1
    NUM_REFINEMENTS_MAX_TRIES = 1
elif TEST_NUMBER == 5:
    BOUNDING_BOX_OPTIMIZATION = False
elif TEST_NUMBER == 6:
    BOUNDING_BOX_OPTIMIZATION = False
    angle = uniform(-22.5,22.5)
    print(angle)
    X_BODY = X_BODY * np.cos(np.deg2rad(angle))
    Y_BODY = Y_BODY + X_BODY * np.sin(np.deg2rad(angle))
elif TEST_NUMBER == 7:
    NUM_REFINEMENT_POSES = 2
    NUM_REFINEMENTS_MAX_TRIES = 1
elif TEST_NUMBER == 8:
    NUM_REFINEMENT_POSES = 4
    NUM_REFINEMENTS_MAX_TRIES = 5
    BOUNDING_BOX_OPTIMIZATION = True
    SHUFFLE = False
    DETECTION_DISTANCE = 0.75
    X_BODY = 1.4
    Y_BODY = 0.85
    ANGLE_BODY = 175

file_path = filename=os.path.join(LOGGING_PATH, f"switches_experiment_{TEST_NUMBER}_LEVEL_{LEVEL}_RUN_{RUN}.log")
if os.path.exists(file_path):
    raise FileExistsError(f"The file '{file_path}' already exists.")

logging.getLogger("understanding-spot.192.168.50.3").disabled = True
logging.basicConfig(filename=file_path, filemode='w', format='%(name)s - %(levelname)s - %(message)s')

logging.info(f"ANGLE_BODY: {ANGLE_BODY}")
logging.info(f"X_BODY: {X_BODY}")
logging.info(f"Y_BODY: {Y_BODY}")
logging.info(f"X_CABINET: {X_CABINET}")
logging.info(f"Y_CABINET: {Y_CABINET}")
logging.info(f"Z_CABINET: {Z_CABINET}")
logging.info(f"STAND_DISTANCE: {STAND_DISTANCE}")
logging.info(f"GRIPPER_WIDTH: {GRIPPER_WIDTH}")
logging.info(f"GRIPPER_HEIGHT: {GRIPPER_HEIGHT}")
logging.info(f"ADVANCED_AFFORDANCE: {ADVANCED_AFFORDANCE}")
logging.info(f"FORCES: {FORCES}")
logging.info(f"LOGGING_PATH: {LOGGING_PATH}")
logging.info(f"TEST_NUMBER: {TEST_NUMBER}")
logging.info(f"RUN: {RUN}")
logging.info(f"LEVEL: {LEVEL}")
logging.info(f"DETECTION_DISTANCE: {DETECTION_DISTANCE}")
logging.info(f"SHUFFLE: {SHUFFLE}")
logging.info(f"NUM_REFINEMENT_POSES: {NUM_REFINEMENT_POSES}")
logging.info(f"NUM_REFINEMENTS_MAX_TRIES: {NUM_REFINEMENTS_MAX_TRIES}")

light_switch_detection = LightSwitchDetection()

def refinement(pose: Pose3D, frame_name: str, bb_optimization: bool = True):

    depth_image_response, color_response = get_camera_rgbd(
        in_frame="image", vis_block=False, cut_to_size=False
    )

    ref_boxes = light_switch_detection.predict_light_switches(color_response[0], vis_block=True)

    #################################
    # refined boxes
    #################################
    if bb_optimization:
        boxes = []
        for ref_box in ref_boxes:
            bb = np.array([ref_box.xmin, ref_box.ymin, ref_box.xmax, ref_box.ymax])
            bb_refined = refine_bounding_box(color_response[0], bb, vis_block=True)
            bb_refined = BBox(bb_refined[0], bb_refined[1], bb_refined[2], bb_refined[3])
            boxes.append(bb_refined)
        ref_boxes = boxes
    ###############################
    # refined boxes
    ###############################

    refined_posess = calculate_light_switch_poses(ref_boxes, depth_image_response, frame_name, frame_transformer)
    # filter refined poses
    distances = np.linalg.norm(
        np.array([refined_pose.coordinates for refined_pose in refined_posess]) - pose.coordinates, axis=1)

    # handle not finding the correct bounding box
    if distances.min() > 0.05:  # 0.05
        return None, None, None
    else:
        idx = np.argmin(distances)
        refined_pose = refined_posess[idx]
        refined_box = ref_boxes[idx]
        return refined_pose, refined_box, color_response[0]

class _Push_Light_Switch(ControlFunction):
    def __call__(
        self,
        config: Config,
        sdk: Sdk,
        *args,
        **kwargs,
    ) -> str:


        if TEST_NUMBER == 2 or TEST_NUMBER == 3:
            logging.info(f"angle: {angle}")

        logging.info(f"ANGLE_BODY: {ANGLE_BODY}")
        logging.info(f"X_BODY: {X_BODY}")
        logging.info(f"Y_BODY: {Y_BODY}")
        logging.info(f"X_CABINET: {X_CABINET}")
        logging.info(f"Y_CABINET: {Y_CABINET}")
        logging.info(f"Z_CABINET: {Z_CABINET}")
        logging.info(f"STAND_DISTANCE: {STAND_DISTANCE}")
        logging.info(f"GRIPPER_WIDTH: {GRIPPER_WIDTH}")
        logging.info(f"GRIPPER_HEIGHT: {GRIPPER_HEIGHT}")
        logging.info(f"ADVANCED_AFFORDANCE: {ADVANCED_AFFORDANCE}")
        logging.info(f"FORCES: {FORCES}")
        logging.info(f"LOGGING_PATH: {LOGGING_PATH}")
        logging.info(f"TEST_NUMBER: {TEST_NUMBER}")
        logging.info(f"RUN: {RUN}")
        logging.info(f"LEVEL: {LEVEL}")
        logging.info(f"DETECTION_DISTANCE: {DETECTION_DISTANCE}")
        logging.info(f"SHUFFLE: {SHUFFLE}")
        logging.info(f"NUM_REFINEMENT_POSES: {NUM_REFINEMENT_POSES}")
        logging.info(f"NUM_REFINEMENTS_MAX_TRIES: {NUM_REFINEMENTS_MAX_TRIES}")

        start_time = time.time()

        set_gripper_camera_params('640x480')
        frame_name = localize_from_images(config, vis_block=False)

        end_time_localization = time.time()
        logging.info(f"Localization time: {end_time_localization - start_time}")

        pose = Pose2D(np.array([X_BODY, Y_BODY]))
        pose.set_rot_from_angle(ANGLE_BODY, degrees=True)
        move_body(
            pose=pose,
            frame_name=frame_name,
        )

        cabinet_pose = Pose3D((X_CABINET, Y_CABINET, Z_CABINET))
        cabinet_pose.set_rot_from_rpy((0,0,ANGLE_BODY), degrees=True)

        carry()

        set_gripper_camera_params('1920x1080')
        time.sleep(1)
        gaze(cabinet_pose, frame_name, gripper_open=True)

        depth_image_response, color_response = get_camera_rgbd(
            in_frame="image",
            vis_block=False,
            cut_to_size=False,
        )
        set_gripper_camera_params('1280x720')
        stow_arm()

        boxes = predict_light_switches(color_response[0], vis_block=True)
        logging.info(f"INITIAL LIGHT SWITCH DETECTION")
        logging.info(f"Number of detected switches: {len(boxes)}")
        end_time_detection = time.time()
        logging.info(f"Detection time: {end_time_detection - end_time_localization}")

        if SHUFFLE:
            random.shuffle(boxes)

        poses = calculate_light_switch_poses(boxes, depth_image_response, frame_name, frame_transformer)
        logging.info(f"Number of calculated poses: {len(poses)}")
        end_time_pose_calculation = time.time()
        logging.info(f"Pose calculation time: {end_time_pose_calculation - end_time_detection}")


        for idx, pose in enumerate(poses):
            pose_start_time = time.time()
            body_add_pose_refinement_right = Pose3D((-STAND_DISTANCE, -0.00, -0.00))
            body_add_pose_refinement_right.set_rot_from_rpy((0, 0, 0), degrees=True)
            p_body = pose.copy() @ body_add_pose_refinement_right.copy()
            move_body(p_body.to_dimension(2), frame_name)
            logging.info(f"Moved body to switch {idx+1} of {len(poses)}")
            end_time_move_body = time.time()
            logging.info(f"Move body time: {end_time_move_body - pose_start_time}")

            carry_arm()
            #################################
            # refine handle position
            #################################

            x_offset = -0.3 # -0.2

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
                    break
                count += 1
                time.sleep(1)

            try:
                refined_pose = average_pose3Ds(refined_poses)
            except:
                logging.warning(f"Refinement failed, no average pose could be calculated")
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
            # interaction based on affordance
            #################################
            offsets = []
            if affordance_dict["button type"] == "rotating switch":
                turn_light_switch(refined_pose, frame_name)
            elif affordance_dict["button type"] == "push button switch":
                if affordance_dict["button count"] == "single":
                    if affordance_dict["interaction inference from symbols"] == "top/bot push":
                        offsets.append([0.0, 0.0, GRIPPER_HEIGHT/2])
                        offsets.append([0.0, 0.0, -GRIPPER_HEIGHT/2])
                    elif affordance_dict["interaction inference from symbols"] == "left/right push":
                        offsets.append([0.0, GRIPPER_WIDTH/2, 0.0])
                        offsets.append([0.0, -GRIPPER_WIDTH/2, 0.0])
                    elif affordance_dict["interaction inference from symbols"] == "no symbols present" or affordance_dict["interaction inference from symbols"] == "center push":
                        offsets.append([0.0, 0.0, 0.0])
                    else:
                        logging.warning(f"AFFORDANCE ERROR: {affordance_dict} NOT EXPECTED")
                        continue
                elif affordance_dict["button count"] == "double":
                    if affordance_dict["button position (wrt. other button!)"] == "buttons side-by-side":
                        if affordance_dict["interaction inference from symbols"] == "top/bot push":
                            offsets.append([0.0, GRIPPER_WIDTH/2, GRIPPER_HEIGHT/2])
                            offsets.append([0.0, GRIPPER_WIDTH/2, -GRIPPER_HEIGHT/2])
                            offsets.append([0.0, -GRIPPER_WIDTH/2, GRIPPER_HEIGHT/2])
                            offsets.append([0.0, -GRIPPER_WIDTH/2, -GRIPPER_HEIGHT/2])
                        elif affordance_dict["interaction inference from symbols"] == "left/right push":
                            logging.warning(f"AFFORDANCE ERROR: {affordance_dict} NOT EXPECTED")
                            continue
                        elif affordance_dict["interaction inference from symbols"] == "no symbols present" or affordance_dict["interaction inference from symbols"] == "center push":
                            offsets.append([0.0, GRIPPER_WIDTH/2, 0.0])
                            offsets.append([0.0, -GRIPPER_WIDTH/2, 0.0])
                        else:
                            logging.warning(f"AFFORDANCE ERROR: {affordance_dict} NOT EXPECTED")
                            continue
                    elif affordance_dict["button position (wrt. other button!)"] == "buttons stacked vertically":
                        if affordance_dict["interaction inference from symbols"] == "no symbols present" or affordance_dict["interaction inference from symbols"] == "center push":
                            offsets.append([0.0, 0.0, GRIPPER_HEIGHT/2])
                            offsets.append([0.0, 0.0, -GRIPPER_HEIGHT/2])
                        else:
                            logging.warning(f"AFFORDANCE ERROR: {affordance_dict} NOT EXPECTED")
                            continue
                    elif affordance_dict["button position (wrt. other button!)"] == "none":
                        logging.warning(f"AFFORDANCE ERROR: {affordance_dict} NOT EXPECTED")
                        continue
                    else:
                        logging.warning(f"AFFORDANCE ERROR: {affordance_dict} NOT EXPECTED")
                        continue
                for offset_coords in offsets:
                    pose_offset = copy.deepcopy(refined_pose)
                    pose_offset.coordinates += np.array(offset_coords)
                    push_light_switch(pose_offset, frame_name, z_offset=True, forces=FORCES)
                logging.info(f"Tried interaction with switch {idx+1} of {len(poses)}")
            else:
                print("THATS NOT A LIGHT SWITCH!")

            stow_arm()
            logging.info(f"Interaction with switch {idx+1} of {len(poses)} finished")
            end_time_switch = time.time()
            logging.info(f"Switch interaction time: {end_time_switch - end_time_affordance}")
            end_time_total = time.time()
            logging.info(f"total time per switch: {end_time_total - pose_start_time}")
            a = 2

        return frame_name


def main():
    config = Config()
    take_control_with_function(config, function=_Push_Light_Switch(), body_assist=True)


if __name__ == "__main__":
    main()

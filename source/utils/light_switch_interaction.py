import numpy as np
import cv2
import supervision as sv
from ultralytics import YOLO, YOLOWorld
from ultralytics.data.dataset import YOLODataset
import cv2
import matplotlib.pyplot as plt
from utils.object_detetion import BBox
from typing import List, Tuple, Union, Dict
import logging
import time
import copy
import textwrap
import re

from robot_utils.advanced_movement import push_light_switch, turn_light_switch
from utils.coordinates import Pose3D, Pose2D, average_pose3Ds
from robot_utils.basic_movements import carry_arm, stow_arm, move_arm, gaze
from robot_utils.video import get_camera_rgbd


from utils.pose_utils import calculate_light_switch_poses
from utils.bounding_box_refinement import refine_bounding_box
from utils.affordance_detection_light_switch import compute_advanced_affordance_VLM_GPT4, check_lamp_state
from bosdyn.api.image_pb2 import ImageResponse


class LightSwitchDetection:
    def __init__(self):
        pass

    def _filter_detections_YOLOWorld(self, detections):

        # squaredness filter
        squaredness = (np.minimum(detections.xyxy[:,2] - detections.xyxy[:,0], detections.xyxy[:,3] - detections.xyxy[:,1])/
                    np.maximum(detections.xyxy[:,2] - detections.xyxy[:,0], detections.xyxy[:,3] - detections.xyxy[:,1]))

        idx_dismiss = np.where(squaredness < 0.95)[0]

        filtered_detections = sv.Detections.empty()
        filtered_detections.class_id = np.delete(detections.class_id, idx_dismiss)
        filtered_detections.confidence = np.delete(detections.confidence, idx_dismiss)
        filtered_detections.data['class_name'] = np.delete(detections.data['class_name'], idx_dismiss)
        filtered_detections.xyxy = np.delete(detections.xyxy, idx_dismiss, axis=0)

        return filtered_detections

    def _filter_detections_ultralytics(self, detections, filter_squaredness=True, filter_area=True, filter_within=True):

        detections = detections[0].cpu()
        xyxy = detections.boxes.xyxy.numpy()

        # filter squaredness outliers
        if filter_squaredness:
            squaredness = (np.minimum(xyxy[:, 2] - xyxy[:, 0],
                                    xyxy[:, 3] - xyxy[:, 1]) /
                        np.maximum(xyxy[:, 2] - xyxy[:, 0],
                                    xyxy[:, 3] - xyxy[:, 1]))

            keep_1 = squaredness > 0.5
            xyxy = xyxy[keep_1, :]

        #filter area outliers
        if filter_area:
            areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
            keep_2 = areas < 3*np.median(areas)
            xyxy = xyxy[keep_2, :]

        # filter bounding boxes within larger ones
        if filter_within:
            centers = np.array([(xyxy[:, 0] + xyxy[:, 2]) / 2, (xyxy[:, 1] + xyxy[:, 3]) / 2]).T
            keep_3 = np.ones(xyxy.shape[0], dtype=bool)
            x_in_box = (xyxy[:, 0:1] <= centers[:, 0]) & (centers[:, 0] <= xyxy[:, 2:3])
            y_in_box = (xyxy[:, 1:2] <= centers[:, 1]) & (centers[:, 1] <= xyxy[:, 3:4])
            centers_in_boxes = x_in_box & y_in_box
            np.fill_diagonal(centers_in_boxes, False)
            pairs = np.argwhere(centers_in_boxes)
            idx_remove = pairs[np.where(areas[pairs[:, 0]] - areas[pairs[:, 1]] < 0), 0].flatten()
            keep_3[idx_remove] = False
            xyxy = xyxy[keep_3, :]

        bbox = xyxy
        return bbox


    def predict_light_switches(self, image: np.ndarray, model_type: str = "yolov8", vis_block: bool = False) -> List[BBox]:
        """
        Returns a list of bounding boxes of the light switches in the image.  

        Args:
            image (np.ndarray): The image to predict the light switches from.
            model_type (str): The type of the model to use for the prediction.
            vis_block (bool): Whether to visualize the image with the predicted light switches.

        Returns:
            List[BBox]: A list of bounding boxes of the light switches in the image.
        """

        if model_type == "yolo_world":
            model = YOLOWorld("yolov8s-world.pt")
            model.set_classes(["light switch"])

            results_predict = model.predict(image)
            results_predict[0].show()

        elif model_type == "yolov8":
            model = YOLO('../../weights/train27/weights/best.pt')
            results_predict = model.predict(source=image, imgsz=1280, conf=0.15, iou=0.4, max_det=9, agnostic_nms=True,
                                            save=False)

            boxes = self._filter_detections_ultralytics(detections=results_predict)

            if vis_block:
                canv = image.copy()
                for box in boxes:
                    xB = int(box[2])
                    xA = int(box[0])
                    yB = int(box[3])
                    yA = int(box[1])

                    cv2.rectangle(canv, (xA, yA), (xB, yB), (0, 255, 0), 2)

                plt.imshow(cv2.cvtColor(canv, cv2.COLOR_BGR2RGB))
                plt.show()

            bbs = []
            for box in boxes:
                bbs.append(BBox(box[0], box[1], box[2], box[3]))

            return bbs

    def light_switch_affordance_detection(self, refined_box: BBox, color_response: ImageResponse, AFFORDANCE_DICT: dict, API_KEY: str, vis_block: bool = False) -> dict:
        """
        Detect affordances in the refined bounding boxes using GPT-4 Vision.
        
        Args:
            refined_box (BBox): A refined bounding box around the specific light switch
            color_response (ImageResponse): The color image of the light switch
            AFFORDANCE_DICT (dict): Dictionary containing possible affordance values
            api_key (str): OpenAI API key
            
        Returns:
            dict: Affordance dictionary containing the information about the detected light switch
        """
        begin_time_affordance = time.time()
        cropped_image = color_response[int(refined_box.ymin):int(refined_box.ymax), 
                                        int(refined_box.xmin):int(refined_box.xmax)]
        
        if vis_block:
            plt.imshow(cropped_image)
            plt.show()
        
        affordance_dict = compute_advanced_affordance_VLM_GPT4(cropped_image, AFFORDANCE_DICT, API_KEY)
        logging.info(f"Affordance detection finished. Returning affordance dict: {affordance_dict}")
        end_time_affordance = time.time()
        logging.info(f"Affordance happened in: {end_time_affordance - begin_time_affordance}")
        
        return affordance_dict

    def validate_light_switches(self, data_path, model_type: str = "yolov8"):

        if model_type == "yolov8":
            model = YOLO('../../weights/train27/weights/best.pt')
            metrics = model.val(data=data_path, imgsz=1280)
            result = metrics.results_dict

        elif model_type == "yolo_world":
            model = YOLOWorld("yolov8s-world.pt")
            model.set_classes(["light switch"])
            metrics = model.val(data=data_path, imgsz=1280)
            result = metrics.results_dict

        pass


class LightSwitchInteraction:
    def __init__(self, frame_transformer, config):
        self.frame_transformer = frame_transformer
        self.config = config
        self.light_switch_detection = LightSwitchDetection()

    @staticmethod
    def switch_interaction(switch_type: str, refined_pose: Pose3D, offsets: List[List], frame_name: str, FORCES: List[float]) -> None:
        """
        Interact with the light switch based on the type of light switch (saved in the affordance dictionary).

        Args:
            switch_type (str): The type of the light switch (e.g. rotating or push switch)
            refined_pose (Pose3D): The refined pose of the light switch handle
            offsets (List[List]): The offsets for the light switch interaction
            frame_name (str): The name of the frame
        """

        if switch_type == "rotating switch":
            turn_light_switch(refined_pose, frame_name)
            return None

        if switch_type == "push button switch":
            for offset_coords in offsets:
                pose_offset = copy.deepcopy(refined_pose)
                pose_offset.coordinates += np.array(offset_coords)
                push_light_switch(pose_offset, frame_name, z_offset=True, forces=FORCES)
            return None

    def get_average_refined_switch_pose(
        self, 
        pose: Pose3D, 
        frame_name: str, 
        x_offset: float, 
        num_refinement_poses: int = 3, 
        num_refinement_max_tries: int = 1,
        bounding_box_optimization: bool = True) -> Tuple[Pose3D, BBox, ImageResponse]:
        """
        Takes the average of (num_refinement_poses) refined poses close to the light switch.

        Args:
            pose (Pose3D): The pose of the centroid of the light switch handle
            frame_name (str): The name of the frame
            x_offset (float): The x offset of the light switch, calculated based on affordance_dict before
            num_refinement_poses (int): The number of refinement poses to take the average of, default is 3
            num_refinement_max_tries (int): The maximum number of tries to refine the pose, default is 1
            bounding_box_optimization (bool): Whether to optimize the bounding box, default is True

        Returns:
            Tuple[Pose3D, BBox, ImageResponse]: The refined pose, bounding box, and color image
        """
        begin_time_refinement = time.time()
        
        camera_add_pose_refinement_right = Pose3D((x_offset, -0.05, -0.04))
        camera_add_pose_refinement_right.set_rot_from_rpy((0, 0, 0), degrees=True)
        camera_add_pose_refinement_left = Pose3D((x_offset, 0.05, -0.04))
        camera_add_pose_refinement_left.set_rot_from_rpy((0, 0, 0), degrees=True)
        camera_add_pose_refinement_bot = Pose3D((x_offset, -0.0, -0.1))
        camera_add_pose_refinement_bot.set_rot_from_rpy((0, 0, 0), degrees=True)
        camera_add_pose_refinement_top = Pose3D((x_offset, -0.0, -0.02))
        camera_add_pose_refinement_top.set_rot_from_rpy((0, 0, 0), degrees=True)

        if num_refinement_poses == 4:
            ref_add_poses = [camera_add_pose_refinement_right, camera_add_pose_refinement_left,
                                camera_add_pose_refinement_bot, camera_add_pose_refinement_top]
        elif num_refinement_poses == 3:
            ref_add_poses = [camera_add_pose_refinement_right, camera_add_pose_refinement_left,
                                camera_add_pose_refinement_bot]
        elif num_refinement_poses == 2:
            ref_add_poses = [camera_add_pose_refinement_right, camera_add_pose_refinement_left]
        elif num_refinement_poses == 1:
            ref_add_poses = [camera_add_pose_refinement_right]
        else:
            raise Warning(f"Number of refinement poses not supported: {num_refinement_poses}")

        refined_poses = []

        count = 0
        while count < num_refinement_max_tries:
            if len(refined_poses) == 0:
                logging.info(f"Refinement try {count+1} of {num_refinement_max_tries}")
                for idx_ref_pose, ref_pose in enumerate(ref_add_poses):
                    p = pose.copy() @ ref_pose.copy()
                    # The arm will move to each of the 4, 3, 2, or 1 positions close to the light switch that we have set before (increased robustness)
                    move_arm(p, frame_name, body_assist=True)
                    try:
                        # for the refined)box and color_response, only the one from the last frame will be returned
                        refined_pose, refined_box, color_response = self.get_refined_switch_pose(pose, frame_name, bounding_box_optimization) 
                        if refined_pose is not None:
                            refined_poses.append(refined_pose)
                    except:
                        logging.warning(f"Refinement try {count+1} failed at refinement pose {idx_ref_pose+1} of {len(ref_add_poses)}")
                        continue
            else:
                logging.info(f"Refinement exited or finished at try {count} of {num_refinement_max_tries}")
                break
            count += 1
            time.sleep(1)

        # our refined pose is the average of the refined poses that we calculated at different positions close to the light switch
        logging.info(f"Number of refined poses: {len(refined_poses)}")
        refined_pose = average_pose3Ds(refined_poses)
        logging.info(f"Refinement finished for frame {frame_name}, average pose calculated")
        
        end_time_refinement = time.time()
        logging.info(f"Refinement time for frame {frame_name}: {end_time_refinement - begin_time_refinement}")

        return refined_pose, refined_box, color_response


    def get_refined_switch_pose(self, pose: Pose3D, frame_name: str, bb_optimization: bool = True) -> Tuple[Union[Pose3D, None], Union[BBox, None], Union[np.ndarray, None]]:
        """
        Calculates the refined pose and bounding box of the light switch handle.

        Args:
            pose (Pose3D): The pose of the centroid of the light switch handle
            frame_name (str): The name of the frame
            bb_optimization (bool): Whether to optimize the bounding box, default is True

        Returns:
            Tuple[Union[Pose3D, None], Union[BBox, None], Union[np.ndarray, None]]: The refined pose, bounding box, and color image
        """

        # get the depth and color image
        depth_image_response, color_response = get_camera_rgbd(in_frame="image", vis_block=False, cut_to_size=False)
        
        # predict the light switch bounding boxes
        ref_boxes = self.light_switch_detection.predict_light_switches(color_response[0], vis_block=False)

        if bb_optimization:
            boxes = []
            for ref_box in ref_boxes:
                bb = np.array([ref_box.xmin, ref_box.ymin, ref_box.xmax, ref_box.ymax])
                bb_refined = refine_bounding_box(color_response[0], bb, vis_block=False)
                bb_refined = BBox(bb_refined[0], bb_refined[1], bb_refined[2], bb_refined[3])
                boxes.append(bb_refined)
            ref_boxes = boxes

        # calculate the poses of the light switch handles
        refined_posess = calculate_light_switch_poses(ref_boxes, depth_image_response, frame_name, self.frame_transformer)

        # filter refined poses
        distances = np.linalg.norm(
            np.array([refined_pose.coordinates for refined_pose in refined_posess]) - pose.coordinates, axis=1)

        # handle not finding the correct bounding box
        if distances.min() > 0.1:  # 0.05
            return None, None, None
        else:
            idx = np.argmin(distances)
            refined_pose = refined_posess[idx]
            refined_box = ref_boxes[idx]

            return refined_pose, refined_box, color_response[0]

    @staticmethod
    def determine_switch_offsets_and_type(affordance_dict: dict, gripper_height: float, gripper_width: float) -> Union[Tuple[List[List[float]], str], None]:
        """
        Determines the offsets and switch type based on the affordance dictionary (output of light_switch_affordance_detection)

        Args:
            affordance_dict (dict): Affordance dictionary containing the information about the detected light switch
        
        Returns:
            Tuple[List[List[float]], str]: The offsets and switch type
        """

        offsets = []
        if affordance_dict["button type"] == "rotating switch":
            offsets.append([0.0, 0.0, 0.0])
            return offsets, affordance_dict["button type"]

        elif affordance_dict["button type"] == "push button switch":
            if affordance_dict["button count"] == "single":
                if affordance_dict["interaction inference from symbols"] == "top/bot push":
                    offsets.append([0.0, 0.0, gripper_height / 2])
                    offsets.append([0.0, 0.0, -gripper_height / 2])
                elif affordance_dict["interaction inference from symbols"] == "left/right push":
                    offsets.append([0.0, gripper_width / 2, 0.0])
                    offsets.append([0.0, -gripper_width / 2, 0.0])
                elif affordance_dict["interaction inference from symbols"] == "no symbols present" or affordance_dict[
                    "interaction inference from symbols"] == "center push":
                    offsets.append([0.0, 0.0, 0.0])
                else:
                    logging.warning(f"AFFORDANCE ERROR: {affordance_dict} NOT EXPECTED")
                    return None
            elif affordance_dict["button count"] == "double":
                if affordance_dict["button position (wrt. other button!)"] == "buttons side-by-side":
                    if affordance_dict["interaction inference from symbols"] == "top/bot push":
                        offsets.append([0.0, gripper_width / 2, gripper_height / 2])
                        offsets.append([0.0, gripper_width / 2, -gripper_height / 2])
                        offsets.append([0.0, -gripper_width / 2, gripper_height / 2])
                        offsets.append([0.0, -gripper_width / 2, -gripper_height / 2])
                    elif affordance_dict["interaction inference from symbols"] == "left/right push":
                        logging.warning(f"AFFORDANCE ERROR: {affordance_dict} NOT EXPECTED")
                        return None
                    elif affordance_dict["interaction inference from symbols"] == "no symbols present" or affordance_dict[
                        "interaction inference from symbols"] == "center push":
                        offsets.append([0.0, gripper_width / 2, 0.0])
                        offsets.append([0.0, -gripper_width / 2, 0.0])
                    else:
                        logging.warning(f"AFFORDANCE ERROR: {affordance_dict} NOT EXPECTED")
                        return None
                elif affordance_dict["button position (wrt. other button!)"] == "buttons stacked vertically":
                    if affordance_dict["interaction inference from symbols"] == "no symbols present" or affordance_dict[
                        "interaction inference from symbols"] == "center push":
                        offsets.append([0.0, 0.0, gripper_height / 2])
                        offsets.append([0.0, 0.0, -gripper_height / 2])
                    else:
                        logging.warning(f"AFFORDANCE ERROR: {affordance_dict} NOT EXPECTED")
                        return None
                elif affordance_dict["button position (wrt. other button!)"] == "none":
                    logging.warning(f"AFFORDANCE ERROR: {affordance_dict} NOT EXPECTED")
                    return None
                else:
                    logging.warning(f"AFFORDANCE ERROR: {affordance_dict} NOT EXPECTED")
                    return None
            return offsets, affordance_dict["button type"]
        else:
            print("THATS NOT A LIGHT SWITCH!")
            return None

    def check_lamps(self, lamp_poses: List[Pose3D], frame_name: str):
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

    def get_lamp_state_changes(self, lamp_images_1: List[np.ndarray], lamp_images_2: List[np.ndarray], vis_block: bool = True):
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

            resp = check_lamp_state(img_1, img_2, self.config["gpt_api_key"])
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




if __name__ == "__main__":

    data_path = "/home/cvg-robotics/tim_ws/data_switch_detection/20240706_test/data.yaml"
    LightSwitchDetection.validate_light_switches(data_path=data_path, model_type="yolov8")

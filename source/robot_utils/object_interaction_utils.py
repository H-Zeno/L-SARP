import copy
import logging
import os
from pathlib import Path
from utils.coordinates import Pose2D, Pose3D
from utils.recursive_config import Config
from utils.vis import show_two_geometries_colored
# from scripts.temp_scripts.deepseek_exploration import ask_deepseek
import json
import math
import numpy as np
from utils.point_clouds import add_coordinate_system, body_planning_front, body_planning
from utils.openmask_interface import get_mask_points
import open3d as o3d

from planner_core.robot_state import RobotStateSingleton
from robot_utils.frame_transformer import FrameTransformerSingleton

from utils.mask3D_interface import _get_list_of_items
robot_state = RobotStateSingleton()
frame_transformer = FrameTransformerSingleton()

config = Config()

H_FOV = config["robot_parameters"]["H_FOV"]
V_FOV = config["robot_parameters"]["V_FOV"]

active_scene_name = config["robot_planner_settings"]["active_scene"] 
path_to_scene_data = Path(config["robot_planner_settings"]["path_to_scene_data"])
scene_graph_json_path = Path(path_to_scene_data/ active_scene_name / "scene_graph.json")


def _get_distance_to_shelf(index: int=0, load_from_json=False) -> tuple:
    '''
    Get the distance to the front of the furniture for the robot to capture whole shelf.
    This function assumes that we will be interacting with the widest face of the furniture
    
    :param index: Index of the shelf/cabinet from the Deepseek result.
    :return: Tuple of the calculated distance and the centroid of the furniture.
    '''
    
    if load_from_json:
    
        with open(scene_graph_json_path, "r") as file:
            scene_data = json.load(file)
    
        furniture_centroid = scene_data["nodes"][index]["centroid"]
        furniture_dimensions = scene_data["nodes"][index]["dimensions"]

    else: 
        furniture_centroid = robot_state.scene_graph.nodes[index].centroid
        furniture_dimensions = robot_state.scene_graph.nodes[index].dimensions

    # calculate robot distance to furniture
    circle_radius_width = (furniture_dimensions[0] + 0.1) / (2 * math.tan(np.radians(H_FOV / 2))) + furniture_dimensions[1] / 2 
    circle_radius_height = (furniture_dimensions[2] + 0.1) / (2 * math.tan(np.radians(V_FOV / 2))) + furniture_dimensions[1] / 2
    min_distance = 0.5

    if len(furniture_dimensions) < 3:
        raise ValueError("Invalid furniture dimensions: Expected at least 3 elements.")

    logging.info(f"Calculated distance to shelf: {max(circle_radius_width, circle_radius_height)} and the furniture dimensions are {furniture_dimensions}")
    return max(circle_radius_width, circle_radius_height, min_distance), furniture_centroid


def _get_shelf_front(cabinet_pcd: o3d.geometry.PointCloud, cabinet_center: np.ndarray) -> np.ndarray:
    '''
    Get the normal of the front face of the furniture.
    
    :param cabinet_pcd: PointCloud of the shelf/cabinet. (o3d.geometry.PointCloud)
    :return: Normal of the front face.
    '''
    # get furniture oriented bounding box
    obb = cabinet_pcd.get_oriented_bounding_box()
    R = obb.R
    extents = obb.extent

    # get vertical faces
    vertical_faces = []
    for axis in range(3):
        for direction in [1, -1]:
            # calculate face normal
            normal = R[:, axis] * direction
            # check if normal is roughly horizontal (= vertical face)
            if abs(normal[2]) < 0.1:
                # calculate face dimensions
                dim1 = (axis + 1) % 3
                dim2 = (axis + 2) % 3
                area = extents[dim1] * extents[dim2]

                # Snap normal to cardinal direction in XY plane
                snapped_normal = snap_to_cardinal(normal)
                
                objects_in_front = robot_state.scene_graph.get_nodes_in_front_of_object_face(cabinet_center, snapped_normal)
                
                logging.info(f"Found vertical face with original normal {normal}, snapped to {snapped_normal}. Area: {area}. Objects in front: {objects_in_front}")
                
                vertical_faces.append({
                    'normal': snapped_normal,
                    'original_normal': normal,
                    'area': area,
                    'objects_in_front': objects_in_front
                })

    if not vertical_faces:
        raise ValueError("No vertical faces found in shelf structure")
    
    # Remove faces that have objects in front of them
    valid_vertical_faces = [vertical_face for vertical_face in vertical_faces if not vertical_face['objects_in_front']]

    # If there are valid faces without objects, choose the largest one
    if valid_vertical_faces:
        # select largest vertical face as front
        front = max(valid_vertical_faces, key=lambda x: x['area'])
    else:
        # All faces have objects in front - get the face with normal pointing most toward robot
        try:
            robot_pos = frame_transformer.get_current_body_position_in_frame(robot_state.frame_name)
            
            # Direction vector from cabinet to robot
            direction_to_robot = robot_pos - cabinet_center
            direction_to_robot = direction_to_robot / np.linalg.norm(direction_to_robot)
            
            # Calculate the alignment of each face normal with the direction to the robot
            for face in vertical_faces:
                # Dot product - higher value means better alignment toward robot
                face['alignment_to_robot'] = np.dot(face['normal'], direction_to_robot)
            
            logging.warning("All vertical faces have objects in front of them. Selecting the face pointing most toward robot.")
            front = max(vertical_faces, key=lambda x: x['alignment_to_robot'])
        except Exception as e:
            # Fallback if robot position is unavailable
            logging.warning(f"Unable to get robot position: {e}. Selecting face with fewest obstacles.")
            front = min(vertical_faces, key=lambda x: len(x['objects_in_front']))

    # To be more accurate, this should be extended to the face with the most free volume in front of it (for 1 meter distance)
    return front['normal']

def snap_to_cardinal(normal: np.ndarray) -> np.ndarray:
    """
    Snaps a normal vector to the closest cardinal direction in the XY plane.
    
    Args:
        normal: The normal vector to snap
        
    Returns:
        The snapped normal vector (aligned to X or Y axis)
    """
    # Get the XY component and normalize it
    normal_xy = np.array([normal[0], normal[1], 0])
    if np.linalg.norm(normal_xy) < 1e-6:  # Handle zero vector case
        return np.array([1, 0, 0])  # Default to X-axis
    
    normal_xy = normal_xy / np.linalg.norm(normal_xy)
    
    # Determine the closest cardinal direction
    # We check which cardinal direction has the highest dot product
    cardinal_directions = [
        np.array([1, 0, 0]),   # +X
        np.array([0, 1, 0]),   # +Y
        np.array([-1, 0, 0]),  # -X
        np.array([0, -1, 0]),  # -Y
    ]
    
    dot_products = [np.dot(normal_xy, cardinal) for cardinal in cardinal_directions]
    best_idx = np.argmax(np.abs(dot_products))
    
    # Use the sign of the dot product to determine direction
    if dot_products[best_idx] < 0:
        return -cardinal_directions[best_idx]  # Return opposite direction
    else:
        return cardinal_directions[best_idx]

def get_pose_in_front_of_furniture(index: int=0, object_description="cabinet, shelf") -> tuple:
    '''
    Get the interaction pose for the robot in front of an object. 
    Currently this function only supports cabinets and shelves. 
    
    :param index: Index of the furniture from the Deepseek result.
    :param furniture_types: Types of furniture to consider, a string that gets feeded into Clip
    :return: Tuple of the centroid of the furniture and the body pose.
    '''
    # get necessary distance to shelf
    radius, furniture_centroid = _get_distance_to_shelf(index)
    radius = max(0.8, radius)

    furniture_sem_label = robot_state.scene_graph.nodes[index].sem_label
    logging.info(f"Looking for furniture with semantic label: {furniture_sem_label}")

    mask_path_base = config.get_subpath("masks")
    pred_masks_base_path = config.get_subpath("prescans")
    ending = config["pre_scanned_graphs"]["high_res"]
    # mask_csv_folder_path = os.path.join(mask_path_base, ending)
    pred_masks_folder = os.path.join(pred_masks_base_path, ending)

    pc_path = config.get_subpath("aligned_point_clouds")
    ending = config["pre_scanned_graphs"]["high_res"]
    pc_path = os.path.join(str(pc_path), ending, "scene.ply")

    # logging.info("Starting the process of finding the best poses in front of the object.")
    df = _get_list_of_items(str(pred_masks_folder))

    # get all entries for our item label
    entries = df[df["class_label"] == furniture_sem_label]
    if index > len(entries) or index < (-len(entries) + 1):
        index = 0

    # get the mask of the item
    entry = entries.iloc[index]
    path_ending = entry["path_ending"]
    pred_mask_file_path = os.path.join(pred_masks_folder, path_ending)

    with open(pred_mask_file_path, "r", encoding="UTF-8") as file:
        lines = file.readlines()
    good_points_bool = np.asarray([bool(int(line)) for line in lines])
    
    # read the point cloud, select by indices specified in the file
    pc = o3d.io.read_point_cloud(pc_path)

    good_points_idx = np.where(good_points_bool)[0]
    environment_cloud = pc.select_by_index(good_points_idx, invert=True)
    furniture_point_cloud = pc.select_by_index(good_points_idx)


    # # When the object is part of the scene graph, take the mask of this object that we already have from mask3d
    # cabinet_mask = robot_state.scene_graph.nodes[index].mesh_mask
    # cabinet_mask = np.asarray(cabinet_mask, dtype=bool)
    # logging.info(f"Object mask loaded from scene graph and memory for object with index {index}")
    # pcd = o3d.io.read_point_cloud(str(pcd_path))
    # logging.info(f"Point cloud loaded from file {pcd_path}")
    # logging.info(f"Size of the point cloud {len(pcd.points)}. Mask size {len(cabinet_mask)}")
    # logging.info(f"Number of points in the mask {np.sum(cabinet_mask)}")
    # cabinet_pcd = pcd.select_by_index(np.where(cabinet_mask)[0])
    # env_pcd = pcd.select_by_index(np.where(~cabinet_mask)[0])
    # logging.info(f"Environment point cloud loaded from scene graph and memory for object with index {index}")

    # # get all cabinets/shelfs in the environment
    # body_pose = None 

    # for idx in range(0, 7):
    #     # This is how Yasmin implemented it for cabinets and shelves:
    #     cabinet_pcd, env_pcd = get_mask_points(object_description, Config(), idx=idx, vis_block=True)
    #     cabinet_center = np.mean(np.asarray(cabinet_pcd.points), axis=0)

    #     # find correct cabinet/shelf
    #     logging.info(f"Cabinet center (openmask3d selected point clouds): {cabinet_center}, Furniture Centroid (Mask3D): {furniture_centroid}")
    #     # A transformation to the robot's coordinate frame is necessary here (if we use the openmask3d point clouds)
    #     if (np.allclose(cabinet_center, furniture_centroid, atol=0.1)):
    #         print("Object found!")
    #         # get normal of cabinet/shelf front face
    #         front_normal = _get_shelf_front(cabinet_pcd, cabinet_center)
    #         # calculate body position in front of cabinet/shelf
    #         body_pose = body_planning_front(
    #             env_pcd,
    #             cabinet_center,
    #             shelf_normal=front_normal,
    #             min_target_distance=radius,
    #             max_target_distance=radius+0.2,
    #             min_obstacle_distance=0.4,
    #             n=5,
    #             vis_block=True,
    #         )
    #         break

    # if body_pose is None: 
    #     # the correct object not found, feedback to the llm to change the clip embedding description
    #     raise ValueError(f"The description of the object {object_description} did not find a match with an object in the scene graph. Please provide a more accurate description.")


    # # get normal of cabinet/shelf front face (something is going wrong here with the coordinate frames)
    # furniture_point_cloud = point_cloud = o3d.geometry.PointCloud()
    # furniture_point_cloud.points = o3d.utility.Vector3dVector(robot_state.scene_graph.nodes[index].points)

    front_normal = _get_shelf_front(furniture_point_cloud, furniture_centroid)

    # Calculate robot position based on furniture position and normal direction
    body_position_3d = furniture_centroid + front_normal * radius
    body_pose = Pose2D((body_position_3d[0], body_position_3d[1]))
    direction_towards_object = -front_normal
    angle = math.atan2(direction_towards_object[1], direction_towards_object[0])
    body_pose.set_rot_from_angle(angle)
    
    # Caclulcation of the pose in front of an object based on the point clouds, very computationally expensive
    # # calculate body position in front of cabinet/shelf
    # body_pose = body_planning_front(
    #             environment_cloud,
    #             furniture_centroid,
    #             shelf_normal=front_normal,
    #             min_target_distance=radius,
    #             max_target_distance=radius+0.2,
    #             min_obstacle_distance=0.4,
    #             n=5,
    #             vis_block=True,
    #         )

    return furniture_centroid, body_pose


def get_best_poses_in_front_of_object(index: int, object_description: str) -> list[tuple[Pose3D, float]]:

    item_centroid = robot_state.scene_graph.nodes[index].centroid
    item_sem_label = robot_state.scene_graph.nodes[index].sem_label

    mask_path_base = config.get_subpath("masks")
    pred_masks_base_path = config.get_subpath("prescans")
    ending = config["pre_scanned_graphs"]["high_res"]
    # mask_csv_folder_path = os.path.join(mask_path_base, ending)
    pred_masks_folder = os.path.join(pred_masks_base_path, ending)

    pc_path = config.get_subpath("aligned_point_clouds")
    ending = config["pre_scanned_graphs"]["high_res"]
    pc_path = os.path.join(str(pc_path), ending, "scene.ply")

    # logging.info(f"Trying to find the item cloud of the object with description {object_description}.")
    
    logging.info("Starting the process of finding the best poses in front of the object.")
    df = _get_list_of_items(str(pred_masks_folder))

    # get all entries for our item label
    entries = df[df["class_label"] == item_sem_label]
    if index > len(entries) or index < (-len(entries) + 1):
        index = 0

    # get the mask of the item
    entry = entries.iloc[index]
    path_ending = entry["path_ending"]
    pred_mask_file_path = os.path.join(pred_masks_folder, path_ending)

    with open(pred_mask_file_path, "r", encoding="UTF-8") as file:
        lines = file.readlines()
    good_points_bool = np.asarray([bool(int(line)) for line in lines])

    item_cloud = robot_state.scene_graph.nodes[index].points
    
    # read the point cloud, select by indices specified in the file
    pc = o3d.io.read_point_cloud(pc_path)

    good_points_idx = np.where(good_points_bool)[0]
    environment_cloud = pc.select_by_index(good_points_idx, invert=True)
    item_cloud = pc.select_by_index(good_points_idx)

    x = copy.deepcopy(item_cloud)
    x.paint_uniform_color((1, 0, 0))
    y = copy.deepcopy(environment_cloud)
    y = add_coordinate_system(y, (1, 1, 1), (0, 0, 0))

    end_coordinates = Pose3D(item_centroid)

    robot_targets = body_planning(
        environment_cloud,
        end_coordinates,
        min_distance=0.6,
        max_distance=1,
        n_best=10,
        vis_block=True,
    )
    logging.info(f"Found {len(robot_targets)} possible poses in front of the object. They are: {robot_targets}")

    return robot_targets # Each tuple contains (pose, score)


if __name__ == "__main__":
    get_pose_in_front_of_furniture(7, "rectangular stand with a light switch")
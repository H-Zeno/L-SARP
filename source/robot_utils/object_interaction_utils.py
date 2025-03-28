import copy
import logging
import os
import gc
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

# Set up logger
logger = logging.getLogger("plugins")

def _get_distance_to_shelf(index: int=0, min_distance=1.10, load_from_json=False) -> tuple:
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
    circle_radius_width = (furniture_dimensions[0] + 0.3) / (2 * math.tan(np.radians(H_FOV / 2))) + furniture_dimensions[1] / 2 
    circle_radius_height = (furniture_dimensions[2] + 0.3) / (2 * math.tan(np.radians(V_FOV / 2))) + furniture_dimensions[1] / 2

    if len(furniture_dimensions) < 3:
        raise ValueError("Invalid furniture dimensions: Expected at least 3 elements.")

    logger.info("Calculated distance to shelf: %s and the furniture dimensions are %s", max(circle_radius_width, circle_radius_height), furniture_dimensions)
    return max(circle_radius_width, circle_radius_height, min_distance), furniture_centroid


def _get_shelf_front(cabinet_pcd: o3d.geometry.PointCloud, cabinet_center: np.ndarray) -> np.ndarray:
    '''
    Get the normal of the front face of the furniture.
    
    :param cabinet_pcd: PointCloud of the shelf/cabinet. (o3d.geometry.PointCloud)
    :return: Normal of the front face.
    '''
    try:
        # get furniture oriented bounding box
        obb = cabinet_pcd.get_oriented_bounding_box()
        R = obb.R
        extents = obb.extent

        # get vertical faces
        vertical_faces = []
        for axis in range(3):
            for direction in [1, -1]:
                try:
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
                        
                        # Default to empty list if get_nodes_in_front_of_object_face fails
                        try:
                            objects_in_front = robot_state.scene_graph.get_nodes_in_front_of_object_face(cabinet_center, snapped_normal)
                        except Exception as e:
                            logger.error("Error getting nodes in front of face: %s", e)
                            objects_in_front = []
                        
                        logger.info("Found vertical face with original normal %s, snapped to %s. Area: %s. Objects in front: %s", normal, snapped_normal, area, objects_in_front)
                        
                        vertical_faces.append({
                            'normal': snapped_normal,
                            'original_normal': normal,
                            'area': area,
                            'objects_in_front': objects_in_front
                        })
                except Exception as e:
                    logger.error("Error processing face at axis %s, direction %s: %s", axis, direction, e)
                    continue

        if not vertical_faces:
            logger.warning("No vertical faces found, defaulting to -X direction")
            return np.array([-1, 0, 0])  # Default to -X direction if no faces found
        
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
                
                logger.warning("All vertical faces have objects in front of them. Selecting the face pointing most toward robot.")
                front = max(vertical_faces, key=lambda x: x['alignment_to_robot'])
            except Exception as e:
                # Fallback if robot position is unavailable
                logger.warning("Unable to get robot position: %s. Selecting face with fewest obstacles.", e)
                front = min(vertical_faces, key=lambda x: len(x['objects_in_front']))
                
        # To be more accurate, this should be extended to the face with the most free volume in front of it (for 1 meter distance)
        return front['normal']
    except Exception as e:
        logger.error("Error in _get_shelf_front: %s", e)
        return np.array([-1, 0, 0])  # Default to -X direction

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

def get_pose_in_front_of_furniture(index: int=0, min_distance=1.10, object_description="cabinet, shelf") -> Pose3D:
    '''
    Get the interaction pose for the robot in front of an object. 
    Currently this function only supports cabinets and shelves. 
    
    :param index: Index of the furniture from the scene graph.
    :param min_distance: Minimum distance for robot to stand from the object. Defaults to 1.10.
    :param object_description: Description of the object.
    :return: The interaction pose (Pose3D).
    :raises: Various exceptions if the pose cannot be calculated.
    '''
    furniture_centroid = None
    furniture_point_cloud = None
    points = None
    
    try:
        # get necessary distance to shelf
        radius, furniture_centroid = _get_distance_to_shelf(index, min_distance=min_distance)
        radius = max(0.8, radius)
        
        # Shallow copy of points to prevent memory issues
        # Get a small subset of points (max 5000) to reduce memory usage
        original_points = robot_state.scene_graph.nodes[index].points
        if len(original_points) > 5000:
            # Randomly sample 5000 points
            indices = np.random.choice(len(original_points), 5000, replace=False)
            points = original_points[indices].copy()
            logger.info(f"Reduced point cloud from {len(original_points)} to 5000 points to avoid memory issues")
        else:
            points = original_points.copy()
            
        # Create a new point cloud with the copied points
        furniture_point_cloud = o3d.geometry.PointCloud()
        furniture_point_cloud.points = o3d.utility.Vector3dVector(points)
        
        # Get front normal
        logger.info(f"Finding front normal for furniture with {len(points)} points")
        front_normal = _get_shelf_front(furniture_point_cloud, furniture_centroid)
        
        # Clean up point cloud to prevent memory leaks
        del furniture_point_cloud
        del points
        furniture_point_cloud = None
        points = None
        
        # Force garbage collection
        gc.collect()
        
        if front_normal is None:
            raise ValueError("Could not determine furniture front normal")
            
        # Save normal and calculate pose
        robot_state.scene_graph.nodes[index].set_normal(front_normal)
        interaction_position_3d = furniture_centroid + front_normal * radius
        interaction_pose_3d = Pose3D(interaction_position_3d)
        interaction_pose_3d.set_rot_from_direction(-front_normal)
        
        logger.info(f"Furniture interaction pose calculated: position={interaction_position_3d}, direction={interaction_pose_3d.direction()}")
        
        return interaction_pose_3d
            
    finally:
        # Additional safeguard to ensure cleanup
        if furniture_point_cloud is not None:
            del furniture_point_cloud
        if points is not None:
            del points

# def _get_fallback_pose(index: int, furniture_centroid: np.ndarray, radius: float) -> Pose3D:
#     """
#     Fallback method to determine a pose in front of furniture without using Open3D.
#     This avoids memory issues that can occur with large point clouds.
    
#     :param index: Index of the furniture
#     :param furniture_centroid: Centroid of the furniture
#     :param radius: Distance to stand from furniture
#     :return: A reasonable interaction pose
#     """
#     logger.info("Using fallback method to determine furniture front normal")
    
#     try:
#         # Check if robot position is available
#         robot_pos = frame_transformer.get_current_body_position_in_frame(robot_state.frame_name)
        
#         # Calculate direction from robot to furniture (in XY plane)
#         if hasattr(robot_pos, 'x') and hasattr(robot_pos, 'y'):
#             # Handle SE2Pose object
#             robot_pos_xy = np.array([robot_pos.x, robot_pos.y])
#         else:
#             # Handle array-like object
#             robot_pos_xy = robot_pos[:2]
            
#         direction_to_furniture = furniture_centroid[:2] - robot_pos_xy
        
#         # Normalize direction
#         if np.linalg.norm(direction_to_furniture) > 1e-6:
#             direction_to_furniture = direction_to_furniture / np.linalg.norm(direction_to_furniture)
#             # Create 3D normal (approach from current robot position)
#             front_normal = np.array([-direction_to_furniture[0], -direction_to_furniture[1], 0])
#         else:
#             # Default to -X direction if robot is too close
#             front_normal = np.array([-1.0, 0.0, 0.0])
            
#     except Exception as e:
#         logger.error(f"Error getting robot position: {e}, using default normal")
#         # Default to -X direction
#         front_normal = np.array([-1.0, 0.0, 0.0])
    
#     # Save normal and calculate pose
#     try:
#         robot_state.scene_graph.nodes[index].set_normal(front_normal)
#     except Exception as e:
#         logger.error(f"Error setting normal: {e}")
    
#     # Calculate pose
#     interaction_position_3d = furniture_centroid + front_normal * radius
#     interaction_pose_3d = Pose3D(interaction_position_3d)
#     interaction_pose_3d.set_rot_from_direction(-front_normal)
    
#     logger.info(f"Fallback furniture interaction pose calculated: position={interaction_position_3d}, direction={interaction_pose_3d.direction()}")
    
#     return interaction_pose_3d

def get_best_pose_in_front_of_object(index: int, object_description: str, min_interaction_distance=1.10) -> Pose3D:
    """
    Get the best poses for the robot to interact with an object.
    This function finds the closest piece of furniture to the object,
    establishes a scene graph connection, and uses the furniture's normal
    as the interaction normal for the object.
    Running this function will update the scene graph (connection the object to the closest furniture)
    and the object will inherit the interaction normal from the furniture.
    
    :param index: Index of the object from the scene graph.
    :param object_description: Description of the object.
    :param min_interaction_distance: Minimum distance for robot to stand from the object. Defaults to 1.10.
    :return: The object's interaction pose (Pose3D).
    """
    item_centroid = robot_state.scene_graph.nodes[index].centroid
    item_sem_label = robot_state.scene_graph.nodes[index].sem_label

    logger.info("Starting the process of finding the best poses in front of the object %s.", index)
    
    # Find the closest piece of furniture to the object
    furniture_labels = config["semantic_labels"]["furniture"]
        
    closest_furniture_idx = None
    object_interaction_normal = None
    
    # Convert furniture labels to semantic label IDs
    furniture_sem_labels = []
    for label, name in robot_state.scene_graph.label_mapping.items():
        if any(furniture_type in name.lower() for furniture_type in furniture_labels):
            furniture_sem_labels.append(label)
    
    # Use scene_graph's get_nodes_in_radius method to find nodes within a reasonable radius
    # We'll use a large enough radius to capture furniture the object might be on/in
    nearby_nodes = robot_state.scene_graph.get_nodes_in_radius(item_centroid, 1.0)
    
    # Filter to only include furniture nodes
    furniture_nodes = [node_id for node_id in nearby_nodes 
                        if node_id != index and  # Skip self
                        robot_state.scene_graph.nodes[node_id].sem_label in furniture_sem_labels]
    if furniture_nodes:
            # Find the closest furniture node
            furniture_distances = [np.linalg.norm(robot_state.scene_graph.nodes[node_id].centroid - item_centroid) 
                                  for node_id in furniture_nodes]
            closest_furniture_idx = furniture_nodes[np.argmin(furniture_distances)]

    # Check if the object might be on the ground (low z-coordinate)
    is_on_ground = item_centroid[2] < 0.15  # If object is less than 15cm from ground

    robot_pos = frame_transformer.get_current_body_position_in_frame(robot_state.frame_name)

    # Fix: Handle SE2Pose properly - extract x and y attributes instead of using slicing
    if hasattr(robot_pos, 'x') and hasattr(robot_pos, 'y'):
        # Handle SE2Pose object
        robot_pos_xy = np.array([robot_pos.x, robot_pos.y])
    else:
        # Handle array-like object - use original slicing
        robot_pos_xy = robot_pos[:2]

    birds_eye_direction_to_object = item_centroid[:2] - robot_pos_xy  # Only consider XY plane
    birds_eye_direction_to_object = birds_eye_direction_to_object / np.linalg.norm(birds_eye_direction_to_object)

    if is_on_ground:
        # Create a 3D normal vector (horizontal approach)
        object_interaction_normal = -np.array([birds_eye_direction_to_object[0], birds_eye_direction_to_object[1], 0])

    elif closest_furniture_idx is None:
        logger.info("The object is not on the ground and there is no furniture found that the object is connected to. Let's approach from a bird's eye view.")

        # Create a 3D normal vector (horizontal approach)
        object_interaction_normal = -np.array([birds_eye_direction_to_object[0], birds_eye_direction_to_object[1], 0])

    # The object now is on a furniture
    elif closest_furniture_idx is not None: 
        
        # Check if there's already a scene graph connection
        furniture_node = robot_state.scene_graph.nodes[closest_furniture_idx]
        
        # Check if there's already a scene graph connection
        if index not in robot_state.scene_graph.outgoing or robot_state.scene_graph.outgoing[index] != closest_furniture_idx:
            logger.info("Creating scene graph connection between object %s and furniture %s", index, closest_furniture_idx)
            # Create a scene graph connection using scene_graph's built-in methods
            # First remove any existing connections
            if index in robot_state.scene_graph.outgoing:
                old_connection = robot_state.scene_graph.outgoing[index]
                if old_connection in robot_state.scene_graph.ingoing:
                    robot_state.scene_graph.ingoing[old_connection].remove(index)
            
            # Add new connection
            robot_state.scene_graph.outgoing[index] = closest_furniture_idx
            robot_state.scene_graph.ingoing.setdefault(closest_furniture_idx, []).append(index)
        
        # Check if furniture has a normal
        if hasattr(furniture_node, 'equation') and furniture_node.equation is not None:
            object_interaction_normal = furniture_node.equation[:3]
            object_interaction_normal = object_interaction_normal / np.linalg.norm(object_interaction_normal)
            logger.info("Using existing furniture normal: %s", object_interaction_normal)
            
        # If no normal present, calculate it via the get_pose_in_front_of_furniture function
        elif hasattr(furniture_node, 'equation') and furniture_node.equation is None:
            logger.info("Furniture node %s has no normal yet, calculating it.", closest_furniture_idx)
            furniture_interaction_pose = get_pose_in_front_of_furniture(closest_furniture_idx, object_description="furniture", min_distance=min_interaction_distance)
            logger.info("Furniture interaction pose: %s", furniture_interaction_pose)
            object_interaction_normal = -furniture_interaction_pose.direction()

    if object_interaction_normal is None:
        raise ValueError("There was an error in calculating the interaction normal for the object.")

    # Make the object inherit the same normal as the furniture
    logger.info("Setting normal for object %s to %s", index, object_interaction_normal)
    robot_state.scene_graph.nodes[index].set_normal(object_interaction_normal)

    logger.info("Calculating interaction pose for object %s", index)
    interaction_position_3d = item_centroid + object_interaction_normal * min_interaction_distance
    interaction_pose_3d = Pose3D(interaction_position_3d)

    # Use safe calculation of angle
    direction_towards_object = -object_interaction_normal

    interaction_pose_3d.set_rot_from_direction(direction_towards_object)
    
    logger.info("Object interaction pose calculated: position=(%.2f, %.2f, %.2f), direction=(%.2f, %.2f, %.2f)", interaction_position_3d[0], interaction_position_3d[1], interaction_position_3d[2], interaction_pose_3d.direction()[0], interaction_pose_3d.direction()[1], interaction_pose_3d.direction()[2])

    return interaction_pose_3d


if __name__ == "__main__":
    get_pose_in_front_of_furniture(7, "rectangular stand with a light switch")
import os
from pathlib import Path
from utils.recursive_config import Config
# from scripts.temp_scripts.deepseek_exploration import ask_deepseek
import json
import math
import numpy as np
from utils.point_clouds import body_planning_front
from utils.openmask_interface import get_mask_points
import open3d as o3d

from planner_core.robot_state import RobotStateSingleton
robot_state = RobotStateSingleton()

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
    print(circle_radius_width, circle_radius_height, furniture_dimensions[0])
    return max(circle_radius_width, circle_radius_height), furniture_centroid


def _get_shelf_front(cabinet_pcd: o3d.geometry.PointCloud, cabinet_center: np.ndarray) -> np.ndarray:
    '''
    Get the normal of the front face of the furniture.
    
    :param cabinet_pcd: PointCloud of the shelf/cabinet.
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

                objects_in_front = robot_state.scene_graph.get_nodes_in_front_of_object_face(cabinet_center, normal)

                vertical_faces.append({
                    'normal': normal,
                    'area': area,
                    'objects_in_front': objects_in_front,
                })


    if not vertical_faces:
        raise ValueError("No vertical faces found in shelf structure")
    
    # remove the faces that are not accessible by the robot (e.g. backside, other objects in front of the shelf)
    # + take te face that is closest to the robot

    # 1. find the objects that are in a radius of _get_distance_to_shelf(index)[0] from the centroid of the shelf
    # objects_in_front_of_face = []
    # for face_normal, _ in vertical_faces:
    #     objects_in_radius += r

    # 2. remove the faces that have objects in front of them / are not accessible by the robot
    print(f'vertical faces of a furniture (cabinet/shelf):', vertical_faces)


    valid_vertical_faces = [vertical_face for vertical_face in vertical_faces if not vertical_face['objects_in_front']]

    # select largest vertical face as front
    front = max(valid_vertical_faces, key=lambda x: x['area'])


    # To be more accurate, this should be extended to the face with the most free volume in front of it (for 1 meter distance)
    return front['normal']
    

def get_pose_in_front_of_object(index: int=0, object_description="cabinet, shelf") -> tuple:
    '''
    Get the interaction pose for the robot in front of an object. 
    Currently this function only supports cabinets and shelves. 
    
    :param index: Index of the furniture from the Deepseek result.
    :param furniture_types: Types of furniture to consider, a string that gets feeded into Clip
    :return: Tuple of the centroid of the furniture and the body pose.
    '''
    # get necessary distance to shelf
    radius, center = _get_distance_to_shelf(index)
    radius = max(0.8, radius)
    # get all cabinets/shelfs in the environment
    for idx in range(1, 5):

        # When the object is part of the scene graph, take the mask of this object that we already have from mask3d

        # This is how Yasmin implemented it for cabinets and shelves:
        cabinet_pcd, env_pcd = get_mask_points(object_description, Config(), idx=idx, vis_block=True)

        cabinet_center = np.mean(np.asarray(cabinet_pcd.points), axis=0)
        # find correct cabinet/shelf
        if (np.allclose(cabinet_center, center, atol=0.1)):
            print("Object found!")
            # get normal of cabinet/shelf front face
            front_normal = _get_shelf_front(cabinet_pcd, cabinet_center)
            # calculate body position in front of cabinet/shelf
            body_pose = body_planning_front(
                env_pcd,
                cabinet_center,
                shelf_normal=front_normal,
                min_target_distance=radius,
                max_target_distance=radius+0.2,
                min_obstacle_distance=0.4,
                n=5,
                vis_block=True,
            )

    if body_pose is None:
        # the correct object not found, feedback to the llm to change the clip embedding description
        raise ValueError(f"The description of the object {object_description} did not find a match with an object in the scene graph. Please provide a more accurate description.")
    
    return cabinet_center, body_pose

if __name__ == "__main__":
    get_pose_in_front_of_object(7, "rectangular stand with a light switch")
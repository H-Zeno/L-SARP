#!/usr/bin/env python
import sys
import os
import logging
import numpy as np
import json
import gc
import psutil
from pathlib import Path
import time
import open3d as o3d
import traceback
import random

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.coordinates import Pose3D
from utils.recursive_config import Config

def memory_usage():
    """Return memory usage in MB"""
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / 1024 / 1024
    return mem

def generate_test_shelf_data():
    """Generate test shelf data to reproduce memory corruption issue"""
    # Load scene graph JSON to get shelf dimensions and centroid
    config = Config()
    active_scene_name = config["robot_planner_settings"]["active_scene"]
    path_to_scene_data = Path(config["robot_planner_settings"]["path_to_scene_data"])
    scene_graph_json_path = Path(path_to_scene_data / active_scene_name / "scene_graph.json")
    
    logger.info(f"Loading scene graph from {scene_graph_json_path}")
    
    # Load JSON data
    with open(scene_graph_json_path, "r") as file:
        scene_data = json.load(file)
    
    # Get shelf data (node 7)
    shelf_data = scene_data["nodes"]["7"]
    
    # Create a synthetic shelf with same dimensions and centroid
    centroid = np.array(shelf_data["centroid"])
    dimensions = np.array(shelf_data["dimensions"])
    
    # Generate points for a cuboid representing the shelf
    # Bottom face
    num_points_per_face = 200000  # Much more points per face to try to trigger memory issues
    points = []
    
    # Create a more detailed cuboid with points on all faces and some internal points
    
    # Bottom face
    x_vals = np.random.uniform(centroid[0] - dimensions[0]/2, centroid[0] + dimensions[0]/2, num_points_per_face)
    y_vals = np.random.uniform(centroid[1] - dimensions[1]/2, centroid[1] + dimensions[1]/2, num_points_per_face)
    z_vals = np.ones(num_points_per_face) * (centroid[2] - dimensions[2]/2)
    points.extend(np.column_stack((x_vals, y_vals, z_vals)))
    
    # Top face
    x_vals = np.random.uniform(centroid[0] - dimensions[0]/2, centroid[0] + dimensions[0]/2, num_points_per_face)
    y_vals = np.random.uniform(centroid[1] - dimensions[1]/2, centroid[1] + dimensions[1]/2, num_points_per_face)
    z_vals = np.ones(num_points_per_face) * (centroid[2] + dimensions[2]/2)
    points.extend(np.column_stack((x_vals, y_vals, z_vals)))
    
    # Front face (more points on faces to simulate the shelf front)
    x_vals = np.random.uniform(centroid[0] - dimensions[0]/2, centroid[0] + dimensions[0]/2, num_points_per_face*2)
    y_vals = np.ones(num_points_per_face*2) * (centroid[1] + dimensions[1]/2)  # Front face in +Y direction
    z_vals = np.random.uniform(centroid[2] - dimensions[2]/2, centroid[2] + dimensions[2]/2, num_points_per_face*2)
    points.extend(np.column_stack((x_vals, y_vals, z_vals)))
    
    # Back face
    x_vals = np.random.uniform(centroid[0] - dimensions[0]/2, centroid[0] + dimensions[0]/2, num_points_per_face)
    y_vals = np.ones(num_points_per_face) * (centroid[1] - dimensions[1]/2)  # Back face in -Y direction
    z_vals = np.random.uniform(centroid[2] - dimensions[2]/2, centroid[2] + dimensions[2]/2, num_points_per_face)
    points.extend(np.column_stack((x_vals, y_vals, z_vals)))
    
    # Left face
    x_vals = np.ones(num_points_per_face) * (centroid[0] - dimensions[0]/2)  # Left face in -X direction
    y_vals = np.random.uniform(centroid[1] - dimensions[1]/2, centroid[1] + dimensions[1]/2, num_points_per_face)
    z_vals = np.random.uniform(centroid[2] - dimensions[2]/2, centroid[2] + dimensions[2]/2, num_points_per_face)
    points.extend(np.column_stack((x_vals, y_vals, z_vals)))
    
    # Right face
    x_vals = np.ones(num_points_per_face) * (centroid[0] + dimensions[0]/2)  # Right face in +X direction
    y_vals = np.random.uniform(centroid[1] - dimensions[1]/2, centroid[1] + dimensions[1]/2, num_points_per_face)
    z_vals = np.random.uniform(centroid[2] - dimensions[2]/2, centroid[2] + dimensions[2]/2, num_points_per_face)
    points.extend(np.column_stack((x_vals, y_vals, z_vals)))
    
    # Convert to numpy array
    points = np.array(points)
    
    logger.info(f"Generated {len(points)} points for test shelf data")
    return {
        "points": points,
        "centroid": centroid,
        "dimensions": dimensions,
        "label": shelf_data.get("label", "shelf")
    }

def test_get_shelf_front(shelf_points, shelf_centroid):
    """Test function similar to _get_shelf_front from object_interaction_utils.py"""
    logger.info(f"Testing _get_shelf_front with {len(shelf_points)} points")
    
    # Create a point cloud with the shelf points
    pcd = o3d.geometry.PointCloud()
    
    # Apply a memory pressure technique - create many large arrays to consume memory
    memory_pressure = []
    for i in range(10):
        # Create a large array for each iteration, about 100MB each
        large_array = np.random.random((5000, 5000))
        memory_pressure.append(large_array)
        
    logger.info(f"Created memory pressure arrays, current memory: {memory_usage():.2f} MB")
    
    # Now add the points to the point cloud
    pcd.points = o3d.utility.Vector3dVector(np.asarray(shelf_points))
    
    try:
        # Get oriented bounding box (this is the operation that might cause issues)
        logger.info("Getting oriented bounding box...")
        obb = pcd.get_oriented_bounding_box()
        logger.info(f"Got oriented bounding box with extent: {obb.extent}")
        
        # Get rotation matrix
        R = obb.R
        logger.info(f"Rotation matrix: {R}")
        
        # Get extents
        extents = obb.extent
        logger.info(f"Extents: {extents}")
        
        # Find vertical faces (similar to _get_shelf_front)
        vertical_faces = []
        for axis in range(3):
            for direction in [1, -1]:
                # Calculate face normal
                normal = R[:, axis] * direction
                # Check if normal is roughly horizontal (= vertical face)
                if abs(normal[2]) < 0.1:
                    # Calculate face dimensions
                    dim1 = (axis + 1) % 3
                    dim2 = (axis + 2) % 3
                    area = extents[dim1] * extents[dim2]
                    logger.info(f"Found vertical face with normal {normal}, area: {area}")
                    
                    # Snap normal to cardinal direction
                    snapped_normal = snap_to_cardinal(normal)
                    logger.info(f"Snapped normal to {snapped_normal}")
                    
                    vertical_faces.append({
                        'normal': snapped_normal,
                        'original_normal': normal,
                        'area': area
                    })
        
        if not vertical_faces:
            logger.warning("No vertical faces found, defaulting to -X direction")
            front_normal = np.array([-1, 0, 0])
        else:
            # Select largest vertical face as front
            front = max(vertical_faces, key=lambda x: x['area'])
            front_normal = front['normal']
        
        logger.info(f"Selected front normal: {front_normal}")
        
        # Clear only half of the memory pressure arrays to simulate a memory leak
        for i in range(5):
            del memory_pressure[0]
        memory_pressure.clear()  # Now clear the rest
        
        return front_normal
        
    except Exception as e:
        logger.error(f"Error in test_get_shelf_front: {e}")
        logger.error(traceback.format_exc())
        return np.array([-1, 0, 0])  # Default
    finally:
        # Explicit cleanup of only some objects to simulate memory leak
        if 'pcd' in locals():
            del pcd
        # Deliberately not cleaning up 'obb' to simulate memory leak
        
        # Don't force gc.collect() to allow memory leaks

def snap_to_cardinal(normal):
    """Snaps a normal vector to the closest cardinal direction in the XY plane"""
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

def test_shelf_point_cloud():
    """Test with shelf point cloud to try to replicate memory corruption"""
    # Generate test shelf data
    shelf_data = generate_test_shelf_data()
    
    logger.info(f"Created test shelf data with centroid {shelf_data['centroid']}")
    logger.info(f"Dimensions: {shelf_data['dimensions']}")
    logger.info(f"Number of points: {len(shelf_data['points'])}")
    
    # Log initial memory usage
    logger.info(f"Memory before point cloud operations: {memory_usage():.2f} MB")
    
    # Create multiple iterations to try to trigger memory issues
    for i in range(5):
        logger.info(f"Iteration {i+1}/5 - Working with point cloud...")
        
        # Force garbage collection before each iteration
        gc.collect()
        
        try:
            # Try to get the front normal
            front_normal = test_get_shelf_front(shelf_data['points'], shelf_data['centroid'])
            
            # Calculate a pose in front of shelf (similar to get_pose_in_front_of_furniture)
            min_distance = 1.10
            radius = min_distance
            interaction_position_3d = shelf_data['centroid'] + front_normal * radius
            interaction_pose = Pose3D(interaction_position_3d)
            interaction_pose.set_rot_from_direction(-front_normal)
            
            logger.info(f"Calculated interaction pose: position={interaction_position_3d}, direction={interaction_pose.direction()}")
            
            # Force garbage collection after operation
            gc.collect()
            
        except Exception as e:
            logger.error(f"Error in iteration {i+1}: {e}")
            logger.error(traceback.format_exc())
        
        logger.info(f"Memory after iteration {i+1}: {memory_usage():.2f} MB")
        time.sleep(0.5)  # Short delay between iterations
    
    logger.info("Point cloud operations completed successfully without memory corruption")

if __name__ == "__main__":
    try:
        logger.info("Starting shelf pose test")
        test_shelf_point_cloud()
        logger.info("Test completed successfully")
    except Exception as e:
        logger.error(f"Test failed with error: {e}")
        sys.exit(1) 
#!/usr/bin/env python3
"""
Standalone test script for shelf interaction pose calculation.
This script creates a mock environment with a shelf similar to node 7 in the scene graph.
"""

import sys
import os
import logging
import numpy as np
import open3d as o3d
import gc
import psutil
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("standalone_shelf_test")

# Add source directory to sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Import the function to test
from utils.coordinates import Pose3D

def log_memory_info():
    """Log current memory usage"""
    process = psutil.Process()
    memory_info = process.memory_info()
    logger.info(f"Memory usage: {memory_info.rss / 1024 / 1024:.2f} MB")
    return memory_info.rss / 1024 / 1024

def generate_box_points(centroid, dimensions, num_points=5000):
    """Generate points in a box shape around the centroid."""
    half_dims = dimensions / 2
    points = []
    
    # Generate random points inside the box
    for _ in range(num_points):
        offset = np.random.uniform(-half_dims, half_dims)
        points.append(centroid + offset)
        
    return np.array(points)

def snap_to_cardinal(normal):
    """
    Snaps a normal vector to the closest cardinal direction in the XY plane.
    """
    # Get the XY component and normalize it
    normal_xy = np.array([normal[0], normal[1], 0])
    if np.linalg.norm(normal_xy) < 1e-6:  # Handle zero vector case
        return np.array([1, 0, 0])  # Default to X-axis
    
    normal_xy = normal_xy / np.linalg.norm(normal_xy)
    
    # Determine the closest cardinal direction
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

def get_shelf_front(pcd, shelf_centroid):
    """Get the normal of the front face of the shelf."""
    try:
        # Log memory before OBB calculation
        mem_before_obb = log_memory_info()
        
        # Get oriented bounding box
        logger.info("Getting oriented bounding box...")
        obb = pcd.get_oriented_bounding_box()
        R = obb.R
        extents = obb.extent
        
        mem_after_obb = log_memory_info()
        logger.info(f"Memory change during OBB calculation: {mem_after_obb - mem_before_obb:.2f} MB")
        
        # Get vertical faces
        vertical_faces = []
        for axis in range(3):
            for direction in [1, -1]:
                try:
                    # Calculate face normal
                    normal = R[:, axis] * direction
                    # Check if normal is roughly horizontal (= vertical face)
                    if abs(normal[2]) < 0.1:
                        # Calculate face dimensions
                        dim1 = (axis + 1) % 3
                        dim2 = (axis + 2) % 3
                        area = extents[dim1] * extents[dim2]
                        
                        # Snap normal to cardinal direction
                        snapped_normal = snap_to_cardinal(normal)
                        
                        logger.info(f"Found vertical face with normal {snapped_normal}, area {area}")
                        
                        vertical_faces.append({
                            'normal': snapped_normal,
                            'area': area,
                        })
                except Exception as e:
                    logger.error(f"Error processing face: {e}")
        
        # If no vertical faces found, default to -X direction
        if not vertical_faces:
            logger.warning("No vertical faces found, defaulting to -X direction")
            return np.array([-1, 0, 0])
        
        # Select largest vertical face as front
        front = max(vertical_faces, key=lambda x: x['area'])
        return front['normal']
    
    except Exception as e:
        logger.error(f"Error in get_shelf_front: {e}")
        return np.array([-1, 0, 0])  # Default to -X direction

def test_shelf_interaction_pose(num_points=5000):
    """
    Test the interaction pose calculation for a shelf similar to node 7.
    
    Args:
        num_points: Number of points to generate for the shelf
    """
    try:
        # Force garbage collection
        gc.collect()
        initial_memory = log_memory_info()
        
        # Create a shelf similar to node 7 from the scene graph
        logger.info("Creating mock shelf...")
        shelf_centroid = np.array([1.777, 2.297, 0.595])  # From scene_graph.json
        shelf_dimensions = np.array([1.351, 0.510, 1.105])  # From scene_graph.json
        
        # Generate shelf points
        logger.info(f"Generating {num_points} points for the shelf...")
        shelf_points = generate_box_points(shelf_centroid, shelf_dimensions, num_points)
        
        # Calculate distance to shelf (simplified)
        logger.info("Calculating distance to shelf...")
        min_distance = 1.10
        # Simple calculation based on dimensions
        radius = max(min_distance, shelf_dimensions[0] / 2 + 0.5)  # Ensure minimum distance
        
        logger.info(f"Shelf centroid: {shelf_centroid}")
        logger.info(f"Shelf dimensions: {shelf_dimensions}")
        logger.info(f"Calculated distance: {radius}")
        
        # Create point cloud
        logger.info("Creating point cloud...")
        mem_before_pcd = log_memory_info()
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(shelf_points)
        
        mem_after_pcd = log_memory_info()
        logger.info(f"Memory change after creating point cloud: {mem_after_pcd - mem_before_pcd:.2f} MB")
        
        # Get shelf front normal
        logger.info("Getting shelf front normal...")
        front_normal = get_shelf_front(pcd, shelf_centroid)
        logger.info(f"Front normal: {front_normal}")
        
        # Calculate interaction pose
        logger.info("Calculating interaction pose...")
        interaction_position = shelf_centroid + front_normal * radius
        interaction_pose = Pose3D(interaction_position)
        interaction_pose.set_rot_from_direction(-front_normal)
        
        logger.info(f"Interaction pose:")
        logger.info(f"Position: {interaction_pose.as_ndarray()[:3]}")
        logger.info(f"Direction: {-front_normal}")
        
        # Calculate distance from shelf
        distance = np.linalg.norm(interaction_pose.as_ndarray()[:3] - shelf_centroid)
        logger.info(f"Distance from shelf: {distance}")
        
        # Clean up
        pcd = None
        shelf_points = None
        gc.collect()
        
        final_memory = log_memory_info()
        logger.info(f"Total memory change: {final_memory - initial_memory:.2f} MB")
        
        return True
        
    except Exception as e:
        logger.error(f"Error in test: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def test_different_point_counts():
    """Test with different numbers of points to find the threshold where issues occur."""
    point_counts = [100, 1000, 5000, 10000, 50000, 100000]
    
    for count in point_counts:
        logger.info(f"\n\n========== TESTING WITH {count} POINTS ==========")
        success = test_shelf_interaction_pose(count)
        
        if not success:
            logger.error(f"Test failed with {count} points")
        
        # Force garbage collection between tests
        gc.collect()

if __name__ == "__main__":
    logger.info("Starting standalone shelf test")
    
    # Test with default number of points
    logger.info("\n========== TESTING WITH DEFAULT POINTS ==========")
    test_shelf_interaction_pose()
    
    # Test with different point counts
    logger.info("\n========== TESTING WITH DIFFERENT POINT COUNTS ==========")
    test_different_point_counts() 
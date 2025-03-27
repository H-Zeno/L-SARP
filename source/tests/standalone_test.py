#!/usr/bin/env python3
"""
Standalone test script for the get_pose_in_front_of_furniture function.
This script creates a mock scene graph with a cabinet node, without relying on existing data.
"""

import sys
import os
import logging
import numpy as np
import open3d as o3d
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("standalone_test")

# Add source directory to sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Imports needed for testing
from utils.coordinates import Pose3D
from LostFound.src.graph_nodes import ObjectNode

def generate_box_points(centroid, dimensions, num_points=1000):
    """Generate points in a box shape around the centroid."""
    half_dims = dimensions / 2
    points = []
    
    # Generate random points inside the box
    for _ in range(num_points):
        offset = np.random.uniform(-half_dims, half_dims)
        points.append(centroid + offset)
        
    return np.array(points)

def get_shelf_front(pcd, centroid):
    """Simplified version of the _get_shelf_front function for testing."""
    # Get oriented bounding box
    obb = pcd.get_oriented_bounding_box()
    R = obb.R
    extents = obb.extent
    
    # Find the largest face that's vertical
    best_normal = None
    best_area = 0
    
    for axis in range(3):
        for direction in [1, -1]:
            # Calculate face normal
            normal = R[:, axis] * direction
            
            # Check if normal is roughly horizontal (vertical face)
            if abs(normal[2]) < 0.1:
                # Calculate face dimensions
                dim1 = (axis + 1) % 3
                dim2 = (axis + 2) % 3
                area = extents[dim1] * extents[dim2]
                
                if area > best_area:
                    best_area = area
                    best_normal = normal
    
    # If no vertical face found, default to -X direction
    if best_normal is None:
        return np.array([-1, 0, 0])
    
    # Normalize the normal
    best_normal = best_normal / np.linalg.norm(best_normal)
    return best_normal

def get_pose_in_front_of_furniture_simplified(furniture_point_cloud, furniture_centroid, min_distance=1.10):
    """Simplified version of get_pose_in_front_of_furniture for testing."""
    # Calculate distance to furniture (simplified for testing)
    radius = max(1.0, min_distance)
    
    # Create Open3D point cloud from numpy array
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(furniture_point_cloud)
    
    # Get front normal
    front_normal = get_shelf_front(pcd, furniture_centroid)
    logger.info(f"Front normal: {front_normal}")
    
    # Calculate interaction pose
    interaction_position = furniture_centroid + front_normal * radius
    interaction_pose = Pose3D(interaction_position)
    interaction_pose.set_rot_from_direction(-front_normal)
    
    logger.info(f"Interaction position: {interaction_position}")
    logger.info(f"Interaction direction: {-front_normal}")
    
    return interaction_pose

def main():
    """Main function to run the standalone test."""
    logger.info("Starting standalone test")
    
    # Create a mock cabinet
    cabinet_centroid = np.array([-0.15, 0.03, 0.20])
    cabinet_dimensions = np.array([0.56, 0.52, 0.92])
    cabinet_points = generate_box_points(cabinet_centroid, cabinet_dimensions)
    
    logger.info(f"Created mock cabinet with centroid {cabinet_centroid} and dimensions {cabinet_dimensions}")
    logger.info(f"Generated {len(cabinet_points)} points")
    
    try:
        # Test the simplified function
        logger.info("Getting pose in front of furniture...")
        interaction_pose = get_pose_in_front_of_furniture_simplified(
            cabinet_points, cabinet_centroid)
        
        # Print result
        logger.info(f"Interaction pose position: {interaction_pose.position()}")
        logger.info(f"Interaction pose direction: {interaction_pose.direction()}")
        
        # Calculate distance from furniture
        distance = np.linalg.norm(interaction_pose.position() - cabinet_centroid)
        logger.info(f"Distance from furniture: {distance}")
        
        # Check if the pose is facing the furniture
        direction_to_furniture = cabinet_centroid - interaction_pose.position()
        direction_to_furniture = direction_to_furniture / np.linalg.norm(direction_to_furniture)
        alignment = np.dot(interaction_pose.direction(), direction_to_furniture)
        logger.info(f"Alignment with furniture (dot product): {alignment}")
        
        logger.info("Test completed successfully")
        
    except Exception as e:
        logger.error(f"Error in test: {e}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main() 
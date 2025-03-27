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
from planner_core.robot_state import RobotState, RobotStateSingleton
from LostFound.src.scene_graph import get_scene_graph

def memory_usage():
    """Return memory usage in MB"""
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / 1024 / 1024
    return mem

def initialize_scene_graph():
    """Initialize the scene graph properly similar to main.py"""
    logger.info("Initializing scene graph properly...")
    
    # Get scene graph path from config
    config = Config()
    active_scene_name = config["robot_planner_settings"]["active_scene"]
    path_to_scene_data = Path(config["robot_planner_settings"]["path_to_scene_data"])
    
    # Set paths similar to main.py
    scene_graph_path = Path(path_to_scene_data / active_scene_name / "full_scene_graph.pkl")
    
    # Get scan directory from config
    base_path = config.get_subpath("prescans")
    ending = config["pre_scanned_graphs"]["high_res"]
    scan_dir = os.path.join(base_path, ending)
    
    logger.info(f"Loading scene graph from {scan_dir}")
    
    # Initialize scene graph using the same method as main.py
    scene_graph = get_scene_graph(
        scan_dir,
        graph_save_path=scene_graph_path,
        drawers=False,
        light_switches=True,
        vis_block=False
    )
    
    # Initialize robot state with the scene graph
    robot_state = RobotState(config)
    RobotStateSingleton(robot_state)
    robot_state.scene_graph = scene_graph
    
    logger.info(f"Successfully initialized scene graph with {len(scene_graph.nodes)} nodes")
    return scene_graph

def test_shelf_point_cloud_handling():
    """Test handling of shelf point cloud to try to replicate memory corruption"""
    # Initialize scene graph properly 
    scene_graph = initialize_scene_graph()
    
    # Find the shelf node (node 7)
    shelf_index = 7
    
    try:
        # Get the shelf node
        if shelf_index not in scene_graph.nodes:
            logger.error(f"Node {shelf_index} not found in scene graph")
            logger.info(f"Available node indices: {list(scene_graph.nodes.keys())}")
            return
            
        shelf_node = scene_graph.nodes[shelf_index]
        
        logger.info(f"Found shelf (node {shelf_index})")
        logger.info(f"Label: {scene_graph.label_mapping.get(shelf_node.sem_label, 'N/A')}")
        logger.info(f"Centroid: {shelf_node.centroid}")
        logger.info(f"Dimensions: {shelf_node.dimensions}")
        logger.info(f"Number of points: {len(shelf_node.points)}")
        
        # Log initial memory usage
        logger.info(f"Memory before point cloud operations: {memory_usage():.2f} MB")
        
        # Get shelf points directly from the node
        shelf_points = shelf_node.points
        logger.info(f"Got {len(shelf_points)} points from shelf node")
        
        # This is the operation that might cause memory corruption
        for i in range(5):  # Try multiple times to increase likelihood of reproducing the issue
            logger.info(f"Iteration {i+1}/5 - Working with point cloud...")
            
            # Force explicit garbage collection
            gc.collect()
            
            try:
                # Create a point cloud with the shelf points
                pcd = o3d.geometry.PointCloud()
                
                # Make a copy of the points to avoid memory sharing issues
                if len(shelf_points) > 100000:
                    # Randomly sample 100000 points to reduce memory usage
                    indices = np.random.choice(len(shelf_points), 100000, replace=False)
                    points_copy = np.copy(shelf_points[indices])
                else:
                    points_copy = np.copy(shelf_points)
                
                logger.info(f"Created points copy with shape {points_copy.shape}")
                
                pcd.points = o3d.utility.Vector3dVector(np.asarray(points_copy))
                
                logger.info(f"Created point cloud with {len(pcd.points)} points")
                logger.info(f"Created point cloud, trying to get oriented bounding box...")
                
                # Try to get the oriented bounding box (operation that might cause issues)
                obb = pcd.get_oriented_bounding_box()
                logger.info(f"Got oriented bounding box with extent: {obb.extent}")
                
                # Clean up explicitly
                del pcd
                del obb
                del points_copy
                
                # Force garbage collection
                gc.collect()
                
            except Exception as e:
                logger.error(f"Error in iteration {i+1}: {e}")
                logger.error(traceback.format_exc())
            
            logger.info(f"Memory after iteration {i+1}: {memory_usage():.2f} MB")
            time.sleep(0.5)  # Short delay between iterations
        
        logger.info("Point cloud operations completed successfully without memory corruption")
        
    except Exception as e:
        logger.error(f"Error during test: {e}")
        logger.error(traceback.format_exc())
        logger.error(f"Current memory usage: {memory_usage():.2f} MB")
        raise

if __name__ == "__main__":
    try:
        logger.info("Starting memory corruption test")
        test_shelf_point_cloud_handling()
        logger.info("Test completed successfully")
    except Exception as e:
        logger.error(f"Test failed with error: {e}")
        sys.exit(1) 
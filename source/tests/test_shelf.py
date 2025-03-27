#!/usr/bin/env python3
"""
Test script focusing specifically on node 7 (shelf) that's causing memory issues.
This script includes detailed diagnostics to help identify the root cause.
"""

import sys
import os
import logging
import numpy as np
import traceback
from pathlib import Path
import gc
import psutil

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("test_shelf")

# Add source directory to sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Import the function to test
from robot_utils.object_interaction_utils import (
    get_pose_in_front_of_furniture, 
    _get_distance_to_shelf,
    _get_shelf_front,
    snap_to_cardinal
)
from planner_core.robot_state import RobotStateSingleton
import open3d as o3d

def get_object_size(obj):
    """Get the size of an object in memory (approximate)"""
    import sys
    return sys.getsizeof(obj)

def log_memory_info():
    """Log current memory usage"""
    process = psutil.Process()
    memory_info = process.memory_info()
    logger.info(f"Memory usage: {memory_info.rss / 1024 / 1024:.2f} MB")
    return memory_info.rss / 1024 / 1024

def test_shelf_step_by_step():
    """Test each step of get_pose_in_front_of_furniture with node 7 (shelf)"""
    try:
        # Get reference to robot state
        robot_state = RobotStateSingleton()
        
        # Force garbage collection before starting
        gc.collect()
        initial_memory = log_memory_info()
        
        # Get shelf node info
        shelf_index = 7
        if shelf_index in robot_state.scene_graph.nodes:
            shelf = robot_state.scene_graph.nodes[shelf_index]
            shelf_label = robot_state.scene_graph.label_mapping.get(shelf_index, "unknown")
            
            logger.info(f"Testing shelf with index {shelf_index} and label {shelf_label}")
            logger.info(f"Centroid: {shelf.centroid}")
            logger.info(f"Dimensions: {shelf.dimensions}")
            logger.info(f"Number of points: {len(shelf.points)}")
            
            # Step 1: Get distance to shelf
            logger.info("\n==== STEP 1: Get distance to shelf ====")
            mem_before = log_memory_info()
            radius, furniture_centroid = _get_distance_to_shelf(shelf_index)
            mem_after = log_memory_info()
            
            logger.info(f"Distance to shelf: {radius}")
            logger.info(f"Furniture centroid: {furniture_centroid}")
            logger.info(f"Memory change: {mem_after - mem_before:.2f} MB")
            
            # Step 2: Create point cloud and get shelf front normal
            logger.info("\n==== STEP 2: Create point cloud and get shelf front ====")
            mem_before = log_memory_info()
            
            logger.info("Creating point cloud...")
            furniture_point_cloud = o3d.geometry.PointCloud()
            furniture_point_cloud.points = o3d.utility.Vector3dVector(shelf.points)
            logger.info("Point cloud created")
            
            mem_after_cloud = log_memory_info()
            logger.info(f"Memory change after creating point cloud: {mem_after_cloud - mem_before:.2f} MB")
            
            logger.info("Getting shelf front normal...")
            front_normal = _get_shelf_front(furniture_point_cloud, furniture_centroid)
            
            mem_after_normal = log_memory_info()
            logger.info(f"Memory change after getting normal: {mem_after_normal - mem_after_cloud:.2f} MB")
            
            logger.info(f"Shelf front normal: {front_normal}")
            
            # Step 3: Calculate interaction pose
            logger.info("\n==== STEP 3: Calculate interaction pose ====")
            mem_before = log_memory_info()
            
            logger.info("Calculating interaction position...")
            interaction_position_3d = furniture_centroid + front_normal * radius
            
            logger.info("Creating interaction pose...")
            from utils.coordinates import Pose3D
            interaction_pose_3d = Pose3D(interaction_position_3d)
            
            logger.info("Setting pose rotation...")
            interaction_pose_3d.set_rot_from_direction(-front_normal)
            
            mem_after = log_memory_info()
            
            logger.info(f"Interaction pose calculated:")
            logger.info(f"Position: {interaction_pose_3d.position()}")
            logger.info(f"Direction: {interaction_pose_3d.direction()}")
            logger.info(f"Memory change: {mem_after - mem_before:.2f} MB")
            
            # Cleanup
            logger.info("\n==== Cleanup ====")
            furniture_point_cloud = None
            gc.collect()
            
            final_memory = log_memory_info()
            logger.info(f"Total memory change: {final_memory - initial_memory:.2f} MB")
            
            return True
        else:
            logger.error(f"Shelf with index {shelf_index} not found in scene graph")
            return False
    
    except Exception as e:
        logger.error(f"Error testing shelf: {e}")
        logger.error(traceback.format_exc())
        return False

def test_full_function():
    """Test the complete get_pose_in_front_of_furniture function with node 7"""
    try:
        # Force garbage collection before starting
        gc.collect()
        initial_memory = log_memory_info()
        
        # Test the full function
        logger.info("\n==== Testing full get_pose_in_front_of_furniture function ====")
        interaction_pose = get_pose_in_front_of_furniture(7)
        
        # Log results
        logger.info(f"Interaction pose position: {interaction_pose.position()}")
        logger.info(f"Interaction pose direction: {interaction_pose.direction()}")
        
        # Cleanup
        gc.collect()
        final_memory = log_memory_info()
        logger.info(f"Total memory change: {final_memory - initial_memory:.2f} MB")
        
        return True
    except Exception as e:
        logger.error(f"Error testing full function: {e}")
        logger.error(traceback.format_exc())
        return False

def test_with_simplified_code():
    """Test with a simplified version of the code to isolate the issue"""
    try:
        # Get reference to robot state
        robot_state = RobotStateSingleton()
        
        # Get shelf node info
        shelf_index = 7
        if shelf_index not in robot_state.scene_graph.nodes:
            logger.error(f"Shelf with index {shelf_index} not found")
            return False
            
        # Get shelf details    
        shelf = robot_state.scene_graph.nodes[shelf_index]
        shelf_centroid = shelf.centroid
        shelf_points = shelf.points
        
        # Force garbage collection before starting
        gc.collect()
        initial_memory = log_memory_info()
        
        # Create a minimal version of the function to test
        logger.info("\n==== Testing with simplified code ====")
        
        # Create point cloud directly
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(shelf_points)
        
        # Get OBB (this is where the crash might happen)
        logger.info("Getting oriented bounding box...")
        try:
            obb = pcd.get_oriented_bounding_box()
            logger.info(f"OBB created successfully: center={obb.center}, extent={obb.extent}")
        except Exception as e:
            logger.error(f"Error creating OBB: {e}")
            return False
            
        # Clean up
        pcd = None
        gc.collect()
        final_memory = log_memory_info()
        logger.info(f"Memory change: {final_memory - initial_memory:.2f} MB")
        
        return True
    except Exception as e:
        logger.error(f"Error in simplified test: {e}")
        logger.error(traceback.format_exc())
        return False

def monitor_memory_growth():
    """Run the function multiple times to check for memory leaks"""
    process = psutil.Process()
    memory_samples = []
    
    # Force initial garbage collection
    gc.collect()
    memory_samples.append(process.memory_info().rss / 1024 / 1024)
    logger.info(f"Initial memory usage: {memory_samples[0]:.2f} MB")
    
    # Run tests multiple times
    for i in range(5):
        logger.info(f"\n\n======== Iteration {i+1} ========")
        success = test_full_function()
        
        # Force garbage collection
        gc.collect()
        memory_usage = process.memory_info().rss / 1024 / 1024
        memory_samples.append(memory_usage)
        
        logger.info(f"Memory after iteration {i+1}: {memory_usage:.2f} MB")
        logger.info(f"Change from start: {memory_usage - memory_samples[0]:.2f} MB")
        
        if not success:
            logger.error("Test failed, stopping")
            break
            
    # Summarize memory pattern
    logger.info("\nMemory usage pattern:")
    for i, mem in enumerate(memory_samples):
        logger.info(f"  Sample {i}: {mem:.2f} MB")
        
    # Check for memory growth
    if len(memory_samples) > 2:
        growth_rate = (memory_samples[-1] - memory_samples[0]) / (len(memory_samples) - 1)
        logger.info(f"Average memory growth per iteration: {growth_rate:.2f} MB")

if __name__ == "__main__":
    logger.info("Starting test_shelf.py")
    
    # First test step by step to identify problematic part
    logger.info("\n\n========== TEST STEP BY STEP ==========")
    test_shelf_step_by_step()
    
    # Then test the simplified code to isolate the issue
    logger.info("\n\n========== TEST SIMPLIFIED CODE ==========")
    test_with_simplified_code()
    
    # Finally monitor memory growth across multiple calls
    logger.info("\n\n========== MONITOR MEMORY GROWTH ==========")
    monitor_memory_growth() 
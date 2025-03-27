#!/usr/bin/env python3
"""
Simple script to test the get_pose_in_front_of_furniture function with the cabinet (node 17).
"""

import sys
import os
import logging
import numpy as np
import traceback
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("test_cabinet")

# Add source directory to sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Import the function to test
from robot_utils.object_interaction_utils import get_pose_in_front_of_furniture
from planner_core.robot_state import RobotStateSingleton

def test_cabinet():
    """Test getting a pose in front of the cabinet (node 17)."""
    try:
        # Get reference to robot state
        robot_state = RobotStateSingleton()
        
        # Log memory usage before
        import psutil
        process = psutil.Process()
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        logger.info(f"Memory usage before: {memory_before:.2f} MB")
        
        # Get cabinet node info
        cabinet_index = 17
        if cabinet_index in robot_state.scene_graph.nodes:
            cabinet = robot_state.scene_graph.nodes[cabinet_index]
            cabinet_label = robot_state.scene_graph.label_mapping.get(cabinet_index, "unknown")
            
            logger.info(f"Testing cabinet with index {cabinet_index} and label {cabinet_label}")
            logger.info(f"Centroid: {cabinet.centroid}")
            logger.info(f"Dimensions: {cabinet.dimensions}")
            logger.info(f"Number of points: {len(cabinet.points)}")
            
            # Test step by step
            logger.info("Step 1: Getting pose")
            interaction_pose = get_pose_in_front_of_furniture(cabinet_index)
            
            # Check pose
            logger.info(f"Pose position: {interaction_pose.position()}")
            logger.info(f"Pose direction: {interaction_pose.direction()}")
            
            # Log memory usage after
            memory_after = process.memory_info().rss / 1024 / 1024  # MB
            logger.info(f"Memory usage after: {memory_after:.2f} MB")
            logger.info(f"Memory change: {memory_after - memory_before:.2f} MB")
            
            return True
        else:
            logger.error(f"Cabinet with index {cabinet_index} not found in scene graph")
            return False
    
    except Exception as e:
        logger.error(f"Error testing cabinet: {e}")
        logger.error(traceback.format_exc())
        return False

def debug_memory_usage():
    """Monitor memory usage through multiple function calls."""
    import gc
    import psutil
    import time
    
    process = psutil.Process()
    
    # Force garbage collection
    gc.collect()
    
    # Monitor memory usage
    memory_samples = []
    
    # Initial sample
    memory_samples.append(process.memory_info().rss / 1024 / 1024)
    logger.info(f"Initial memory usage: {memory_samples[-1]:.2f} MB")
    
    # Try accessing the cabinet multiple times
    for i in range(10):
        logger.info(f"Iteration {i+1}")
        
        # Get pose
        success = test_cabinet()
        
        # Force garbage collection
        gc.collect()
        
        # Record memory
        memory_samples.append(process.memory_info().rss / 1024 / 1024)
        logger.info(f"Memory after iteration {i+1}: {memory_samples[-1]:.2f} MB")
        logger.info(f"Change from start: {memory_samples[-1] - memory_samples[0]:.2f} MB")
        
        # Short pause
        time.sleep(1)
        
        if not success:
            logger.error("Test failed, stopping iterations")
            break
    
    # Log memory usage pattern
    logger.info("Memory usage pattern:")
    for i, mem in enumerate(memory_samples):
        logger.info(f"  Sample {i}: {mem:.2f} MB")

if __name__ == "__main__":
    logger.info("Starting test_cabinet.py")
    
    # Debug memory usage pattern
    debug_memory_usage() 
#!/usr/bin/env python3
"""
Simple script to test the get_pose_in_front_of_furniture function with the shelf (node 7).
This script will initialize the robot state and test the actual function.
"""

import os
import sys
import json
import logging
import numpy as np
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("test_shelf_interaction")

# Add source to Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
source_dir = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.insert(0, source_dir)

# Import required modules
from utils.coordinates import Pose3D
from utils.recursive_config import Config
from planner_core.robot_state import RobotState, RobotStateSingleton
from LostFound.src.scene_graph import SceneGraph
from LostFound.src.graph_nodes import ObjectNode
from robot_utils.object_interaction_utils import get_pose_in_front_of_furniture

def create_scene_graph_from_json(json_file_path):
    """Create a scene graph from a JSON file."""
    logger.info(f"Creating scene graph from JSON file: {json_file_path}")
    
    try:
        # Load the JSON file
        with open(json_file_path, 'r') as f:
            data = json.load(f)
        
        # Create a new scene graph
        scene_graph = SceneGraph()
        
        # Process nodes
        for node_id_str, node_data in data["nodes"].items():
            node_id = int(node_id_str)
            
            # Get node attributes
            centroid = np.array(node_data["centroid"])
            label = node_data["label"]
            dimensions = np.array(node_data["dimensions"])
            confidence = node_data.get("confidence", 1.0)
            movable = node_data.get("movable", True)
            
            # Create points around centroid to simulate point cloud
            # (since we don't have actual points in the JSON)
            points = generate_box_points(centroid, dimensions)
            
            # Create a mock node
            node = ObjectNode(
                object_id=node_id,
                color=(0.5, 0.5, 0.5),
                sem_label=label,
                points=points,
                tracking_points=points[:10],
                mesh_mask=np.ones(len(points), dtype=bool),
                confidence=confidence,
                movable=movable
            )
            
            # Add the node to the scene graph
            scene_graph.nodes[node_id] = node
            scene_graph.label_mapping[node_id] = label
        
        # Process connections (outgoing and ingoing)
        if "outgoing" in data:
            for src_str, dst in data["outgoing"].items():
                src = int(src_str)
                scene_graph.outgoing[src] = int(dst) if isinstance(dst, (int, str)) else dst
        
        if "ingoing" in data:
            for dst_str, src_list in data["ingoing"].items():
                dst = int(dst_str)
                scene_graph.ingoing[dst] = [int(src) for src in src_list]
        
        logger.info(f"Successfully created scene graph with {len(scene_graph.nodes)} nodes")
        return scene_graph
    
    except Exception as e:
        logger.error(f"Error creating scene graph from JSON: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None

def generate_box_points(centroid, dimensions, num_points=5000):
    """Generate points in a box shape around the centroid."""
    half_dims = dimensions / 2
    points = []
    
    # Generate random points inside the box
    for _ in range(num_points):
        offset = np.random.uniform(-half_dims, half_dims)
        points.append(centroid + offset)
        
    return np.array(points)

def initialize_robot_state():
    """Initialize the robot state with a scene graph."""
    logger.info("Initializing robot state...")
    
    # Load config
    config = Config()
    
    # Get path to scene graph
    active_scene_name = config["robot_planner_settings"]["active_scene"]
    path_to_scene_data = Path(config["robot_planner_settings"]["path_to_scene_data"])
    scene_graph_json_path = path_to_scene_data / active_scene_name / "scene_graph.json"
    
    logger.info(f"Loading scene graph from {scene_graph_json_path}")
    
    # Check if the scene graph exists
    if not scene_graph_json_path.exists():
        logger.error(f"Scene graph file not found: {scene_graph_json_path}")
        return False
    
    # Initialize robot state
    robot_state = RobotStateSingleton()
    
    # Create scene graph from JSON
    scene_graph = create_scene_graph_from_json(scene_graph_json_path)
    
    if scene_graph is None:
        logger.error("Failed to create scene graph")
        return False
    
    # Set up the robot state with the loaded scene graph
    robot_state.scene_graph = scene_graph
    robot_state.frame_name = "map"
    
    logger.info(f"Successfully initialized robot state with scene graph containing {len(scene_graph.nodes)} nodes")
    
    # Verify node 7 exists
    if 7 in scene_graph.nodes:
        shelf = scene_graph.nodes[7]
        logger.info(f"Found shelf (node 7)")
        logger.info(f"Label: {shelf.sem_label}")
        logger.info(f"Centroid: {shelf.centroid}")
        logger.info(f"Dimensions: {shelf.dimensions}")
        logger.info(f"Number of points: {len(shelf.points)}")
    else:
        logger.error("Shelf (node 7) not found in scene graph")
        return False
    
    return True

def test_get_pose_function():
    """Test the get_pose_in_front_of_furniture function with the shelf (node 7)."""
    logger.info("\n=== Testing get_pose_in_front_of_furniture with shelf (node 7) ===")
    
    import gc
    gc.collect()  # Force garbage collection before starting
    
    try:
        # Get the pose in front of the shelf
        logger.info("Calling get_pose_in_front_of_furniture with node 7...")
        interaction_pose = get_pose_in_front_of_furniture(7)
        
        # Print the result
        logger.info(f"Successfully calculated interaction pose for shelf")
        logger.info(f"Position: {interaction_pose.as_ndarray()[:3]}")
        logger.info(f"Direction: {interaction_pose.direction()}")
        
        # Get shelf info
        robot_state = RobotStateSingleton()
        shelf_centroid = robot_state.scene_graph.nodes[7].centroid
        
        # Calculate distance from shelf
        position = interaction_pose.as_ndarray()[:3]
        distance = np.linalg.norm(position - shelf_centroid)
        logger.info(f"Distance from shelf: {distance}")
        
        return True
    except Exception as e:
        logger.error(f"Error testing get_pose_in_front_of_furniture: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def main():
    """Main function."""
    logger.info("Starting test_shelf_interaction.py")
    
    # Initialize robot state
    if not initialize_robot_state():
        logger.error("Failed to initialize robot state. Exiting.")
        return
    
    # Run the test multiple times to check for memory issues
    num_iterations = 5
    logger.info(f"Running test {num_iterations} times to check for memory issues")
    
    import gc
    import psutil
    process = psutil.Process()
    memory_samples = []
    
    # Force initial garbage collection
    gc.collect()
    memory_samples.append(process.memory_info().rss / 1024 / 1024)
    logger.info(f"Initial memory usage: {memory_samples[0]:.2f} MB")
    
    success_count = 0
    for i in range(num_iterations):
        logger.info(f"\n======== Iteration {i+1}/{num_iterations} ========")
        success = test_get_pose_function()
        
        if success:
            success_count += 1
        
        # Force garbage collection
        gc.collect()
        memory_usage = process.memory_info().rss / 1024 / 1024
        memory_samples.append(memory_usage)
        
        logger.info(f"Memory after iteration {i+1}: {memory_usage:.2f} MB")
        logger.info(f"Change from start: {memory_usage - memory_samples[0]:.2f} MB")
    
    # Summary
    logger.info("\n======== Test Summary ========")
    logger.info(f"Successful runs: {success_count}/{num_iterations}")
    
    # Summarize memory pattern
    logger.info("\nMemory usage pattern:")
    for i, mem in enumerate(memory_samples):
        logger.info(f"  Sample {i}: {mem:.2f} MB")
        
    # Check for memory growth
    if len(memory_samples) > 2:
        growth_rate = (memory_samples[-1] - memory_samples[0]) / (len(memory_samples) - 1)
        logger.info(f"Average memory growth per iteration: {growth_rate:.2f} MB")
    
    if success_count == num_iterations:
        logger.info("All tests completed successfully")
    else:
        logger.error(f"Some tests failed: {success_count}/{num_iterations} succeeded")

if __name__ == "__main__":
    main() 
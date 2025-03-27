#!/usr/bin/env python3
"""
Test script for object_interaction_utils.py functions.
Specifically tests the get_pose_in_front_of_furniture function.
"""

import sys
import os
import logging
import numpy as np
import open3d as o3d
import unittest
import json
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("test_object_interaction")

# Add source directory to sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Import the module we want to test
from robot_utils.object_interaction_utils import (
    get_pose_in_front_of_furniture, 
    _get_distance_to_shelf,
    _get_shelf_front,
    snap_to_cardinal
)
from utils.coordinates import Pose3D
from utils.recursive_config import Config
from planner_core.robot_state import RobotStateSingleton, RobotState
from LostFound.src.scene_graph import SceneGraph
from LostFound.src.graph_nodes import ObjectNode

# Mocked functions and classes for testing
class MockConfig(Config):
    def __init__(self):
        self.config_data = {
            "robot_parameters": {
                "H_FOV": 70,
                "V_FOV": 43
            },
            "robot_planner_settings": {
                "active_scene": "SEMANTIC_CORNER_WITH_BED",
                "path_to_scene_data": "data_scene"
            },
            "pre_scanned_graphs": {
                "high_res": "high_res"
            },
            "semantic_labels": {
                "furniture": ["cabinet", "shelf", "table", "desk"]
            }
        }
    
    def __getitem__(self, key):
        return self.config_data.get(key)
    
    def get_subpath(self, key):
        if key == "prescans":
            return "data/prescans"
        elif key == "aligned_point_clouds":
            return "data/aligned_point_clouds"
        return "data"

class MockSceneGraph(SceneGraph):
    def __init__(self):
        self.nodes = {}
        self.label_mapping = {
            8: "cabinet",
            7: "shelf",
            3: "table"
        }
        self.outgoing = {}
        self.ingoing = {}
        self.frame_name = "map"

    def get_nodes_in_front_of_object_face(self, node_centroid, face_normal):
        # Mock implementation - return empty list (no objects in front)
        return []
    
    def get_nodes_in_radius(self, point, radius):
        # Return all node ids
        return list(self.nodes.keys())

class MockRobotState:
    def __init__(self):
        self.scene_graph = MockSceneGraph()
        self.frame_name = "map"

class MockFrameTransformer:
    def get_current_body_position_in_frame(self, frame_name):
        # Return a default position
        return np.array([0.0, 0.0, 0.0])

class TestObjectInteractionUtils(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures before each test method"""
        # Create a mock scene graph with furniture nodes
        self.mock_robot_state = MockRobotState()
        self.mock_config = MockConfig()
        
        # Override the global RobotStateSingleton
        import robot_utils.object_interaction_utils as oi_utils
        oi_utils.robot_state = self.mock_robot_state
        oi_utils.config = self.mock_config
        oi_utils.frame_transformer = MockFrameTransformer()
        
        # Load scene graph from JSON file for realistic testing
        try:
            scene_graph_path = Path(self.mock_config["robot_planner_settings"]["path_to_scene_data"]) / self.mock_config["robot_planner_settings"]["active_scene"] / "scene_graph.json"
            if scene_graph_path.exists():
                with open(scene_graph_path, "r") as f:
                    self.scene_data = json.load(f)
                logger.info(f"Loaded scene graph from {scene_graph_path}")
                
                # Add furniture nodes to our mock scene graph
                for node_id, node_data in self.scene_data["nodes"].items():
                    if "cabinet" in node_data["label"].lower() or "shelf" in node_data["label"].lower():
                        node_id = int(node_id)
                        self._add_mock_node(node_id, node_data)
            else:
                # Create mock furniture if we can't load from file
                self._create_mock_furniture()
        except Exception as e:
            logger.warning(f"Could not load scene graph from file: {e}. Using mock furniture instead.")
            self._create_mock_furniture()

    def _add_mock_node(self, node_id, node_data):
        """Add a node from scene graph data to our mock scene graph"""
        centroid = np.array(node_data["centroid"])
        dimensions = np.array(node_data["dimensions"])
        
        # Create some points around the centroid to simulate a point cloud
        points = self._generate_box_points(centroid, dimensions)
        
        # Create a mock node
        node = ObjectNode(
            object_id=node_id,
            color=(0.5, 0.5, 0.5),
            sem_label=node_data["label"],
            points=points,
            tracking_points=points[:10],
            mesh_mask=np.ones(len(points), dtype=bool),
            confidence=node_data.get("confidence", 1.0),
            movable=node_data.get("movable", True)
        )
        
        # Add the node to the scene graph
        self.mock_robot_state.scene_graph.nodes[node_id] = node
        self.mock_robot_state.scene_graph.label_mapping[node_id] = node_data["label"]
    
    def _generate_box_points(self, centroid, dimensions, num_points=1000):
        """Generate points in a box shape around the centroid"""
        half_dims = dimensions / 2
        points = []
        
        # Generate random points inside the box
        for _ in range(num_points):
            offset = np.random.uniform(-half_dims, half_dims)
            points.append(centroid + offset)
            
        return np.array(points)
    
    def _create_mock_furniture(self):
        """Create mock furniture for testing if we couldn't load from file"""
        # Cabinet
        cabinet_centroid = np.array([-0.15, 0.03, 0.20])
        cabinet_dimensions = np.array([0.56, 0.52, 0.92])
        cabinet_points = self._generate_box_points(cabinet_centroid, cabinet_dimensions)
        
        cabinet = ObjectNode(
            object_id=17,
            color=(0.5, 0.5, 0.5),
            sem_label="cabinet",
            points=cabinet_points,
            tracking_points=cabinet_points[:10],
            mesh_mask=np.ones(len(cabinet_points), dtype=bool),
            confidence=0.9,
            movable=True
        )
        
        # Shelf
        shelf_centroid = np.array([1.77, 2.30, 0.59])
        shelf_dimensions = np.array([1.35, 0.51, 1.10])
        shelf_points = self._generate_box_points(shelf_centroid, shelf_dimensions)
        
        shelf = ObjectNode(
            object_id=7,
            color=(0.5, 0.5, 0.5),
            sem_label="shelf",
            points=shelf_points,
            tracking_points=shelf_points[:10],
            mesh_mask=np.ones(len(shelf_points), dtype=bool),
            confidence=0.95,
            movable=False
        )
        
        # Add nodes to scene graph
        self.mock_robot_state.scene_graph.nodes[17] = cabinet
        self.mock_robot_state.scene_graph.nodes[7] = shelf
        
        # Add label mappings
        self.mock_robot_state.scene_graph.label_mapping[17] = "cabinet"
        self.mock_robot_state.scene_graph.label_mapping[7] = "shelf"

    def test_snap_to_cardinal(self):
        """Test the snap_to_cardinal function"""
        # Test cardinal directions
        self.assertTrue(np.allclose(snap_to_cardinal(np.array([1, 0, 0])), np.array([1, 0, 0])))
        self.assertTrue(np.allclose(snap_to_cardinal(np.array([0, 1, 0])), np.array([0, 1, 0])))
        self.assertTrue(np.allclose(snap_to_cardinal(np.array([-1, 0, 0])), np.array([-1, 0, 0])))
        self.assertTrue(np.allclose(snap_to_cardinal(np.array([0, -1, 0])), np.array([0, -1, 0])))
        
        # Test diagonals
        self.assertTrue(np.allclose(snap_to_cardinal(np.array([0.7, 0.7, 0])), np.array([1, 0, 0])) or 
                        np.allclose(snap_to_cardinal(np.array([0.7, 0.7, 0])), np.array([0, 1, 0])))
        
        # Test with z component
        self.assertTrue(np.allclose(snap_to_cardinal(np.array([0.7, 0.2, 0.5])), np.array([1, 0, 0])))
        
        # Test zero vector
        self.assertTrue(np.allclose(snap_to_cardinal(np.array([0, 0, 0])), np.array([1, 0, 0])))

    def test_get_distance_to_shelf(self):
        """Test the _get_distance_to_shelf function"""
        # Test with a valid node
        for node_id in self.mock_robot_state.scene_graph.nodes:
            distance, centroid = _get_distance_to_shelf(node_id)
            
            # Check that distance is reasonable
            self.assertGreater(distance, 0.8)  # Should be at least the minimum
            
            # Check that centroid matches node centroid
            self.assertTrue(np.allclose(centroid, self.mock_robot_state.scene_graph.nodes[node_id].centroid))
            
            logger.info(f"Distance to node {node_id} ({self.mock_robot_state.scene_graph.label_mapping.get(node_id, 'unknown')}): {distance}")

    def test_get_shelf_front(self):
        """Test the _get_shelf_front function"""
        # Test with valid point cloud
        for node_id, node in self.mock_robot_state.scene_graph.nodes.items():
            # Create point cloud
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(node.points)
            
            # Get front normal
            front_normal = _get_shelf_front(pcd, node.centroid)
            
            # Check that normal is a unit vector
            self.assertAlmostEqual(np.linalg.norm(front_normal), 1.0, places=5)
            
            # Check that normal is roughly horizontal (z component near 0)
            self.assertLess(abs(front_normal[2]), 0.1)
            
            logger.info(f"Front normal for node {node_id} ({self.mock_robot_state.scene_graph.label_mapping.get(node_id, 'unknown')}): {front_normal}")
    
    def test_get_pose_in_front_of_furniture(self):
        """Test the get_pose_in_front_of_furniture function"""
        # Test with valid furniture nodes
        for node_id in self.mock_robot_state.scene_graph.nodes:
            logger.info(f"Testing get_pose_in_front_of_furniture with node {node_id} ({self.mock_robot_state.scene_graph.label_mapping.get(node_id, 'unknown')})")
            
            try:
                # Get interaction pose
                interaction_pose = get_pose_in_front_of_furniture(node_id)
                
                # Check that pose is valid
                self.assertIsInstance(interaction_pose, Pose3D)
                
                # Check that pose is in front of furniture
                furniture_centroid = self.mock_robot_state.scene_graph.nodes[node_id].centroid
                pose_position = interaction_pose.position()
                
                # Calculate vector from furniture to pose
                direction = pose_position - furniture_centroid
                distance = np.linalg.norm(direction)
                
                # Verify the distance is reasonable
                self.assertGreater(distance, 0.8)
                
                # Verify the height is reasonable (should be close to furniture height)
                self.assertAlmostEqual(pose_position[2], furniture_centroid[2], delta=0.5)
                
                # Verify the pose is facing the furniture
                # The pose's negative direction vector should roughly point toward the furniture
                pose_direction = interaction_pose.direction()
                furniture_direction = -direction / distance  # Unit vector pointing toward furniture
                
                # Dot product should be close to 1 (vectors pointing in same direction)
                dot_product = np.dot(pose_direction, furniture_direction)
                self.assertGreater(dot_product, 0.8)
                
                logger.info(f"Interaction pose: {pose_position}, direction: {pose_direction}")
                logger.info(f"Distance from furniture: {distance}")
                
                # Verify the furniture now has a normal set
                self.assertIsNotNone(self.mock_robot_state.scene_graph.nodes[node_id].equation)
                
            except Exception as e:
                logger.error(f"Error testing node {node_id}: {e}")
                raise
    
    def test_invalid_furniture_index(self):
        """Test with an invalid furniture index"""
        # Get the highest index in the scene graph
        max_index = max(self.mock_robot_state.scene_graph.nodes.keys())
        
        # Test with an index that doesn't exist
        interaction_pose = get_pose_in_front_of_furniture(max_index + 1)
        
        # Should return a default pose
        self.assertIsInstance(interaction_pose, Pose3D)
    
    def test_memory_usage(self):
        """Test memory usage during repeated calls"""
        import gc
        import psutil
        
        process = psutil.Process()
        
        # Get initial memory usage
        gc.collect()
        initial_memory = process.memory_info().rss / 1024 / 1024  # Convert to MB
        
        # Call function multiple times
        for i in range(5):
            for node_id in self.mock_robot_state.scene_graph.nodes:
                interaction_pose = get_pose_in_front_of_furniture(node_id)
                
            # Force garbage collection
            gc.collect()
            
            # Check memory usage
            current_memory = process.memory_info().rss / 1024 / 1024  # Convert to MB
            logger.info(f"Memory usage after iteration {i+1}: {current_memory:.2f} MB (change: {current_memory - initial_memory:.2f} MB)")
            
            # Memory should not increase significantly over iterations
            # Allow some leeway for normal Python memory management
            self.assertLess(current_memory - initial_memory, 50)  # Less than 50MB increase

if __name__ == "__main__":
    print("Running test_object_interaction.py")
    unittest.main() 
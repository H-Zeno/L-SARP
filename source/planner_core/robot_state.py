import base64
import io
import numpy as np
import cv2
from typing import List, Dict, Any, Optional, Tuple
from collections import deque
from dataclasses import dataclass, field
import yaml

from PIL import Image
import math
import json
import pickle
from pathlib import Path
import os
from dotenv import dotenv_values

# Import Spot SDK
import bosdyn.client
import bosdyn.client.util
from bosdyn.client.robot import Robot
from bosdyn.client.image import ImageClient
from bosdyn.client.robot_state import RobotStateClient
from bosdyn.client.frame_helpers import VISION_FRAME_NAME, BODY_FRAME_NAME, ODOM_FRAME_NAME, GRAV_ALIGNED_BODY_FRAME_NAME, get_a_tform_b
from bosdyn.client import math_helpers

# Import scene graph
from source.LostFound.src.scene_graph import SceneGraph

@dataclass
class RobotState:
    """
    Represents the current state of the Spot robot, serving as short-term memory for the agentic framework.
    
    This class maintains real-time information about the robot's state, including:
    - Live video feed from the front camera
    - Semantic scene graph with objects in field of view highlighted
    - Current room location
    - History of recent actions
    - Current task
    
    It provides methods to update this state and format it for use by the agentic framework.
    """
    
    # Robot connection and configuration
    spot_config: Dict[str, Any] = field(default_factory=dict)
    robot: Optional[Robot] = None
    image_client: Optional[ImageClient] = None
    robot_state_client: Optional[RobotStateClient] = None
    
    # Current video frame from the robot's camera
    current_frame: Optional[np.ndarray] = None
    frame_base64: Optional[str] = None
    available_cameras: List[str] = field(default_factory=list)
    
    # Scene graph with semantic information about the environment
    scene_graph: Optional[SceneGraph] = None
    objects_in_view: List[int] = field(default_factory=list)
    
    # Robot location and navigation information
    current_room: str = "unknown"
    robot_pose: Optional[np.ndarray] = None
    
    # Kinematic state information
    odom_tform_body: Optional[math_helpers.SE3Pose] = None
    vision_tform_body: Optional[math_helpers.SE3Pose] = None
    gravity_aligned_tform_body: Optional[math_helpers.SE3Pose] = None
    
    # Task and action history
    current_task: str = ""
    action_history: deque = field(default_factory=lambda: deque(maxlen=5))
    
    # Additional state information that might be added in the future
    additional_state: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize the robot state with configuration from environment file."""
        self._load_spot_config()
        
    def _load_spot_config(self) -> None:
        """Load Spot robot configuration from environment file."""
        try:
            env_path = Path(os.path.dirname(os.path.abspath(__file__))).parent.parent / ".environment.yaml"
            with open(env_path, 'r') as file:
                config = yaml.safe_load(file)
                self.spot_config = config.get('spot', {})
        except Exception as e:
            print(f"Error loading Spot configuration: {e}")
            self.spot_config = {}
    
    def connect_to_spot(self) -> None:
        """Establish connection to the Spot robot."""
        try:
            # Create robot object and authenticate
            sdk = bosdyn.client.create_standard_sdk('RobotStateClient')
            self.robot = sdk.create_robot(self.spot_config.get('wifi_default_address'))
            self.robot.authenticate(
                self.spot_config.get('spot_admin_console_username'),
                self.spot_config.get('spot_admin_console_password')
            )
            
            # Establish time sync with the robot
            bosdyn.client.util.authenticate(self.robot)
            self.robot.time_sync.wait_for_sync()
            
            # Initialize clients
            self.image_client = self.robot.ensure_client(ImageClient.default_service_name)
            self.robot_state_client = self.robot.ensure_client(RobotStateClient.default_service_name)
            
            # Get available cameras
            self._get_available_cameras()
            
            # Update the robot's pose from kinematic state
            self.update_pose_from_kinematic_state()
            
            print("Successfully connected to Spot robot")
        except Exception as e:
            print(f"Failed to connect to Spot robot: {e}")
            self.robot = None
    
    def _get_available_cameras(self) -> None:
        """Get a list of available cameras on the robot."""
        if not self.robot or not self.image_client:
            print("Robot not connected. Cannot get available cameras.")
            return
        
        try:
            sources = self.image_client.list_image_sources()
            self.available_cameras = [source.name for source in sources]
            print(f"Available cameras: {self.available_cameras}")
        except Exception as e:
            print(f"Error getting available cameras: {e}")
            self.available_cameras = []
    
    def update_video_feed(self, camera_name: str = "frontright_fisheye_image") -> None:
        """
        Update the current video frame from the specified robot camera.
        
        Args:
            camera_name: Name of the camera to use. Defaults to "frontright_fisheye_image".
                         Use one of the available cameras from self.available_cameras.
        """
        if not self.robot or not self.image_client:
            print("Robot not connected. Cannot update video feed.")
            return
        
        if not self.available_cameras:
            self._get_available_cameras()
            
        if camera_name not in self.available_cameras:
            print(f"Camera {camera_name} not available. Using first available camera.")
            if self.available_cameras:
                camera_name = self.available_cameras[0]
            else:
                print("No cameras available.")
                return
        
        try:
            # Get image from the specified camera
            image_response = self.image_client.get_image_from_sources([camera_name])[0]
            
            # Convert the image data to a PIL Image
            pil_image = Image.open(io.BytesIO(image_response.shot.image.data))
            
            # Convert PIL Image to numpy array for OpenCV processing
            self.current_frame = np.array(pil_image)
            
            # Convert to base64 for inclusion in prompts
            buffered = io.BytesIO()
            pil_image.save(buffered, format="JPEG")
            self.frame_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
            
            # Update objects in field of view based on the new frame
            self._update_objects_in_view()
            
            print(f"Updated video feed from camera: {camera_name}")
        except Exception as e:
            print(f"Error updating video feed: {e}")
    
    def update_pose_from_kinematic_state(self) -> None:
        """
        Update the robot's pose information from its kinematic state.
        
        This method retrieves the robot's current kinematic state and extracts
        the transformation matrices that define the robot's position and orientation
        in different reference frames.
        """
        if not self.robot or not self.robot_state_client:
            print("Robot not connected. Cannot update pose from kinematic state.")
            return
            
        try:
            # Get the robot's current state
            robot_state = self.robot_state_client.get_robot_state()
            
            # Extract the transform snapshot from the kinematic state
            transforms_snapshot = robot_state.kinematic_state.transforms_snapshot
            
            # Get the robot's pose in different reference frames
            # 1. Odometry frame - useful for relative motion
            self.odom_tform_body = get_a_tform_b(transforms_snapshot, 
                                                ODOM_FRAME_NAME, 
                                                BODY_FRAME_NAME)
            
            # 2. Vision frame - useful for global positioning if using visual odometry
            try:
                self.vision_tform_body = get_a_tform_b(transforms_snapshot, 
                                                      VISION_FRAME_NAME, 
                                                      BODY_FRAME_NAME)
            except Exception as e:
                print(f"Could not get vision frame transform: {e}")
                self.vision_tform_body = None
            
            # 3. Gravity aligned body frame - useful for commands that need to be aligned with gravity
            self.gravity_aligned_tform_body = get_a_tform_b(transforms_snapshot, 
                                                           ODOM_FRAME_NAME, 
                                                           GRAV_ALIGNED_BODY_FRAME_NAME)
            
            # Update the robot_pose with the odometry transform as a 4x4 matrix
            self.robot_pose = self.odom_tform_body.to_matrix()
            # the boston dynamics example shows the use of the gravity aligned frame
            
            # After updating the pose, update which objects are in view
            self._update_objects_in_view()
            
            print("Updated robot pose from kinematic state")
        except Exception as e:
            print(f"Error updating pose from kinematic state: {e}")
    
    def get_position(self) -> Tuple[float, float, float]:
        """
        Get the robot's current position in the odometry frame.
        
        Returns:
            Tuple[float, float, float]: The (x, y, z) position of the robot
        """
        if self.odom_tform_body is None:
            print("Robot pose not available. Call update_pose_from_kinematic_state first.")
            return (0.0, 0.0, 0.0)
            
        return (self.odom_tform_body.x, self.odom_tform_body.y, self.odom_tform_body.z)
    
    def get_orientation_quaternion(self) -> Tuple[float, float, float, float]:
        """
        Get the robot's current orientation as a quaternion in the odometry frame.
        
        Returns:
            Tuple[float, float, float, float]: The orientation as a quaternion (w, x, y, z)
        """
        if self.odom_tform_body is None:
            print("Robot pose not available. Call update_pose_from_kinematic_state first.")
            return (1.0, 0.0, 0.0, 0.0)  # Identity quaternion
            
        return (self.odom_tform_body.rot.w, self.odom_tform_body.rot.x, 
                self.odom_tform_body.rot.y, self.odom_tform_body.rot.z)
    
    def get_orientation_euler(self) -> Tuple[float, float, float]:
        """
        Get the robot's current orientation as Euler angles (roll, pitch, yaw) in the odometry frame.
        
        Returns:
            Tuple[float, float, float]: The orientation as Euler angles (roll, pitch, yaw) in radians
        """
        if self.odom_tform_body is None:
            print("Robot pose not available. Call update_pose_from_kinematic_state first.")
            return (0.0, 0.0, 0.0)
            
        # Convert quaternion to Euler angles
        euler = self.odom_tform_body.rot.to_euler_zxy()
        return (euler.roll, euler.pitch, euler.yaw)
    
    def load_scene_graph(self, scene_graph_path: str) -> None:
        """
        Load a scene graph from a file.
        
        Args:
            scene_graph_path: Path to the scene graph file (JSON or pickle)
        """
        try:
            if scene_graph_path.endswith('.json'):
                with open(scene_graph_path, 'r') as f:
                    scene_data = json.load(f)
                    
                # Create a new scene graph and populate it with the loaded data
                self.scene_graph = SceneGraph()
                
                # Set basic properties
                self.scene_graph.pose = np.array(scene_data.get('pose', None))
                self.scene_graph.min_confidence = scene_data.get('min_confidence', 0.2)
                self.scene_graph.k = scene_data.get('k', 2)
                self.scene_graph.immovable = scene_data.get('immovable', [])
                
                # Load nodes
                for node_data in scene_data.get('nodes', []):
                    # In a real implementation, you would create proper node objects
                    # This is a simplified version
                    node_id = node_data.get('object_id')
                    self.scene_graph.nodes[node_id] = node_data
                    self.scene_graph.ids.append(node_id)
                
                # Load connections
                self.scene_graph.outgoing = scene_data.get('outgoing', {})
                self.scene_graph.ingoing = scene_data.get('ingoing', {})
                
            elif scene_graph_path.endswith('.pkl'):
                with open(scene_graph_path, 'rb') as f:
                    self.scene_graph = pickle.load(f)
            else:
                print(f"Unsupported scene graph file format: {scene_graph_path}")
                
            if self.scene_graph:
                print(f"Successfully loaded scene graph with {len(self.scene_graph.nodes)} nodes")
        except Exception as e:
            print(f"Error loading scene graph: {e}")
            self.scene_graph = None
    
    def _update_objects_in_view(self) -> None:
        """
        Update the list of objects that are currently in the robot's field of view.
        
        This uses the robot's current pose and camera parameters to determine which
        objects from the scene graph are visible in the current camera frame.
        """
        if not self.scene_graph or not self.robot_pose:
            return
        
        self.objects_in_view = []
        
        # Camera parameters (these would be obtained from the actual robot in production)
        # Field of view in radians (typical values for Spot's front camera)
        horizontal_fov = math.radians(70)
        vertical_fov = math.radians(43)
        
        # Extract camera position and orientation from robot pose
        camera_position = self.robot_pose[:3, 3]
        camera_forward = self.robot_pose[:3, 2]  # Z-axis is forward in camera frame
        camera_up = self.robot_pose[:3, 1]       # Y-axis is up in camera frame
        camera_right = self.robot_pose[:3, 0]    # X-axis is right in camera frame
        
        # Check each object in the scene graph
        for node_id, node in self.scene_graph.nodes.items():
            # Get object centroid
            if isinstance(node, dict):  # If loaded from JSON
                if 'centroid' not in node:
                    continue
                centroid = np.array(node.get('centroid', [0, 0, 0]))
            else:  # If using actual SceneGraph object
                if not hasattr(node, 'centroid'):
                    continue
                centroid = node.centroid
            
            # Vector from camera to object
            direction = centroid - camera_position
            distance = np.linalg.norm(direction)
            
            # Avoid division by zero
            if distance < 1e-6:  # Small epsilon value
                continue
                
            # Normalize direction vector
            direction = direction / distance
            
            # Project direction onto camera axes
            forward_proj = np.dot(direction, camera_forward)
            
            # Only consider objects in front of the camera
            if forward_proj <= 0:
                continue
                
            # Calculate horizontal and vertical angles
            right_proj = np.dot(direction, camera_right)
            up_proj = np.dot(direction, camera_up)
            
            horizontal_angle = math.atan2(right_proj, forward_proj)
            vertical_angle = math.atan2(up_proj, forward_proj)
            
            # Check if object is within field of view
            if (abs(horizontal_angle) <= horizontal_fov / 2 and 
                abs(vertical_angle) <= vertical_fov / 2):
                self.objects_in_view.append(node_id)
                
                # Mark the object as in field of view in the scene graph
                if isinstance(node, dict):
                    node['in_field_of_view'] = True
                else:
                    # Assuming the ObjectNode class has an attribute for this
                    # If not, you would need to add this attribute to the class
                    if hasattr(node, 'in_field_of_view'):
                        node.in_field_of_view = True
    
    def update_room_location(self, room: str) -> None:
        """
        Update the current room location of the robot.
        
        Args:
            room: Name of the current room (e.g., "kitchen", "living room")
        """
        self.current_room = room
    
    def update_robot_pose(self, pose: np.ndarray) -> None:
        """
        Update the robot's current pose in the world.
        
        Args:
            pose: 4x4 transformation matrix representing the robot's pose
        """
        self.robot_pose = pose
        # After updating the pose, update which objects are in view
        self._update_objects_in_view()
    
    def set_current_task(self, task: str) -> None:
        """
        Set the current task the robot is working on.
        
        Args:
            task: Description of the current task
        """
        self.current_task = task
    
    def add_action(self, action: str) -> None:
        """
        Add an action to the robot's action history.
        
        Args:
            action: Description of the action taken
        """
        self.action_history.append(action)
    
    def add_additional_state(self, key: str, value: Any) -> None:
        """
        Add additional state information to the robot state.
        This allows for extensibility of the robot state.
        
        Args:
            key: Key for the additional state information
            value: Value of the additional state information
        """
        self.additional_state[key] = value
    
    def get_additional_state(self, key: str, default: Any = None) -> Any:
        """
        Get additional state information from the robot state.
        
        Args:
            key: Key for the additional state information
            default: Default value to return if key is not found
            
        Returns:
            The value associated with the key, or the default value if not found
        """
        return self.additional_state.get(key, default)
    
    def format_for_prompt(self) -> str:
        """
        Format the robot state as a string for inclusion in a prompt to the agentic framework.
        
        Returns:
            str: Formatted robot state information
        """
        prompt = "# ROBOT STATE\n\n"
        
        # Current task
        prompt += f"## Current Task\n{self.current_task}\n\n"
        
        # Current location
        prompt += f"## Current Location\nRoom: {self.current_room}\n\n"
        
        # Robot position and orientation
        prompt += "## Robot Pose\n"
        if self.odom_tform_body:
            position = self.get_position()
            orientation = self.get_orientation_euler()
            prompt += f"Position (x, y, z): ({position[0]:.2f}, {position[1]:.2f}, {position[2]:.2f})\n"
            prompt += f"Orientation (roll, pitch, yaw): ({orientation[0]:.2f}, {orientation[1]:.2f}, {orientation[2]:.2f}) rad\n"
        else:
            prompt += "Pose information not available.\n"
        prompt += "\n"
        
        # Action history
        prompt += "## Recent Actions\n"
        if self.action_history:
            for i, action in enumerate(reversed(self.action_history), 1):
                prompt += f"{i}. {action}\n"
        else:
            prompt += "No recent actions.\n"
        prompt += "\n"
        
        # Objects in view
        prompt += "## Objects in Field of View\n"
        if self.objects_in_view and self.scene_graph:
            for obj_id in self.objects_in_view:
                node = self.scene_graph.nodes.get(obj_id)
                if node:
                    if isinstance(node, dict):  # If loaded from JSON
                        label = node.get('sem_label', 'unknown')
                    else:  # If using actual SceneGraph object
                        label = self.scene_graph.label_mapping.get(node.sem_label, node.sem_label)
                    prompt += f"- {label} (ID: {obj_id})\n"
        else:
            prompt += "No objects in view.\n"
        prompt += "\n"
        
        # Scene graph summary
        if self.scene_graph:
            prompt += "## Scene Graph Summary\n"
            prompt += f"Total objects: {len(self.scene_graph.nodes)}\n"
            
            # Count objects by type
            object_types: Dict[str, int] = {}
            for node_id, node in self.scene_graph.nodes.items():
                if isinstance(node, dict):  # If loaded from JSON
                    label = node.get('sem_label', 'unknown')
                else:  # If using actual SceneGraph object
                    label = self.scene_graph.label_mapping.get(node.sem_label, node.sem_label)
                
                object_types[label] = object_types.get(label, 0) + 1
            
            prompt += "Object types:\n"
            for obj_type, count in object_types.items():
                prompt += f"- {obj_type}: {count}\n"
            prompt += "\n"
        
        # Available cameras
        if self.available_cameras:
            prompt += "## Available Cameras\n"
            for camera in self.available_cameras:
                prompt += f"- {camera}\n"
            prompt += "\n"
        
        # Additional state information
        if self.additional_state:
            prompt += "## Additional State Information\n"
            for key, value in self.additional_state.items():
                prompt += f"- {key}: {value}\n"
            prompt += "\n"
        
        # Camera feed reference (in a real implementation, this would be a URL or base64 image)
        if self.frame_base64:
            prompt += "## Camera Feed\n"
            prompt += "[Camera feed is available as a base64-encoded image]\n\n"
        
        return prompt
    
    def format_for_json(self) -> Dict[str, Any]:
        """
        Format the robot state as a JSON-serializable dictionary.
        
        Returns:
            Dict[str, Any]: JSON-serializable representation of the robot state
        """
        state_dict = {
            "current_task": self.current_task,
            "current_room": self.current_room,
            "action_history": list(self.action_history),
            "objects_in_view": self.objects_in_view,
            "available_cameras": self.available_cameras,
            "additional_state": self.additional_state
        }
        
        # Add position and orientation if available
        if self.odom_tform_body:
            position = self.get_position()
            orientation_quat = self.get_orientation_quaternion()
            orientation_euler = self.get_orientation_euler()
            
            state_dict["position"] = {
                "x": position[0],
                "y": position[1],
                "z": position[2]
            }
            
            state_dict["orientation"] = {
                "quaternion": {
                    "w": orientation_quat[0],
                    "x": orientation_quat[1],
                    "y": orientation_quat[2],
                    "z": orientation_quat[3]
                },
                "euler": {
                    "roll": orientation_euler[0],
                    "pitch": orientation_euler[1],
                    "yaw": orientation_euler[2]
                }
            }
        
        # Add camera frame as base64 if available
        if self.frame_base64:
            state_dict["camera_frame"] = self.frame_base64
        
        return state_dict
    
    def save_current_frame(self, file_path: str) -> bool:
        """
        Save the current camera frame to a file.
        
        Args:
            file_path: Path where the image should be saved
            
        Returns:
            bool: True if successful, False otherwise
        """
        if self.current_frame is None:
            print("No current frame to save.")
            return False
        
        try:
            # Convert numpy array to PIL Image
            pil_image = Image.fromarray(self.current_frame)
            pil_image.save(file_path)
            print(f"Saved current frame to {file_path}")
            return True
        except Exception as e:
            print(f"Error saving current frame: {e}")
            return False

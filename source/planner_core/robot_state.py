# Standard library imports
import logging
import os
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Third-party imports
import numpy as np
from dotenv import dotenv_values
from PIL import Image

# Bosdyn SDK imports
from bosdyn import client as bosdyn_client
from bosdyn.api import estop_pb2
from bosdyn.api.spot import robot_command_pb2 as spot_command_pb2
from bosdyn.client import math_helpers, util as bosdyn_util
from bosdyn.client.estop import EstopClient

# Local imports
from robot_utils.frame_transformer import FrameTransformerSingleton
from robot_utils.video import (
    frame_coordinate_from_depth_image,
    get_camera_rgbd,
    localize_from_images,
    relocalize,
    select_points_from_bounding_box,
    set_gripper,
    set_gripper_camera_params
)
from source.LostFound.src.scene_graph import SceneGraph
from utils import environment
from utils.recursive_config import Config

from bosdyn.client.robot_command import (
    RobotCommandClient,
    blocking_selfright,
    blocking_stand,
)
from bosdyn.client.robot_state import RobotStateClient
from bosdyn.client.frame_helpers import VISION_FRAME_NAME, BODY_FRAME_NAME, ODOM_FRAME_NAME, GRAV_ALIGNED_BODY_FRAME_NAME, get_a_tform_b

# Import singletons
from utils.singletons import (
    _SingletonWrapper,
    GraphNavClientSingleton,
    ImageClientSingleton,
    RobotCommandClientSingleton,
    RobotSingleton,
    RobotStateClientSingleton,
    WorldObjectClientSingleton,
)

frame_transformer = FrameTransformerSingleton()
graph_nav_client = GraphNavClientSingleton()
image_client = ImageClientSingleton()
robot_command_client = RobotCommandClientSingleton()
robot = RobotSingleton()
robot_state_client = RobotStateClientSingleton()
world_object_client = WorldObjectClientSingleton()


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
    config = Config()
    
    # Scene graph with semantic information about the environment
    scene_graph: Optional[SceneGraph] = None


    # objects_in_view: List[int] = field(default_factory=list)
    
    # Current video frame from the robot's camera
    # current_image: Optional[np.ndarray] = None
    # current_image_metadata: Optional[Dict[str, Any]] = None
    # available_cameras: List[str] = field(default_factory=list)
    
    # # Video feed refresh parameters
    # _video_refresh_thread: Optional[threading.Thread] = None
    # _stop_refresh_event: Optional[threading.Event] = None
    # _refresh_rate_hz: float = 10.0  # Default refresh rate in Hz
    # _current_camera: str = "frontright_fisheye_image"
    # _refresh_lock: threading.RLock = field(default_factory=threading.RLock)
    # _is_refreshing: bool = False
    
    # # Pose update parameters
    # _pose_update_thread: Optional[threading.Thread] = None
    # _stop_pose_update_event: Optional[threading.Event] = None
    # _pose_update_rate_hz: float = 5.0  # Default pose update rate in Hz
    # _is_updating_pose: bool = False
    
    def __init__(self, scene_graph: SceneGraph):

        self.connect_to_spot()
        self.current_room: str = "unknown"
        self.scene_graph = scene_graph  # Explicitly set the scene_graph attribute

        # # The equivalent of this was written for the frame_transformer class. Why is it used there?
        # robot_state = RobotStateSingleton()
        # robot_state.set_instance(self)

    def connect_to_spot(self) -> None:
        """Establish connection to the Spot robot."""

        # Setup adapted from github.com/boston-dynamics/spot-sdk/blob/master/python/examples/hello_spot/hello_spot.py
        spot_env_config = environment.get_environment_config(self.config, ["spot"])
        robot_config = self.config["robot_parameters"]
        self.sdk = bosdyn_client.create_standard_sdk("understanding-spot")
    
        # setup logging
        bosdyn_util.setup_logging(robot_config["verbose"])

        # setup robot
        global robot
        robot.set_instance(self.sdk.create_robot(spot_env_config["wifi_default_address"]))
        environment.set_robot_password(self.config)
        bosdyn_util.authenticate(robot)

        # Establish time sync with the robot. 
        robot.time_sync.wait_for_sync()

        # Verify the robot is not estopped and that an external application has registered and holds
        # an estop endpoint.
        self._verify_estop()

        # The robot state client will allow us to get the robot's state information, and construct
        # a command using frame information published by the robot.
        global robot_state_client
        robot_state_client.set_instance(robot.ensure_client(RobotStateClient.default_service_name))
        
        global robot_command_client
        robot_command_client.set_instance(
            robot.ensure_client(RobotCommandClient.default_service_name)
        )

        lease_client = robot.ensure_client(
        bosdyn_client.lease.LeaseClient.default_service_name
        )

        with bosdyn_client.lease.LeaseKeepAlive(
            lease_client, must_acquire=True, return_at_exit=True
        ):

            robot.power_on(timeout_sec=20)
            assert robot.is_powered_on(), "Robot power on failed."
            # robot.logger.info("Robot powered on.")

            battery_states = robot_state_client.get_robot_state().battery_states[0]
            percentage = battery_states.charge_percentage.value
            estimated_time = battery_states.estimated_runtime.seconds / 60
            if percentage < 20.0:
                robot.logger.info(
                    f"\033[91mCurrent battery percentage at {percentage}%.\033[0m"
                )
            else:
                robot.logger.info(f"Current battery percentage at {percentage}%.")
            robot.logger.info(f"Estimated time left {estimated_time:.2f} min.")

            #################################
            # localization of spot based on camera images and depth scans
            #################################
            start_time = time.time()
            set_gripper_camera_params('640x480')

            self.frame_name = localize_from_images(self.config, vis_block=False)
            print("====================================")
            print(f"Frame name: {self.frame_name}")
            print("====================================")
            end_time_localization = time.time()
            logging.info(f"Spot localization succesfull. Localization time: {end_time_localization - start_time}")

            logging.info("Successfully connected to Spot robot")

        # self.image_client = self.robot.ensure_client(ImageClient.default_service_name)

        # # Get available cameras
        # self._get_available_cameras()
        
        # # Update the robot's pose from kinematic state
        # self.update_pose_from_kinematic_state()
        
        # # Get default refresh rate from config if available
        # default_refresh_rate = robot_config.get("video_refresh_rate_hz", 10.0)
        # default_camera = robot_config.get("default_camera", "frontright_fisheye_image")
        # default_pose_rate = robot_config.get("pose_update_rate_hz", 5.0)
        
        # # Start the video refresh thread with the configured refresh rate
        # self.start_video_refresh(refresh_rate_hz=default_refresh_rate, camera_name=default_camera)
        
        # # Start the pose update thread with the configured update rate
        # self.start_pose_update(update_rate_hz=default_pose_rate)
        

    def _verify_estop(self):
        """Verify the robot is not estopped"""
        # https://github.com/boston-dynamics/spot-sdk/blob/master/python/examples/arm_joint_move/arm_joint_move.py

        client = robot.ensure_client(EstopClient.default_service_name)
        if client.get_status().stop_level != estop_pb2.ESTOP_LEVEL_NONE:
            error_message = (
                "Robot is estopped. Please use an external E-Stop client, such as the "
                "estop SDK example, to configure E-Stop."
            )
            robot.logger.error(error_message)
            raise EStopError(error_message)

    
    def _get_available_cameras(self) -> None:
        """Get a list of available cameras on the robot."""
        if not robot or not image_client:
            print("Robot not connected. Cannot get available cameras.")
            return
        
        try:
            sources = image_client.list_image_sources()
            self.available_cameras = [source.name for source in sources]
            print(f"Available cameras: {self.available_cameras}")
        except Exception as e:
            print(f"Error getting available cameras: {e}")
            self.available_cameras = []
    
    # def update_video_feed(self, camera_name: str = "frontright_fisheye_image") -> None:
    #     """
    #     Update the current video frame from the specified robot camera.
        
    #     Args:
    #         camera_name: Name of the camera to use. Defaults to "frontright_fisheye_image".
    #                      Use one of the available cameras from self.available_cameras.
    #     """
    #     # Use the lock to ensure thread safety when updating the image
    #     with self._refresh_lock:
    #         if not self.robot or not self.image_client:
    #             print("Robot not connected. Cannot update video feed.")
    #             return
            
    #         if not self.available_cameras:
    #             self._get_available_cameras()
                
    #         if camera_name not in self.available_cameras:
    #             print(f"Camera {camera_name} not available. Using first available camera.")
    #             if self.available_cameras:
    #                 camera_name = self.available_cameras[0]
    #             else:
    #                 print("No cameras available.")
    #                 return
            
    #         try:
    #             # Get image from the specified camera
    #             image_response = self.image_client.get_image_from_sources([camera_name])[0]

    #             # Convert the image data to a PIL Image
    #             pil_image = Image.open(io.BytesIO(image_response.shot.image.data))
                
    #             # Convert PIL Image to numpy array for OpenCV processing
    #             self.current_image = np.array(pil_image)
                
    #             # Store metadata from the image response
    #             self.current_image_metadata = {
    #                 'timestamp': image_response.shot.acquisition_time,
    #                 'camera_name': camera_name,
    #                 'frame_name': image_response.shot.frame_name_in_robot
    #             }
        
    #             # Update objects in field of view based on the new frame
    #             self._update_objects_in_view()
                
    #             # Only print updates when manually called, not from the refresh thread
    #             if not threading.current_thread().name == "VideoRefreshThread":
    #                 print(f"Updated video feed from camera: {camera_name}")
    #         except Exception as e:
    #             print(f"Error updating video feed: {e}")


    # def get_position(self) -> Tuple[float, float, float]:
    #     """
    #     Get the robot's current position in the odometry frame.
        
    #     Returns:
    #         Tuple[float, float, float]: The (x, y, z) position of the robot
    #     """
    #     if self.odom_tform_body is None:
    #         print("Robot pose not available. Call update_pose_from_kinematic_state first.")
    #         return (0.0, 0.0, 0.0)
            
    #     return (self.odom_tform_body.x, self.odom_tform_body.y, self.odom_tform_body.z)
    
    
    # def get_orientation_euler(self) -> Tuple[float, float, float]:
    #     """
    #     Get the robot's current orientation as Euler angles (roll, pitch, yaw) in the odometry frame.
        
    #     Returns:
    #         Tuple[float, float, float]: The orientation as Euler angles (roll, pitch, yaw) in radians
    #     """
    #     if self.odom_tform_body is None:
    #         print("Robot pose not available. Call update_pose_from_kinematic_state first.")
    #         return (0.0, 0.0, 0.0)
            
    #     # Convert quaternion to Euler angles
    #     euler = self.odom_tform_body.rot.to_euler_zxy()
    #     return (euler.roll, euler.pitch, euler.yaw)
    
    # def _update_objects_in_view(self) -> None:
    #     """
    #     Update the list of objects that are currently in the robot's field of view.
        
    #     This uses the robot's current pose and camera parameters to determine which
    #     objects from the scene graph are visible in the current camera frame.
    #     """
    #     # Use the lock to ensure thread safety when updating objects in view
    #     with self._refresh_lock:
    #         if not self.scene_graph or not self.robot_pose:
    #             return
            
    #         self.objects_in_view = []
            
    #         # Camera parameters (these would be obtained from the actual robot in production)
    #         # Field of view in radians (typical values for Spot's front camera)
    #         horizontal_fov = math.radians(70)
    #         vertical_fov = math.radians(43)
            
    #         # Extract camera position and orientation from robot pose
    #         camera_position = self.robot_pose[:3, 3]
    #         camera_forward = self.robot_pose[:3, 2]  # Z-axis is forward in camera frame
    #         camera_up = self.robot_pose[:3, 1]       # Y-axis is up in camera frame
    #         camera_right = self.robot_pose[:3, 0]    # X-axis is right in camera frame
            
    #         # Check each object in the scene graph
    #         for node_id, node in self.scene_graph.nodes.items():
    #             # Get object centroid
    #             if isinstance(node, dict):  # If loaded from JSON
    #                 if 'centroid' not in node:
    #                     continue
    #                 centroid = np.array(node.get('centroid', [0, 0, 0]))
    #             else:  # If using actual SceneGraph object
    #                 if not hasattr(node, 'centroid'):
    #                     continue
    #                 centroid = node.centroid
                
    #             # Vector from camera to object
    #             direction = centroid - camera_position
    #             distance = np.linalg.norm(direction)
                
    #             # Avoid division by zero
    #             if distance < 1e-6:  # Small epsilon value
    #                 continue
                    
    #             # Normalize direction vector
    #             direction = direction / distance
                
    #             # Project direction onto camera axes
    #             forward_proj = np.dot(direction, camera_forward)
                
    #             # Only consider objects in front of the camera
    #             if forward_proj <= 0:
    #                 continue
                    
    #             # Calculate horizontal and vertical angles
    #             right_proj = np.dot(direction, camera_right)
    #             up_proj = np.dot(direction, camera_up)
                
    #             horizontal_angle = math.atan2(right_proj, forward_proj)
    #             vertical_angle = math.atan2(up_proj, forward_proj)
                
    #             # Check if object is within field of view
    #             if (abs(horizontal_angle) <= horizontal_fov / 2 and 
    #                 abs(vertical_angle) <= vertical_fov / 2):
    #                 self.objects_in_view.append(node_id)
                    
    #                 # Mark the object as in field of view in the scene graph
    #                 if isinstance(node, dict):
    #                     node['in_field_of_view'] = True
    #                 else:
    #                     # Assuming the ObjectNode class has an attribute for this
    #                     # If not, you would need to add this attribute to the class
    #                     if hasattr(node, 'in_field_of_view'):
    #                         node.in_field_of_view = True
    
    def update_room_location(self, room: str) -> None:
        """
        Update the current room location of the robot.
        
        Args:
            room: Name of the current room (e.g., "kitchen", "living room")
        """
        self.current_room = room
    

    
    
    # def save_current_image(self, file_path: str) -> bool:
    #     """
    #     Save the current camera frame to a file.
        
    #     Args:
    #         file_path: Path where the image should be saved
            
    #     Returns:
    #         bool: True if successful, False otherwise
    #     """
    #     if self.current_image is None:
    #         print("No current frame to save.")
    #         return False
        
    #     try:
    #         # Convert numpy array to PIL Image
    #         pil_image = Image.fromarray(self.current_image)
    #         pil_image.save(file_path)
    #         print(f"Saved current frame to {file_path}")
    #         return True
    #     except Exception as e:
    #         print(f"Error saving current frame: {e}")
    #         return False

    # def start_video_refresh(self, refresh_rate_hz: float = 10.0, camera_name: str = "frontright_fisheye_image") -> bool:
    #     """
    #     Start asynchronous video feed refresh at the specified rate.
        
    #     Args:
    #         refresh_rate_hz: Refresh rate in Hz (updates per second)
    #         camera_name: Name of the camera to use
            
    #     Returns:
    #         bool: True if refresh started successfully, False otherwise
    #     """
    #     with self._refresh_lock:
    #         if self._is_refreshing:
    #             print("Video refresh is already running. Stop it first to change parameters.")
    #             return False
                
    #         if not self.robot or not self.image_client:
    #             print("Robot not connected. Cannot start video refresh.")
    #             return False
                
    #         # Update parameters
    #         self._refresh_rate_hz = max(0.1, refresh_rate_hz)  # Minimum 0.1 Hz (one update every 10 seconds)
    #         self._current_camera = camera_name
            
    #         # Reset stop event
    #         self._stop_refresh_event.clear()
            
    #         # Create and start the refresh thread
    #         self._video_refresh_thread = threading.Thread(
    #             target=self._video_refresh_worker,
    #             daemon=True,  # Thread will exit when main program exits
    #             name="VideoRefreshThread"
    #         )
    #         self._video_refresh_thread.start()
    #         self._is_refreshing = True
            
    #         print(f"Started video refresh at {self._refresh_rate_hz} Hz using camera: {self._current_camera}")
    #         return True
            
    # def stop_video_refresh(self) -> None:
    #     """
    #     Stop the asynchronous video feed refresh.
    #     """
    #     with self._refresh_lock:
    #         if not self._is_refreshing:
    #             print("Video refresh is not running.")
    #             return
                
    #         # Signal the thread to stop
    #         self._stop_refresh_event.set()
            
    #         # Wait for the thread to finish (with timeout)
    #         if self._video_refresh_thread and self._video_refresh_thread.is_alive():
    #             self._video_refresh_thread.join(timeout=2.0)
                
    #         self._is_refreshing = False
    #         print("Stopped video refresh.")
            
    # def _video_refresh_worker(self) -> None:
    #     """
    #     Background worker function that periodically updates the video feed.
    #     This runs in a separate thread.
    #     """
    #     print(f"Video refresh thread started with refresh rate: {self._refresh_rate_hz} Hz")
        
    #     # Calculate sleep time in seconds
    #     sleep_time = 1.0 / self._refresh_rate_hz
        
    #     while not self._stop_refresh_event.is_set():
    #         try:
    #             # Update the video feed
    #             self.update_video_feed(camera_name=self._current_camera)
                
    #             # Sleep for the calculated time, but check for stop event periodically
    #             # This allows for more responsive stopping
    #             self._stop_refresh_event.wait(timeout=sleep_time)
    #         except Exception as e:
    #             print(f"Error in video refresh thread: {e}")
    #             # Sleep a bit longer on error to avoid spamming
    #             time.sleep(1.0)
                
    #     print("Video refresh thread stopped.")

    # def cleanup(self) -> None:
    #     """
    #     Clean up resources and stop background threads before shutting down.
    #     This should be called when the robot is being disconnected or the program is exiting.
    #     """
    #     # Stop the video refresh thread if it's running
    #     self.stop_video_refresh()
        
    #     # Stop the pose update thread if it's running
    #     self.stop_pose_update()
        
    #     # Add any other cleanup tasks here
    #     print("Robot state cleanup completed.")

    # def set_video_refresh_parameters(self, refresh_rate_hz: Optional[float] = None, camera_name: Optional[str] = None) -> bool:
    #     """
    #     Update the video refresh parameters without stopping the refresh thread.
        
    #     Args:
    #         refresh_rate_hz: New refresh rate in Hz (updates per second), or None to keep current rate
    #         camera_name: New camera name to use, or None to keep current camera
            
    #     Returns:
    #         bool: True if parameters were updated successfully, False otherwise
    #     """
    #     with self._refresh_lock:
    #         # If refresh is not running, just update the parameters
    #         if not self._is_refreshing:
    #             if refresh_rate_hz is not None:
    #                 self._refresh_rate_hz = max(0.1, refresh_rate_hz)
    #             if camera_name is not None:
    #                 self._current_camera = camera_name
    #             print(f"Updated video refresh parameters: rate={self._refresh_rate_hz} Hz, camera={self._current_camera}")
    #             return True
                
    #         # If refresh is running, we need to restart it with new parameters
    #         current_rate = self._refresh_rate_hz
    #         current_camera = self._current_camera
            
    #         # Update parameters if provided
    #         if refresh_rate_hz is not None:
    #             self._refresh_rate_hz = max(0.1, refresh_rate_hz)
    #         if camera_name is not None:
    #             self._current_camera = camera_name
                
    #         # Only restart if parameters actually changed
    #         if (current_rate != self._refresh_rate_hz or current_camera != self._current_camera):
    #             # Stop and restart the refresh thread
    #             self.stop_video_refresh()
    #             success = self.start_video_refresh(self._refresh_rate_hz, self._current_camera)
    #             return success
            
    #         return True

    # def get_video_refresh_status(self) -> Dict[str, Any]:
    #     """
    #     Get the current status of the video refresh thread and its parameters.
        
    #     Returns:
    #         Dict[str, Any]: Dictionary containing refresh status information
    #     """
    #     with self._refresh_lock:
    #         status = {
    #             "is_refreshing": self._is_refreshing,
    #             "refresh_rate_hz": self._refresh_rate_hz,
    #             "current_camera": self._current_camera,
    #             "available_cameras": self.available_cameras,
    #             "last_frame_timestamp": None
    #         }
            
    #         # Add timestamp of the last frame if available
    #         if self.current_image_metadata and 'timestamp' in self.current_image_metadata:
    #             status["last_frame_timestamp"] = self.current_image_metadata['timestamp']
                
    #         return status

    # def start_pose_update(self, update_rate_hz: float = 5.0) -> bool:
    #     """
    #     Start asynchronous pose update at the specified rate.
        
    #     Args:
    #         update_rate_hz: Update rate in Hz (updates per second)
            
    #     Returns:
    #         bool: True if update started successfully, False otherwise
    #     """
    #     with self._refresh_lock:
    #         if self._is_updating_pose:
    #             print("Pose update is already running. Stop it first to change parameters.")
    #             return False
                
    #         if not self.robot or not self.robot_state_client:
    #             print("Robot not connected. Cannot start pose update.")
    #             return False
                
    #         # Update parameters
    #         self._pose_update_rate_hz = max(0.1, update_rate_hz)  # Minimum 0.1 Hz (one update every 10 seconds)
            
    #         # Reset stop event
    #         self._stop_pose_update_event.clear()
            
    #         # Create and start the update thread
    #         self._pose_update_thread = threading.Thread(
    #             target=self._pose_update_worker,
    #             daemon=True,  # Thread will exit when main program exits
    #             name="PoseUpdateThread"
    #         )
    #         self._pose_update_thread.start()
    #         self._is_updating_pose = True
            
    #         print(f"Started pose update at {self._pose_update_rate_hz} Hz")
    #         return True
            
    # def stop_pose_update(self) -> None:
    #     """
    #     Stop the asynchronous pose update.
    #     """
    #     with self._refresh_lock:
    #         if not self._is_updating_pose:
    #             print("Pose update is not running.")
    #             return
                
    #         # Signal the thread to stop
    #         self._stop_pose_update_event.set()
            
    #         # Wait for the thread to finish (with timeout)
    #         if self._pose_update_thread and self._pose_update_thread.is_alive():
    #             self._pose_update_thread.join(timeout=2.0)
                
    #         self._is_updating_pose = False
    #         print("Stopped pose update.")
            
    # def _pose_update_worker(self) -> None:
    #     """
    #     Background worker function that periodically updates the pose.
    #     This runs in a separate thread.
    #     """
    #     print(f"Pose update thread started with update rate: {self._pose_update_rate_hz} Hz")
        
    #     # Calculate sleep time in seconds
    #     sleep_time = 1.0 / self._pose_update_rate_hz
        
    #     while not self._stop_pose_update_event.is_set():
    #         try:
    #             # Update the pose
    #             self.update_pose_from_kinematic_state()
                
    #             # Sleep for the calculated time, but check for stop event periodically
    #             # This allows for more responsive stopping
    #             self._stop_pose_update_event.wait(timeout=sleep_time)
    #         except Exception as e:
    #             print(f"Error in pose update thread: {e}")
    #             # Sleep a bit longer on error to avoid spamming
    #             time.sleep(1.0)
                
    #     print("Pose update thread stopped.")

    # def get_pose_update_status(self) -> Dict[str, Any]:
    #     """
    #     Get the current status of the pose update thread and its parameters.
        
    #     Returns:
    #         Dict[str, Any]: Dictionary containing pose update status information
    #     """
    #     with self._refresh_lock:
    #         status = {
    #             "is_updating_pose": self._is_updating_pose,
    #             "update_rate_hz": self._pose_update_rate_hz,
    #             "position": self.get_position() if self.odom_tform_body else None,
    #             "orientation_euler": self.get_orientation_euler() if self.odom_tform_body else None,
    #             "orientation_quaternion": self.get_orientation_quaternion() if self.odom_tform_body else None
    #         }
    #         return status

class EStopError(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)

class RobotStateSingleton(_SingletonWrapper):
    """
    Singleton for RobotState to allow for persistent storage and easy access.
    For more information on singleton see utils/singletons.py
    """

    _type_of_class = RobotState

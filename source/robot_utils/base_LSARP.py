"""
This code acts as the jump-off point for all later scripts.
It exists because there is a bunch of framework stuff we have to do before being able to control the robot
(including, but not limited to creating the sdk, syncing the robot, authenticating to the robot, verifying E-Stop, ...).
This code does all this framework code. For our custom code, you should create an object inheriting from
ControlFunction, in which the __call__() method defines the actual actions.
At the bottom of the script call take_control_with_function(f: ControlFunction, **kwargs).
"""

from __future__ import annotations

import time
from typing import Optional
import logging

from bosdyn import client as bosdyn_client
from bosdyn.api import estop_pb2
from bosdyn.api.spot import robot_command_pb2 as spot_command_pb2
from bosdyn.client import Sdk
from bosdyn.client import util as bosdyn_util
from bosdyn.client.estop import EstopClient
from bosdyn.client.robot_command import (
    RobotCommandClient,
    blocking_selfright,
    blocking_stand,
)
from bosdyn.client.image import ImageClient

from bosdyn.client.robot_state import RobotStateClient
from bosdyn.client import Sdk

from utils import environment
from robot_utils.basic_movements import move_body
from robot_utils.frame_transformer import FrameTransformerSingleton
from robot_utils.video import (
    get_d_pictures,
    get_greyscale_pictures,
    get_rgb_pictures,
    localize_from_images,
    set_gripper_camera_params
)

from utils.coordinates import Pose3D
from utils.logger import LoggerSingleton, TimedFileLogger
from utils.recursive_config import Config
from utils.singletons import (
    GraphNavClientSingleton,
    ImageClientSingleton,
    RobotCommandClientSingleton,
    RobotSingleton,
    RobotStateClientSingleton,
    RobotLeaseClientSingleton,
    WorldObjectClientSingleton,
    reset_singletons,
)
from source.planner_core.robot_state import RobotStateSingleton

frame_transformer = FrameTransformerSingleton()
graph_nav_client = GraphNavClientSingleton()
image_client = ImageClientSingleton()
robot_command_client = RobotCommandClientSingleton()
robot = RobotSingleton()
robot_state_client = RobotStateClientSingleton()
robot_lease_client = RobotLeaseClientSingleton()
world_object_client = WorldObjectClientSingleton()
logger = LoggerSingleton()

robot_state = RobotStateSingleton()


ALL_SINGLETONS = (
    frame_transformer,
    graph_nav_client,
    image_client,
    robot_command_client,
    robot,
    robot_state_client,
    robot_lease_client,
    world_object_client,
)

config = Config()

class ControlFunction(typing.Protocol):
    """
    This class defines all control functions. It gets as input all that you need for controlling the robot.
    :return: FrameTransformer, FrameName used for returning to origin
    """

    def __call__(
        self,
        config: Config,
        *args,
        **kwargs,
    ) -> str:
        pass


def take_control_with_function(
    config: Config,
    function: ControlFunction,
    *args,
    **kwargs,
):
    """
    Code wrapping all ControlFunctions, handles all kinds of framework (see description at beginning of file).
    :param config: config file for the robot
    :param function: ControlFunction specifying the actual actions
    :param args: other args for ControlFunction
    :param stand: whether to stand (and self-right if applicable) in beginning of script
    :param power_off: whether to power off after successful execution of actions
    :param body_assist: TODO: might not be useful at all
    :param return_to_start: whether to return to start at the end of execution
    :param kwargs: other keyword-args for ControlFunction
    """

    # Verify the estop
    verify_estop()

    # if stand:
    #     # Here, we want the robot so self-right, otherwise it cannot stand
    #     # robot.logger.info("Commanding robot to self-right.")
    #     blocking_selfright(robot_command_client)
    #     # robot.logger.info("Self-righted")

    #     # Tell the robot to stand up. The command service is used to issue commands to a robot.
    #     # The set of valid commands for a robot depends on hardware configuration. See
    #     # RobotCommandBuilder for more detailed examples on command building. The robot
    #     # command service requires timesync between the robot and the client.
    #     # robot.logger.info("Commanding robot to stand...")

    #     if body_assist:
    #         body_control = spot_command_pb2.BodyControlParams(
    #             body_assist_for_manipulation=spot_command_pb2.BodyControlParams.BodyAssistForManipulation(
    #                 enable_hip_height_assist=True, enable_body_yaw_assist=True
    #             )
    #         )
    #         params = spot_command_pb2.MobilityParams(body_control=body_control)
    #     else:
    #         params = None
    #     blocking_stand(robot_command_client, timeout_sec=10, params=params)
    #     # robot.logger.info("Robot standing.")
    #     time.sleep(3)

    # Execute the specific control function
    return_values = function(
        config,
        *args,
        **kwargs,
    )

def initialize_robot_connection():
    """ 1. Generate a Robot object
        2. Authentication and time sync
        3. Instantiates the robot state client, robot command client and the lease client
        4. Verifies the estop
    """
    spot_env_config = environment.get_environment_config(config, ["spot"])
    robot_config = config["robot_parameters"]
    sdk = bosdyn_client.create_standard_sdk("understanding-spot")

    # setup logging
    bosdyn_util.setup_logging(robot_config["verbose"])

    # setup robot
    global robot
    robot.set_instance(sdk.create_robot(spot_env_config["wifi_default_address"]))

    environment.set_robot_password(config)
    bosdyn_util.authenticate(robot)

    # Establish time sync with the robot. 
    robot.time_sync.wait_for_sync()

    # The robot state client will allow us to get the robot's state information, and construct
    # a command using frame information published by the robot.
    global robot_state_client
    robot_state_client.set_instance(robot.ensure_client(RobotStateClient.default_service_name))
    
    global robot_command_client
    robot_command_client.set_instance(
        robot.ensure_client(RobotCommandClient.default_service_name)
    )
    global lease_client
    robot_lease_client.set_instance(robot.ensure_client(
    bosdyn_client.lease.LeaseClient.default_service_name
    ))

    verify_estop()


def spot_initial_localization() -> None:
    """Initial localization of the spot robot in the frame (relative to the fiducial)
    using the camera images and depth scans."""
    
    #################################
    # localization of spot based on camera images and depth scans
    #################################
    start_time = time.time()
    set_gripper_camera_params('640x480')

    robot_state.frame_name = localize_from_images(config, vis_block=False) # localize from images instantiates the image client
    print("====================================")
    print(f"Frame name: {robot_state.frame_name}")
    print("====================================")
    end_time_localization = time.time()
    logging.info(f"Spot localization succesfull. Localization time: {end_time_localization - start_time}")


def power_on(): 
    """Power on the robot."""
    robot.power_on(timeout_sec=20)
    assert robot.is_powered_on(), "Robot power on failed."
    robot.logger.info("Robot powered on.")

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


def safe_power_off():
    """Sit and power off robot """
    robot.logger.info('Powering off robot...')
    robot.power_off(cut_immediately=False, timeout_sec=20)
    assert not robot.is_powered_on(), 'Robot power off failed.'
    robot.logger.info('Robot safely powered off.')


def verify_estop():
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


class EStopError(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)

        
def update_image_state(image_source: Optional[str] = None) -> None:
    """Updates the image that the planning framework has access to."""

    if image_source is None:
        image_source = robot_state.default_image_source
    
    if image_source in robot_state.hand_image_sources:
        robot_state.image_state = get_rgb_pictures(image_sources=[image_source], gripper_open=True)[0]
    else: 
        robot_state.image_state = get_greyscale_pictures(image_sources=[image_source], gripper_open=False)[0]


def update_depth_image_state(image_source: Optional[str] = None) -> None:
    if image_source is None:
        image_source = robot_state.default_image_source
    
    if image_source in robot_state.hand_image_sources:
        robot_state.depth_image_state = get_d_pictures(image_sources=[image_source], gripper_open=True)[0]
    else:
        robot_state.depth_image_state = get_d_pictures(image_sources=[image_source], gripper_open=False)[0]
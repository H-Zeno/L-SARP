# Standard library imports
import asyncio
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from dotenv import dotenv_values

# Third-party imports
from bosdyn import client as bosdyn_client
from semantic_kernel.contents.chat_history import ChatHistory
from semantic_kernel.agents import ChatHistoryAgentThread
from semantic_kernel.functions.kernel_arguments import KernelArguments

from robot_utils.base_LSARP import (
    initialize_robot_connection,
    spot_initial_localization,
    power_on,
    safe_power_off
)
from robot_utils.frame_transformer import FrameTransformerSingleton

from planner_core.robot_planner import RobotPlanner, RobotPlannerSingleton
from planner_core.robot_state import RobotState, RobotStateSingleton
from LostFound.src.scene_graph import get_scene_graph

# Local imports
from configs.scenes_and_plugins_config import Scene
from utils.agent_utils import invoke_agent
from utils.recursive_config import Config
from utils.singletons import RobotLeaseClientSingleton
from utils.logging_utils import setup_logging


# Initialize singletons
robot_state = RobotStateSingleton()
robot_planner = RobotPlannerSingleton()
robot_lease_client = RobotLeaseClientSingleton()
frame_transformer = FrameTransformerSingleton()

# Set up the configuration
config = Config()


env_variables = dotenv_values(".env_core_planner")
connection_string = env_variables.get("AZURE_APP_INSIGHTS_CONNECTION_STRING")

# Set up logging with OpenTelemetry integration if configured
enable_opentelemetry = config.get("logging_settings", {}).get("enable_opentelemetry", False)
service_name = config.get("logging_settings", {}).get("service_name", "L-SARP")

# Set up logging
_, logger_main = setup_logging(
    enable_opentelemetry=enable_opentelemetry,
    service_name=service_name,
    connection_string=connection_string
)
logger = logger_main

# Set debug level based on config
debug = config.get("robot_planner_settings", {}).get("debug", True)
if debug:
    logger.setLevel(logging.DEBUG)

async def main():
    """Main entry point for the robot control system.
    
    This function initializes the robot, loads scene data, and handles either
    online live instructions or offline predefined goals based on configuration.
    """
    active_scene_name = config["robot_planner_settings"]["active_scene"]
    active_scene = Scene[active_scene_name]
    if active_scene not in Scene:
        raise ValueError(
            f"Selected active scene '{active_scene}' (mentioned in config.yaml) "
            "not found in Scene enum"
        )
    logger.info("Loading robot planner configurations for scene: '%s'", active_scene.value)

    path_to_scene_data = Path(config["robot_planner_settings"]["path_to_scene_data"])
    
    if not path_to_scene_data.exists():
        logger.info("Scene data directory not found at %s", path_to_scene_data)
        logger.info("Creating it now...")
        path_to_scene_data.mkdir(parents=True, exist_ok=True)

    base_path = config.get_subpath("prescans")
    ending = config["pre_scanned_graphs"]["high_res"]
    scan_dir = os.path.join(base_path, ending)

    # Loading/computing the scene graph
    scene_graph_path = Path(path_to_scene_data / active_scene_name / "full_scene_graph.pkl")
    scene_graph_json_path = Path(path_to_scene_data / active_scene_name / "scene_graph.json")
    logger.info("Loading scene graph from %s. This may take a few seconds...", scan_dir)
    scene_graph = get_scene_graph(
        scan_dir,
        graph_save_path=scene_graph_path,
        drawers=False,
        light_switches=True,
        vis_block=False
    )
    scene_graph.save_as_json(scene_graph_json_path)

    # Create timestamp for the responses file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    responses = {}
    separator = "======================="

    ############################################################
    # Start the connection to the robot
    ############################################################
    use_robot = config["robot_planner_settings"]["use_with_robot"]
    
    if use_robot:
        initialize_robot_connection()

    # Define a context manager helper class for when we're not using the robot
    class DummyContextManager:
        def __enter__(self):
            return None
        def __exit__(self, exc_type, exc_val, exc_tb):
            pass
    
    # Choose the appropriate context manager based on whether we're using the robot
    context_manager = bosdyn_client.lease.LeaseKeepAlive(
        robot_lease_client, must_acquire=True, return_at_exit=True
    ) if use_robot else DummyContextManager()
    
    # Use the context manager
    with context_manager:
        
        if use_robot:
            power_on()
            spot_initial_localization()

        # Initialize the robot state with the scene graph
        robot_planner.set_instance(RobotPlanner(scene=active_scene))
        robot_state.set_instance(RobotState(scene_graph_object=scene_graph))

        ### Online Live Instruction ###
        if config["robot_planner_settings"]["task_instruction_mode"] == "online_live_instruction":
            initial_prompt = f"""
            Hey, I'm Spot, your intelligent robot assistant. Currently I am located in the {active_scene.value} scene. I am happy to help you with:
            - Navigating through the environment
            - Understanding the scene
            - Executing tasks and actions
            
            What goal would you like me to achieve?
            """
            # TODO: Implement online live instruction handling
        
        ### Offline Predefined Goals ###
        elif config["robot_planner_settings"]["task_instruction_mode"] == "offline_predefined_instruction":
            # Process existing predefined goals
            goals_path = Path("configs/goals.json")
            
            # Load the predefined goals
            if not goals_path.exists():
                raise FileNotFoundError(f"Goals file not found at {goals_path}")
            
            with open(goals_path, "r") as file:
                try:
                    goals = json.load(file)
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON format in goals file: {e}")
            
            # Process each goal
            for nr, goal in goals.items():
                goal_text = f"{nr}. {goal}"
                logger.info(separator)
                logger.info(goal_text)

                # Task execution prompt template
                task_completion_prompt_template = """
                It is your job to complete the following task: {task}
                
                This task is part of the following plan: {plan}

                The following tasks have already been completed: {tasks_completed}

                Here is the scene graph:
                {scene_graph}
                
                Here is the robot's current position:
                {robot_position}
                """

                # Set the goal and create the initial plan
                time_before = datetime.now()
                initial_plan, chain_of_thought = await robot_planner.create_task_plan_from_goal(goal)
                time_after = datetime.now()
                inference_time = time_after - time_before
                logger.info("Time taken for inference: %s", inference_time)

                # Record the results
                goal_response = {
                    'inference_time (seconds)': str(inference_time.seconds),
                    'zero_shot_plan': initial_plan,
                    'chain_of_thought': chain_of_thought
                }

                # Save the plan results
                plan_gen_save_path = Path(path_to_scene_data / active_scene_name / "initial_plans.json")
                plan_gen_save_path.parent.mkdir(parents=True, exist_ok=True)
                
                if not plan_gen_save_path.exists():
                    with open(plan_gen_save_path, 'w', encoding='utf-8') as file:
                        json.dump({}, file)

                with open(plan_gen_save_path, 'r') as file:
                    existing_data = json.load(file)
                    existing_goal_responses = existing_data.get(goal, {})
                    existing_goal_responses[robot_planner.task_planner_agent.service_id] = goal_response
                    existing_data[goal] = existing_goal_responses
                    new_data = json.dumps(existing_data, indent=2)

                with open(plan_gen_save_path, 'w') as file:
                    file.write(new_data)

                # Main Agentic Loop - Execute the plan
                execution_thread = None
                
                while True:
                    for task in robot_planner.plan["tasks"]:
                        robot_planner.task = task
                        logger.info("%s\nExecuting task: %s\n%s", separator, task, separator)
                        if use_robot:
                            logger.info("Current robot frame: %s", robot_state.frame_name)

                        # Format the task execution prompt
                        task_execution_prompt = task_completion_prompt_template.format(
                            task=task,
                            plan=robot_planner.plan,
                            tasks_completed=robot_planner.tasks_completed,
                            scene_graph=str(scene_graph.scene_graph_to_dict()),
                            robot_position="Not available" if not use_robot else str(frame_transformer.get_current_body_position_in_frame(robot_state.frame_name))
                        )
                        
                        # Execute the task using thread-based approach for better context management
                        task_completion_response, execution_thread = await invoke_agent(
                            agent=robot_planner.task_execution_agent,
                            thread=execution_thread,
                            input_text_message=task_execution_prompt,
                            input_image_message=robot_state.get_current_image_content()
                        )
                        
                        logger.info("Task completion response: %s", task_completion_response)

                        # Break out of the task execution loop when replanning
                        if robot_planner.replanned:
                            robot_planner.replanned = False
                            break

                        robot_planner.tasks_completed.append(task)
                
                        # Check if all tasks are completed
                        if len(robot_planner.tasks_completed) == len(robot_planner.plan["tasks"]):
                            # Goal completed
                            logger.info("Goal completed successfully!")
                            break
                    
                    # If we've completed all tasks or not replanning, break out of the main loop
                    if not robot_planner.replanned:
                        break

            logger.info("Finished processing Offline Predefined Goals")

        else:
            raise ValueError(
                f"Invalid task instruction mode: {config['robot_planner_settings']['task_instruction_mode']}"
            )

        if use_robot:
            safe_power_off()

if __name__ == "__main__":
    asyncio.run(main())

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

from robot_utils.base_LSARP import (
    initialize_robot_connection,
    spot_initial_localization,
    power_on,
    safe_power_off
)
from robot_utils.frame_transformer import FrameTransformerSingleton

from planner_core.robot_planner import RobotPlanner, RobotPlannerSingleton
from planner_core.robot_state import RobotState, RobotStateSingleton
from planner_core.agents import TaskPlannerAgent, TaskExecutionAgent, GoalCompletionCheckerAgent

from LostFound.src.scene_graph import get_scene_graph

from robot_plugins.replanning import ReplanningPlugin
from robot_plugins.goal_checker import TaskPlannerGoalChecker

# Local imports
from configs.scenes_and_plugins_config import Scene
from configs.agent_instruction_prompts import (
    TASK_EXECUTION_PROMPT_TEMPLATE,
)
from utils.agent_utils import invoke_agent
from utils.recursive_config import Config
from utils.singletons import RobotLeaseClientSingleton
from utils.logging_utils import setup_logging

from configs.goal_execution_log_models import (
    GoalExecutionLogs,
    TaskPlannerAgentLogs,
    TaskExecutionAgentLogs,
    GoalCompletionCheckerAgentLogs,
    TaskExecutionLogs,
)


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
setup_logging(
    enable_opentelemetry=enable_opentelemetry,
    service_name=service_name,
    connection_string=connection_string,
)
logger = logging.getLogger("main")

# Set debug level based on config
debug = config.get("robot_planner_settings", {}).get("debug", True)
if debug:
    logger.setLevel(logging.DEBUG)

# async def reduce_and_log_chat_history(chat_thread, thread_name):
#     """Helper function to reduce chat history and log the results."""
#     try:
#         initial_messages = await chat_thread.get_messages()
#         initial_count = len(initial_messages.messages)
#         logger.info(f"@ {thread_name} History count BEFORE reduction attempt: {initial_count}") # Log count before

#         is_reduced = await chat_thread.reduce()
#         chat_history = await chat_thread.get_messages()
#         final_count = len(chat_history.messages) # Get final count

#         if is_reduced:
#             # Use initial_count and final_count in the log message
#             logger.info(f"@ {thread_name} History reduced from {initial_count} to {final_count} messages.")
#             for msg in chat_history.messages:
#                 if msg.metadata and msg.metadata.get("__summary__"):
#                     logger.info(f"\t{thread_name} Summary: {msg.content}")
#                     break # Assuming only one summary message needs logging
#         else:
#              # Log that history wasn't reduced and the count remains final_count
#              logger.info(f"@ {thread_name} History not reduced. Count remains: {final_count}")

#         # Log final count using the variable
#         logger.info(f"@ {thread_name} Final Message Count: {final_count}\n") 
#     except AgentThreadOperationException:
#         logger.warning(f"Could not reduce chat history for {thread_name} as the thread is not active.")
#         # Optionally, still log the current message count if the thread object allows access
#         try:
#             chat_history = await chat_thread.get_messages()
#             # Use final_count variable here too if needed, or recalculate
#             final_count_except = len(chat_history.messages)
#             logger.info(f"@ {thread_name} Final Message Count (reduction skipped): {final_count_except}\n")
#         except Exception as e:
#             logger.warning(f"Could not retrieve messages for {thread_name} after failed reduction: {e}")

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
    origninal_scene_graph = get_scene_graph(
        scan_dir,
        graph_save_path=scene_graph_path,
        drawers=False,
        light_switches=True,
        vis_block=False
    )
    origninal_scene_graph.save_as_json(scene_graph_json_path)

    # Create timestamp for the responses file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
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
            
            # Initialize the execution logs directory
            execution_logs_dir = Path(path_to_scene_data / active_scene_name)
            execution_logs_dir.mkdir(parents=True, exist_ok=True)
            execution_logs_path = execution_logs_dir / f"execution_logs_{timestamp}.json"
            
            # Process each goal
            for nr, goal in goals.items():
                
                # Initialize a fresh robot planner instance for each goal
                robot_planner.set_instance(RobotPlanner(
                    task_planner_agent=TaskPlannerAgent(), 
                    task_execution_agent=TaskExecutionAgent(), 
                    goal_completion_checker_agent=GoalCompletionCheckerAgent(), 
                    scene=active_scene))
                
                # Initialize the robot state with the scene graph
                robot_state.set_instance(RobotState(scene_graph_object=origninal_scene_graph))
        
                goal_text = f"{nr}. {goal}"
                logger.info(separator)
                logger.info(goal_text)
                
                # Loop for task execution and potential replanning
                goal_start_time = datetime.now()
                
                # Set the goal and create the initial plan
                await robot_planner.create_task_plan_from_goal(goal)

                while True:
                    planned_tasks = robot_planner.plan["tasks"]
                    for task in planned_tasks:
                        
                        robot_planner.task = task
                        logger.info("%s\nExecuting task: %s\n%s", separator, task, separator)
                        
                        if use_robot:
                            logger.info("Current robot frame: %s", robot_state.frame_name)

                        # Format the task execution prompt
                        task_execution_prompt = TASK_EXECUTION_PROMPT_TEMPLATE.format(
                            task=task,
                            plan=robot_planner.plan,
                            tasks_completed=robot_planner.tasks_completed,
                            scene_graph=str(robot_state.scene_graph.scene_graph_to_dict()),
                            robot_position="Not available" if not use_robot else str(frame_transformer.get_current_body_position_in_frame(robot_state.frame_name))
                        )
                        
                        # Execute the task using thread-based approach for better context management
                        task_completion_response, robot_planner.task_execution_chat_thread, agent_response_logs = await invoke_agent(
                            agent=robot_planner.task_execution_agent,
                            thread=robot_planner.task_execution_chat_thread,
                            input_text_message=task_execution_prompt,
                            input_image_message=robot_state.get_current_image_content()
                        )
                        
                        robot_planner.task_execution_logs.append(
                            TaskExecutionLogs(
                                task_description=task.get("task_description"),
                                reasoning=task.get("reasoning", ""),
                                plan_id=robot_planner.replanning_count,
                                agent_invocation=agent_response_logs,
                                relevant_objects_identified_by_planner=[obj.get('sem_label', str(obj)) + ' (object id: ' + str(obj.get('object_id', str(obj))) + ')' for obj in task.get("relevant_objects", [])]
                            ))

                        # Break out of the task execution loop when replanning
                        if robot_planner.replanned:
                            logger.info("The task planner decided to replan. Breaking out of the task execution loop.")
                            robot_planner.replanned = False
                            # When replanned, the task is not completed
                            break
                        
                        # Now the completion of a task is seen as completing one task execution agent invocation
                        # log the completion of the task (since no replanning took place)
                        robot_planner.task_execution_logs[-1].completed = True
                        robot_planner.task_execution_logs[-1].agent_invocation.agent_invocation_end_time = datetime.now()
                        robot_planner.tasks_completed.append(task.get("task_description"))
                        
                        # Reset the thread for the next task to ensure clean context
                        robot_planner.task_execution_chat_thread = None
                        
                    # Check if the goal is completed, this will set the robot_planner.goal_completed flag
                    goal_completion_response = await TaskPlannerGoalChecker().check_if_goal_is_completed(explanation="All planned tasks seem to have been completed.")
                    
                    if robot_planner.goal_completed:
                        logger.info("Goal completed successfully!")
                        break
                    
                    else:
                        logger.info("Goal is not completed yet. Replanning...")
                        await ReplanningPlugin().update_task_plan(goal_completion_response)
                        
                # Goal Completed, save logging details.
                goal_end_time = datetime.now()
                goal_duration = (goal_end_time - goal_start_time).total_seconds()
                
                # Task Planner Agent Logs
                task_planner_agent_logs = TaskPlannerAgentLogs(
                    ai_service_id=robot_planner.task_planner_agent.service_id,
                    initial_plan=robot_planner.initial_plan_log,
                    updated_plans=robot_planner.plan_generation_logs,
                    total_replanning_count=robot_planner.replanning_count,
                    task_planner_invocations=robot_planner.task_planner_invocations
                )
                
                # Task Execution Agent Logs
                task_execution_agent_logs = TaskExecutionAgentLogs(
                    ai_service_id=robot_planner.task_execution_agent.service_id,
                    task_logs=robot_planner.task_execution_logs
                )
                
                # Goal Completion Checker Agent Logs
                goal_completion_checker_agent_logs = GoalCompletionCheckerAgentLogs(
                    ai_service_id=robot_planner.goal_completion_checker_agent.service_id,
                    completion_check_logs=robot_planner.goal_completion_checker_logs
                )
                
                # Goal Execution Log
                goal_execution_log = GoalExecutionLogs(
                    goal=goal,
                    goal_number=nr,
                    start_time=goal_start_time,
                    end_time=goal_end_time,
                    duration_seconds=goal_duration,
                    task_planner_agent=task_planner_agent_logs,
                    task_execution_agent=task_execution_agent_logs,
                    goal_completion_checker_agent=goal_completion_checker_agent_logs
                )
                
                # Save the execution logs after each goal
                if execution_logs_path.exists():
                    with open(execution_logs_path, 'r') as file:
                        existing_execution_logs = json.load(file)
                else:
                    existing_execution_logs = {}
                        
                existing_execution_logs[nr] = json.loads(goal_execution_log.model_dump_json())
                
                with open(execution_logs_path, 'w', encoding='utf-8') as file:
                    json.dump(existing_execution_logs, file, indent=2)
                

            logger.info("Finished processing Offline Predefined Goals")
            

        else:
            raise ValueError(
                f"Invalid task instruction mode: {config['robot_planner_settings']['task_instruction_mode']}"
            )

        if use_robot:
            safe_power_off()


if __name__ == "__main__":
    asyncio.run(main())

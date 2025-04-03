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
from semantic_kernel import Kernel
from semantic_kernel.agents import ChatHistoryAgentThread
from semantic_kernel.contents import ChatHistory, ChatMessageContent, AuthorRole
from semantic_kernel.exceptions import AgentThreadOperationException
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion

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
    HISTORY_SUMMARY_REDUCER_INSTRUCTIONS
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

summary_kernel = Kernel()
summary_kernel.add_service(OpenAIChatCompletion(
    service_id="small_cheap_model",
    api_key=dotenv_values(".env_core_planner").get("OPENAI_API_KEY"),
    ai_model_id="gpt-4o-mini"
))


# will only reduce every (threshold - untouched_messages - 1) messages
async def reduce_and_log_chat_history(chat_thread, thread_name, threshold=10, untouched_messages=3):
    """Helper function to reduce chat history and log the results."""
    try:
        initial_messages = await chat_thread.get_messages()
        initial_count = len(initial_messages.messages)
        
        if initial_count <= threshold:
            logger.info(f"@ {thread_name} History count count is below threshold of {threshold}: {initial_count}")
            return
        
        logger.info("History count above threshold, attempting to reduce...")
        logger.info(f"@ {thread_name} History count BEFORE reduction attempt: {initial_count}") # Log count before
        # logger.info(f"@ {thread_name} History (Before): {initial_messages.messages}")
        
        # Summarize all messages except the last 'untouched_messages'
        prompt = HISTORY_SUMMARY_REDUCER_INSTRUCTIONS.format(chat_history=initial_messages.messages[:-untouched_messages])
        summary_result = await summary_kernel.invoke_prompt(prompt) # Added await
        summary_content = str(summary_result) # Extract string content from the result
        logger.info(f"@ {thread_name} Summary: {summary_content}")
  
        # Create the new chat history: summary + newest untouched messages
        chat_history = ChatHistory()
        chat_history.add_message(ChatMessageContent(role=AuthorRole.USER, content=summary_content)) # Add summary as system message
        
        for msg in initial_messages.messages[-untouched_messages:]:
            chat_history.add_message(msg)
        
        # Restore logging for reduction status
        final_count = len(chat_history.messages)

        logger.info(f"@ {thread_name} Final Message Count AFTER reduction: {final_count}")
        
        chat_thread._chat_history = chat_history
        
    except AgentThreadOperationException:
        logger.warning(f"Could not reduce chat history for {thread_name} as the thread is not active.")
        # Optionally, still log the current message count if the thread object allows access
        try:
            chat_history = await chat_thread.get_messages()
            # Use final_count variable here too if needed, or recalculate
            final_count_except = len(chat_history.messages)
            logger.info(f"@ {thread_name} Final Message Count (reduction skipped): {final_count_except}\n")
        except Exception as e:
            logger.warning(f"Could not retrieve messages for {thread_name} after failed reduction: {e}")

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
        drawers=True,
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
            goals_path = Path(config["robot_planner_settings"]["goals_path"])
            dataset_name = goals_path.stem
                
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
            execution_logs_path = execution_logs_dir / f"execution_logs_{dataset_name}_{timestamp}.json"
            
            # Process each goal
            for nr, goal_dict in goals.items():
                goal = goal_dict["goal"]
                complexity = goal_dict["complexity"]
                
                try: 

                    # Load the goal/query text
                    goal_text = f"{nr}. {goal}"
                    logger.info(separator)
                    logger.info(goal_text)
                    
                    # Loop for task execution and potential replanning
                    goal_start_time = datetime.now()
                    
                    # Reset the robot state
                    robot_state = RobotStateSingleton()
                    robot_state.set_instance(RobotState(scene_graph_object=origninal_scene_graph))
            
                    # Reset the robot planner
                    robot_planner = RobotPlannerSingleton()
                    robot_planner.set_instance(RobotPlanner(
                        task_planner_agent=TaskPlannerAgent(), 
                        task_execution_agent=TaskExecutionAgent(), 
                        goal_completion_checker_agent=GoalCompletionCheckerAgent(), 
                        scene=active_scene))
                    await robot_planner.create_task_plan_from_goal(goal)
                    
                    # # print all the attributes of the robot_planner
                    # for attr in dir(robot_planner):
                    #     logger.info(f"{attr}: {getattr(robot_planner, attr)}")

                    if robot_planner.goal_completed:
                        raise ValueError("Goal marked as completed before starting to solve it!")
                    
                    # Begin of while loop: solving one specific goal
                    while True:
                        if robot_planner.replanning_count > robot_planner.max_replanning_count:
                            logger.info("Replanning count exceeded max_replanning_count. Breaking out of the while loop.")
                            robot_planner.goal_failed_max_tries = True
                            break
                        
                        # Reset the replanning flag
                        robot_planner.replanned = False
                        
                        # Get the planned tasks
                        planned_tasks = robot_planner.plan["tasks"]
                        
                        # Execute each task
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
                                robot_position=str(robot_state.virtual_robot_pose) if not use_robot else str(frame_transformer.get_current_body_position_in_frame(robot_state.frame_name)),
                                core_memory=str(robot_state.core_memory)
                            )
                            
                            # Execute the task using thread-based approach for better context management
                            task_completion_response, robot_planner.task_execution_chat_thread, agent_response_logs = await invoke_agent(
                                agent=robot_planner.task_execution_agent,
                                thread=robot_planner.task_execution_chat_thread,
                                input_text_message=task_execution_prompt,
                                input_image_message=robot_state.get_current_image_content()
                            )
                            
                            if task.get("relevant_objects") is not None:
                                relevant_objects_identified_by_planner = [obj.get('sem_label', str(obj)) + ' (object id: ' + str(obj.get('object_id', str(obj))) + ')' for obj in task.get("relevant_objects", [])]
                            else:
                                relevant_objects_identified_by_planner = None
                            
                            # Log the task execution
                            robot_planner.task_execution_logs.append(
                                TaskExecutionLogs(
                                    task_description=task.get("task_description"),
                                    reasoning=task.get("reasoning", ""),
                                    plan_id=robot_planner.replanning_count,
                                    agent_invocation=agent_response_logs,
                                    relevant_objects_identified_by_planner=relevant_objects_identified_by_planner
                                ))
                            
                            if not robot_planner.replanned:
                                # Now the completion of a task is seen as completing one task execution agent invocation
                                robot_planner.task_execution_logs[-1].completed = True
                                robot_planner.task_execution_logs[-1].agent_invocation.agent_invocation_end_time = datetime.now()
                                robot_planner.tasks_completed.append(task.get("task_description"))
                                    
                                # Reduce the chat history
                                await reduce_and_log_chat_history(robot_planner.task_execution_chat_thread, "Task Execution Agent")
                                
                                # Check if the goal is completed
                                if robot_planner.goal_completed:
                                    logger.info("Goal completed successfully!")
                                    break

                            else:
                                # Break out of the task execution loop when a replanning got invoked during the task executor's invocation
                                # When replanned, the task is not completed
                                logger.info("The task planner decided to replan. Breaking out of the task execution loop.")
                                break
                            
                        # Check if the goal is completed, this will set the robot_planner.goal_completed flag
                        if not robot_planner.replanned and not robot_planner.goal_failed_max_tries:
                            
                            if robot_planner.goal_completed:
                                logger.info("Goal completed, marked by the goal checker invoked by the task execution agent.")
                                break
                            
                            else: 
                                # Check if the goal is completed after all planned tasks have been completed
                                goal_completion_response = await TaskPlannerGoalChecker().check_if_goal_is_completed(explanation="All planned tasks seem to have been completed.")
                                logger.info("Goal completion check after completing all planned tasks - goal_completed: %s", robot_planner.goal_completed)
                                
                                if robot_planner.goal_completed:
                                    # Goal is completed, we have to break out of the while loop
                                    logger.info("Goal completed successfully!")
                                    break
                                
                                else:
                                    logger.info("Goal is not completed yet. Replanning...")
                                    await ReplanningPlugin().update_task_plan(goal_completion_response)
        
                    # End of while loop
                            
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
                        goal_completed=robot_planner.goal_completed,
                        goal_failed_max_tries=robot_planner.goal_failed_max_tries,
                        complexity=complexity,
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

                
                except Exception as e:
                    logger.error("Error processing goal %s: %s", nr, e)
                
                    # Save the error log
                    if execution_logs_path.exists():
                        with open(execution_logs_path, 'r') as file:
                            existing_execution_logs = json.load(file)
                    else:
                        existing_execution_logs = {}
                            
                    existing_execution_logs[nr] = {
                        "error": str(e)
                    }
                    
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

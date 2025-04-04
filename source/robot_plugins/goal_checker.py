import logging
import time
from datetime import datetime
from typing import Annotated, Dict, Tuple, List

from semantic_kernel.agents import ChatHistoryAgentThread
from semantic_kernel.functions.kernel_function_decorator import kernel_function
from semantic_kernel.contents.utils.author_role import AuthorRole
from semantic_kernel.contents import ChatMessageContent
from configs.agent_instruction_prompts import (
    TASK_PLANNER_AGENT_GOAL_CHECK_PROMPT_TEMPLATE,
    TASK_EXECUTION_AGENT_GOAL_CHECK_PROMPT_TEMPLATE
)
from configs.goal_execution_log_models import (
    GoalCompletionCheckerLogs,
)
from planner_core.robot_planner import RobotPlannerSingleton
from planner_core.robot_state import RobotStateSingleton
from robot_utils.frame_transformer import FrameTransformerSingleton
from utils.agent_utils import invoke_agent
from utils.recursive_config import Config

# Get singleton instances
robot_state = RobotStateSingleton()
robot_planner = RobotPlannerSingleton()
frame_transformer = FrameTransformerSingleton()

# Set up config
config = Config()
use_robot = config.get("robot_planner_settings", {}).get("use_with_robot", False)

# Set up logger
logger = logging.getLogger("plugins")

termination_keyword = config.get("robot_planner_settings", {}).get("termination_keyword", "COMPLETED")

class TaskExecutionGoalChecker:
    """This plugin should only be called when the task execution agent thinks that the goal is completed."""
    
    @kernel_function(description="Function to call when you (the task execution agent) think that by completing the current task, you have actually completed the overall goal.")
    async def check_if_goal_is_completed(self, explanation: Annotated[str, "A detailed explanation of why you think the goal is completed."]) -> str:
        """Check if the goal is completed."""
        
        # Explicitly reset the flag at the start of each check
        # robot_planner.goal_completed = False
        # logger.info("Goal flag explicitly reset to False at start of TaskExecutionGoalChecker check")
        
        check_if_goal_is_completed_prompt = TASK_EXECUTION_AGENT_GOAL_CHECK_PROMPT_TEMPLATE.format(
            task=robot_planner.task,
            goal=robot_planner.goal,
            explanation=explanation,
            plan=robot_planner.plan,
            tasks_completed=robot_planner.tasks_completed,
            scene_graph=str(robot_state.scene_graph.scene_graph_to_dict()),
            robot_position=str(robot_state.virtual_robot_pose) if not use_robot else str(frame_transformer.get_current_body_position_in_frame(robot_state.frame_name)),
            core_memory=str(robot_state.core_memory)
        )
        
        logger.debug("========================================")
        logger.debug(f"Goal checker prompt (task execution): {check_if_goal_is_completed_prompt}")
        logger.debug("========================================")

        response, robot_planner.planning_chat_thread, agent_response_logs = await invoke_agent(
            agent=robot_planner.goal_completion_checker_agent,
            thread=robot_planner.planning_chat_thread,
            input_text_message=check_if_goal_is_completed_prompt,
            input_image_message=robot_state.get_current_image_content()
        )
        agent_response_logs.plan_id = robot_planner.replanning_count
        logger.info("Task execution goal checker response: %s", response)
        
        # We save the goal check in the task execution chat history
        await robot_planner.task_execution_chat_thread.on_new_message(ChatMessageContent(role=AuthorRole.ASSISTANT, content="The Goal Checker has completed its analysis and here is its response to your query: " + str(response)))
        
        # Check for termination keyword and set flag if found
        has_termination_keyword = termination_keyword.lower() in str(response).lower()
        logger.info("Termination keyword '%s' found in response: %s", termination_keyword, has_termination_keyword)
        
        if has_termination_keyword:
            robot_planner.goal_completed = True
            logger.info("Goal completed flag set to True by TaskExecutionGoalChecker")
        
        robot_planner.goal_completion_checker_logs.append(
            GoalCompletionCheckerLogs(
                completion_check_requested_by_agent="TaskExecutionAgent",
                completion_check_request=explanation,
                completion_check_agent_invocation=agent_response_logs,
                completion_check_final_response=str(response)
            )
        )
        
        return str(response)

class TaskPlannerGoalChecker:
    """This plugin should only be called when the task planner thinks that the goal is completed."""
    
    @kernel_function(description="Function to call when the task planner thinks that the goal is completed.")
    async def check_if_goal_is_completed(self, explanation: Annotated[str, "A detailed explanation of why the task planner thinks the goal is completed."]) -> str:
        """Check if the goal is completed."""
        
        # Explicitly reset the flag at the start of each check
        # robot_planner.goal_completed = False
        # logger.info("Goal flag explicitly reset to False at start of TaskPlannerGoalChecker check")
        
        check_if_goal_is_completed_prompt = TASK_PLANNER_AGENT_GOAL_CHECK_PROMPT_TEMPLATE.format(
            goal=robot_planner.goal,
            plan=robot_planner.plan,
            tasks_completed=robot_planner.tasks_completed,
            explanation=explanation,
            scene_graph=str(robot_state.scene_graph.scene_graph_to_dict()),
            robot_position=str(robot_state.virtual_robot_pose) if not use_robot else str(frame_transformer.get_current_body_position_in_frame(robot_state.frame_name)),
            core_memory=str(robot_state.core_memory)
        )
        logger.debug("========================================")
        logger.debug(f"Goal checker prompt (task planner): {check_if_goal_is_completed_prompt}")
        logger.debug("========================================")
        
        response, robot_planner.planning_chat_thread, agent_response_logs = await invoke_agent(
            agent=robot_planner.goal_completion_checker_agent,
            thread=robot_planner.planning_chat_thread,
            input_text_message=check_if_goal_is_completed_prompt,
            input_image_message=robot_state.get_current_image_content()
        )
        
        agent_response_logs.plan_id = robot_planner.replanning_count
        logger.info("Task planner goal completion checker response: %s", response)
        
        # Check for termination keyword and set flag if found
        has_termination_keyword = termination_keyword.lower() in str(response).lower()
        logger.info("Termination keyword '%s' found in response: %s", termination_keyword, has_termination_keyword)
        
        if has_termination_keyword:
            robot_planner.goal_completed = True
            logger.info("Goal completed flag set to True by TaskPlannerGoalChecker")
        
        robot_planner.goal_completion_checker_logs.append(
            GoalCompletionCheckerLogs(
                completion_check_requested_by_agent="TaskPlannerAgent",
                completion_check_request=explanation,
                completion_check_agent_invocation=agent_response_logs,
                completion_check_final_response=str(response)
            )
        )
        
        return str(response)


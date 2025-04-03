import logging
from typing import Annotated

from semantic_kernel.functions.kernel_function_decorator import kernel_function
from semantic_kernel.contents import ChatMessageContent
from semantic_kernel.contents.utils.author_role import AuthorRole
from utils.agent_utils import invoke_agent
from planner_core.robot_state import RobotStateSingleton
from robot_utils.frame_transformer import FrameTransformerSingleton
from utils.recursive_config import Config
from planner_core.robot_planner import RobotPlannerSingleton

# Get singleton instances
robot_state = RobotStateSingleton()
robot_planner = RobotPlannerSingleton()
frame_transformer = FrameTransformerSingleton()

config = Config()
use_robot = config.get("robot_planner_settings", {}).get("use_with_robot", False)

# Set up logger
logger = logging.getLogger("plugins")


class TaskPlannerCommunicationPlugin:
    """A plugin to communicate with the task planner.
    
    Consult the task planner when:
    1. Something (unexpected) goes wrong/different to the plan and a re-plan is needed.
    2. Instructions are not clear and you need to ask the task planner for clarification.
    3. Only for the task execution agent: By completing a task, you think that the overall goal is now actually completed (no more action needed).
    """
    
    @kernel_function(description="Function to call when you need to communicate with the task planner.")
    async def communicate_with_task_planner(self, request: Annotated[str, "A detailed explanation of the current situation and what you need from the task planner."]) -> Annotated[str, "The response from the task planner."]:
        """Communicate with the task planner."""
        
    
        # In invoking the task planner agent, it is possible that it will invoke a replanning. 
        response, robot_planner.planning_chat_thread, agent_response_logs = await invoke_agent(
            agent=robot_planner.task_planner_agent, 
            thread=robot_planner.planning_chat_thread,
            input_text_message=request, 
            input_image_message=robot_state.get_current_image_content()
        )
        
        agent_response_logs.plan_id = robot_planner.replanning_count
        robot_planner.task_planner_invocations.append(agent_response_logs)
        
        # Add to the task execution chat history
        await robot_planner.task_execution_chat_thread.on_new_message(ChatMessageContent(role=AuthorRole.USER, content=request))
        await robot_planner.task_execution_chat_thread.on_new_message(ChatMessageContent(role=AuthorRole.ASSISTANT, content=str(response)))
        
        logger.info("========================================")
        logger.info(f"Task planner response to communication request: {response}")
        logger.info("========================================")
        
        return None
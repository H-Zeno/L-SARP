# Standard library imports
import json
import logging
import re
import sys
from typing import Annotated, Dict, Tuple

# Third-party imports
from dotenv import dotenv_values
from langchain.output_parsers import PydanticOutputParser
from semantic_kernel.contents import ChatHistory
from semantic_kernel.functions import kernel_function
from semantic_kernel.contents.utils.author_role import AuthorRole

# Local imports
from configs.agent_instruction_prompts import (
    CREATE_TASK_PLANNER_PROMPT_TEMPLATE
)

from configs.scenes_and_plugins_config import Scene
from planner_core.robot_state import RobotStateSingleton
from utils.agent_utils import invoke_agent
from robot_utils.frame_transformer import FrameTransformerSingleton
from utils.recursive_config import Config
from utils.singletons import _SingletonWrapper

from planner_core.json_object_models import TaskPlannerResponse

# Initialize robot state singleton
robot_state = RobotStateSingleton()
frame_transformer = FrameTransformerSingleton()

config = Config()
dotenv_values(".env_core_planner")
use_robot = config.get("robot_planner_settings", {}).get("use_with_robot", False)
debug = config.get("robot_planner_settings", {}).get("debug", True)

# Just get the logger, configuration is handled in main.py
logger = logging.getLogger("main")

class RobotPlanner:
    """
    Plugin that handles the planning of the robot tasks based on a specific goal or query that is given by the user.
    """
    
    def __init__(self, task_planner_agent, task_execution_agent, goal_completion_checker_agent, scene: Scene = None,) -> None:
        """
        Constructor for the RobotPlanner class that handles plugin initialization and planning.
        """
        # Set the configurations for the scene
        self.scene = scene
        
        # Initialize agent references to None - they will be loaded lazily when needed
        self.task_planner_agent = task_planner_agent
        self.task_execution_agent = task_execution_agent
        self.goal_completion_checker_agent = goal_completion_checker_agent

        # Planner states
        self.goal = None
        self.goal_completed = False
        self.plan = None
        self.replanned = False
        
        self.planning_chat_history = ChatHistory()
        self.task_execution_chat_history = ChatHistory()
        self.task = None
        self.tasks_completed = []
        self.actions_taken = []

        # # Task planning variables
        self.json_format_agent_thread = None
        
    async def _create_task_plan(self, additional_message: Annotated[str, "Additional message to add to the task generation prompt"] = "") -> str:
        """Create a task plan based on the current goal and robot state."""
        parser = PydanticOutputParser(pydantic_object=TaskPlannerResponse)
        model_desc = parser.get_format_instructions()
        
        plan_generation_prompt = CREATE_TASK_PLANNER_PROMPT_TEMPLATE.format(
            goal=self.goal, 
            model_description=model_desc,
            scene_graph=str(robot_state.scene_graph.scene_graph_to_dict()), 
            robot_position="Not available" if not use_robot else str(frame_transformer.get_current_body_position_in_frame(robot_state.frame_name))
        )

        logger.info("========================================")
        logger.info(f"Plan generation prompt: {plan_generation_prompt}")
        logger.info("========================================")

        plan_response, self.json_format_agent_thread = await invoke_agent(
            agent=self.task_planner_agent, 
            thread=self.json_format_agent_thread,
            input_text_message=additional_message + plan_generation_prompt, 
            input_image_message=robot_state.get_current_image_content()
        )
        
        logger.debug("========================================")
        logger.debug(f"Initial plan full response: {str(plan_response)}")
        logger.debug("========================================")

        # Convert ChatMessageContent to string
        plan_response_str = str(plan_response)

        # Define the pattern to extract everything before ```json
        pattern_before_json = r"(.*?)```json"
        match_before_json = re.search(pattern_before_json, plan_response_str, re.DOTALL)

        # Extract and assign to reasoning variable
        chain_of_thought = match_before_json.group(1).strip() if match_before_json else ""

        pattern = r"```json\s*(.*?)\s*```"
        match = re.search(pattern, plan_response_str, re.DOTALL)
        if match:
            json_content_inside = match.group(1)
            plan_json_str = str(json_content_inside).replace('```json', '').replace('```', '').strip()
        else:
            logger.info('No ```json``` block found in response. Using the whole response as JSON.')
            plan_json_str = str(plan_response_str).replace('```json', '').replace('```', '').strip()

        try:
            logger.debug("========================================")
            logger.debug(f"Plan JSON string: {plan_json_str}")   
            logger.debug("========================================")
            self.plan = json.loads(plan_json_str)
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from response: {e}")
            await self._create_task_plan(additional_message="Failed to parse JSON from response with error: " + str(e) + ". Please try again.")
        
        self.planning_chat_history.add_message({
            "role": AuthorRole.USER,
            "content": "Initial plan:" + str(self.plan)
        })
        logger.info(f"Initial plan: {json.dumps(self.plan, indent=2)}")
        # Log the reasoning content
        logger.info(f"Chain of thought of initial plan (in case of reasoning model): {chain_of_thought}")

        self.json_format_agent_thread = None # Reset the chat history
        return chain_of_thought


    async def create_task_plan_from_goal(self, goal: Annotated[str, "The goal to be achieved by the robot"]) -> Tuple[Dict, str]:
        """
        Sets the goal for the robot planner and creates an initial task plan.
        """
        self.goal = goal
        self.goal_completed = False
        self.plan = None
        self.replanned = False
        self.planning_chat_history = ChatHistory()
        self.task_execution_chat_history = ChatHistory()
        self.tasks_completed = []
        self.task = None
        self.actions_taken = []

        # # Task planning variables
        self.json_format_agent_thread = None
        
        # Create initial plan
        chain_of_thought = await self._create_task_plan()
        
        logger.info(f"Goal set to: {self.goal}. Initial plan created.")

        return self.plan, chain_of_thought

    
   
class RobotPlannerSingleton(_SingletonWrapper):
    """Singleton wrapper for the RobotPlanner class."""
    _type_of_class = RobotPlanner
    
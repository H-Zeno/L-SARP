# Standard library imports
import asyncio
from datetime import datetime
import json
import logging
import re
import sys
from abc import ABC
from collections.abc import AsyncIterable
from openai import AsyncOpenAI
from utils.recursive_config import Config
from typing import Annotated, Dict, List, Tuple, Any, ClassVar, TYPE_CHECKING, Union, Optional

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

# Third-party imports
from dotenv import dotenv_values
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from semantic_kernel import Kernel
from semantic_kernel.agents import AgentGroupChat, ChatCompletionAgent, ChatHistoryAgentThread
from semantic_kernel.agents.strategies import (
    KernelFunctionSelectionStrategy,
    KernelFunctionTerminationStrategy,
)
from semantic_kernel.agents.strategies.termination.termination_strategy import TerminationStrategy
from semantic_kernel.connectors.ai.function_choice_behavior import FunctionChoiceBehavior
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion, OpenAIChatPromptExecutionSettings, OpenAIPromptExecutionSettings
from semantic_kernel.contents import ChatHistory, ChatMessageContent
from semantic_kernel.functions import KernelArguments, KernelFunctionFromPrompt, kernel_function
from semantic_kernel.services.ai_service_client_base import AIServiceClientBase


# Local imports
from configs.agent_instruction_prompts import (
    GOAL_COMPLETION_CHECKER_AGENT_INSTRUCTIONS,
    TASK_EXECUTION_AGENT_INSTRUCTIONS,
    TASK_PLANNER_AGENT_INSTRUCTIONS,
    CREATE_TASK_PLANNER_PROMPT_TEMPLATE,
    UPDATE_TASK_PLANNER_PROMPT_TEMPLATE,
    CHECK_IF_GOAL_IS_COMPLETED_PROMPT_TEMPLATE,
)

# from configs.plugin_configs import plugin_configs
from configs.scenes_and_plugins_config import Scene
from planner_core.robot_state import RobotStateSingleton
from robot_plugins.item_interactions import ItemInteractionsPlugin
from robot_plugins.navigation import NavigationPlugin
from robot_plugins.inspection import InspectionPlugin
from utils.agent_utils import invoke_agent, invoke_agent_group_chat
from robot_utils.frame_transformer import FrameTransformerSingleton

from utils.singletons import _SingletonWrapper

# Initialize robot state singleton
robot_state = RobotStateSingleton()
frame_transformer = FrameTransformerSingleton()

config = Config()
dotenv_values(".env_core_planner")
use_robot = config.get("robot_planner_settings", {}).get("use_with_robot", False)
debug = config.get("robot_planner_settings", {}).get("debug", True)

# Just get the logger, configuration is handled in main.py
logger = logging.getLogger("main")

class SceneGraphObject(BaseModel):
    """A node in the scene graph."""
    object_id: int
    sem_label: str
    centroid: List[float]
    movable: bool

class TaskResponse(BaseModel):
    """A task to be completed."""
    task_description: str = Field(description="A clear description of the task to be completed.")
    reasoning: str = Field(description="A concise reasoning behind the task, especially answering the 'why?' question.")
    function_calls_involved: List[str] = Field(description="A list of function calls involved in completing the task, including their arguments.")
    relevant_objects: List[SceneGraphObject] = Field(description="A list of relevant objects from the scene graph that the robot could interact with to complete the task.")

class TaskPlannerResponse(BaseModel):
    """A response from the task planner agent."""
    tasks : List[TaskResponse]


class RobotAgentBase(ChatCompletionAgent, ABC):
    """Base class for all robot agents."""
    service_id: ClassVar[str] = "gpt4o"
    
    def _add_retrieval_plugins(self, kernel: Kernel) -> Kernel:
        """
        Adds all the enabled plugins to the kernel.
        Scenes_and_plugins_config.py contains the plugin configurations for each scene.
        """
        # Add Enabled Plugins to the kernel
        for plugin_name in self._enabled_retrieval_plugins:
            if plugin_name in self._retrieval_plugins_configs:
                factory_func, args, kernel_name = self._retrieval_plugins_configs[plugin_name]
                plugin = factory_func(*args)
                kernel.add_plugin(plugin, plugin_name=kernel_name)
        return kernel

    def _add_action_plugins(self, kernel: Kernel) -> Kernel:
        """
        Adds all the action plugins to the kernel
        """
        kernel.add_plugin(ItemInteractionsPlugin(), plugin_name="item_interactions")
        kernel.add_plugin(NavigationPlugin(), plugin_name="navigation")
        kernel.add_plugin(InspectionPlugin(), plugin_name="object_inspection")
        
        return kernel

    def _add_task_planner_communication_plugins(self, kernel: Kernel) -> Kernel:
        """
        Adds all the task planner communication plugins to the kernel
        """
        # Register the task planning methods from this class as a plugin
        kernel.add_plugin(self, plugin_name="task_planner_communication")

        return kernel
    
    def _create_kernel(self, action_plugins=False, retrieval_plugins=False, task_planner_communication=False) -> Kernel:
        """Create and configure a kernel with all the AI services that we support."""
        logger.info(f"Creating kernel with service ID: {self.service_id}")
        kernel = Kernel()
        
        if self.service_id == "gpt4o":
            # General Multimodal Intelligence model (GPT4o)
            kernel.add_service(OpenAIChatCompletion(
                service_id=self.service_id,
                api_key=dotenv_values(".env_core_planner").get("OPENAI_API_KEY"),
                ai_model_id="gpt-4o-2024-11-20"
            ))

        elif self.service_id == "o3-mini":
            # Reasoning models
            kernel.add_service(OpenAIChatCompletion(
                service_id="o3-mini",
                api_key=dotenv_values(".env_core_planner").get("OPENAI_API_KEY"),
                ai_model_id="o3-mini-2025-01-31"
            ))
        
        elif self.service_id == "o1":
            # Reasoning models
            kernel.add_service(OpenAIChatCompletion( 
                service_id="o1",
                api_key=dotenv_values(".env_core_planner").get("OPENAI_API_KEY"),
                ai_model_id="o1-2024-12-17"
            ))
        
        elif self.service_id == "deepseek-r1":
            # Reasoning models
            kernel.add_service(OpenAIChatCompletion(
                service_id="deepseek-r1",
                ai_model_id="deepseek-ai/deepseek-r1",
                async_client=AsyncOpenAI(
                api_key=dotenv_values(".env_core_planner").get("DEEPSEEK_API_KEY"),
                base_url="https://integrate.api.nvidia.com/v1"
            )
            ))
        
        elif self.service_id == "small_cheap_model":
            # Small and cheap model for the processing of certain user responses
            kernel.add_service(OpenAIChatCompletion(
                service_id="small_cheap_model",
            api_key=dotenv_values(".env_core_planner").get("OPENAI_API_KEY"),
            ai_model_id="gpt-4o-mini"
        ))
        
        if action_plugins:
            kernel = self._add_action_plugins(kernel)
            
        if retrieval_plugins:
            kernel = self._add_retrieval_plugins(kernel)
            
        if task_planner_communication:
            kernel = self._add_task_planner_communication_plugins(kernel)
            
        return kernel
    

    # @override
    # async def invoke(
    #     self,
    #     history: ChatHistory,
    #     arguments: KernelArguments | None = None,
    #     kernel: "Kernel | None" = None,
    #     **kwargs: Any,
        
    # ) -> AsyncIterable[ChatMessageContent]:
        
    #     ### Here we can potentially filter out internal messages from other agents: !
    #     # # Since the history contains internal messages from other agents,
    #     # # we will do our best to filter out those. Unfortunately, there will
    #     # # be a side effect of losing the context of the conversation internal
    #     # # to the agent when the conversation is handed back to the agent, i.e.
    #     # # previous function call results.
    #     # filtered_chat_history = ChatHistory()
    #     # for message in history:
    #     #     content = message.content
    #     # # We don't want to add messages whose text content is empty.
    #     # # Those messages are likely messages from function calls and function results.
    #     # if content:
    #     #     filtered_chat_history.add_message(message)
                
    #     async for response in super().invoke(history, arguments=arguments, kernel=kernel, **kwargs):
    #         yield response


class TaskPlannerAgent(RobotAgentBase):
    """Agent responsible for planning tasks based on goals."""
    service_id = "gpt4o"

    def __init__(self):
        kernel = self._create_kernel(action_plugins=True, retrieval_plugins=False, planning_plugins=False)
        
        settings = kernel.get_prompt_execution_settings_from_service_id(service_id=self.service_id)
        settings.function_choice_behavior = FunctionChoiceBehavior.Auto()
        
        parser = PydanticOutputParser(pydantic_object=TaskPlannerResponse)
        model_desc = parser.get_format_instructions()
        
        super().__init__(
            kernel=kernel,
            arguments=KernelArguments(settings=settings),
            name="TaskPlannerAgent",
            instructions=TASK_PLANNER_AGENT_INSTRUCTIONS.format(model_description=model_desc),
            description="Select me to plan sequential tasks that the robot should perform to complete the goal."
        )


class TaskExecutionAgent(RobotAgentBase):
    """Agent responsible for executing tasks."""
    service_id = "gpt4o"

    def __init__(self):
        kernel = self._create_kernel(action_plugins=True, retrieval_plugins=False, planning_plugins=True)
        
        settings = kernel.get_prompt_execution_settings_from_service_id(service_id=self.service_id)
        settings.function_choice_behavior = FunctionChoiceBehavior.Auto()
        
        super().__init__(
            name="TaskExecutionAgent",
            instructions=TASK_EXECUTION_AGENT_INSTRUCTIONS,
            kernel=kernel,
            arguments=KernelArguments(settings=settings),
            description="Select me to execute tasks that are generated/planned by the TaskPlannerAgent."
        )


class GoalCompletionCheckerAgent(RobotAgentBase):
    """Agent responsible for checking if goals have been completed."""
    service_id = "gpt4o"
    
    def __init__(self):
        kernel = self._create_kernel(action_plugins=False, retrieval_plugins=False, planning_plugins=True)
        
        settings = kernel.get_prompt_execution_settings_from_service_id(service_id=self.service_id)
        settings.function_choice_behavior = FunctionChoiceBehavior.Auto()
        
        super().__init__(
            kernel=kernel,
            instructions=GOAL_COMPLETION_CHECKER_AGENT_INSTRUCTIONS.format(termination_keyword=config.get("robot_planner_settings", {}).get("termination_keyword", "COMPLETED")),
            arguments=KernelArguments(settings=settings),
            name="GoalCompletionCheckerAgent",
            description="Select me to check if goals have been completed."
        )

# Cool: it would be nice to test how a group chat vs. responsibility handover mechanism would perform against each other.
# 1. Predefined logic deterines who can now decide/speak
# 2. An agent can decide who to give the responsibility at the moment (group chat), might be more flexible, but might gave worse performance (we don't know)


class ApprovalTerminationStrategy(TerminationStrategy):
    """A strategy for determining when an agent should terminate."""

    async def should_agent_terminate(self, agent, history):
        """Check if the agent should terminate."""
        return "Approved! The goal is completed!" in history[-1].content.lower()


class RobotPlanner:
    """
    Class that handles the planning of the robot using specialized agents.
    """

    def __init__(self, scene: Scene) -> None:
        """
        Constructor for the RobotPlanner class that handles plugin initialization and planning.
        """
        # Set the configurations for the scene
        self.scene = scene

        # Initialize agents
        self.task_planner_agent = TaskPlannerAgent()
        self.task_execution_agent = TaskExecutionAgent()
        self.goal_completion_checker_agent = GoalCompletionCheckerAgent()

        # Planner states
        self.goal = None
        self.goal_completed = False
        self.plan = None
        self.replanned = False
        self.planning_chat_history = ChatHistory()
        self.task = None
        self.tasks_completed = []
        self.actions_taken = []

        # Task planning variables
        self.json_format_agent_thread = None

    async def _create_task_plan(self, additional_message: Annotated[str, "Additional message to add to the task generation prompt"] = "") -> str:
        """Create a task plan based on the current goal and robot state."""
        # Check if json_format_agent_thread exists
        # if self.json_format_agent_thread is None:
        #     logger.info("Initializing json_format_agent_thread as it is None")
        #     self.json_format_agent_thread = ChatHistoryAgentThread()
            
        plan_generation_prompt = CREATE_TASK_PLANNER_PROMPT_TEMPLATE.format(
            goal=self.goal, 
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
        
        self.planning_chat_history.add_user_message("Initial plan:" + str(self.plan))
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
        self.plan = None
        self.replanned = False
        self.planning_chat_history = ChatHistory()
        self.tasks_completed = []
        
        # Create initial plan
        chain_of_thought = await self._create_task_plan()
        
        logger.info(f"Goal set to: {self.goal}. Initial plan created.")

        return self.plan, chain_of_thought

    @kernel_function(description="Function to call when something happens that doesn't follow the initial plan generated by the task planning agent.")
    async def update_task_plan(self, issue_description: Annotated[str, "A detailed description of the current situation and what went different to the original plan."]) -> None:
        """Update the task plan based on issues encountered during execution."""
            
        self.replanned = True
        
        update_plan_prompt = UPDATE_TASK_PLANNER_PROMPT_TEMPLATE.format(
            goal=self.goal,
            previous_plan=self.plan,
            issue_description=issue_description, 
            tasks_completed=', '.join(map(str, self.tasks_completed)), 
            planning_chat_history=self.planning_chat_history, 
            scene_graph=str(robot_state.scene_graph.scene_graph_to_dict()),
            robot_position="Not available" if not use_robot else str(frame_transformer.get_current_body_position_in_frame(robot_state.frame_name))
        )

        updated_plan_response, self.json_format_agent_thread = await invoke_agent(
            agent=self.task_planner_agent, 
            thread=self.json_format_agent_thread,
            input_text_message=update_plan_prompt, 
            input_image_message=robot_state.get_current_image_content()
        )

        logger.info("========================================")
        logger.info(f"Reasoning about the updated plan: {str(updated_plan_response)}")
        logger.info("========================================")

        # Convert ChatMessageContent to string
        updated_plan_response_str = str(updated_plan_response)

        # Define the pattern to extract everything before ```json
        pattern_before_json = r"(.*?)```json"
        match_before_json = re.search(pattern_before_json, updated_plan_response_str, re.DOTALL)

        # Extract and assign to reasoning variable
        chain_of_thought = match_before_json.group(1).strip() if match_before_json else ""

        pattern = r"```json\s*(.*?)\s*```"
        match = re.search(pattern, updated_plan_response_str, re.DOTALL)

        if match:
            json_content_inside = match.group(1)
            updated_plan_json_str = str(json_content_inside).replace('```json', '').replace('```', '').strip()
        else:
            logger.info('No ```json``` block found in response. Using the whole response as JSON.')
            updated_plan_json_str = str(updated_plan_response_str).replace('```json', '').replace('```', '').strip()
            
        try:
            self.plan = json.loads(updated_plan_json_str)
           
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from response: {e}")
            await self.update_task_plan("Failed to parse JSON from response with error: " + str(e) + ". Please try again.")
        
        self.planning_chat_history.add_user_message("Issue description with previous plan:" + issue_description)
        self.planning_chat_history.add_user_message("Updated plan:" + str(self.plan))
        logger.info("========================================")
        logger.info(f"Extracted updated plan: {json.dumps(self.plan, indent=2)}")
        logger.info("========================================")
        self.json_format_agent_thread = None
        
        return chain_of_thought

    @kernel_function(description="Function to call when you (the task execution agent) think that by completing the current task, you have actually completed the overall goal.")
    async def check_if_goal_is_completed(self, explanation: Annotated[str, "A detailed explanation of why you think the goal is completed."]) -> None:
        """Check if the goal is completed."""
        
        #
        agent = self.kernel
        
        
        check_if_goal_is_completed_prompt = CHECK_IF_GOAL_IS_COMPLETED_PROMPT_TEMPLATE.format(
            task=self.task,
            goal=self.goal,
            explanation=explanation,
            termination_keyword=config.get("robot_planner_settings", {}).get("termination_keyword", "COMPLETED"),
            plan=self.plan
        )
        
        thread = ChatHistoryAgentThread()

        response, thread = await invoke_agent(
            agent=self.goal_completion_checker_agent,
            thread=thread,
            input_text_message=check_if_goal_is_completed_prompt,
            input_image_message=robot_state.get_current_image_content()
        )
        
        if "COMPLETED" in response.lower():
            self.goal_completed = True
            logger.info("Goal completion flag set to True")



class RobotPlannerSingleton(_SingletonWrapper):
    """Singleton wrapper for the RobotPlanner class."""
    _type_of_class = RobotPlanner
    
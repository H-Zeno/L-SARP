# Standard library imports
import asyncio
from datetime import datetime
import json
import logging
import re
from openai import AsyncOpenAI
from utils.recursive_config import Config
import yaml
from pathlib import Path
from typing import Annotated, Dict, List, Optional, Tuple

# Third-party imports
from dotenv import dotenv_values
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from semantic_kernel import Kernel
from semantic_kernel.agents import AgentGroupChat, ChatCompletionAgent
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
)
# from configs.plugin_configs import plugin_configs
from configs.scenes_and_plugins_config import Scene
from source.LostFound.src.utils import scene_graph_to_json
from source.planner_core.robot_state import RobotStateSingleton
from source.robot_plugins.item_interactions import ItemInteractionsPlugin
from source.robot_plugins.navigation import NavigationPlugin
from source.robot_plugins.inspection import InspectionPlugin
from source.utils.agent_utils import invoke_agent, invoke_agent_group_chat
# from robot_utils.frame_transformer import FrameTransformerSingleton

# Initialize robot state singleton
robot_state = RobotStateSingleton()
# frame_transformer = FrameTransformerSingleton()

config = Config()


# Just get the logger, configuration is handled in main.py
logger = logging.getLogger(__name__)

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



class RobotPlanner:
    """
    Class that handles the planning of the robot.
    """
    # Loading scene graph from json file
    scene_graph_path = Path(config["robot_planner_settings"]["path_to_scene_data"]) / config["robot_planner_settings"]["active_scene"] / "scene_graph.json"
    with open(scene_graph_path, "r") as file:
        scene_graph_data = json.load(file)
    
    def __init__(
        self, 
        scene: Scene,

    ) -> None:
        """
        Constructor for the RobotPlanner class that handles plugin initialization and planning.
        """
        # Load settings
        self._planner_settings = dotenv_values(".env_core_planner")

        # Set the configurations for the retrieval plugins
        self.scene = scene
        # self._enabled_retrieval_plugins = self.scene.retrieval_plugins
        # self._retrieval_plugins_configs = plugin_configs

        # Create the kernel
        self.kernel = Kernel()

        # Planner states
        self.goal = None
        self.plan = None
        self.replanned = False
        self.planning_chat_history = ChatHistory()
        self.task = None
        self.tasks_completed = []

        # Task planning variables
        self.json_format_chat_history = None
        
        # Initialize the system
        self.setup_services()
        # self.add_retrieval_plugins()
        self.add_action_plugins()
        self.add_planning_plugins()
        self.initialize_task_planner_agent()
        self.initialize_task_execution_agent()
        self.initialize_goal_completion_checker_agent()

    # def add_retrieval_plugins(self) -> None:
    #     """
    #     Adds all the enabled plugins to the kernel.
    #     Scenes_and_plugins_config.py contains the plugin configurations for each scene.
    #     """
        
    #     # Add Enabled Plugins to the kernel
    #     for plugin_name in self._enabled_retrieval_plugins:
    #         if plugin_name in self._retrieval_plugins_configs:
    #             factory_func, args, kernel_name = self._retrieval_plugins_configs[plugin_name]
    #             plugin = factory_func(*args)
    #             self.kernel.add_plugin(plugin, plugin_name=kernel_name)

    def add_action_plugins(self) -> None:
        """
        Adds all the action plugins to the kernel
        """
        self.kernel.add_plugin(ItemInteractionsPlugin(), plugin_name="item_interactions")
        self.kernel.add_plugin(NavigationPlugin(), plugin_name="navigation")
        self.kernel.add_plugin(InspectionPlugin(), plugin_name="object_inspection")

    def add_planning_plugins(self) -> None:
        """
        Adds all the planning plugins to the kernel
        """
        # Register the task planning methods from this class as a plugin
        self.kernel.add_plugin(self, plugin_name="task_planner")

    def setup_services(self) -> None:
        """
        Set up AI services with appropriate models and API keys.
        Configures both the main GPT-4 model and the auxiliary reasoning model.
        """

        # General Multimodal Intelligence model (GPT4o)
        self.kernel.add_service(OpenAIChatCompletion(
            service_id="gpt4o",
            api_key=self._planner_settings.get("OPENAI_API_KEY"),
            ai_model_id="gpt-4o-2024-11-20"))

        # # Set up highly intelligent Google Gemini model for answering question
        # self.kernel.add_service(GoogleAIChatCompletion(
        #     service_id="gpt4o",
        #     api_key=dotenv_values().get("GEMINI_API_KEY"),
        #     gemini_model_id="gemini-2.0-flash"))

        # Reasoning models
        self.kernel.add_service(OpenAIChatCompletion(
            service_id="o3-mini",
            api_key=self._planner_settings.get("OPENAI_API_KEY"),
            ai_model_id="o3-mini-2025-01-31"))
        
        self.kernel.add_service(OpenAIChatCompletion( 
            service_id="o1",
            api_key=self._planner_settings.get("OPENAI_API_KEY"),
            ai_model_id="o1-2024-12-17"))
        
        self.kernel.add_service(OpenAIChatCompletion(
            service_id="deepseek-r1",
            ai_model_id="deepseek-ai/deepseek-r1",
            async_client=AsyncOpenAI(
                api_key=self._planner_settings.get("DEEPSEEK_API_KEY"),
                base_url="https://integrate.api.nvidia.com/v1"
            )
        ))

        # Small and cheap model for the processing of certain user responses
        self.kernel.add_service(OpenAIChatCompletion(
            service_id="small_cheap_model",
            api_key=self._planner_settings.get("OPENAI_API_KEY"),
            ai_model_id="gpt-4o-mini"))


    def initialize_task_planner_agent(self) -> None:
        """
        Initializes the task generation agent.
        """
        parser = PydanticOutputParser(pydantic_object=TaskPlannerResponse)
        model_desc = parser.get_format_instructions()

        # Create task generation agent with auto function calling
        task_generation_endpoint_settings = OpenAIChatPromptExecutionSettings(
            max_tokens=int(self._planner_settings.get("MAX_TOKENS")),
            temperature=float(self._planner_settings.get("TEMPERATURE")),
            top_p=float(self._planner_settings.get("TOP_P")),
            structured_json_response=True,
            function_choice_behavior=FunctionChoiceBehavior.Auto() # auto function calling
        )
        self.task_planner_agent = ChatCompletionAgent(
            service_id="o1",
            kernel=self.kernel,
            name="TaskPlannerAgent",
            instructions=TASK_PLANNER_AGENT_INSTRUCTIONS.format(model_description=model_desc),
            # execution_settings=task_generation_endpoint_settings
        )

    def initialize_task_execution_agent(self) -> None:
        """
        Initializes the task execution agent.
        """

        task_execution_endpoint_settings = OpenAIChatPromptExecutionSettings(
            service_id="gpt4o",
            max_tokens=int(self._planner_settings.get("MAX_TOKENS")),
            temperature=float(self._planner_settings.get("TEMPERATURE")),
            top_p=float(self._planner_settings.get("TOP_P")),
            function_choice_behavior=FunctionChoiceBehavior.Auto(),
        )

        # Create task execution agent with auto function calling
        self.task_execution_agent = ChatCompletionAgent(
            service_id="gpt4o",
            kernel=self.kernel,
            name="TaskExecutionAgent",
            instructions=TASK_EXECUTION_AGENT_INSTRUCTIONS,
            execution_settings=task_execution_endpoint_settings
        )

    def initialize_goal_completion_checker_agent(self) -> None:
        """
        Initializes the goal completion checker agent.
        """
        goal_completion_checker_endpoint_settings = OpenAIChatPromptExecutionSettings(
            service_id="gpt4o",
            max_tokens=int(self._planner_settings.get("MAX_TOKENS")),
            temperature=float(self._planner_settings.get("TEMPERATURE")),
            top_p=float(self._planner_settings.get("TOP_P")),
            function_choice_behavior=FunctionChoiceBehavior.Auto()
        )
        self.goal_completion_checker_agent = ChatCompletionAgent(
            service_id="gpt4o",
            kernel=self.kernel,
            name="GoalCompletionCheckerAgent",
            instructions=GOAL_COMPLETION_CHECKER_AGENT_INSTRUCTIONS,
            execution_settings=goal_completion_checker_endpoint_settings
        )

    # def setup_task_completion_group_chat(self):
    #     # Create chat for requirement analysis

    #     selection_function = KernelFunctionFromPrompt(
    #         function_name="selection",
    #         prompt=f"""
    #         Examine the provided RESPONSE and choose the next participant.
    #         State only the name of the chosen participant without explanation.
    #         Never choose the participant named in the RESPONSE.

    #         Choose only from these participants:
    #         - {self.task_planner_agent.name}
    #         - {self.task_execution_agent.name}

    #         Rules:
    #         - If the {self.task_execution_agent.name} is experiencing a problem with executing the task, please choose the {self.task_planner_agent.name} to help and adjust the plan.

    #         RESPONSE:
    #         {{{{$lastmessage}}}}
    #         """
    #     )
    #     termination_keyword = "Task Completed!"

    #     termination_function = KernelFunctionFromPrompt(
    #         function_name="termination",
    #         prompt=f"""
    #         Examine the RESPONSE and determine if the task is completed. To determine if the task is completed, you have access to the following information:
    #         1. The task that has to be completed
    #         2. The plan generated by the {self.task_planner_agent.name}
    #         3. The tasks that have been completed so far
    #         4. The current state of the environment

    #         If the task is achieved, respond with a single word without explanation: {termination_keyword}
    #         If the goal is not yet achieved, respond with a single word without explanation: CONTINUE

    #         1. Task to be completed: {self.task}
    #         2. The plan generated by the {self.task_planner_agent.name}:
    #         3. The tasks that have been completed so far: {self.tasks_completed}
    #         4. The current state of the environment (scene graph): {robot_state.scene_graph}

    #         RESPONSE:
    #         {{{{$lastmessage}}}}
    #         """
    #     )

    #     self.task_completion_group_chat = AgentGroupChat(
    #         agents=[self.task_planner_agent, self.task_execution_agent],
    #         selection_strategy=KernelFunctionSelectionStrategy(
    #         initial_agent=self.task_planner_agent,
    #         function=selection_function,
    #         kernel=self.kernel,
    #         result_parser=lambda result: str(result.value[0]).strip() if result.value[0] is not None else self.task_execution_agent.name,
    #         history_variable_name="lastmessage",
    #     ),
    #     termination_strategy=KernelFunctionTerminationStrategy(
    #         agents=[self.goal_completion_checker_agent],
    #         function=termination_function,
    #         kernel=self.kernel,
    #         result_parser=lambda result: termination_keyword in str(result.value[0]).lower(),
    #         history_variable_name="lastmessage",
    #         maximum_iterations=10,
    #     ),
    #     )
    #     logger.debug(f" Agent Group Chat successfully setup.")

    #     return self.task_completion_group_chat

    async def _create_task_plan(self, additional_message: Annotated[str, "Additional message to add to the task generation prompt"] = "") -> str:
        # Check if json_format_chat_history exists


        if self.json_format_chat_history == None:
            logger.info("Initializing json_format_chat_history as it is None")
            self.json_format_chat_history = ChatHistory()
            
        plan_generation_prompt = CREATE_TASK_PLANNER_PROMPT_TEMPLATE.format(goal=self.goal, 
                                                                             scene_graph=self.scene_graph_data, 
                                                                             robot_position=str("(0,0,0)"))
        # scene_graph=scene_graph_to_json(robot_state.scene_graph)
        # frame_transformer.get_current_body_position_in_frame(robot_state.frame_name)

        logger.info("========================================")
        logger.info(f"Plan generation prompt: {plan_generation_prompt}")
        logger.info("========================================")

        plan_response, self.json_format_chat_history = await invoke_agent(agent=self.task_planner_agent, 
                                                                          chat_history=self.json_format_chat_history,
                                                                          input_text_message=additional_message + plan_generation_prompt, 
                                                                          input_image_message=robot_state.get_current_image_content())
        logger.info("========================================")
        logger.info(f"Initial plan full response: {str(plan_response)}")
        logger.info("========================================")


        # # Define the pattern to extract everything before ```json
        # pattern_before_json = r"(.*?)```json"
        # match_before_json = re.search(pattern_before_json, plan_response, re.DOTALL)

        # # Extract and assign to reasoning variable
        # chain_of_thought = match_before_json.group(1).strip() if match_before_json else ""
        chain_of_thought = ""

        pattern = r"```json\s*(.*?)\s*```"
        match = re.search(pattern, plan_response, re.DOTALL)
        if match:
            json_content_inside = match.group(1)
            plan_json_str = str(json_content_inside).replace('```json', '').replace('```', '').strip()
        else:
            logger.info('No ```json``` block found in response. Using the whole response as JSON.')
            plan_json_str = str(plan_response).replace('```json', '').replace('```', '').strip()

        try:
            logger.info("========================================")
            logger.info(f"Plan JSON string: {plan_json_str}")   
            logger.info("========================================")
            self.plan = json.loads(plan_json_str)
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from response: {e}")
            await self._create_task_plan(additional_message="Failed to parse JSON from response with error: " + str(e) + ". Please try again.")
        
        self.planning_chat_history.add_user_message("Initial plan:" + str(self.plan))
        logger.info(f"Initial plan: {json.dumps(self.plan, indent=2)}")
        # Log the reasoning content
        logger.info(f"Chain of thought of initial plan (in case of reasoning model): {chain_of_thought}")

        self.json_format_chat_history = None # Reset the chat history
        return chain_of_thought

    @kernel_function(description="Function to call when something happens that doesn't follow the initial plan generated by the task planning agent.")
    async def update_task_plan(self, issue_description: Annotated[str, "Detailed description of the current explanation and what went different to the original plan"]) -> None:
        if self.json_format_chat_history == None:
            self.json_format_chat_history = ChatHistory()
            
        self.replanned = True
        
        update_plan_prompt = UPDATE_TASK_PLANNER_PROMPT_TEMPLATE.format(goal=self.goal,
                                                                        previous_plan=self.plan,
                                                                        issue_description=issue_description, 
                                                                        tasks_completed=', '.join(map(str, self.tasks_completed)), 
                                                                        planning_chat_history=self.planning_chat_history, 
                                                                        scene_graph=self.scene_graph_data,
                                                                        robot_position=str("(0,0,0)"))
        # scene_graph=scene_graph_to_json(robot_state.scene_graph)
        # robot_position=frame_transformer.get_current_body_position_in_frame(robot_state.frame_name)

        updated_plan_response, self.json_format_chat_history = await invoke_agent(agent=self.task_planner_agent, 
                                                                                  chat_history=self.json_format_chat_history,
                                                                                  input_text_message=update_plan_prompt, 
                                                                                  input_image_message=robot_state.get_current_image_content())

        logger.info("========================================")
        logger.info(f"Reasoning about the updated plan: {str(updated_plan_response)}")
        logger.info("========================================")

        # # Define the pattern to extract everything before ```json
        # pattern_before_json = r"(.*?)```json"
        # match_before_json = re.search(pattern_before_json, updated_plan_response, re.DOTALL)

        # # Extract and assign to reasoning variable
        # chain_of_thought = match_before_json.group(1).strip() if match_before_json else ""

        chain_of_thought = ""

        pattern = r"```json\s*(.*?)\s*```"
        match = re.search(pattern, updated_plan_response, re.DOTALL)

        if match:
            json_content_inside = match.group(1)
            updated_plan_json_str = str(json_content_inside).replace('```json', '').replace('```', '').strip()
        else:
            logger.info('No ```json``` block found in response. Using the whole response as JSON.')
            updated_plan_json_str = str(updated_plan_response).replace('```json', '').replace('```', '').strip()
            
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
        self.json_format_chat_history = None
        
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


class ApprovalTerminationStrategy(TerminationStrategy):
    """A strategy for determining when an agent should terminate."""

    async def should_agent_terminate(self, agent, history):
        """Check if the agent should terminate."""
        return "Approved! The goal is completed!" in history[-1].content.lower()



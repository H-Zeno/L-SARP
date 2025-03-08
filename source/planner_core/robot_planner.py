# Standard library imports
import asyncio
import json
import logging
import re
import yaml
from pathlib import Path
from typing import Annotated, List, Optional, Tuple

# Third-party imports
from dotenv import dotenv_values
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel
from semantic_kernel import Kernel
from semantic_kernel.agents import AgentGroupChat, ChatCompletionAgent
from semantic_kernel.agents.strategies import (
    KernelFunctionSelectionStrategy,
    KernelFunctionTerminationStrategy,
)
from semantic_kernel.agents.strategies.termination.termination_strategy import TerminationStrategy
from semantic_kernel.connectors.ai.function_choice_behavior import FunctionChoiceBehavior
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion, OpenAIChatPromptExecutionSettings
from semantic_kernel.contents import ChatHistory, ChatMessageContent
from semantic_kernel.functions import KernelArguments, KernelFunctionFromPrompt, kernel_function
from semantic_kernel.services.ai_service_client_base import AIServiceClientBase


# Local imports
from configs.agent_instruction_prompts import (
    GOAL_COMPLETION_CHECKER_AGENT_INSTRUCTIONS,
    TASK_EXECUTION_AGENT_INSTRUCTIONS,
    TASK_PLANNER_AGENT_INSTRUCTIONS,
)
# from configs.plugin_configs import plugin_configs
from configs.scenes_and_plugins_config import Scene
from source.LostFound.src.utils import scene_graph_to_json
from source.planner_core.robot_state import RobotStateSingleton
from source.robot_plugins.item_interactions import ItemInteractionsPlugin
from source.robot_plugins.navigation import NavigationPlugin
from source.utils.agent_utils import invoke_agent, invoke_agent_group_chat

# Initialize robot state singleton
robot_state = RobotStateSingleton()

# Just get the logger, configuration is handled in main.py
logger = logging.getLogger(__name__)


create_task_planning_prompt_template = """
            1. Please generate a task plan to complete the following goal: {goal}

            2. Here is the scene graph:
            {scene_graph}

            3. Here is the robot's current position:
            (0, 0, 0)

            Make sure that the plan contains the actual function calls that should be made and is as clear and concise as possible.
            When future steps of the the plan are dependent on the outcome of previous steps, mention [DEPENDENT ON PREVIOUS STEPS] in the plan.
            """

update_task_planning_prompt_template = """
            1. There was an issue with the previous generated plan to achieve the following goal: {goal}

            2. This was the previous plan: {previous_plan}

            3.  This is the current description of the issue from the task execution agent:
            {issue_description}
            
            4. These are the tasks that have been completed so far:
            {tasks_completed}

            5. Here is the history of the past plans that have been generated to achiev the goal: {planning_chat_history}
            
            6. Here is the scene graph:
            {scene_graph}

            7. Here is the robot's current position:
            (0, 0, 0)

            Make sure that the plan contains the actual function calls that should be made and is as clear and concise as possible.
            """
class Task(BaseModel):
    """A task to be completed."""
    task: str
    dependent_on_outcome_of_previous_task : bool


class TaskPlannerResponse(BaseModel):
    """A response from the task planner agent."""
    tasks : List[Task]


class RobotPlanner:
    """
    Class that handles the planning of the robot.
    """
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
        
        # Initialize the system
        self.setup_services()
        # self.add_retrieval_plugins()
        self.add_action_plugins()
        self.add_planning_plugins()
        self.initialize_task_planner_agent()
        self.initialize_task_execution_agent()
        self.initialize_goal_completion_checker_agent()

    def _load_config(self) -> dict:
        """
        Load configuration from config.yaml file.
        
        """
        with open(Path(self._planner_settings.get("PROJECT_DIR")) / 'configs' / 'config.yaml', 'r') as file:
            return yaml.safe_load(file)

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

        # Set up highly intelligent OpenAI model for answering question
        self.kernel.add_service(OpenAIChatCompletion(
            service_id="general_intelligence",
            api_key=self._planner_settings.get("OPENAI_API_KEY"),
            ai_model_id="gpt-4o-2024-11-20"))

        # # Set up highly intelligent Google Gemini model for answering question
        # self.kernel.add_service(GoogleAIChatCompletion(
        #     service_id="general_intelligence",
        #     api_key=dotenv_values().get("GEMINI_API_KEY"),
        #     gemini_model_id="gemini-2.0-flash"))

        # # Set Up Reasoning Model for the analysis of the experiences to requirements matching
        # self.kernel.add_service(OpenAIChatCompletion(
        #     service_id="opneai_reasoning_model",
        #     api_key=self._planner_settings.get("OPENAI_API_KEY"),
        #     ai_model_id="gpt-4o-2024-11-20")) # will be replaced by o3 mini in the future!
        
        # self.kernel.add_service(OpenAIChatCompletion(
        #     service_id="deepseek-r1",
        #     ai_model_id="deepseek-ai/deepseek-r1",
        #     async_client=AsyncOpenAI(
        #         api_key=self._planner_settings.get("DEEPSEEK_API_KEY"),
        #         base_url="https://integrate.api.nvidia.com/v1"
        #     ),
            
        # ))

        # Set Up small and cheap model for the processing of certain user responses
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
            service_id="general_intelligence",
            max_tokens=int(self._planner_settings.get("MAX_TOKENS")),
            temperature=float(self._planner_settings.get("TEMPERATURE")),
            top_p=float(self._planner_settings.get("TOP_P")),
            function_choice_behavior=FunctionChoiceBehavior.Auto() # auto function calling
        )
        self.task_planner_agent = ChatCompletionAgent(
            service_id="general_intelligence",
            kernel=self.kernel,
            name="TaskPlannerAgent",
            instructions=TASK_PLANNER_AGENT_INSTRUCTIONS.format(model_description=model_desc),
            execution_settings=task_generation_endpoint_settings
        )

    def initialize_task_execution_agent(self) -> None:
        """
        Initializes the task execution agent.
        """

        task_execution_endpoint_settings = OpenAIChatPromptExecutionSettings(
            service_id="general_intelligence",
            max_tokens=int(self._planner_settings.get("MAX_TOKENS")),
            temperature=float(self._planner_settings.get("TEMPERATURE")),
            top_p=float(self._planner_settings.get("TOP_P")),
            function_choice_behavior=FunctionChoiceBehavior.Auto()
        )

        # Create task execution agent with auto function calling
        self.task_execution_agent = ChatCompletionAgent(
            service_id="general_intelligence",
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
            service_id="general_intelligence",
            max_tokens=int(self._planner_settings.get("MAX_TOKENS")),
            temperature=float(self._planner_settings.get("TEMPERATURE")),
            top_p=float(self._planner_settings.get("TOP_P")),
            function_choice_behavior=FunctionChoiceBehavior.Auto()
        )
        self.goal_completion_checker_agent = ChatCompletionAgent(
            service_id="general_intelligence",
            kernel=self.kernel,
            name="GoalCompletionCheckerAgent",
            instructions=GOAL_COMPLETION_CHECKER_AGENT_INSTRUCTIONS,
            execution_settings=goal_completion_checker_endpoint_settings
        )

    def setup_task_completion_group_chat(self):
        # Create chat for requirement analysis

        selection_function = KernelFunctionFromPrompt(
            function_name="selection",
            prompt=f"""
            Examine the provided RESPONSE and choose the next participant.
            State only the name of the chosen participant without explanation.
            Never choose the participant named in the RESPONSE.

            Choose only from these participants:
            - {self.task_planner_agent.name}
            - {self.task_execution_agent.name}

            Rules:
            - If the {self.task_execution_agent.name} is experiencing a problem with executing the task, please choose the {self.task_planner_agent.name} to help and adjust the plan.

            RESPONSE:
            {{{{$lastmessage}}}}
            """
        )
        termination_keyword = "Task Completed!"

        termination_function = KernelFunctionFromPrompt(
            function_name="termination",
            prompt=f"""
            Examine the RESPONSE and determine if the task is completed. To determine if the task is completed, you have access to the following information:
            1. The task that has to be completed
            2. The plan generated by the {self.task_planner_agent.name}
            3. The tasks that have been completed so far
            4. The current state of the environment

            If the task is achieved, respond with a single word without explanation: {termination_keyword}
            If the goal is not yet achieved, respond with a single word without explanation: CONTINUE

            1. Task to be completed: {self.task}
            2. The plan generated by the {self.task_planner_agent.name}:
            3. The tasks that have been completed so far: {self.tasks_completed}
            4. The current state of the environment (scene graph): {robot_state.scene_graph}

            RESPONSE:
            {{{{$lastmessage}}}}
            """
        )

        self.task_completion_group_chat = AgentGroupChat(
            agents=[self.task_planner_agent, self.task_execution_agent],
            selection_strategy=KernelFunctionSelectionStrategy(
            initial_agent=self.task_planner_agent,
            function=selection_function,
            kernel=self.kernel,
            result_parser=lambda result: str(result.value[0]).strip() if result.value[0] is not None else self.task_execution_agent.name,
            history_variable_name="lastmessage",
        ),
        termination_strategy=KernelFunctionTerminationStrategy(
            agents=[self.goal_completion_checker_agent],
            function=termination_function,
            kernel=self.kernel,
            result_parser=lambda result: termination_keyword in str(result.value[0]).lower(),
            history_variable_name="lastmessage",
            maximum_iterations=10,
        ),
        )
        logger.debug(f" Agent Group Chat successfully setup.")

        return self.task_completion_group_chat

    async def create_task_plan(self, additional_message: Annotated[str, "Additional message to add to the task generation prompt"] = "", internal_chat_history: ChatHistory = None) -> None:
        if internal_chat_history is None:
            internal_chat_history = ChatHistory()
            
        plan_generation_prompt = create_task_planning_prompt_template.format(goal=self.goal, tasks_completed=', '.join(map(str, self.tasks_completed)), scene_graph=scene_graph_to_json(robot_state.scene_graph))

        logger.info("========================================")
        logger.info(f"Plan generation prompt: {plan_generation_prompt}")
        logger.info("========================================")
        plan_response, json_format_chat_history = await invoke_agent(self.task_planner_agent, additional_message + plan_generation_prompt, chat_history=internal_chat_history)

        logger.info("========================================")
        logger.info(f"Reasoning about the initial plan: {str(plan_response)}")
        logger.info("========================================")

        plan_json_str = str(plan_response).replace('```json', '').replace('```', '').strip()

        try:
            self.plan = json.loads(plan_json_str)

            self.planning_chat_history.add_user_message("Initial plan:" + str(self.plan))
            logger.info(f"Extracted plan: {json.dumps(self.plan, indent=2)}")
        
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from response: {e}")
            await self.create_task_plan("Failed to parse JSON from response with error: " + str(e) + ". Please try again.", internal_chat_history=internal_chat_history)
            
        return None

    @kernel_function(description="Function to call when something happens that doesn't follow the initial plan generated by the task planning agent.")
    async def update_task_plan(self, issue_description: Annotated[str, "Detailed description of the current explanation and what went different to the original plan"], internal_chat_history: ChatHistory = None) -> None:
        if internal_chat_history is None:
            internal_chat_history = ChatHistory()
            
        self.replanned = True
        
        update_plan_prompt = update_task_planning_prompt_template.format(goal=self.goal, previous_plan=self.plan, issue_description=issue_description, tasks_completed=', '.join(map(str, self.tasks_completed)), planning_chat_history=self.planning_chat_history, scene_graph=scene_graph_to_json(robot_state.scene_graph))

        updated_plan_response, json_format_chat_history = await invoke_agent(self.task_planner_agent, update_plan_prompt, chat_history=internal_chat_history)

        logger.info("========================================")
        logger.info(f"Reasoning about the updated plan: {str(updated_plan_response)}")
        logger.info("========================================")

        updated_plan_json_str = str(updated_plan_response).replace('```json', '').replace('```', '').strip()

        try:
            self.plan = json.loads(updated_plan_json_str)
            self.planning_chat_history.add_user_message("Issue description with previous plan:" + issue_description)
            self.planning_chat_history.add_user_message("Updated plan:" + str(self.plan))
            logger.info("========================================")
            logger.info(f"Extracted updated plan: {json.dumps(self.plan, indent=2)}")
            logger.info("========================================")
        
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from response: {e}")
            await self.update_task_plan("Failed to parse JSON from response with error: " + str(e) + ". Please try again.", internal_chat_history=json_format_chat_history)
                
        return None

    async def set_goal(self, goal: Annotated[str, "The goal to be achieved by the robot"]) -> None:
        """
        Sets the goal for the robot planner and creates an initial task plan.
        """
        self.goal = goal
        self.plan = None
        self.replanned = False
        self.planning_chat_history = ChatHistory()
        self.tasks_completed = []
        
        # Create initial plan
        await self.create_task_plan()
        
        logger.info(f"Goal set to: {self.goal}. Initial plan created.")


class ApprovalTerminationStrategy(TerminationStrategy):
    """A strategy for determining when an agent should terminate."""

    async def should_agent_terminate(self, agent, history):
        """Check if the agent should terminate."""
        return "Approved! The goal is completed!" in history[-1].content.lower()



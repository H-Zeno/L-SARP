from pathlib import Path
from typing import Optional, Tuple, List
from dotenv import dotenv_values

from semantic_kernel import Kernel
from semantic_kernel.agents import ChatCompletionAgent
from semantic_kernel.services.ai_service_client_base import AIServiceClientBase
from semantic_kernel.connectors.ai.open_ai import OpenAIChatPromptExecutionSettings, OpenAIChatCompletion
from semantic_kernel.connectors.ai.function_choice_behavior import FunctionChoiceBehavior

from semantic_kernel.contents import ChatHistory
from semantic_kernel.functions import KernelArguments
from configs.plugin_configs import plugin_configs

from configs.scenes_and_plugins_config import Scene
import logging

logger = logging.getLogger(__name__)

class RobotPlanner:
    def __init__(
        self, 
        scene: Scene
    ) -> None:
        """
        Constructor for the RobotPlanner class that handles plugin initialization and planning.

        Args:
            task_execution_service (AIServiceClientBase): The AI service client (e.g. OpenAI) that will
                be used by the semantic kernel for the execution of the task that have to be completed.
            task_generation_service (AIServiceClientBase): The AI service client (e.g. OpenAI) that will
                be used by the semantic kernel for the generation of the task that have to be completed.
            task_execution_endpoint_settings (OpenAIChatPromptExecutionSettings): The settings for the request to the AI service.
            task_generation_endpoint_settings (OpenAIChatPromptExecutionSettings): The settings for the request to the AI service.
            enabled_plugins (List[str]): List of plugin names that should be enabled for the
                current scene, e.g. ["nav", "text", "sql", "image"].
            plugin_configs (dict): Configuration dictionary for plugins containing tuples of
                (factory_function, arguments, kernel_name) for each plugin.
        """
        # Load settings
        self._settings = dotenv_values(".env_core_planner")

        # Set plugin configurations
        self._enabled_plugins = self.scene.plugins
        self._plugin_configs = plugin_configs

        # Agent instructions
        self._task_execution_agent_instructions = Path("configs/task_execution_agent_instructions.txt").read_text()
        self._task_generation_agent_instructions = Path("configs/task_generation_agent_instructions.txt").read_text()
        self._goal_completion_checker_agent_instructions = Path("configs/goal_completion_checker_agent_instructions.txt").read_text()


    def add_plugins_to_kernel(self) -> None:
        """
        Adds all the enabled plugins to the kernel.
        Scenes_and_plugins_config.py contains the plugin configurations for each scene.
        """
        # Addd our kernel Service
        self.kernel = Kernel()
        
        # Add Enabled Plugins to the kernel
        for plugin_name in self._enabled_plugins:
            if plugin_name in self._plugin_configs:
                factory_func, args, kernel_name = self._plugin_configs[plugin_name]
                plugin = factory_func(*args)
                self.kernel.add_plugin(plugin, plugin_name=kernel_name)


    def initialize_task_generation_agent(self) -> None:
        """
        Initializes the task generation agent.
        """
        task_generation_service = OpenAIChatCompletion(
            service_id="task_generator",
            api_key=self._settings.get("API_KEY"),
            org_id=self._settings.get("ORG_ID"),
            ai_model_id=self._settings.get("AI_MODEL_ID")
        )
        self.kernel.add_service(task_generation_service)

        # Create task generation agent with auto function calling
        task_generation_endpoint_settings = OpenAIChatPromptExecutionSettings(
            service_id="task_generator",
            max_tokens=int(self._settings.get("MAX_TOKENS")),
            temperature=float(self._settings.get("TEMPERATURE")),
            top_p=float(self._settings.get("TOP_P")),
            function_choice_behavior=FunctionChoiceBehavior.Auto() # auto function calling
        )
        self.task_generation_agent = ChatCompletionAgent(
            service_id="task_generator",
            kernel=self.kernel,
            name="TaskGenerationAgent",
            instructions=self._task_generation_agent_instructions,
            execution_settings=task_generation_endpoint_settings
        )


    def initialize_task_execution_agent(self) -> None:
        """
        Initializes the task execution agent.
        """
        task_execution_service = OpenAIChatCompletion(
            service_id="task_executer",
            api_key=self._settings.get("API_KEY"),
            org_id=self._settings.get("ORG_ID"),
            ai_model_id=self._settings.get("AI_MODEL_ID")
        )
        
        self.kernel.add_service(task_execution_service)

        task_execution_endpoint_settings = OpenAIChatPromptExecutionSettings(
            service_id="task_executer",
            max_tokens=int(self._settings.get("MAX_TOKENS")),
            temperature=float(self._settings.get("TEMPERATURE")),
            top_p=float(self._settings.get("TOP_P")),
            function_choice_behavior=FunctionChoiceBehavior.Auto()
        )

        # Create task execution agent with auto function calling
        self.task_execution_agent = ChatCompletionAgent(
            service_id="task_executer",
            kernel=self.kernel,
            name="TaskExecutionAgent",
            instructions=self._task_execution_agent_instructions,
            execution_settings=task_execution_endpoint_settings
        )


    def initialize_goal_completion_checker_agent(self) -> None:
        """
        Initializes the goal completion checker agent.
        """
        goal_completion_checker_service = OpenAIChatCompletion(
            service_id="goal_completion_checker",
            api_key=self._settings.get("API_KEY"),
            org_id=self._settings.get("ORG_ID"),
            ai_model_id=self._settings.get("AI_MODEL_ID")
        )
        self.kernel.add_service(goal_completion_checker_service)

        goal_completion_checker_endpoint_settings = OpenAIChatPromptExecutionSettings(
            service_id="goal_completion_checker",
            max_tokens=int(self._settings.get("MAX_TOKENS")),
            temperature=float(self._settings.get("TEMPERATURE")),
            top_p=float(self._settings.get("TOP_P")),
            function_choice_behavior=FunctionChoiceBehavior.Auto()
        )

        # Create goal_completion_checker agent
        self.goal_completion_checker_agent = ChatCompletionAgent(
            service_id="goal_completion_checker",
            kernel=self.kernel,
            name="GoalCompletionCheckerAgent",
            instructions=self._goal_completion_checker_agent_instructions,
            execution_settings=goal_completion_checker_endpoint_settings
        )


    async def invoke_task_generation_agent(self, goal: str, completed_tasks: list = None, env_state: str = None) -> Tuple:
        """
        Invokes the task generation agent to generate a list of tasks to complete based on the goal that is provided.
        
        Args:
            goal (str): The main goal to accomplish
            completed_tasks (list, optional): List of tasks already completed
            env_state (str, optional): Current state of the environment
        """
        # Create structured user message following the template
        task_generation_user_message = f"""
Goal: {goal}

Tasks Completed: 
{chr(10).join([f"- {task}" for task in completed_tasks]) if completed_tasks else "No tasks completed yet"}

Environment State:
{env_state if env_state else "Initial state - no environment data available"}
"""

        # Create chat history for this interaction
        self._goal_generator_history = ChatHistory()
        self._goal_generator_history.add_user_message(task_generation_user_message)

        # Invoke agent and get response
        async for response in self._task_generation_agent.invoke(self._goal_generator_history):
            return response.contents

    async def invoke_robot_on_task(self, task: str) -> Tuple[str, str]:
        """
        The robot achieves the given task using automatic tool calling via an agent.

        Args:
            task (str): task to be executed

        Returns:
            Tuple[str, str]: final response and chat history
        """
        if self.kernel is None:
            raise ValueError("You need to set the Semantic Kernel first")

        try:
            # Add task to chat history
            self._task_executer_history = ChatHistory()
            self._task_executer_history.add_user_message(task)

            # Invoke agent and get response with function calls
            async for response in self._task_execution_agent.invoke(self._task_executer_history):
                # Store response in history
                self._task_executer_history.add_assistant_message(response.content)
                return response.content, self._task_executer_history

        except Exception as e:
            logger.error(f"Error during task execution: {str(e)}")
            raise RuntimeError(f"Task execution failed: {str(e)}")

    async def invoke_goal_completion_checker_agent(self, goal: str, completed_tasks: list = None, env_state: str = None) -> Tuple:
        """
        Invokes the goal completion checker agent to check if the goal has been completed.
        The goal completion checker agent usually only gets activated 1 or 2 steps before the task generation agent plans the task to be completed.
        """
        pass

    # Check out: get access/insight on the plan that was made (e.g. telemetry support)


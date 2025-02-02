from pathlib import Path
from typing import Optional, Tuple, List

from semantic_kernel import Kernel
from semantic_kernel.agents import ChatCompletionAgent
from semantic_kernel.services.ai_service_client_base import AIServiceClientBase
from semantic_kernel.connectors.ai.open_ai import OpenAIChatPromptExecutionSettings

from semantic_kernel.contents import ChatHistory
from semantic_kernel.functions import KernelArguments

import logging

logger = logging.getLogger(__name__)

class RobotPlanner:
    def __init__(
        self, 
        task_execution_service: AIServiceClientBase,
        task_generation_service: AIServiceClientBase,
        task_execution_endpoint_settings: OpenAIChatPromptExecutionSettings,
        task_generation_endpoint_settings: OpenAIChatPromptExecutionSettings,
        enabled_plugins: List[str],
        plugin_configs: dict
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
        self._task_execution_service = task_execution_service
        self._task_generation_service = task_generation_service
        self._task_execution_endpoint_settings = task_execution_endpoint_settings
        self._task_generation_endpoint_settings = task_generation_endpoint_settings
        self._enabled_plugins = enabled_plugins
        self._plugin_configs = plugin_configs
        self._task_execution_agent_instructions = Path("configs/task_execution_agent_instructions.txt").read_text()
        self._task_generation_agent_instructions = Path("configs/task_generation_agent_instructions.txt").read_text()

    def set_kernel(self) -> None:
        """
        Sets up the kernel: adds the kernel services, the enabled plugins and the planner.

        """
        # Addd our kernel Service
        self._kernel = Kernel()
        self._kernel.add_service(self._task_execution_service)
        self._kernel.add_service(self._task_generation_service)

        # Add Enabled Plugins to the kernel
        for plugin_name in self._enabled_plugins:
            if plugin_name in self._plugin_configs:
                factory_func, args, kernel_name = self._plugin_configs[plugin_name]
                plugin = factory_func(*args)
                self._kernel.add_plugin(plugin, plugin_name=kernel_name)

        # Create task generation agent with auto function calling
        self._task_generation_agent = ChatCompletionAgent(
            service_id="task_generator",
            kernel=self._kernel,
            name="TaskGenerationAgent",
            instructions=self._task_generation_agent_instructions,
            execution_settings=self._task_generation_endpoint_settings
        )
        
        # Create task execution agent with auto function calling
        self._task_execution_agent = ChatCompletionAgent(
            service_id="task_executer",
            kernel=self._kernel,
            name="TaskExecutionAgent",
            instructions=self._task_execution_agent_instructions,
            execution_settings=self._task_execution_endpoint_settings
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
        if self._kernel is None:
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

    # Check out: get access/insight on the plan that was made (e.g. telemetry support)

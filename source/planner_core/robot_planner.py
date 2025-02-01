from pathlib import Path
from typing import Optional, Tuple, List

from semantic_kernel import Kernel
from semantic_kernel.services.ai_service_client_base import AIServiceClientBase
from semantic_kernel.connectors.ai.open_ai import OpenAIChatPromptExecutionSettings

from semantic_kernel.contents import ChatHistory
from semantic_kernel.functions import KernelArguments

import logging

logger = logging.getLogger(__name__)

class RobotPlanner:
    def __init__(
        self, 
        kernel_service: AIServiceClientBase,
        request_settings: OpenAIChatPromptExecutionSettings,
        enabled_plugins: List[str],
        plugin_configs: dict
    ) -> None:
        """
        Constructor for the RobotPlanner class that handles plugin initialization and planning.

        Args:
            kernel_service (AIServiceClientBase): The AI service client (e.g. OpenAI) that will
                be used by the semantic kernel for planning and execution.
            request_settings (OpenAIChatPromptExecutionSettings): The settings for the request to the AI service.
            enabled_plugins (List[str]): List of plugin names that should be enabled for the
                current scene, e.g. ["nav", "text", "sql", "image"].
            plugin_configs (dict): Configuration dictionary for plugins containing tuples of
                (factory_function, arguments, kernel_name) for each plugin.
        """
        self._kernel_service = kernel_service
        self._request_settings = request_settings
        self._enabled_plugins = enabled_plugins
        self._plugin_configs = plugin_configs
        self._system_prompt = Path("configs/system_prompt.txt").read_text()

    def set_kernel(self) -> None:
        """
        Sets up the kernel: adds the kernel services, the enabled plugins and the planner.

        """
        # Addd our kernel Service
        self._kernel = Kernel()
        self._kernel.add_service(self._kernel_service)
        
        # Add Enabled Plugins to the kernel
        for plugin_name in self._enabled_plugins:
            if plugin_name in self._plugin_configs:
                factory_func, args, kernel_name = self._plugin_configs[plugin_name]
                plugin = factory_func(*args)
                self._kernel.add_plugin(plugin, plugin_name=kernel_name)
        
        # Pass the request settings to the kernel arguments
        self._arguments = KernelArguments(settings=self._request_settings)
        
        # Create a chat history to store the system message, initial messages, and the conversation
        history = ChatHistory()
        history.add_system_message(self._system_prompt)


    async def invoke_robot_on_task(self, task: str) -> Tuple[str, str]:
        """
        The robot achieves the given task using automatic tool calling.

        Args:
            task (str): task to be answered

        Returns:
            Tuple[str, str]: final response (called final_answer) to the task and the function calls made,
            a question will be asked to the user if the task is not yet completed
        """

        if self._kernel is None:
            raise ValueError("You need to set the Semantic Kernel first")
        
        # Add the chat history to the arguments
        self._arguments["chat_history"] = self._history
        self._arguments["task"] = task

        try:
            # Get the response from the robot
            # The response is either a confirmation or a question to the user
            # The question to the user still has to be implemented
            response = await self._kernel.invoke(arguments=self._arguments)

            # Make sure we have a valid response
            if response and hasattr(response, 'final_answer'):
                return response.final_answer, response.chat_history[0].content if response.chat_history else ""
            else:
                raise ValueError("Invalid response format from planner")
            
        except Exception as e:
            logger.error(f"Error during planner invocation: {str(e)}")
            raise RuntimeError(f"Planner failed to process the question: {str(e)}")

        # Check out: get access/insight on the plan that was made (e.g. telemetry support)

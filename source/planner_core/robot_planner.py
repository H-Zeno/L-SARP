from pathlib import Path
from typing import Optional, Tuple, List

from semantic_kernel import Kernel
from semantic_kernel.services.ai_service_client_base import AIServiceClientBase
from semantic_kernel.planners import (
    FunctionCallingStepwisePlanner,
    FunctionCallingStepwisePlannerOptions,
)

import logging

logger = logging.getLogger(__name__)

class RobotPlanner:
    def __init__(
        self, 
        kernel_service: AIServiceClientBase,
        enabled_plugins: List[str],
        plugin_configs: dict
    ) -> None:
        """
        Constructor for the RobotPlanner class that handles plugin initialization and planning.

        Args:
            kernel_service (AIServiceClientBase): The AI service client (e.g. OpenAI) that will
                be used by the semantic kernel for planning and execution.
            enabled_plugins (List[str]): List of plugin names that should be enabled for the
                current scene, e.g. ["nav", "text", "sql", "image"].
            plugin_configs (dict): Configuration dictionary for plugins containing tuples of
                (factory_function, arguments, kernel_name) for each plugin.
        """
        self._kernel_service = kernel_service
        self._enabled_plugins = enabled_plugins
        self._plugin_configs = plugin_configs

    def set_kernel(self) -> None:
        """
        Sets up the kernel: adds the kernel services, the enabled plugins and the planner.

        """
        self._kernel = Kernel()
        self._kernel.add_service(self._kernel_service)
        
        # Add Enabled Plugins to the kernel
        for plugin_name in self._enabled_plugins:
            if plugin_name in self._plugin_configs:
                factory_func, args, kernel_name = self._plugin_configs[plugin_name]
                plugin = factory_func(*args)
                self._kernel.add_plugin(plugin, plugin_name=kernel_name)

        # Set up planner
        options = FunctionCallingStepwisePlannerOptions(
            max_iterations=5,
            min_iteration_time_ms=2000,
            max_tokens=5000,
        )
        self._planner = FunctionCallingStepwisePlanner(
            service_id=self._kernel_service.service_id,
            options=options,
        )

    async def invoke_planner(self, question: str) -> Tuple[str, str]:
        """
        Gets the answer for the given question using automatic tool calling.

        Args:
            question (str): question to be answered

        Returns:
            Tuple[str, str]: final answer to the question and the function calls made
        """
        if self._kernel is None:
            raise ValueError("You need to set the Semantic Kernel first")

        try:
            # Get the response from the AI
            response = await self._planner.invoke(self._kernel, question)
            
            # Make sure we have a valid response
            if response and hasattr(response, 'final_answer'):
                return response.final_answer, response.chat_history[0].content if response.chat_history else ""
            else:
                raise ValueError("Invalid response format from planner")
            
        except Exception as e:
            logger.error(f"Error during planner invocation: {str(e)}")
            raise RuntimeError(f"Planner failed to process the question: {str(e)}")

        # Check out: get access/insight on the plan that was made (e.g. telemetry support)

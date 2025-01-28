from pathlib import Path
from typing import Optional, Tuple, List

from semantic_kernel import Kernel
from semantic_kernel.services.ai_service_client_base import AIServiceClientBase
from semantic_kernel.planners import (
    FunctionCallingStepwisePlanner,
    FunctionCallingStepwisePlannerOptions,
)
from semantic_kernel.connectors.ai.open_ai.utils import get_tool_call_object

from misc.scenes_enum import Scene
from plugins.plugins_factory import PluginsFactory

DEFAULT_PLUGINS = ["nav", "text", "sql", "image"]

class RobotPlanner:
    def __init__(
        self, 
        plugins_factory: PluginsFactory, 
        path_to_data: Path, 
        kernel_service: AIServiceClientBase,
        enabled_plugins: List[str]
    ) -> None:
        """
        Constructor

        Args:
            plugins_factory (PluginsFactory): factory for plugins
            path_to_data (Path): path to the folder containing the data for RAG/plugins
        """
        self._plugins_factory = plugins_factory
        self._path_to_data = path_to_data
        self._kernel_service = kernel_service
        self._enabled_plugins = enabled_plugins
        self._kernel: Optional[Kernel] = None
        self._plugins = {}

    def set_scene(
        self, scene_choice: Scene, nav_vis_path: Optional[Path] = None
    ) -> None:
        """
        Sets up the scene for the chatbot.

        Args:
            scene_choice (Scene): selected scene to be set up
            nav_vis_path (Optional[Path], optional): path to the folder where the navigation
                visualization should be saved.
        """
        plugin_configs = {
            "nav": (
                self._plugins_factory.get_nav_plugin,
                [self._path_to_data / f"{scene_choice.value}/nav_data/navmesh.txt", nav_vis_path],
                "navigation"
            ),
            "text": (
                self._plugins_factory.get_text_plugin,
                [Path(f".TEXT_DIR/{scene_choice.value}"), self._path_to_data / f"{scene_choice.value}/text_data"],
                "text"
            ),
            "sql": (
                self._plugins_factory.get_sql_plugin,
                [self._path_to_data / f"{scene_choice.value}/sql_data/sql_db_data.json", Path(f".SQL_DIR/{scene_choice.value}")],
                "sql"
            ),
            "image": (
                self._plugins_factory.get_image_plugin,
                [Path(f".IMAGE_DIR/{scene_choice.value}"), self._path_to_data / f"{scene_choice.value}/img_data"],
                "image"
            )
        }

        self._kernel = Kernel()
        self._kernel.add_service(self._kernel_service)

        for plugin_name in self._enabled_plugins:
            if plugin_name in plugin_configs:
                factory_func, args, kernel_name = plugin_configs[plugin_name]
                plugin = factory_func(*args)
                self._plugins[plugin_name] = plugin
                self._kernel.add_plugin(plugin, plugin_name=kernel_name)

        # Setup planner
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
            raise ValueError("You need to set the SK first")

        # Get the response from the AI
        result = await self._planner.invoke(self._kernel, question)
        return result.final_answer, result.chat_history[0].content

        # Check out: get access/insight on the plan that was made (e.g. telemetry support)

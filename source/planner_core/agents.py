# Standard library imports
import logging
import sys
from abc import ABC
from collections.abc import AsyncIterable
from typing import Any, ClassVar, Optional, Dict, List, Tuple, Callable

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

# Third-party imports
from dotenv import dotenv_values
from openai import AsyncOpenAI
from semantic_kernel import Kernel
from semantic_kernel.agents import ChatCompletionAgent
from semantic_kernel.connectors.ai.function_choice_behavior import FunctionChoiceBehavior
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
from semantic_kernel.functions import KernelArguments

# Local imports
from configs.agent_instruction_prompts import (
    GOAL_COMPLETION_CHECKER_AGENT_INSTRUCTIONS,
    TASK_EXECUTION_AGENT_INSTRUCTIONS,
    TASK_PLANNER_AGENT_INSTRUCTIONS
)
from robot_plugins.inspection import InspectionPlugin
from robot_plugins.item_interactions import ItemInteractionsPlugin
from robot_plugins.navigation import NavigationPlugin
from robot_plugins.task_planner_communication import TaskPlannerCommunicationPlugin
from robot_plugins.goal_checker import TaskPlannerGoalChecker, TaskExecutionGoalChecker
from robot_plugins.maths import MathematicalOperationsPlugin
from robot_plugins.replanning import ReplanningPlugin
from robot_plugins.core_memory import CoreMemoryPlugin
from utils.recursive_config import Config


# Initialize logger
logger = logging.getLogger("main")

# Load configuration
config = Config()


class RobotAgentBase(ChatCompletionAgent, ABC):
    """Base class for all robot agents."""
    service_id: ClassVar[str] = "gpt4o"
    
    def __init__(self, *args, **kwargs):
        # First initialize the parent class
        super().__init__(*args, **kwargs)
        
        # Initialize retrieval plugin configurations after parent initialization
        self._enabled_retrieval_plugins = []
        self._retrieval_plugins_configs = {}
    
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
    
    def _create_kernel(self, action_plugins=False, retrieval_plugins=False, task_planner_communication=False, do_maths=False, core_memory=False) -> Kernel:
        """Create and configure a kernel with all the AI services that we support."""
        logger.info(f"Creating kernel with service ID: {self.service_id}")
        kernel = Kernel()
        
        if self.service_id == "gpt4o":
            # General Multimodal Intelligence model (GPT4o)
            kernel.add_service(OpenAIChatCompletion(
                service_id="gpt4o",
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
            kernel.add_plugin(TaskPlannerCommunicationPlugin(), plugin_name="task_planner_communication")
            
        if do_maths:
            kernel.add_plugin(MathematicalOperationsPlugin(), plugin_name="mathematical_operations")
        
        if core_memory:
            kernel.add_plugin(CoreMemoryPlugin(), plugin_name="core_memory")
        
        return kernel
    

class TaskPlannerAgent(RobotAgentBase):
    """Agent responsible for planning tasks based on goals."""
    service_id = config.get("robot_planner_settings", {}).get("task_planner_service_id", "")

    def __init__(self):
        kernel = self._create_kernel(action_plugins=True, retrieval_plugins=False, task_planner_communication=False, do_maths=True, core_memory=True)
        
        # Add the goal completion checker plugin
        kernel.add_plugin(TaskPlannerGoalChecker(), plugin_name="goal_checker")
        
        # Add the task planning (and replanning) plugins
        kernel.add_plugin(ReplanningPlugin(), plugin_name="task_planning")
        
        # Load the correct prompt execution settings
        settings = kernel.get_prompt_execution_settings_from_service_id(service_id=self.service_id)
        settings.function_choice_behavior = FunctionChoiceBehavior.Auto()
        
        
        super().__init__(
            kernel=kernel,
            arguments=KernelArguments(settings=settings),
            name="TaskPlannerAgent",
            instructions=TASK_PLANNER_AGENT_INSTRUCTIONS,
            description="Select me to plan sequential tasks that the robot should perform to complete the goal."
        )


class TaskExecutionAgent(RobotAgentBase):
    """Agent responsible for executing tasks."""
    service_id = config.get("robot_planner_settings", {}).get("task_execution_service_id", "")

    def __init__(self):
        kernel = self._create_kernel(action_plugins=True, retrieval_plugins=False, task_planner_communication=True, do_maths=True, core_memory=True)
        
        # Add the goal completion checker plugin
        kernel.add_plugin(TaskExecutionGoalChecker(), plugin_name="goal_checker")
        
        # Load the correct prompt execution settings
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
    service_id = config.get("robot_planner_settings", {}).get("goal_completion_checker_service_id", "")
    
    def __init__(self):
        kernel = self._create_kernel(action_plugins=False, retrieval_plugins=False, task_planner_communication=False, do_maths=True)
        
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




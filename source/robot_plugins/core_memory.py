import numpy as np
from typing import Annotated
from utils.recursive_config import Config

from semantic_kernel.functions.kernel_function_decorator import kernel_function

from planner_core.robot_state import RobotStateSingleton

config = Config()
robot_state = RobotStateSingleton()


class CoreMemoryPlugin:
    """Use this plugin to store core information that you acquired through interaction with the environment. TThe can be certain things that you discovered, insights or results from a series of important calculations or a reasoning process."""
    
    @kernel_function(description="Use this function to store core information that you acquired through interaction with the environment. The information can be certain things that you discovered, insights or results from a series of important calculations or a reasoning process.")
    def store_core_information(self, agent_name: Annotated[str, "The name of the agent that is storing this information."], information: Annotated[str, "A 1-2 sentence summary of the core information you want to store."]) -> str:
        """Store the core information in the robot's core memory."""
        agent_name = agent_name.lower()
        robot_state.core_memory.setdefault(agent_name, "")  # Ensure the key exists
        robot_state.core_memory[agent_name] += information
        return f"Core information stored: {information}"
    
    
    @kernel_function(description="Use this function to retrieve core information from the robot's core memory.")
    def retrieve_core_information(self) -> str:
        """Retrieve the core information from the robot's core memory."""
        return robot_state.core_memory
    
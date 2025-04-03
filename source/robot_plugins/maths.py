import numpy as np
from typing import Annotated, List
from utils.recursive_config import Config

from semantic_kernel.functions.kernel_function_decorator import kernel_function

from planner_core.robot_state import RobotStateSingleton

config = Config()
robot_state = RobotStateSingleton()

class MathematicalOperationsPlugin:
    """This plugin contains functions to perform mathematical operations on objects in the scene graph. Please use this plugin to:
    - calculate the euclidean distance between two objects
    - calculate the bounding box volume of an object
    - perform a mathematical operation on two numbers
    - sort a list of numbers in ascending or descending order"""
    
    @kernel_function(description="This function returns the Euclidean distance between two objects in the scene graph.")
    def euclidean_distance(self, object_id_1: Annotated[int, "The ID of the first object in the scene graph"], object_id_2: Annotated[int, "The ID of the second object in the scene graph"]) -> float:
        """
        This function returns the Euclidean distance between two objects in the scene graph.
        """
        return np.linalg.norm(robot_state.scene_graph.nodes[object_id_1].centroid - robot_state.scene_graph.nodes[object_id_2].centroid)
    
    @kernel_function(description="This function allows you to calculate the bounding box volume of an object in the scene graph.")
    def bounding_box_volume(self, object_id: Annotated[int, "The ID of the object in the scene graph"]) -> float:
        """
        This function allows you to calculate the bounding box volume of an object in the scene graph.
        """
        volume = robot_state.scene_graph.nodes[object_id].dimensions[0] * robot_state.scene_graph.nodes[object_id].dimensions[1] * robot_state.scene_graph.nodes[object_id].dimensions[2]
        return volume
    
    @kernel_function(description="Use this function to do a mathematical operation on two numbers.")
    def do_math_operation(self, operation: Annotated[str, "The operation to perform on the two numbers. Choose from 'add', 'subtract', 'multiply', 'divide'"], number_1: Annotated[float, "The first number to perform the operation on"], number_2: Annotated[float, "The second number to perform the operation on"]) -> float:
        """
        This function allows you to perform a mathematical operation on two numbers.
        """
        if operation == "add":
            return number_1 + number_2
        elif operation == "subtract":
            return number_1 - number_2
        elif operation == "multiply":
            return number_1 * number_2
        elif operation == "divide":
            return number_1 / number_2
        else:
            return "Invalid operation"
    
    @kernel_function(description="Use this function to sort a list of numbers in ascending or descending order.")
    def sort_list(self, list_to_sort: Annotated[List[float], "The list of numbers to sort"], ascending: Annotated[bool, "Whether to sort the list in ascending or descending order"]) -> List[float]:
        """
        This function allows you to sort a list of numbers in ascending or descending order.
        """
        return sorted(list_to_sort, reverse=not ascending)
    



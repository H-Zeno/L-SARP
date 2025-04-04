import numpy as np
from typing import Annotated, List, Union
from utils.recursive_config import Config
import ast

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
    def euclidean_distance_between_two_objects(self, object_id_1: Annotated[int, "The ID of the first object in the scene graph"], object_id_2: Annotated[int, "The ID of the second object in the scene graph"]) -> float:
        """
        This function returns the Euclidean distance between two objects in the scene graph.
        """
        return np.linalg.norm(robot_state.scene_graph.nodes[object_id_1].centroid - robot_state.scene_graph.nodes[object_id_2].centroid)
    
    @kernel_function(description="This function returns the Euclidean distance between two coordinates.")
    def euclidean_distance_between_coordinates(self, coordinates_1: Annotated[str, "The coordinates of the first object e.g. '[3.4, 8.1, 2.1]' "], coordinates_2: Annotated[str, "The coordinates of the second object e.g. '[3.4, 8.1, 2.1]' "]) -> float:
        """
        This function returns the Euclidean distance between two coordinates.
        """
        return np.linalg.norm(np.array(ast.literal_eval(coordinates_1)) - np.array(ast.literal_eval(coordinates_2)))
    
    @kernel_function(description="This function allows you to calculate the bounding box volume of an object in the scene graph.")
    def object_bounding_box_volume(self, object_id: Annotated[int, "The ID of the object in the scene graph"]) -> float:
        """
        This function allows you to calculate the bounding box volume of an object in the scene graph.
        """
        volume = robot_state.scene_graph.nodes[object_id].dimensions[0] * robot_state.scene_graph.nodes[object_id].dimensions[1] * robot_state.scene_graph.nodes[object_id].dimensions[2]
        return volume
    
    @kernel_function(description="Use this function to do elementary mathematical operations on two numbers.")
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
    def sort_list(self, list_to_sort: Annotated[str, "A string representation of a list containing the numbers to sort (e.g., '[4, 2, 3]')"], ascending: Annotated[bool, "Whether to sort the list in ascending or descending order"]) -> List[float]:
        """
        This function allows you to sort a list of numbers in ascending or descending order.
        """
        # convert the string to a numpy array
        list_to_sort_np = np.array(ast.literal_eval(list_to_sort))
        return sorted(list_to_sort_np, reverse=not ascending)
    
    @kernel_function(description="Use this function to get the n closest objects to a given object in the scene graph. Use 'all' if you want to return all objects.")
    def n_closest_objects_to_object(self, object_id: Annotated[int, "The ID of the object in the scene graph"], n: Annotated[str, "The number of closest objects to return. Write 'all' if you want to return all objects."]) -> Annotated[List[str], "A list of the n closest objects to the given object (sorted) and their distance from the given object."]:
        """
        This function returns the IDs of the n closest objects to a given object in the scene graph (sorted by distance).
        """
        scene_nodes = robot_state.scene_graph.nodes
        if object_id not in scene_nodes:
            return [f"Error: Object with ID {object_id} not found in the scene graph."]

        if n == "all":
            # Exclude the object itself if present
            num_nodes_to_consider = len(scene_nodes) - 1 if object_id in scene_nodes else len(scene_nodes)
            n = num_nodes_to_consider
        else:
            try:
                n = int(n)
                if n <= 0:
                    return ["Error: n must be a positive integer or 'all'."]
            except ValueError:
                return ["Error: n must be a positive integer or 'all'."]

        # Calculate distances to all other nodes
        distances = []
        for node_id, node_data in scene_nodes.items():
            if node_id == object_id: # Don't compare the object to itself
                continue
            distance = self.euclidean_distance_between_two_objects(object_id, node_id)
            # Store the semantic label ID (sem_label) for now
            distances.append((node_id, distance, node_data.sem_label))

        # Sort by distance
        distances.sort(key=lambda item: item[1])

        # Take the top n
        closest_n = distances[:n]

        # Format the response string using the label mapping
        response = []
        for node_id, dist, sem_label in closest_n:
            # Get the string label from the mapping
            label = robot_state.scene_graph.label_mapping.get(sem_label, f"Unknown Label ID: {sem_label}")
            response.append(f"Object ID: {node_id} - Semantic Label: {label} - Distance: {dist:.4f}")

        return response
    
    @kernel_function(description="Use this function to get the n closest objects to a given coordinate in the scene graph. Use 'all' if you want to return all objects. This function could be used to find the closest object(s) to the robot's current position.")
    def n_closest_objects_to_coordinate(self, coordinate: Annotated[str, "The coordinates of which you want to calculate the closest objects to (can be this of the robot). e.g. '[3.4, 8.1, 2.1]' "], n: Annotated[str, "The number of closest objects to return. Write 'all' if you want to return all objects."]) -> Annotated[List[str], "A list of the n closest objects to the given coordinates (sorted) and their distance from the given coordinates."]:
        """
        This function returns the IDs of the n closest objects to a given coordinate in the scene graph (sorted by distance).
        """
        scene_nodes = robot_state.scene_graph.nodes
        
        if n == "all":
            n = len(scene_nodes)
        else:
            try:
                n = int(n)
                if n <= 0:
                    return ["Error: n must be a positive integer or 'all'."]
            except ValueError:
                return [f"Error: Invalid value for n '{n}'. Must be a positive integer or 'all'."]
            
        # Parse the input coordinate string once before the loop
        try:
            target_coord_np = np.array(ast.literal_eval(coordinate))
            if target_coord_np.shape != (3,):
                 raise ValueError("Coordinate must be a 3D point.")
        except (ValueError, SyntaxError, TypeError) as e:
            return [f"Error: Invalid input coordinate string '{coordinate}': {e}"]

        # Calculate distances to all other nodes
        distances = []
        for node_id, node_data in scene_nodes.items():
            object_coordinates_np = node_data.centroid
            # Calculate distance directly using numpy
            distance = np.linalg.norm(object_coordinates_np - target_coord_np)
            distances.append((node_id, distance, node_data.sem_label))

        # Sort by distance
        distances.sort(key=lambda item: item[1])

        # Take the top n
        closest_n = distances[:n]

        # Format the response string using the label mapping
        response = []
        for node_id, dist, sem_label in closest_n:
            # Get the string label from the mapping
            label = robot_state.scene_graph.label_mapping.get(sem_label, f"Unknown Label ID: {sem_label}")
            response.append(f"Object ID: {node_id} - Semantic Label: {label} - Distance: {dist:.4f}")

        return response





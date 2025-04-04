from typing import List, Optional, Union, Dict, Any
from pydantic import BaseModel, Field

class SceneGraphObject(BaseModel):
    """A node in the scene graph."""
    sem_label: str = Field(description="The semantic label of the object.")
    object_id: Optional[Union[int, str]] = Field(description="The id of the object in the scene graph.")
    centroid: Optional[Union[List[float], str]] = Field(description="The centroid (list of 3 floats) of the object in the scene graph. Only provide the centroid if it is available.")
    movable: bool

class TaskResponse(BaseModel):
    """A task to be completed."""
    task_description: str = Field(description="A clear description of the task to be completed.")
    reasoning: str = Field(description="A concise reasoning behind the task, especially answering the 'why?' question.")
    # function_calls_involved: List[Union[str, Dict[str, Any]]] = Field(description="A list of function calls involved in completing the task, including their arguments.")
    # relevant_objects: Optional[List[SceneGraphObject]] = Field(description="A list of relevant objects from the scene graph that the robot could interact with to complete the task.")

class TaskPlannerResponse(BaseModel):
    """A response from the task planner agent."""
    tasks: List[TaskResponse] 
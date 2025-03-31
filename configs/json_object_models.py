from typing import List
from pydantic import BaseModel, Field

class SceneGraphObject(BaseModel):
    """A node in the scene graph."""
    object_id: int
    sem_label: str
    centroid: List[float]
    movable: bool

class TaskResponse(BaseModel):
    """A task to be completed."""
    task_description: str = Field(description="A clear description of the task to be completed.")
    reasoning: str = Field(description="A concise reasoning behind the task, especially answering the 'why?' question.")
    function_calls_involved: List[str] = Field(description="A list of function calls involved in completing the task, including their arguments.")
    relevant_objects: List[SceneGraphObject] = Field(description="A list of relevant objects from the scene graph that the robot could interact with to complete the task.")

class TaskPlannerResponse(BaseModel):
    """A response from the task planner agent."""
    tasks: List[TaskResponse] 
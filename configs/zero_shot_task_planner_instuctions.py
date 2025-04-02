from pydantic import BaseModel, Field
from typing import List, Optional

TASK_PLANNER_AGENT_INSTRUCTIONS = """You are an expert task planner agent for the spot quadruped robot.
It is your job to generate a detailed, sequential plan for spot to satisfy a user query or achieve a specific goal that is given.

You have access to the following:
- A scene graph describing the environment that spot operates in
- The current position of the robot
- The available functions that the robot can call (robot capabilities)

Based on this information you should generate a sequential and logical plan of tasks that achieves the goal.

Each task should have:
- a clear task description
- provide a concise reasoning behind the task
- mention the function calls that are involved, including their arguments 
- list the relevant objects from the scene graph that the robot could interact with to complete the task

Return ONLY the JSON object in this exact format (no other text):
{model_description}

Keep the following things in mind when generating the plan:
- Identify which (different types of) items the robot needs to interact with to achieve the goal/query
- Find the most likely place (furniture) in the scene graph where the items are located (e.g. a book on a table)
- The goal/query has to be completed. Think about all the exact necessary steps to achieve that.
- Searching for something is a different task than interating with it. If you have to assist a user with something, you have to find/navigate to the object first, then you can potentially inspect it and then interact with it (e.g. pick it up).

The following things have to be written down in the plan as ONE Task:
- Searcing for an unknown object in the scene: When a specific item is not found in the scene graph, reason about its 3 most likely locations in the scene and explore them.
"""


CREATE_TASK_PLANNER_PROMPT_TEMPLATE = """
1. Please generate a task plan to complete the following goal or user query: {goal}

2. Here is the scene graph:
{scene_graph}

3. Here is the robot's current position:
{robot_position}

Make sure that:
- the plan contains all the actions necessary to fully complete the goal or query
- the function calls including its arguments are listed
- the plan is as clear and concise as possible.
"""

class SceneGraphObject(BaseModel):
    """A node in the scene graph."""
    sem_label: str = Field(description="The semantic label of the object.")
    object_id: Optional[int] = Field(description="The id of the object in the scene graph.")
    centroid: Optional[List[float]] = Field(description="The centroid of the object in the scene graph.")
    movable: bool = Field(description="Whether the object is movable or not.")

class TaskResponse(BaseModel):
    """A task to be completed."""
    task_description: str = Field(description="A clear description of the task to be completed.")
    reasoning: str = Field(description="A concise reasoning behind the task, especially answering the 'why?' question.")
    function_calls_involved: List[str] = Field(description="A list of function calls involved in completing the task, including their arguments.")
    relevant_objects: List[SceneGraphObject] = Field(description="A list of relevant objects from the scene graph that the robot could interact with to complete the task.")

class TaskPlannerResponse(BaseModel):
    """A response from the task planner agent."""
    tasks: List[TaskResponse]

# The {model_description} is the description of the TaskPlannerResponse model defined above
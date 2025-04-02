from typing import Dict, Optional, List, Any
from pydantic import BaseModel, model_validator
from datetime import datetime

from configs.json_object_models import TaskPlannerResponse


########################################################

class ToolCall(BaseModel):
    """Logs for a tool call made by an agent."""
    tool_call_name: str
    tool_call_arguments: Dict
    tool_call_result: Optional[str] = None
    
class AgentResponse(BaseModel):
    """Logs for an agent response. Must contain exactly one content type."""
    text_content: Optional[str] = None
    tool_call_content: Optional[ToolCall] = None

    @model_validator(mode='before')
    @classmethod
    def check_exclusive_content(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure exactly one content field is populated."""
        content_fields = [
            'text_content',
            'tool_call_content',
        ]
        # Count how many content fields are present (not None)
        present_fields_count = sum(1 for field in content_fields if values.get(field) is not None)

        # Raise error if count is not exactly 1
        if present_fields_count > 1:
            present_fields = [field for field in content_fields if values.get(field) is not None]
            raise ValueError(
                f"AgentResponse can not have more than one content field set. "
                f"Found {present_fields_count} fields set: {present_fields}"
            )
        return values

class AgentResponseLogs(BaseModel):
    """Logs to track the full response of an agent invokation"""
    request: str
    plan_id: Optional[int] = None
    agent_responses: List[AgentResponse]
    agent_invocation_start_time: datetime
    agent_invocation_end_time: datetime
    agent_invocation_duration_seconds: float
    
########################################################

class PlanGenerationLogs(BaseModel):
    """Logs for the plan generation."""
    plan_id: int
    plan: TaskPlannerResponse
    plan_generation_start_time: datetime
    plan_generation_end_time: datetime
    plan_generation_duration_seconds: float
    issue_description: Optional[str] = None
    chain_of_thought: Optional[str] = None
    
class TaskPlannerAgentLogs(BaseModel):
    """Logs for the task planner agent."""
    ai_service_id: str
    total_replanning_count: int
    initial_plan: PlanGenerationLogs
    updated_plans: List[PlanGenerationLogs]
    task_planner_invocations: List[AgentResponseLogs]
    
########################################################

class TaskExecutionLogs(BaseModel): 
    """Logs for a task."""
    task_description: str
    reasoning: str
    plan_id: int # the id of the plan that this task belongs to
    relevant_objects_identified_by_planner: List[str]
    agent_invocation: AgentResponseLogs
    completed: bool = False

# What do we do when the task is not completed?
# What happens when we replan? 
# In case of replanning -> Task not successfull, mark the replanning time as the end
# Asking the task planner for extra information -> keep the timer going 

class TaskExecutionAgentLogs(BaseModel):
    """Logs for the task execution agent."""
    ai_service_id: str
    task_logs: List[TaskExecutionLogs]

########################################################

class GoalCompletionCheckerLogs(BaseModel):
    """Logs for a completion check."""
    completion_check_requested_by_agent: str
    completion_check_request: str
    completion_check_agent_invocation: AgentResponseLogs
    completion_check_final_response: str

class GoalCompletionCheckerAgentLogs(BaseModel):
    """Logs for the goal completion checker agent."""
    ai_service_id: str
    completion_check_logs: List[GoalCompletionCheckerLogs]
    
########################################################

class GoalExecutionLogs(BaseModel):
    """Logs for one goal execution."""
    goal: str
    goal_number: int
    complexity: int
    ambiguity: int
    dataset_name: Optional[str] = None
    start_time: datetime
    end_time: datetime
    duration_seconds: float
    task_planner_agent: TaskPlannerAgentLogs
    task_execution_agent: TaskExecutionAgentLogs
    goal_completion_checker_agent: GoalCompletionCheckerAgentLogs

    
    
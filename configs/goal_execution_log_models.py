from typing import Dict, Optional, List
from pydantic import BaseModel
from datetime import datetime

from configs.json_object_models import TaskPlannerResponse

class ToolCall(BaseModel):
    """Logs for a tool call made by an agent."""
    tool_call_name: str
    tool_call_arguments: Dict
    tool_call_reasoning: str # Why did this tool get called?
    tool_call_response: str  
    tool_call_start_time: datetime
    tool_call_end_time: datetime
    tool_call_duration_seconds: float
    
########################################################

class PlanGenerationLogs(BaseModel):
    """Logs for the plan generation."""
    plan: TaskPlannerResponse
    plan_generation_start_time: datetime
    plan_generation_end_time: datetime
    plan_generation_duration_seconds: float
    plan_generation_reasoning: Optional[str] = None
    chain_of_thought: Optional[str] = None
    
class TaskPlannerAgentLogs(BaseModel):
    """Logs for the task planner agent."""
    ai_service_id: str
    initial_plan: PlanGenerationLogs
    updated_plans: List[PlanGenerationLogs]
    total_replanning_count: int
    tool_calls: List[ToolCall]
    
########################################################

class TaskExecutionLogs(BaseModel):
    """Logs for a task."""
    task_description: str
    reasoning: str
    task_start_time: datetime
    task_end_time: datetime
    task_duration_seconds: float
    tool_calls_made: List[ToolCall]
    relevant_objects: List[str]
    completed: bool = False

class TaskExecutionAgentLogs(BaseModel):
    """Logs for the task execution agent."""
    ai_service_id: str
    task_logs: List[TaskExecutionLogs]

########################################################

class GoalCompletionCheckerLogs(BaseModel):
    """Logs for a completion check."""
    completion_check_requested_by_agent: str
    completion_check_request: str
    completion_check_response: str
    completion_check_start_time: datetime
    completion_check_end_time: datetime
    completion_check_duration_seconds: float

class GoalCompletionCheckerAgentLogs(BaseModel):
    """Logs for the goal completion checker agent."""
    ai_service_id: str
    completion_check_logs: List[GoalCompletionCheckerLogs]
    
########################################################
class GoalExecutionLogs(BaseModel):
    """Logs for one goal execution."""
    goal: str
    goal_number: int
    start_time: datetime
    end_time: datetime
    duration_seconds: float
    task_planner_agent: TaskPlannerAgentLogs
    task_execution_agent: TaskExecutionAgentLogs
    goal_completion_checker_agent: GoalCompletionCheckerAgentLogs

    
    
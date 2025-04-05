import pandas as pd
from typing import List, Tuple
from configs.goal_execution_log_models import GoalExecutionLogsCollection

def generate_task_summary_tables_per_goal(
    goal_logs_collection: GoalExecutionLogsCollection,
) -> List[Tuple[int, pd.DataFrame, str]]:
    """
    Generates a list of pandas DataFrames and their LaTeX representations,
    one table per goal, summarizing executed tasks for that goal.

    Args:
        goal_logs_collection: The collection of goal execution logs.

    Returns:
        A list of tuples, where each tuple contains:
            - goal_number (int): The number of the goal.
            - df_tasks (pd.DataFrame): DataFrame with task details for the goal.
            - latex_table (str): LaTeX string representation of the table for the goal.
    """
    all_tables_data = []

    for goal_log in goal_logs_collection.goal_execution_logs:
        goal_number = goal_log.goal_number
        goal_description = goal_log.goal # Get the goal description
        table_data_goal = []

        if not goal_log.task_execution_agent.task_logs:
            # Handle cases where a goal might have no tasks logged
            # Optionally, create an empty table or skip this goal
            print(f"Goal {goal_number} has no tasks logged, skipping table generation.")
            continue # Skip to the next goal

        for task_log in goal_log.task_execution_agent.task_logs:
            response_list = []
            if task_log.agent_invocation.agent_responses:
                n_responses = len(task_log.agent_invocation.agent_responses)
                for i, agent_response in enumerate(task_log.agent_invocation.agent_responses):
                    tool_call_str = None
                    text_str = None
                    if agent_response.tool_call_content:
                        tool_name = agent_response.tool_call_content.tool_call_name
                        tool_args = agent_response.tool_call_content.tool_call_arguments
                        tool_args_str = ', '.join([f'{k}={v}' for k, v in tool_args.items()])
                        tool_call_str = f'{tool_name}({tool_args_str})'
                    elif agent_response.text_content:
                        # Shorten text for display
                        text_preview = agent_response.text_content[:30] + '...' if len(agent_response.text_content) > 30 else agent_response.text_content
                        if i == n_responses - 1: # Final response
                            text_str = f'Final Response: {text_preview}'
                        else:
                            text_str = f'Text: {text_preview}'
                    response_list.append(tool_call_str or text_str or "Empty Response")
            else:
                response_list.append("No agent response logged")

            # Join the responses with <br> for HTML display
            formatted_responses_html = '<br>'.join(response_list)

            table_data_goal.append({
                "Task Description": task_log.task_description,
                "Tool Calls / Responses": formatted_responses_html, # Use the <br>-joined string for DataFrame
                "Plan ID": task_log.plan_id,
                "Inference Time (s)": f"{task_log.agent_invocation.agent_invocation_duration_seconds:.3f}" # Format time
            })

        # After processing all tasks, add the final goal completion check response
        completion_checker_logs = goal_log.goal_completion_checker_agent.completion_check_logs
        if completion_checker_logs:
            last_check_log = completion_checker_logs[-1] # Get the last check log
            if last_check_log.completion_check_agent_invocation.agent_responses:
                last_checker_response = last_check_log.completion_check_agent_invocation.agent_responses[-1]
                if last_checker_response.text_content:
                    checker_time = last_check_log.completion_check_agent_invocation.agent_invocation_duration_seconds
                    # Use text content directly, assuming it doesn't need <br>
                    checker_response_text = last_checker_response.text_content.strip()
                    table_data_goal.append({
                        "Task Description": "Final Goal Completion Check",
                        "Tool Calls / Responses": checker_response_text, # Use plain text
                        "Plan ID": "N/A",
                        "Inference Time (s)": f"{checker_time:.3f}"
                    })
                else:
                     print(f"Goal {goal_number}: Last completion check response had no text content.")
            else:
                 print(f"Goal {goal_number}: Last completion check invocation had no agent responses.")
        else:
             print(f"Goal {goal_number}: No completion check logs found.")

        # Create DataFrame for the current goal (uses <br> for display)
        df_tasks_goal = pd.DataFrame(table_data_goal)

        # Prepare DataFrame for LaTeX output by replacing <br> with \n
        df_latex = df_tasks_goal.copy()
        # Ensure the column exists and has string data before replacing
        if 'Tool Calls / Responses' in df_latex.columns:
            df_latex['Tool Calls / Responses'] = df_latex['Tool Calls / Responses'].astype(str).str.replace('<br>', '\n', regex=False)
        
        # Generate LaTeX representation from the modified DataFrame
        latex_table_goal = df_latex.to_latex(
            index=False,
            caption=f'Summary of Executed Tasks for Goal {goal_number}: {goal_description}',
            label=f'tab:task_summary_goal_{goal_number}',
            column_format='p{6cm}p{5cm}cc',
            escape=True, # escape=True handles \n correctly
            header=["Task Description", "Tool Call / Response", "Plan ID", "Time (s)"]
        )

        all_tables_data.append((goal_number, df_tasks_goal, latex_table_goal))

    return all_tables_data 
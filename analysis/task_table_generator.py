import pandas as pd
import json
import os
import sys
from IPython.display import display, HTML
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
                tool_invocations = 0
                for i, agent_response in enumerate(task_log.agent_invocation.agent_responses):
                    tool_call_str = None
                    text_str = None
                    if agent_response.tool_call_content:
                        tool_invocations += 1
                        tool_name = agent_response.tool_call_content.tool_call_name
                        tool_args = agent_response.tool_call_content.tool_call_arguments
                        tool_args_str = ', '.join([f'{k}={v}' for k, v in tool_args.items()])
                        tool_call_str = f'{tool_invocations}. {tool_name}({tool_args_str})'
                        
                    elif agent_response.text_content:
                        continue
                        # Shorten text for display
                        text_preview = agent_response.text_content[:50] + '...' if len(agent_response.text_content) > 30 else agent_response.text_content
                        if i == n_responses - 1: # Final response
                            text_str = f'Final Response: {text_preview}'
                        else:
                            text_str = f'{text_preview}'
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
                    checker_response_text = last_checker_response.text_content.strip()
                    table_data_goal.append({
                        "Task Description": "Final Goal Completion Check",
                        "Tool Calls / Responses": checker_response_text, # Keep checker response plain
                        "Plan ID": "---", # Changed from "N/A"
                        "Inference Time (s)": f"{checker_time:.3f}"
                    })
                else:
                     print(f"Goal {goal_number}: Last completion check response had no text content.")
            else:
                 print(f"Goal {goal_number}: Last completion check invocation had no agent responses.")
        else:
             print(f"Goal {goal_number}: No completion check logs found.")


        # Create DataFrame for the current goal (uses numbered <br> for display)
        df_tasks_goal = pd.DataFrame(table_data_goal)

        # # --- Prepare for LaTeX Output ---
        df_latex = df_tasks_goal.copy()
        
        # Ensure the column exists and has string data before processing
        if 'Tool Calls / Responses' in df_latex.columns:
            # Convert to string type first
            df_latex['Tool Calls / Responses'] = df_latex['Tool Calls / Responses'].astype(str)
            # Replace <br> used for HTML numbering with LaTeX's \newline for explicit breaks within p{} columns
            df_latex['Tool Calls / Responses'] = df_latex['Tool Calls / Responses'].str.replace('<br>', ' \\newline ', regex=False)
            # Manually escape underscores for LaTeX
            df_latex['Tool Calls / Responses'] = df_latex['Tool Calls / Responses'].str.replace('_', '\\_', regex=False)

        # Generate LaTeX representation from the modified DataFrame
        # NOTE: Requires \usepackage{array} in LaTeX preamble for p{} columns.
        latex_table_goal = df_latex.to_latex(
            index=False,
            caption=f'Summary of Executed Tasks for Goal {goal_number}: {goal_description}',
            label=f'tab:task_summary_goal_{goal_number}',
            # Use p{width} columns for text wrapping and specify widths.
            column_format='@{}p{0.20\\linewidth}p{0.60\\linewidth}p{0.08\\linewidth}l@{}',
            escape=False, # Set escape to False as we handle underscores manually
            # Longtable can help if the table spans multiple pages
            # longtable=True, # Uncomment if needed
            header=["Task Description", "Tool Call / Response", "Plan ID", "Time (s)"],
            multicolumn=True,
            multicolumn_format='c'
        )

        # --- Post-process LaTeX string to add \midrule between ALL rows --- 
        lines = latex_table_goal.strip().split('\n')
        processed_lines = []

        # # Keep \toprule
        # processed_lines.append(lines[0]) 

        n_tasks = len(goal_log.task_execution_agent.task_logs)
        
        task_idx = 0
        # Process lines between toprule and bottomrule
        for line in lines:
            # Skip the default midrule added by pandas
            if line.strip() == '\\midrule':
                continue
           
            # Add the actual content line (header or data row)
            processed_lines.append(line)
            # If the line added was a row (ends with \\), add a midrule after it
            if line.strip().endswith('\\'):
                task_idx += 1
                if task_idx <= n_tasks + 1:
                    processed_lines.append('\\midrule')
                

        # Add \bottomrule
        # processed_lines.append(lines[-1])
        
        latex_table_goal_with_midrules = '\n'.join(processed_lines)
        # --------------------------------------------------------------------

        all_tables_data.append((goal_number, df_tasks_goal, latex_table_goal_with_midrules))

    return all_tables_data 

if __name__ == "__main__":
    path_for_log_analysis = '/local/home/zhamers/L-SARP/data_scene/SEMANTIC_CORNER_WITH_BED/execution_logs_goals_12_full_run_1_20250405_032700.json'
    # Load the JSON data from the file
    with open(path_for_log_analysis) as f:
        raw_logs_data = json.load(f)

    # Extract the log entries (values) and put them in the expected structure
    formatted_logs_data = {"goal_execution_logs": list(raw_logs_data.values())}
    
    # Validate the formatted data using the Pydantic model
    goal_execution_logs_collection = GoalExecutionLogsCollection.model_validate(formatted_logs_data)

    # Generate the list of tables (one per goal)
    # The function now returns a list of tuples: (goal_number, dataframe, latex_string)
    list_of_tables = generate_task_summary_tables_per_goal(goal_execution_logs_collection)

    # Display/print each table
    for goal_num, df_goal_summary, latex_goal_summary in list_of_tables:
        # Print a header for the notebook output
        print(f"\n--- Goal {goal_num} Task Execution Summary Table ---")
        display(HTML(df_goal_summary.to_html(escape=False, index=False)))

        # Print the LaTeX representation for this goal's table
        print(f"\n\nLaTeX Representation for Goal {goal_num}:\n")
        print(latex_goal_summary)
        print("-" * 80) # Separator line for readability
        break
import os
import json
import logging
from typing import List

from utils.logging_utils import setup_logging
from utils.recursive_config import Config
from configs.goal_execution_log_models import GoalExecutionLogsCollection

config = Config()
setup_logging()

logger = logging.getLogger(__name__)

# Get the project root directory
project_root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
path_to_scene_data = config.get('robot_planner_settings').get('path_to_scene_data')
logs_path = config.get('robot_planner_settings').get('path_for_log_analysis')

# Get the path for log analysis
path_for_log_analysis = os.path.join(project_root_dir, path_to_scene_data, logs_path)
logger.info(f"Path for log analysis: {path_for_log_analysis}")

# Load the JSON data from the file
with open(path_for_log_analysis) as f:
    raw_logs_data = json.load(f)

# Extract the log entries (values) and put them in the expected structure
formatted_logs_data = {"goal_execution_logs": list(raw_logs_data.values())}

# Validate the formatted data using the Pydantic model
goal_execution_logs_collection = GoalExecutionLogsCollection.model_validate(formatted_logs_data)


###########################
# Get the standard
###########################


# Analyse the correlation between




# Zoom into the failure cases by visualizing the plans made, actions taken and what the goal completion checker said





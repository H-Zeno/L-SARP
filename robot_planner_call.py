#region Imports
import asyncio
import json
import yaml
import logging
import logging.config
from pathlib import Path
from dotenv import dotenv_values
from datetime import datetime

from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion, OpenAIChatPromptExecutionSettings
from semantic_kernel.connectors.ai.function_choice_behavior import FunctionChoiceBehavior
from semantic_kernel.contents.function_result_content import FunctionResultContent
from semantic_kernel.contents.function_call_content import FunctionCallContent
from semantic_kernel.contents import ChatHistory
from semantic_kernel.functions import KernelArguments

from source.utils.logging_utils import setup_logging
from source.utils.agent_utils import invoke_agent_group_chat, invoke_agent
from configs.scenes_and_plugins_config import Scene
from planner_core.robot_planner import RobotPlanner
from configs.plugin_configs import plugin_configs
#endregion

#region Loading Loggers
logger_plugins, logger_main = setup_logging()
#endregion

#region Config Loading
with open("configs/config.yaml", "r") as f:
    config = yaml.safe_load(f)
#endregion

async def main():
  
    # region Scene Setup
    active_scene_name = config["robot_planner_settings"]["active_scene"]
    active_scene = Scene[active_scene_name]
    if active_scene not in Scene:
        raise ValueError(f"Selected active scene '{active_scene}' (mentioned in config.yaml) not found in Scene enum")
    logger_main.info(f"Loading robot planner configurations for scene: '{active_scene.value}'")

    path_to_scene_data = Path(config["robot_planner_settings"]["path_to_scene_data"])
    if not path_to_scene_data.exists():
        raise FileNotFoundError(f"Scene data directory not found at {path_to_scene_data}")
    # endregion Scene Setup

    #region Robot Planner
    robot_planner = RobotPlanner(scene=active_scene)
    robot_planner.setup_services()
    robot_planner.add_plugins()
    robot_planner.initialize_task_generation_agent()
    robot_planner.initialize_task_execution_agent()
    robot_planner.initialize_goal_completion_checker_agent()

    robot_planner_group_chat =robot_planner.setup_agent_group_chat([robot_planner.task_generation_agent, robot_planner.task_execution_agent, robot_planner.goal_completion_checker_agent])

    ### Online Live Instruction ###
    if config["robot_planner_settings"]["task_instruction_mode"] == "online_live_instruction":
        initial_prompt = f"""
        Hey, I'm Spot, your intelligent robot assistant. Currently I am located in the {active_scene.value} scene. I am happy to help you with:
        - Navigating throught the environment
        - Understand the scene
        - Execute tasks and actions
        
        What goal would you like me to achieve?
        """
        pass
    
    ### Offline Predefined Goals ###
    elif config["robot_planner_settings"]["task_instruction_mode"] == "offline_predefined_instruction":
        # Process existing predefined goals
        goals_path = Path("configs/goals.json")  # Convert string to Path object
        # Create timestamp for the responses file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        responses_path = path_to_scene_data / f"responses_{timestamp}.json"
        responses = {}
        separator = "======================="

        # Load the predefined goals and potential responses (if already generated)
        try:
            if not goals_path.exists():
                raise FileNotFoundError(f"Goals file not found at {goals_path}")
            
            with open(goals_path, "r") as file:
                try:
                    goals = json.load(file)
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON format in goals file: {e}")

            if responses_path.exists():
                with responses_path.open("r") as file:
                    responses = json.load(file)

        except Exception as e:
            logger_main.error(f"Error loading goals: {e}")
            raise
        
        # Handle each Offline Predefined Goal
        for nr, goal in goals.items():
            if nr in responses:
                continue
            goal_text = f"{nr}. {goal}"

            logger_plugins.info(separator)
            logger_plugins.info(goal_text)
            logger_main.info(separator)
            logger_main.info(goal_text)

            try:
                response, robot_planner_group_chat = await invoke_agent_group_chat(robot_planner_group_chat, goal_text)
                
                responses[nr] = {
                    "goal": goal_text,
                    "response": response,
                }

                with responses_path.open("w") as file:
                    json.dump(responses, file, indent=4)

            except Exception as e:
                error_str = f"Error processing goal {goal_text}: {e}"
                logger_plugins.error(error_str)
                logger_main.error(error_str)

        logger_main.info("Finished processing Offline Predefined Goals")

    else:
        raise ValueError(f"Invalid task instruction mode: {config['robot_planner_settings']['task_instruction_mode']}")
    #endregion Robot Planner


    # Querying the room (asking questions) should just be one plugin
    # This means that each plugins needs to contain multiple plugins with itself

    # Plugins Factory get's

    # The robot needs to get its abilities (plugins) and then effectively plan based on multimodal inputs from various sources

    # Based on the specific scene or environement, we might need to give other capabilities to the robot (too many might reduce performance, inference time overhead, etc.)
    # the plugins need to be added to the kernel

    # So the kernel gets now instantiated in the RAG3DCHAT (robot_planner)

    # We have to add the relevant plugins to the kernel based on the configurations of the scene


    # I have to create 3 agents: 1. that determines the next (sub)task to complete the goal, 2. One that selects actions to complete this task and 3. An agent that determines if the overall goal is completed yet or not

    # The game state/environment state (in text form) that is given to the task generation agent always needs to be up-to-date with our scene graph representation
    # There are several ways to highlight the most relevant parts of the scene graph: 
    # 1. CLIP embedding similarity
    # 2. RAG on descriptions of nodes in the scene graph
    # 3. The nodes of the scene graph that are in the field of view should be highlighted/get more attention
    # The nodes in the field of view should be automatically enterred in a text description that makes up the robot state
if __name__ == "__main__":
    asyncio.run(main())

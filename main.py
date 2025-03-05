import asyncio
import json
import os
import yaml
import logging
import logging.config
from pathlib import Path
from dotenv import dotenv_values
from datetime import datetime
import atexit
import re

from source.utils.agent_utils import invoke_agent_group_chat, invoke_agent
from source.utils.recursive_config import Config

from configs.scenes_and_plugins_config import Scene
from planner_core.robot_planner import RobotPlanner
from planner_core.robot_state import RobotState

from LostFound.src.scene_graph import get_scene_graph
from LostFound.src.utils import scene_graph_to_json

from semantic_kernel.contents.chat_history import ChatHistory

# Set up logging - this will be the single source of logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logging.getLogger("kernel").setLevel(logging.DEBUG)

logger = logging.getLogger(__name__)

# Set debug level based on config or environment variable
debug = True  # This could be moved to config
if debug:
    logging.getLogger().setLevel(logging.DEBUG)


#region Config Loading
config = Config()
#endregion


async def main():
  
    # region Scene Setup
    active_scene_name = config["robot_planner_settings"]["active_scene"]
    active_scene = Scene[active_scene_name]
    if active_scene not in Scene:
        raise ValueError(f"Selected active scene '{active_scene}' (mentioned in config.yaml) not found in Scene enum")
    logger.info(f"Loading robot planner configurations for scene: '{active_scene.value}'")

    path_to_scene_data = Path(config["robot_planner_settings"]["path_to_scene_data"])
    if not path_to_scene_data.exists():
        raise FileNotFoundError(f"Scene data directory not found at {path_to_scene_data}")
    # endregion Scene Setup

    # region Load the Scene Graph
    # base_path = config.get_subpath("prescans")
    # ending = config["pre_scanned_graphs"]["high_res"]
    # SCAN_DIR = os.path.join(base_path, ending)

    # scene_graph = get_scene_graph(SCAN_DIR, drawers=True, light_switches=True)

    scene_graph_json = json.load(open('/local/home/zhamers/L-SARP/data/3D-Scene-Understanding/scene_graph.json'))
    scene_graph_string = json.dumps(scene_graph_json)

    # scene_graph.visualize(labels=True, connections=True, centroids=True)
    # endregion Load the Scene Graph

    #region Robot State
    # robot_state = RobotState(scene_graph=scene_graph)
    # robot_state.connect_to_spot(config)

    # # Register a cleanup handler to ensure proper shutdown
    # atexit.register(robot_state.cleanup)
    #endregion Robot State

    #region Robot Planner
    robot_planner = RobotPlanner(scene=active_scene)
    robot_planner.setup_services()
    robot_planner.add_retrieval_plugins()
    robot_planner.add_action_plugins()
    robot_planner.initialize_task_planner_agent()
    robot_planner.initialize_task_execution_agent()
    robot_planner.initialize_goal_completion_checker_agent()

    robot_task_completion_group_chat =robot_planner.setup_task_completion_group_chat()

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
            logger.error(f"Error loading goals: {e}")
            raise
        
        # Handle each Offline Predefined Goal
        for nr, goal in goals.items():
            if nr in responses:
                continue
            goal_text = f"{nr}. {goal}"

            logger.info(separator)
            logger.info(goal_text)

            # 1. We simply take the goal and ask the task execution agent to complete this task

            task_planning_prompt_template = """
            1. Please generate a plan to complete the following goal: {goal}

            2. Tasks completed so far:
            {tasks_completed}

            3. Here is the scene graph:
            {scene_graph}

            4. Here is the robot's current position:
            (0, 0, 0)
            """

            task_completion_prompt_template = """
            It is your job to complete the following task: {task}
            
            This task is part of the following plan: {plan}

            You have access to the following information to reason on how to complete the task:
            1. An up-to-date scene graph representation of the environment
            2. The current location of the robot in the environment

            Here is the scene graph:
            {scene_graph}
            
            Here is the robot's current position:
            (0, 0, 0)
            """
            
            # We need an "okay to follow plan" agent that checks the current action and gives the go-ahead

            valid_plan = False
            tasks_completed = []
            planning_chat_history = ChatHistory()
            while True:
                
                # Create the plan
                if not valid_plan:
                    plan_generation_prompt = task_planning_prompt_template.format(goal=goal_text, tasks_completed=', '.join(map(str, tasks_completed)), scene_graph=scene_graph_string)
                    plan_response, planning_chat_history = await invoke_agent(robot_planner.task_planner_agent, plan_generation_prompt, chat_history=planning_chat_history)
                    logger.info(f"Raw plan response: {str(plan_response)}")

                    # Extract JSON from the response using regex
                    json_match = re.search(r'```json\s*(\{[\s\S]*?\})\s*`*', plan_response, re.DOTALL)
                    if json_match:
                        try:
                            current_plan = json.loads(json_match.group(1))
                            logger.info(f"Extracted plan: {json.dumps(current_plan, indent=2)}")
                        except json.JSONDecodeError as e:
                            logger.error(f"Failed to parse JSON from response: {e}")
                            current_plan = None
                    else:
                        logger.error("No JSON found in the response")
                        raise ValueError("No JSON found in the response")

                robot_planner.plan = current_plan
                for task in current_plan["tasks"]:
                    robot_planner.task = task
                    task_execution_prompt = task_completion_prompt_template.format(task=task, plan=current_plan, scene_graph=scene_graph_string)

                    task_completion_response, robot_task_completion_group_chat = await invoke_agent_group_chat(robot_task_completion_group_chat, task_execution_prompt)
                



                # task_completion_prompt = task_completion_prompt_template.format(goal=goal_text, scene_graph=scene_graph_string)


            #     response, robot_planner_group_chat = await invoke_agent_group_chat(robot_planner_group_chat, task_completion_prompt)
                
            #     responses[nr] = {
            #         "goal": goal_text,
            #         "response": response,
            #     }

            #     with responses_path.open("w") as file:
            #         json.dump(responses, file, indent=4)

            # except Exception as e:
            #     error_str = f"Error processing goal {goal_text}: {e}"
            #     logger.error(error_str)

        logger.info("Finished processing Offline Predefined Goals")

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

# Standard library imports
import asyncio
import json
import logging
import logging.config
import os
from datetime import datetime
from pathlib import Path

# Third-party imports
import yaml
from dotenv import dotenv_values
from semantic_kernel.contents.chat_history import ChatHistory
from bosdyn import client as bosdyn_client

# Local imports
from configs.scenes_and_plugins_config import Scene
from planner_core.robot_planner import RobotPlanner
from planner_core.robot_state import RobotState, RobotStateSingleton
from LostFound.src.scene_graph import get_scene_graph
from utils.agent_utils import invoke_agent, invoke_agent_group_chat
from utils.recursive_config import Config
from robot_utils.base_LSARP import initialize_robot_connection, spot_initial_localization, power_on, safe_power_off

from utils.singletons import RobotLeaseClientSingleton
from planner_core.robot_planner import RobotPlannerSingleton
from robot_utils.frame_transformer import FrameTransformerSingleton

# Initialize singletons
robot_state = RobotStateSingleton()
robot_planner = RobotPlannerSingleton()
robot_lease_client = RobotLeaseClientSingleton()
frame_transformer = FrameTransformerSingleton()

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

config = Config()

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

    base_path = config.get_subpath("prescans")
    ending = config["pre_scanned_graphs"]["high_res"]
    SCAN_DIR = os.path.join(base_path, ending)

    # Loading/computing the scene graph
    scene_graph_path = Path(path_to_scene_data/ active_scene_name / "full_scene_graph.pkl")
    scene_graph_json_path = Path(path_to_scene_data/ active_scene_name / "scene_graph.json")
    logging.info(f"Loading scene graph from {SCAN_DIR}. This may take a few seconds...")
    scene_graph = get_scene_graph(SCAN_DIR, graph_save_path=scene_graph_path, drawers=False, light_switches=True, vis_block=True)
    scene_graph.save_as_json(scene_graph_json_path)

    
    # Process existing predefined goals
    goals_path = Path("configs/goals.json")  # Convert string to Path object
    # Create timestamp for the responses file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # responses_path = path_to_scene_data / f"responses_{timestamp}.json"
    responses = {}
    separator = "======================="

    ############################################################
    # Start the connection to the robot
    ############################################################
    initialize_robot_connection()   

    with bosdyn_client.lease.LeaseKeepAlive(
        robot_lease_client, must_acquire=True, return_at_exit=True
    ):
        power_on()
        spot_initial_localization()

         # Initialze the robot state with the scene graph
        robot_planner.set_instance(RobotPlanner(scene=active_scene))
        robot_state.set_instance(RobotState(scene_graph_object=scene_graph))

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
            # responses_path = path_to_scene_data / f"responses_{timestamp}.json"
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

                # if responses_path.exists():
                #     with responses_path.open("r") as file:
                #         responses = json.load(file)

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

                task_completion_prompt_template = """
                It is your job to complete the following task: {task}
                
                This task is part of the following plan: {plan}

                The following tasks have already been completed: {tasks_completed}

                Here is the scene graph:
                {scene_graph}
                
                Here is the robot's current position:
                {robot_position}
                """

                # Set the goal and create the initial plan
                time_before = datetime.now()
                initial_plan, chain_of_thought = await robot_planner.create_task_plan_from_goal(goal)
                time_after = datetime.now()
                inference_time = time_after - time_before
                logger.info(f"Time taken for inference: {inference_time}")

                goal_response = {'inference_time (seconds)': str(inference_time.seconds), 'zero_shot_plan': initial_plan, 'chain_of_thought': chain_of_thought}

                # Save the initial plan that is generated
                plan_gen_save_path = Path(path_to_scene_data / active_scene_name / "initial_plans.json")
                plan_gen_save_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Create the empty json file if it does not exist
                if not plan_gen_save_path.exists():
                    with open(plan_gen_save_path, 'w') as file:
                        json.dump({}, file)

                with open(plan_gen_save_path, 'r') as file:
                    existing_data = json.load(file)
                    existing_goal_responses = existing_data.get(goal, {})
                    existing_goal_responses[robot_planner.task_planner_agent.service_id] = goal_response
                    existing_data[goal] = existing_goal_responses
                    new_data = json.dumps(existing_data, indent=2)

                with open(plan_gen_save_path, 'w') as file:
                    file.write(new_data)

                # Main Agentic Loop
                while True:
                    
                    task_completion_chat_history = ChatHistory()
                    # Execute the plan of action, with new
                    for task in robot_planner.plan["tasks"]:
                        robot_planner.task = task
                        logger.info(f"{separator}\nExecuting task: {task} \n{separator}")
                        print(f"This is the current robot frame: {robot_state.frame_name}")

                        # Execute the task
                        task_execution_prompt = task_completion_prompt_template.format(task=task, 
                                                                                        plan=robot_planner.plan, 
                                                                                        tasks_completed=robot_planner.tasks_completed,
                                                                                        scene_graph=str(scene_graph.scene_graph_to_dict()),
                                                                                        robot_position=str(frame_transformer.get_current_body_position_in_frame(robot_state.frame_name)))
                                                                                        
                        task_completion_response, task_completion_chat_history  = await invoke_agent(robot_planner.task_execution_agent, 
                                                                                                        chat_history=task_completion_chat_history,
                                                                                                        input_text_message=task_execution_prompt, 
                                                                                                        input_image_message=robot_state.get_current_image_content(),
                                                                                                        debug=debug)
                        
                        logger.info(f"Task completion response: {task_completion_response}")


                        # Break out of the task execution loop when replanning
                        if robot_planner.replanned == True:
                            robot_planner.replanned = False
                            break

                        robot_planner.tasks_completed.append(task)
                
                        # Activate the goal completion checker agent (small quick model)

            logger.info("Finished processing Offline Predefined Goals")

        else:
            raise ValueError(f"Invalid task instruction mode: {config['robot_planner_settings']['task_instruction_mode']}")


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
        safe_power_off()

if __name__ == "__main__":
    asyncio.run(main())

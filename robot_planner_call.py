#region Imports
import asyncio
import json
import yaml
import logging
import logging.config
from pathlib import Path
from dotenv import dotenv_values

from semantic_kernel.utils.settings import openai_settings_from_dot_env
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion

from planner_core.robot_planner import RobotPlanner
from planner_core.config_handler import ConfigHandler
from planner_core.model_factories import OpenAiChatModelFactory, OpenAiModelFactory
from configs.scenes_enum import Scene
from plugins.plugins_factory import PluginsFactory
#endregion

#region Logging Setup
logging.config.fileConfig("configs/logging_conf.ini")
logger_plugins = logging.getLogger("plugins")
logger_main = logging.getLogger("main")
#endregion

#region Config Loading
with open("configs/config.yaml", "r") as f:
    config = yaml.safe_load(f)
#endregion

async def main():
    #region Initializing Plugins
    plugins_dotenv = Path(".env_plugins")
    config_handler = ConfigHandler(plugins_dotenv)
    chat_model_factory = OpenAiChatModelFactory(config_handler)
    model_factory = OpenAiModelFactory(config_handler) # These model factories are used for the plugins (capabilities) of our robot
    #endregion Initializing Plugins

    #region Kernel Services Setup (AI Model)
    settings = dotenv_values(".env")
    kernel_service = OpenAIChatCompletion(
        ai_model_id=settings["PLANNER_CORE_LLM_MODEL_NAME"],
        api_key=settings["PLANNER_CORE_API_KEY"],
        org_id=settings.get("PLANNER_CORE_ORG_ID"),
        service_id="default",
        default_headers={"api-key": settings["PLANNER_CORE_API_KEY"]} if settings.get("PLANNER_CORE_API_KEY") else None
    )
    #endregion Kernel Services Setup (AI Model)

    #region Scene and Plugin Setup
    active_scene = Scene(config["robot_planner_settings"]["active_scene"])
    if active_scene not in Scene:
        raise ValueError(f"Selected active scene '{active_scene}' (mentioned in config.yaml) not found in Scene enum")
    logger_main.info(f"Loading robot planner configurations for scene: '{active_scene.value}'")

    # Define plugin configurations (based on the scene)
    path_to_scene_data = Path(config["robot_planner_settings"]["path_to_scene_data"])
    if not path_to_scene_data.exists():
        raise FileNotFoundError(f"Scene data directory not found at {path_to_scene_data}")

    plugins_factory = PluginsFactory(model_factory, chat_model_factory)
    plugin_configs = {
        "nav": (
            plugins_factory.get_nav_plugin,
            [
                path_to_scene_data / Path(f"{active_scene.value}/nav_data/navmesh.txt"),
                None  # nav_vis_path is None
            ],
            "navigation"
        ),
        "text": (
            plugins_factory.get_text_plugin,
            [
                path_to_scene_data / Path(f"{active_scene.value}/text_data"),
                Path(f".TEXT_DIR/{active_scene.value}")
            ],
            "text"
        ),
        "sql": (
            plugins_factory.get_sql_plugin,
            [
                path_to_scene_data / Path(f"{active_scene.value}/sql_data/sql_db_data.json"),
                Path(f".SQL_DIR/{active_scene.value}")
            ],
            "sql"
        ),
        "image": (
            plugins_factory.get_image_plugin,
            [
                path_to_scene_data / Path(f"{active_scene.value}/img_data"),
                Path(f".IMAGE_DIR/{active_scene.value}")
            ],
            "image"
        )
    }
    #endregion Scene and Plugin Setup

    #region Robot Planner
    try:
        robot_planner = RobotPlanner(
            kernel_service=kernel_service,
            enabled_plugins=active_scene.plugins,
            plugin_configs=plugin_configs
        )
        robot_planner.set_kernel()
    except Exception as e:
        logger_main.error(f"Failed to initialize robot planner: {str(e)}")
        raise

    # Process Offline Predefined Instructions (good for testing and benchmarking)
    if config["robot_planner_settings"]["task_instruction_mode"] == "offline_predefined_instruction":
        instructions_path = path_to_scene_data / "instructions.json"
        responses_path = path_to_scene_data / "responses.json"
        responses = {}
        separator = "======================="

        # Load the predefined instructions and potential responses (if already generated)
        try:
            if not instructions_path.exists():
                raise FileNotFoundError(f"Instructions file not found at {instructions_path}")
            
            with open(instructions_path, "r") as file:
                try:
                    instructions = json.load(file)
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON format in instructions file: {e}")

            if responses_path.exists():
                with responses_path.open("r") as file:
                    responses = json.load(file)

        except Exception as e:
            logger_main.error(f"Error loading instructions: {e}")
            raise
        
        # Handle each Offline Predefined Instruction
        for nr, ins in instructions.items():
            if nr in responses:
                continue
            instruction = f"{nr}. {ins}"

            logger_plugins.info(separator)
            logger_plugins.info(instruction)
            logger_main.info(separator)
            logger_main.info(instruction)

            try:
                final_response, generated_plan = await robot_planner.invoke_planner(instruction)

                logger_main.info(final_response)
                logger_plugins.info(final_response)
                logger_plugins.info("---")
                logger_plugins.info(generated_plan)

                responses[nr] = final_response
                responses[nr]["plan"] = generated_plan

                with responses_path.open("w") as file:
                    json.dump(responses, file, indent=4)

            except Exception as e:
                error_str = f"Error with instruction {instruction}: {e}"
                logger_plugins.error(error_str)
                logger_main.error(error_str)

        logger_main.info("Finished processing Offline Predefined Instructions")

    elif config["robot_planner_settings"]["task_instruction_mode"] == "online_live_instruction":
        # TODO: Implement Online Live Instructions
        pass
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


if __name__ == "__main__":
    asyncio.run(main())
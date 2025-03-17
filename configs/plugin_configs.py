import yaml
import logging
from pathlib import Path

from planner_core.model_factories import OpenAiChatModelFactory, OpenAiModelFactory

from retrieval_plugins.plugins_factory import PluginsFactory
from planner_core.config_handler import ConfigHandler
from configs.scenes_and_plugins_config import Scene

plugins_dotenv = Path(".env_plugins")
config_handler = ConfigHandler(plugins_dotenv)

chat_model_factory = OpenAiChatModelFactory(config_handler)
model_factory = OpenAiModelFactory(config_handler) # These model factories are used for the plugins (capabilities)

#region Config Loading
with open("configs/config.yaml", "r") as f:
    config = yaml.safe_load(f)
#endregion

# Just get the logger, configuration is handled in main.py
logger = logging.getLogger(__name__)

# Load the configurations for this specific scene
active_scene_name = config["robot_planner_settings"]["active_scene"]
active_scene = Scene[active_scene_name]
if active_scene not in Scene:
    raise ValueError(f"Selected active scene '{active_scene}' (mentioned in config.yaml) not found in Scene enum")
logger.info(f"Loading robot planner configurations for scene: '{active_scene.value}'")

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
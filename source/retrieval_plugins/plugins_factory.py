from pathlib import Path
from typing import Optional

from llama_index.embeddings.clip import ClipEmbedding

from retrieval_plugins.text_plugin import TextPlugin
from retrieval_plugins.sql_plugin import SqlPlugin
from retrieval_plugins.image_plugin import ImagePlugin
from retrieval_plugins.nav_plugin import NavPlugin
from planner_core.config_handler import ConfigPrefix
from planner_core.interfaces import AbstractModelFactory, AbstractLlmChatFactory


class PluginsFactory:
    def __init__(
        self,
        model_factory: AbstractModelFactory,
        llm_chat_factory: AbstractLlmChatFactory,
    ) -> None:
        """
        Constructor

        Args:
            model_factory (AbstractModelFactory): factory for creating models (LLMs, embeddings)
            llm_chat_factory (AbstractLlmChatFactory): factory for creating LLM chats
        """
        self._model_factory: AbstractModelFactory = model_factory
        self._chat_model_factory: AbstractLlmChatFactory = llm_chat_factory

    def get_sql_plugin(
        self, json_file_path: Optional[Path] = None, persist_dir: Optional[Path] = None
    ) -> SqlPlugin:
        """
        Creates an instance of the SQL plugin.

        Args:
            json_file_path (Optional[Path]): path to the JSON file with SQL data (detections and room information)
            persist_dir (Optional[Path]): path to the directory where the plugin's context stores/will store its data

        Returns:
            SqlPlugin: instance of the SQL plugin
        """
        sql_llm = self._model_factory.get_llm_model(ConfigPrefix.SQL)
        sql_embed = self._model_factory.get_embed_model(ConfigPrefix.SQL)
        return SqlPlugin(sql_llm, sql_embed, json_file_path, persist_dir)

    def get_image_plugin(
        self, image_dir: Optional[Path] = None, persist_dir: Optional[Path] = None
    ) -> ImagePlugin:
        """
        Creates an instance of the image plugin.

        Args:
            image_dir (Optional[Path]): path to the directory with input images
            persist_dir (Optional[Path]): path to the directory where the plugin's context stores/will store its data

        Returns:
            ImagePlugin: instance of the image plugin
        """
        image_llm = self._model_factory.get_multimodal_llm_model(ConfigPrefix.IMAGES)
        image_embed = ClipEmbedding()
        chat_llm = self._chat_model_factory.get_llm_chat()
        return ImagePlugin(image_llm, chat_llm, image_embed, image_dir, persist_dir)

    def get_text_plugin(
        self, text_dir: Optional[Path] = None, persist_dir: Optional[Path] = None
    ) -> TextPlugin:
        """
        Creates an instance of the text plugin.

        Args:
            text_dir (Optional[Path]): path to the directory with input text files
            persist_dir (Optional[Path]): path to the directory where the plugin's context stores/will store its data

        Returns:
            TextPlugin: instance of the text plugin
        """
        llm_text = self._model_factory.get_llm_model(ConfigPrefix.TEXT)
        text_embed = self._model_factory.get_embed_model(ConfigPrefix.TEXT)
        llm_chat = self._chat_model_factory.get_llm_chat()
        return TextPlugin(llm_text, text_embed, llm_chat, text_dir, persist_dir)

    def get_nav_plugin(
        self, navmesh_path: Path, vis_dir_path: Optional[Path] = None
    ) -> NavPlugin:
        """
        Creates an instance of the navigation plugin.

        Args:
            navmesh_path (Path): path to the navmesh file
            vis_dir_path (Optional[Path]): path to the directory where the plugin will save visualizations of the results

        Returns:
            NavPlugin: instance of the navigation plugin
        """
        llm_nav = self._model_factory.get_llm_model(ConfigPrefix.NAVIGATION)
        return NavPlugin(llm_nav, navmesh_path, vis_dir_path)

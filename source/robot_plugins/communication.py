from semantic_kernel.functions import kernel_function
import datetime
import os
from typing import Annotated
import json
from pathlib import Path
import yaml
from typing import Union
import re
import emoji
import logging
from semantic_kernel.functions.kernel_arguments import KernelArguments
from utils.recursive_config import Config

config = Config()
logger = logging.getLogger("plugins")
# Set debug level based on config
debug = config.get("robot_planner_settings", {}).get("debug", True)
if debug:
    logger.setLevel(logging.DEBUG)

class CommunicationPlugin:
    """A plugin that can be used to communicate with the user"""
    def __init__(self):
        self.debug = True
        logger.info(f"CommunicationPlugin initialized")
    def remove_unnecessary_characters(self, text: Annotated[str, "The text to clean"]) -> str:
        # Remove repeated exclamation marks, question marks, and other symbols
        text = re.sub(r'[!]+', r'!', text)
        text = re.sub(r'[?]+', r'?', text)
        # Remove any repeated whitespace characters (spaces, tabs, etc.)
        text = re.sub(r'\s+', ' ', text)
        # Remove trailing whitespace
        text = text.strip()
        # Remove emojis from the text - using updated emoji module API
        text = emoji.replace_emoji(text, '')
        # Remove last dot or comma since it is probably not needed
        while len(text) > 1 and text[-1] in [",", ".", " "]:
            text = text[:-1]
        return text

    # Here the communication with the WhatsApp API should be implemented

    @kernel_function(description="Function used to inform the user about certain important events in the application process")
    async def inform_user(self, message: Annotated[str, "The message to inform the user about"]) -> str:
        try:
            # Fallback to console
            print('--------------------------------')
            print(message)
            print('--------------------------------')
            return message
        except Exception as e:
            logger.error(f"Error in inform_user: {e}")
            return message

    @kernel_function(description="Function used to ask the user for information")
    async def ask_user(self, question: Annotated[str, "The question to ask the user"]) -> str:
        print("ASKING USER")
        print('--------------------------------')
        response = input(question)
        print('--------------------------------')
        
        return self.remove_unnecessary_characters(response)
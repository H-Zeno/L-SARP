import logging
from semantic_kernel.contents.chat_message_content import ChatMessageContent
from semantic_kernel.contents.utils.author_role import AuthorRole
from semantic_kernel.functions.kernel_arguments import KernelArguments
from semantic_kernel.contents.function_call_content import FunctionCallContent
from semantic_kernel.contents.function_result_content import FunctionResultContent
from semantic_kernel.contents.chat_history import ChatHistory
from semantic_kernel.agents import AgentGroupChat, ChatCompletionAgent
from typing import Tuple

# Just get the logger, configuration is handled in main.py
logger = logging.getLogger(__name__)

def _write_content(content: ChatMessageContent, debug: bool = False) -> None:
    """Write the content to the console."""
    last_item_type = type(content.items[-1]).__name__ if content.items else "(empty)"
    message_content = ""
    if isinstance(last_item_type, FunctionCallContent):
        message_content = f"tool request = {content.items[-1].function_name}"
    elif isinstance(last_item_type, FunctionResultContent):
        message_content = f"function result = {content.items[-1].result}"
    else:
        message_content = str(content.items[-1])
    if debug:
        logger.debug(f"[{last_item_type}] {content.role} : '{message_content}'")


async def invoke_agent(
    agent: ChatCompletionAgent,
    input: str,
    chat_history: ChatHistory,
    save_to_history: bool = True,
    debug: bool = False
    ) -> Tuple[ChatMessageContent, ChatHistory]:
    """Invoke the agent with the user input."""
    if debug:
        logger.debug(f"# {AuthorRole.USER}: '{input}'")
    if save_to_history:
        chat_history.add_user_message(input)

    async for content in agent.invoke(chat_history):
        if not any(isinstance(item, (FunctionCallContent, FunctionResultContent)) for item in content.items):
            if save_to_history:
                if debug:
                    logger.debug(f"# This text is now being saved to the chat history: '{content.content}'")
                chat_history.add_message(content)
        _write_content(content)
    return content.content, chat_history


async def invoke_agent_group_chat(
    group_chat: AgentGroupChat, 
    input_message: str, 
    save_to_history: bool = True,
    debug: bool = True
) -> Tuple[str, AgentGroupChat]:
    """
    Invoke the agent group chat with an input message and collect the response.
    
    Args:
        group_chat (AgentGroupChat): The group chat to invoke
        input_message (str): Message to send to the group chat
        save_to_history (bool): Whether to save the message to chat history
        debug (bool): Whether to print debug messages
        
    Returns:
        tuple: (response string, updated group chat)
    """
    if debug:
        logger.debug(f"# {AuthorRole.USER}: '{input_message}'")
    
    if save_to_history:
        await group_chat.add_chat_message(
            ChatMessageContent(role=AuthorRole.USER, content=input_message)
        )
        
    response = ""
    async for content in group_chat.invoke():
        response += content.content
        if debug:
            logger.debug(f"### DEBUG: Group chat content added to response: {content.content}")

    if debug:
        logger.debug(f"### DEBUG: Group chat response: {response}")

    if save_to_history:
        await group_chat.add_chat_message(
            ChatMessageContent(role=AuthorRole.ASSISTANT, content=response)
        )

    return response, group_chat
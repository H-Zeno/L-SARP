# Standard library imports
import logging
from typing import Tuple

# Third-party imports
from semantic_kernel.agents import AgentGroupChat, ChatCompletionAgent
from semantic_kernel.contents import ChatHistory, ChatMessageContent, ImageContent, TextContent
from semantic_kernel.contents.chat_message_content import ChatMessageContent
from semantic_kernel.contents.function_call_content import FunctionCallContent
from semantic_kernel.contents.function_result_content import FunctionResultContent
from semantic_kernel.contents.utils.author_role import AuthorRole
from semantic_kernel.functions.kernel_arguments import KernelArguments

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
    chat_history: ChatHistory,
    input_text_message: str,
    input_image_message: ImageContent = None,
    save_to_history: bool = True,
    debug: bool = False
    ) -> Tuple[ChatMessageContent, ChatHistory]:
    """Invoke the agent with the user input."""
    if debug:
        logger.debug(f"# {AuthorRole.USER}: '{input_text_message}'")
    
    if chat_history is None:
        raise ValueError("chat_history cannot be None")

    if save_to_history == False:
        init_history = chat_history

    if input_image_message is not None:
        chat_history.add_message(
            ChatMessageContent(role=AuthorRole.USER, items=[TextContent(text=input_text_message), input_image_message])
        )
    else:
        chat_history.add_message(
            ChatMessageContent(role=AuthorRole.USER, content=input_text_message)
        )

    async for content in agent.invoke(chat_history):
        if not any(isinstance(item, (FunctionCallContent, FunctionResultContent)) for item in content.items):
            if debug:
                logger.debug(f"# This text is now being saved to the chat history: '{content.content}'")
            chat_history.add_message(content)
        _write_content(content)

    if save_to_history == False:
        chat_history = init_history

    return content.content, chat_history


async def invoke_agent_group_chat(
    group_chat: AgentGroupChat, 
    input_text_message: str, 
    input_image_message: ImageContent = None,
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
        logger.debug(f"# {AuthorRole.USER}: '{input_text_message}'")
    
    if save_to_history == False:
        init_hitory = group_chat.history

    if input_image_message is not None:
        await group_chat.add_chat_message(
            ChatMessageContent(role=AuthorRole.USER, items=[TextContent(text=input_text_message), input_image_message])
        )
    else:
        await group_chat.add_chat_message(
            ChatMessageContent(role=AuthorRole.USER, content=input_text_message)
        )
        
    response = ""
    async for content in group_chat.invoke():
        response += content.content
        if debug:
            logger.debug(f"### DEBUG: Group chat content added to response: {content.content}")

    if debug:
        logger.debug(f"### DEBUG: Group chat response: {response}")
    
    await group_chat.add_chat_message(
        ChatMessageContent(role=AuthorRole.ASSISTANT, content=response)
    )

    if save_to_history == False:
        group_chat.history = init_hitory

    return response, group_chat
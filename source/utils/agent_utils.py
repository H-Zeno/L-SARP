# Standard library imports
import logging
from typing import Optional, Tuple, AsyncIterator, Union

# Third-party imports
from semantic_kernel.agents import AgentGroupChat, ChatCompletionAgent, ChatHistoryAgentThread
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
        logger.debug("[%s] %s : '%s'", last_item_type, content.role, message_content)


async def invoke_agent(
    agent: ChatCompletionAgent,
    input_text_message: str,
    input_image_message: ImageContent = None,
    thread: ChatHistoryAgentThread = None,
    chat_history: ChatHistory = None,
    save_to_history: bool = True,
    debug: bool = False,
    arguments: KernelArguments = None
    ) -> Tuple[str, Union[ChatHistory, ChatHistoryAgentThread]]:
    """
    Invoke the agent with the user input.
    
    Args:
        agent (ChatCompletionAgent): The agent to invoke
        chat_history (ChatHistory, optional): Chat history to use if not using thread
        input_text_message (str): Text message to send to the agent
        input_image_message (ImageContent, optional): Image content to send with the message
        save_to_history (bool): Whether to save the conversation to history
        debug (bool): Whether to print debug messages
        thread (ChatHistoryAgentThread, optional): Chat thread to use (preferred over chat_history)
        arguments (KernelArguments, optional): Additional arguments to pass to the agent
        
    Returns:
        tuple: (response content, updated chat history or thread)
    """
    if debug:
        logger.debug(f"# {AuthorRole.USER}: '{input_text_message}'")
    
    # Create message with text and optional image
    message = None
    if input_text_message is not None:
        if input_image_message is not None:
            message = [TextContent(text=input_text_message), input_image_message]
        else:
            message = input_text_message
    
    # Using thread (preferred) or chat_history
    if thread is not None:
        # Use the thread-based API
        if not save_to_history:
            # Create a copy of messages for temporary use
            orig_messages = await thread.get_messages()
        
        response = await agent.get_response(messages=message, thread=thread, arguments=arguments)
        _write_content(response)
        
        if not save_to_history:
            # Restore original messages
            thread = await thread.create_new(orig_messages.messages)
            
        return response.content, response.thread
        
    else:
        # Using chat_history (legacy approach)
        if chat_history is None:
            chat_history = ChatHistory()
            
        if not save_to_history:
            init_history = ChatHistory()
            for msg in chat_history.messages:
                init_history.add_message(msg)
                
        # Add user message to history
        if input_image_message is not None:
            chat_history.add_message(
                ChatMessageContent(role=AuthorRole.USER, items=[TextContent(text=input_text_message), input_image_message])
            )
        else:
            chat_history.add_message(
                ChatMessageContent(role=AuthorRole.USER, content=input_text_message)
            )

        response_content = None
        # Invoke the agent and capture the last response
        async for content in agent.invoke(chat_history, arguments=arguments):
            chat_history.add_message(content)
            _write_content(content, debug)
            response_content = content.content

        if not save_to_history:
            chat_history = init_history

        return response_content, chat_history


async def invoke_agent_group_chat(
    group_chat: AgentGroupChat, 
    input_text_message: str, 
    input_image_message: ImageContent = None,
    save_to_history: bool = True,
    debug: bool = True,
    agent: ChatCompletionAgent = None
) -> Tuple[str, AgentGroupChat]:
    """
    Invoke the agent group chat with an input message and collect the response.
    
    Args:
        group_chat (AgentGroupChat): The group chat to invoke
        input_text_message (str): Message to send to the group chat
        input_image_message (ImageContent, optional): Image content to send with the message
        save_to_history (bool): Whether to save the message to chat history
        debug (bool): Whether to print debug messages
        agent (ChatCompletionAgent, optional): Specific agent to target in the group chat
        
    Returns:
        tuple: (response string, updated group chat)
    """
    
    if debug:
        logger.debug(f"# {AuthorRole.USER}: '{input_text_message}'")
    
    init_history = None
    if save_to_history is False:
        if hasattr(group_chat.history, 'clone'):
            init_history = group_chat.history.clone()
        else:
            init_history = group_chat.history

    # Add user message to chat
    if input_image_message is not None:
        await group_chat.add_chat_message(
            ChatMessageContent(role=AuthorRole.USER, items=[TextContent(text=input_text_message), input_image_message])
        )
    else:
        await group_chat.add_chat_message(
            ChatMessageContent(role=AuthorRole.USER, content=input_text_message)
        )
        
    response = ""
    
    # Invoke the group chat and collect responses
    async for content in group_chat.invoke(agent=agent):
        response += content.content
        if debug:
            logger.debug(f"### DEBUG: Group chat content added to response: {content.content}")

    if debug:
        logger.debug(f"### DEBUG: Group chat response: {response}")
    
    # Add the response to history
    await group_chat.add_chat_message(
        ChatMessageContent(role=AuthorRole.ASSISTANT, content=response)
    )

    if save_to_history is False and init_history is not None:
        group_chat.history = init_history

    return response, group_chat
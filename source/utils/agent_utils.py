# Standard library imports
import logging
import json
from datetime import datetime
from typing import Tuple

from configs.goal_execution_log_models import ToolCall

# Third-party imports
from semantic_kernel.agents import AgentGroupChat, ChatCompletionAgent, ChatHistoryAgentThread
from semantic_kernel.contents import ChatMessageContent, ImageContent, TextContent
from semantic_kernel.contents.function_call_content import FunctionCallContent
from semantic_kernel.contents.function_result_content import FunctionResultContent
from semantic_kernel.contents.utils.author_role import AuthorRole
from semantic_kernel.functions.kernel_arguments import KernelArguments
from utils.recursive_config import Config


config = Config()
logger = logging.getLogger("main")

def _write_content(content: ChatMessageContent) -> None:
    """Write the content items to the console."""
    if not content.items:
        logger.debug("#DEBUG (write content): [(empty)] %s : '(no content)'", content.role)
        return

    # Iterate through all items in the message content
    for item in content.items:
        item_type_name = type(item).__name__
        message_content = ""
        log_level = logging.INFO # Default log level

        if isinstance(item, FunctionCallContent):
            # Log arguments as well for more detail
            args_str = ", ".join(f"{k}={v}" for k, v in item.arguments.items()) if item.arguments else ""
            # Include id for potential tracking/matching with results
            message_content = f"tool_request(id={item.id}) = {item.function_name}({args_str})"
            log_level = logging.INFO # Function calls are important INFO level
        elif isinstance(item, FunctionResultContent):
            # Potentially long results, maybe truncate or summarize later if needed. Log full for now.
            # Include id for potential tracking/matching with requests
            message_content = f"function_result(id={item.tool_call_id}) = {item.function_name} -> {item.result}"
            log_level = logging.INFO # Results are important INFO level
        elif isinstance(item, TextContent):
            message_content = str(item.text) # Access the text attribute
            log_level = logging.INFO # Keep existing text logging at INFO
        elif isinstance(item, ImageContent):
            message_content = "[ImageContent received]" # Don't log raw image data
            log_level = logging.DEBUG # Image content might be verbose for INFO
        else:
             # Log other types maybe at debug level?
             logger.debug("#Content: [%s] %s - (Unhandled item type in logger)", item_type_name, content.role)
             continue # Skip logging unknown types via the main logger.log call below

        # Log each processed item with its specific type and role
        logger.log(log_level, "#Content: [%s] %s : '%s'", item_type_name, content.role, message_content)


async def invoke_agent(
    agent: ChatCompletionAgent,
    thread: ChatHistoryAgentThread,
    input_text_message: str,
    input_image_message: ImageContent = None,
    save_to_history: bool = True,
    arguments: KernelArguments = None
    ) -> Tuple[str, ChatHistoryAgentThread]:
    """
    Invoke the agent with the user input.
    
    Args:
        agent (ChatCompletionAgent): The agent to invoke
        input_text_message (str): Text message to send to the agent
        input_image_message (ImageContent, optional): Image content to send with the message
        thread (ChatHistoryAgentThread, optional): Chat thread to use
        save_to_history (bool): Whether to save the conversation to history
        debug (bool): Whether to print debug messages
        arguments (KernelArguments, optional): Additional arguments to pass to the agent
        
    Returns:
        tuple: (response content, updated thread)
    """
    logger.info("Agent %s Invoked.", agent.name)
    logger.debug("Exact message sent to agent: %s", input_text_message)
    
    # Create message with text and optional image
    message = None
    if input_text_message is not None:
        if input_image_message is not None:
            # Create a proper ChatMessageContent with role for text+image
            message = ChatMessageContent(role=AuthorRole.USER, items=[TextContent(text=input_text_message), input_image_message])
        else:
            # Create a proper ChatMessageContent with role for text-only
            message = ChatMessageContent(role=AuthorRole.USER, content=input_text_message)
    
    # Save original messages if we shouldn't save to history
    if thread is not None:
        orig_chat_history = await thread.get_messages()
    else:
        orig_chat_history = None
    
    # Start time for tool call tracking
    start_time = datetime.now()
    
    response = await agent.get_response(messages=message, thread=thread, arguments=arguments)
    logger.debug("Raw response from agent: %s", response)
    
    # End time for tool call tracking
    end_time = datetime.now()
    
    # Log all content items, including function calls
    chat_history = await response.thread.get_messages()
    
    # Get the messages that were added during this invocation (the new ones)
    start_idx = len(orig_chat_history.messages) if orig_chat_history is not None else 0
    
    # Track function calls
    tool_calls = []
    for msg in chat_history.messages[start_idx:]:
        # Log the role of each message to better understand the conversation flow
        logger.debug("Processing message with role: %s", msg.role)
        
        # Process all messages regardless of role
        for item in msg.items:
            if isinstance(item, FunctionCallContent):
                logger.info("#Function Call: %s(%s)", item.function_name, item.arguments)
                tool_call_info = ToolCall(
                    tool_call_name=item.function_name,
                    tool_call_arguments=json.loads(item.arguments),
                    tool_call_reasoning="", # save the previous text message as reasoning
                    tool_call_response="", # save the next text message as response (possible, or the actual function result)
                    tool_call_start_time=start_time,
                    tool_call_end_time=end_time,
                    tool_call_duration_seconds=(end_time - start_time).total_seconds()
                )
                tool_calls.append(tool_call_info)

    # Log the final response content
    _write_content(response.content)
    
    # If we shouldn't save to history, create a new thread with the original messages
    if not save_to_history:
        # Create a new thread with the original messages
        new_thread = ChatHistoryAgentThread(chat_history=orig_chat_history, thread_id=thread._thread_id)
        await new_thread.create()
        thread = new_thread
        
    return response.content, response.thread


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
        logger.debug("# %s: '%s'", AuthorRole.USER, input_text_message)
    
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
            logger.debug("### DEBUG: Group chat content added to response: %s", content.content)

    logger.debug("### DEBUG: Group chat response: %s", response)
    
    # Add the response to history
    await group_chat.add_chat_message(
        ChatMessageContent(role=AuthorRole.ASSISTANT, content=response)
    )

    if save_to_history is False and init_history is not None:
        group_chat.history = init_history

    return response, group_chat
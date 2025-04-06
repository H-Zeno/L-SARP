# Standard library imports
import logging
import json
from datetime import datetime
from typing import Tuple, List

from configs.goal_execution_log_models import ToolCall, AgentResponse, AgentResponseLogs

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

def _log_agent_response(request: str, messages: List[ChatMessageContent], start_time: datetime, end_time: datetime) -> AgentResponseLogs:
    """Log all content items from a list of agent messages to the console.
    
    This function handles logging of:
    - Function calls (FunctionCallContent)
    - Function results (FunctionResultContent)
    - Text content (TextContent)
    - # TODO: Image content (ImageContent)
    - Other content types (logged at debug level)
    
    Args:
        request (str): The request to the agent
        messages (List[ChatMessageContent]): The messages from the agent
        start_time (datetime): The start time of the agent response
        end_time (datetime): The end time of the agent response
        
    Returns:
        AgentResponseLogs: The logs for the agent response
    """
    agent_responses = []
    for msg in messages:
        if not msg.items:
            logger.debug("#DEBUG (log response): [(empty)] %s : '(no msg)'", msg.role)
            continue
        
        logger.debug("Processing message with role: %s", msg.role)

        # Iterate through all items in the message msg
        for item in msg.items:
            item_type_name = type(item).__name__
            message_content = ""
            log_level = logging.INFO  # Default log level

            if isinstance(item, FunctionCallContent):
                # Log function calls with arguments and ID for tracking
                message_content = f"Tool Request (Function Call, id={item.id}) = {item.function_name}({item.arguments})"
                log_level = logging.INFO  # Function calls are important INFO level
                # logger.info("(DEBUG) Tool Request (Function Call, id=%s) = %s(%s)", item.id, item.function_name, item.arguments)
                
                # Handle arguments being either a string (needs loading) or already a dict
                parsed_arguments = {}
                if isinstance(item.arguments, str):
                    try:
                        parsed_arguments = json.loads(item.arguments)
                    except json.JSONDecodeError:
                        logger.error("Failed to parse arguments JSON string: %s", item.arguments)
                        # Decide how to handle invalid JSON - maybe log and continue?
                        # For now, we'll keep parsed_arguments as empty dict or raise error?
                elif isinstance(item.arguments, dict):
                    parsed_arguments = item.arguments
                else:
                    logger.warning("Unexpected type for item.arguments: %s", type(item.arguments))
                    # Handle unexpected type if necessary

                tool_call_info = ToolCall(
                    tool_call_name=item.function_name,
                    tool_call_arguments=parsed_arguments,
                )
                
            elif isinstance(item, FunctionResultContent):
                # Log function results with ID for tracking
                message_content = f"Tool Result (Function Result, id={item.id}) = {item.function_name} -> {item.result}"
                log_level = logging.INFO  # Results are important INFO level
                
                # ASSUMPTION: A function result always follows a tool call!!!! # ASSUMPTION
                agent_response = AgentResponse(
                    tool_call_content=tool_call_info,
                    tool_call_result=item.result,
                )
                agent_responses.append(agent_response)
                
                
            elif isinstance(item, TextContent):
                # Log text content
                message_content = str(item.text)
                log_level = logging.DEBUG  # Keep existing text logging at INFO
                
                agent_response = AgentResponse(
                    text_content=item.text,
                )
                agent_responses.append(agent_response)
                
            elif isinstance(item, ImageContent):
                # Log image content presence without raw data
                message_content = "[ImageContent received]"
                log_level = logging.DEBUG  # Image content might be verbose for INFO
                
                # TODO: Add image content to agent response
                
            else:
                # Log other types at debug level
                logger.info("[%s] %s - (Unhandled item type in logger)", item_type_name, msg.role)
                continue  # Skip logging unknown types via the main logger.log call below

            # Log each processed item with its specific type and role
            logger.log(log_level, "[%s] %s : '%s'", item_type_name, msg.role, message_content)

    agent_response_logs = AgentResponseLogs(
        request=request,
        agent_responses=agent_responses,
        agent_invocation_start_time=start_time,
        agent_invocation_end_time=end_time,
        agent_invocation_duration_seconds=(end_time - start_time).total_seconds()
    )
    
    return agent_response_logs


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
        tuple: (response content, updated thread, agent response logs)
    """
    logger.info("Agent %s Invoked.", agent.name)
    logger.debug("Exact message sent to agent: %s", input_text_message)
    
    # Create message with text and optional image
    message = None
    if input_text_message is not None:
        if input_image_message is not None and isinstance(input_image_message, ImageContent):
            # Create a proper ChatMessageContent with role for text+image
            message = ChatMessageContent(role=AuthorRole.USER, items=[TextContent(text=input_text_message), input_image_message])
        else:
            # Create a proper ChatMessageContent with role for text-only
            message = ChatMessageContent(role=AuthorRole.USER, content=input_text_message)
    
    # Save original messages if we shouldn't save to history
    orig_chat_history = None
    start_idx = 0
    if thread is not None:
        orig_chat_history = await thread.get_messages()
        start_idx = len(orig_chat_history.messages)

    # Start time for tool call tracking
    start_time = datetime.now()
    
    response = await agent.get_response(messages=message, thread=thread, arguments=arguments)
    logger.debug("Raw final response message content from agent: %s", response.content)
    
    # End time for tool call tracking
    end_time = datetime.now()
    
    # Get the full chat history after the invocation
    chat_history = await response.thread.get_messages()
    
    # Get the messages that were added during this invocation (the new ones)
    new_messages = chat_history.messages[start_idx+1:] # we ignore the request message, since we log this already
    
    # Log all new messages (including function calls, results, text, etc.)
    agent_response_logs = _log_agent_response(request=input_text_message, messages=new_messages, start_time=start_time, end_time=end_time)
    
    if not save_to_history and orig_chat_history is None:
        logger.debug("Message thread was empty when invoking agent, clearing all message history in the thread (save_to_history is False).")
        response.thread._chat_history.clear()
        
    # If we shouldn't save to history, restore the original thread state
    elif not save_to_history and orig_chat_history is not None:
        logger.debug("Restoring chat history as save_to_history is False.")
        response.thread._chat_history = orig_chat_history # Directly manipulating might be risky?
        # Consider if response.thread needs replacement: thread = new_thread_with_old_history
    
    return response.content, response.thread, agent_response_logs


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

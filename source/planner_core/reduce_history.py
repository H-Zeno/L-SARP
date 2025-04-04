import logging
from dotenv import dotenv_values
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
from semantic_kernel.connectors.ai.google.google_ai import GoogleAIChatCompletion
from semantic_kernel.contents.chat_history import ChatHistory
from semantic_kernel.contents.chat_message_content import ChatMessageContent
from semantic_kernel.contents.utils.author_role import AuthorRole
from semantic_kernel.exceptions.agent_exceptions import AgentThreadOperationException

from configs.agent_instruction_prompts import HISTORY_SUMMARY_REDUCER_INSTRUCTIONS
from utils.recursive_config import Config
from planner_core.robot_planner import RobotPlannerSingleton
robot_planner = RobotPlannerSingleton()

logger = logging.getLogger("main")
config = Config()
summary_kernel = Kernel()


if config.get("robot_planner_settings").get("history_reduction_model_id") == "gemini-2.0-flash":
    summary_kernel.add_service(GoogleAIChatCompletion(
                gemini_model_id="gemini-2.0-flash",
                api_key=dotenv_values(".env_core_planner").get("GOOGLE_API_KEY"),
    ))  
    
else:   
    summary_kernel.add_service(OpenAIChatCompletion(
        api_key=dotenv_values(".env_core_planner").get("OPENAI_API_KEY"),
        ai_model_id=config.get("robot_planner_settings").get("history_reduction_model_id")
    ))


# will only reduce every (threshold - untouched_messages - 1) messages
async def reduce_and_log_chat_history(chat_thread, thread_name, threshold=10, untouched_messages=3):
    """Helper function to reduce chat history and log the results."""
    try:
        initial_messages = await chat_thread.get_messages()
        initial_count = len(initial_messages.messages)
        
        if initial_count <= threshold:
            logger.info(f"@ {thread_name} History count is equal orbelow threshold of {threshold}: {initial_count}")
            return
        
        logger.info("History count above threshold, attempting to reduce...")
        logger.info(f"@ {thread_name} History count BEFORE reduction attempt: {initial_count}") # Log count before
        # logger.info(f"@ {thread_name} History (Before): {initial_messages.messages}")
        
        # Check that the last message is not a tool call 
        while initial_messages.messages[-untouched_messages].role == AuthorRole.TOOL:
            untouched_messages = untouched_messages + 1
            if untouched_messages > initial_count:
                logger.info("No non-tool messages found in the chat history. Exiting reduction.")
                return
        
        # Summarize all messages except the last 'untouched_messages'
        prompt = HISTORY_SUMMARY_REDUCER_INSTRUCTIONS.format(
            goal=robot_planner.goal, 
            plan=robot_planner.plan,
            tasks_completed=robot_planner.tasks_completed, 
            chat_history=initial_messages.messages[:-untouched_messages]
        )
        
        summary_result = await summary_kernel.invoke_prompt(prompt) # Added await
        summary_content = str(summary_result) # Extract string content from the result
        # logger.info(f"@ {thread_name} Summary: {summary_content}")
  
        # Create the new chat history: summary + newest untouched messages
        chat_history = ChatHistory()
        chat_history.add_message(ChatMessageContent(role=AuthorRole.USER, content=summary_content)) # Add summary as system message
        
        for msg in initial_messages.messages[-untouched_messages:]:
            chat_history.add_message(msg)
        
        # Restore logging for reduction status
        final_count = len(chat_history.messages)

        logger.info(f"@ {thread_name} Final Message Count AFTER reduction: {final_count}")
        
        chat_thread._chat_history = chat_history
        
    except AgentThreadOperationException:
        logger.warning(f"Could not reduce chat history for {thread_name} as the thread is not active.")
        # Optionally, still log the current message count if the thread object allows access
        try:
            chat_history = await chat_thread.get_messages()
            # Use final_count variable here too if needed, or recalculate
            final_count_except = len(chat_history.messages)
            logger.info(f"@ {thread_name} Final Message Count (reduction skipped): {final_count_except}\n")
        except Exception as e:
            logger.warning(f"Could not retrieve messages for {thread_name} after failed reduction: {e}")

import semantic_kernel as sk
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
from semantic_kernel.contents.chat_history import ChatHistory
import asyncio
import logging
from utils.user_input import (
    get_yes_no_answer,
    get_wanted_item_mask3d,
    get_n_word_answer,
    confirm_coordinates,
    confirm_move
)

logger_main = logging.getLogger("main")
logger_plugins = logging.getLogger("plugins")


class UserInterface:
    def __init__(self, kernel, robot_planner):
        self.kernel = kernel
        self.chat_history = ChatHistory()
        self.chat_completion = OpenAIChatCompletion()
        self.robot_planner = robot_planner
        
    async def start_conversation(self, initial_prompt=None):
        """Start an interactive conversation with the model"""
        if initial_prompt:
            print("\nAssistant: " + initial_prompt)
            self.chat_history.add_assistant_message(initial_prompt)

        separator = "======================="
        
        while True:
            # Get user input
            user_input = input("\nYou: ").strip()
            
            # Check for exit command
            if user_input.lower() in ['exit', 'quit', 'bye']:
                print("\nRobot Assistant: Goodbye!")
                break
                
            # Add user message to history
            self.chat_history.add_user_message(user_input)
            
            try:
                # Log the instruction
                logger_plugins.info(separator)
                logger_plugins.info(user_input)
                logger_main.info(separator)
                logger_main.info(user_input)
                
                # Use robot planner to process instruction
                final_response, generated_plan = await self.robot_planner.invoke_planner(user_input)
                
                # If plan requires user confirmation or input
                if "requires_confirmation" in generated_plan:
                    confirmed = get_yes_no_answer("Do you want to proceed with this action")
                    if not confirmed:
                        print("\nRobot Assistant: Action cancelled. What else can I help you with?")
                        continue
                
                if "requires_item_selection" in generated_plan:
                    item = get_wanted_item_mask3d()
                    # Update plan with selected item
                    generated_plan["selected_item"] = item
                
                if "requires_movement_confirmation" in generated_plan:
                    start_pose = generated_plan["start_pose"]
                    end_pose = generated_plan["end_pose"]
                    confirmed = confirm_move(start_pose, end_pose)
                    if not confirmed:
                        print("\nRobot Assistant: Movement cancelled. What else can I help you with?")
                        continue
                
                # Log and print responses
                logger_main.info(final_response)
                logger_plugins.info(final_response)
                logger_plugins.info("---")
                logger_plugins.info(generated_plan)
                
                # Print response to user
                print("\nRobot Assistant:", final_response)
                print("\nPlan:", generated_plan)
                
                # Add response to chat history
                self.chat_history.add_assistant_message(final_response)
                
            except Exception as e:
                error_str = f"Error processing instruction: {str(e)}"
                logger_plugins.error(error_str)
                logger_main.error(error_str)
                print(f"\nError: {error_str}")
                print("Please try again.")

def create_kernel():
    """Create and configure the semantic kernel"""
    kernel = sk.Kernel()
    
    # Add your kernel configuration here
    # For example:
    # kernel.add_chat_service("chat-gpt", OpenAIChatCompletion("gpt-3.5-turbo", api_key))
    
    return kernel

async def main():
    # Create and configure the kernel
    kernel = create_kernel()
    
    # Create the interface
    interface = UserInterface(kernel)
    
    # Start conversation with initial prompt
    initial_prompt = "Hello! I'm your AI assistant. How can I help you today?"
    await interface.start_conversation(initial_prompt)

if __name__ == "__main__":
    asyncio.run(main()) 
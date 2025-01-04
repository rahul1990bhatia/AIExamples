"""
Travel Assistant using LangGraph and OpenAI

This module implements an intelligent travel assistant that leverages OpenAI's language models
for natural language understanding and generation. The assistant can handle multiple users,
maintain context, and provide travel-related services through sophisticated intent
classification and information extraction.

Key Features:
- OpenAI-powered natural language understanding
- Contextual conversation management
- Multi-user support
- Sophisticated intent classification
- Smart information extraction
"""

from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime
from enum import Enum
from dataclasses import dataclass
from langgraph.graph import Graph, StateGraph
from langgraph.prebuilt.tool_executor import ToolExecutor
import operator
from typing import Annotated, Sequence, TypedDict, Union
import openai
import json

from dotenv import load_dotenv

_ = load_dotenv()

class Intent(Enum):
    """
    Enumeration of possible conversation intents.
    Additional intents can be easily added to extend functionality.
    """
    FLIGHT_BOOKING = "flight_booking"
    HOTEL_BOOKING = "hotel_booking"
    GENERAL_INQUIRY = "general_inquiry"
    ITINERARY_PLANNING = "itinerary_planning"
    DESTINATION_INFO = "destination_info"
    BUDGET_PLANNING = "budget_planning"
    UNKNOWN = "unknown"

class UserState:
    """
    Maintains the state and context for a single user's conversation.
    
    Attributes:
        user_id (str): Unique identifier for the user
        context (list): List of previous interactions and their context
        current_intent (Intent): Current conversation intent
        collected_info (dict): Information collected during the conversation
        last_interaction (datetime): Timestamp of the last interaction
    """
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.context = []
        self.current_intent = None
        self.collected_info = {}
        self.last_interaction = datetime.now()

    def get_conversation_history(self) -> str:
        """
        Format conversation history for context injection into prompts.
        """
        history = []
        for interaction in self.context[-5:]:  # Last 5 interactions for context
            history.append(f"User ({interaction['timestamp']}): {interaction['message']}")
        return "\n".join(history)

@dataclass
class Message:
    """
    Data class representing a user message.
    """
    user_id: str
    content: str
    timestamp: datetime = None

class TravelAssistant:
    """
    Advanced travel assistant using OpenAI for natural language understanding.
    """
    
    def __init__(self):
        self.users: Dict[str, UserState] = {}
        self.graph = self._create_graph()
        
        # Initialize prompts
        self.intent_classification_prompt = """
        You are a travel assistant analyzing user messages to determine their intent.
        Based on the message and conversation history, classify the intent into one of the following categories:
        - flight_booking: User wants to book a flight
        - hotel_booking: User wants to book accommodation
        - itinerary_planning: User wants help planning a trip
        - destination_info: User wants information about a destination
        - budget_planning: User wants help with travel budgeting
        - general_inquiry: General travel-related questions
        - unknown: Intent cannot be determined

        Conversation History:
        {history}

        Current Message: {message}

        Respond with just the intent category name.
        """

        self.information_extraction_prompt = """
        Extract relevant travel information from the user's message.
        Return a JSON object with any of the following fields if found:
        - departure_city
        - destination_city
        - departure_date
        - return_date
        - number_of_travelers
        - budget_range
        - hotel_preferences
        - specific_requirements

        Message: {message}
        """

        self.response_generation_prompt = """
        You are a helpful travel assistant. Generate a natural and helpful response
        based on the current context and information.

        Conversation History:
        {history}

        Current Intent: {intent}
        Collected Information: {collected_info}
        Missing Information: {missing_info}

        Generate a response that helps move the conversation forward and collect
        necessary information in a natural way.
        """

    def _create_graph(self) -> StateGraph:
        """
        Create and configure the conversation workflow graph with OpenAI-powered nodes.
        """
        workflow = StateGraph(Message)

        # Add nodes for the workflow
        workflow.add_node("intent_classifier", self._classify_intent)
        workflow.add_node("information_extractor", self._extract_information)
        workflow.add_node("context_manager", self._manage_context)
        workflow.add_node("response_generator", self._generate_response)

        # Set the entry point
        workflow.set_entry_point("intent_classifier")

        # Define the flow
        workflow.add_edge("intent_classifier", "information_extractor")
        workflow.add_edge("information_extractor", "context_manager")
        workflow.add_edge("context_manager", "response_generator")

        return workflow.compile()

    def _classify_intent(self, state: Dict) -> Dict:
        """
        Use OpenAI to classify the user's intent based on message content and context.
        """
        message = state["message"]
        user_state = self.users[message.user_id]
        
        # Prepare the prompt with conversation history
        prompt = self.intent_classification_prompt.format(
            history=user_state.get_conversation_history(),
            message=message.content
        )

        # Call OpenAI API
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": message.content}
            ],
            temperature=0.3
        )

        # Extract intent from response
        intent_str = response.choices[0].message.content.strip().lower()
        try:
            intent = Intent(intent_str)
        except ValueError:
            intent = Intent.UNKNOWN

        return {"message": message, "intent": intent}

    def _extract_information(self, state: Dict) -> Dict:
        """
        Use OpenAI to extract relevant information from the user's message.
        """
        message = state["message"]
        
        # Prepare the prompt
        prompt = self.information_extraction_prompt.format(
            message=message.content
        )

        # Call OpenAI API
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": message.content}
            ],
            temperature=0.3
        )

        # Parse extracted information
        try:
            extracted_info = json.loads(response.choices[0].message.content)
        except json.JSONDecodeError:
            extracted_info = {}

        return {**state, "extracted_info": extracted_info}
    
    def _create_user_if_not_exists(self, user_id: str) -> None:
        """
        [Previous method documentation remains the same...]
        """
        if user_id not in self.users:
            self.users[user_id] = UserState(user_id)

    def _manage_context(self, state: Dict) -> Dict:
        """
        Update conversation context with extracted information.
        """
        user_id = state["message"].user_id
        intent = state["intent"]
        extracted_info = state["extracted_info"]
        
        user_state = self.users[user_id]
        user_state.current_intent = intent
        
        # Update collected information
        user_state.collected_info.update(extracted_info)
        
        # Add to context history
        user_state.context.append({
            "timestamp": datetime.now(),
            "intent": intent,
            "message": state["message"].content,
            "extracted_info": extracted_info
        })
        
        return {**state, "user_state": user_state}

    def _generate_response(self, state: Dict) -> Dict:
        """
        Generate a natural language response using OpenAI based on current context.
        """
        user_state = state["user_state"]
        intent = state["intent"]
        
        # Determine missing information based on intent
        missing_info = self._get_missing_info(intent, user_state.collected_info)
        
        # Prepare the prompt
        prompt = self.response_generation_prompt.format(
            history=user_state.get_conversation_history(),
            intent=intent.value,
            collected_info=json.dumps(user_state.collected_info, indent=2),
            missing_info=json.dumps(missing_info, indent=2)
        )

        # Call OpenAI API
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": state["message"].content}
            ],
            temperature=0.7
        )

        return {"response": response.choices[0].message.content}

    def _get_missing_info(self, intent: Intent, collected_info: Dict) -> List[str]:
        """
        Determine what information is still needed based on the intent.
        """
        required_info = {
            Intent.FLIGHT_BOOKING: ["departure_city", "destination_city", "departure_date"],
            Intent.HOTEL_BOOKING: ["destination_city", "check_in_date", "check_out_date"],
            Intent.ITINERARY_PLANNING: ["destination_city", "trip_duration", "interests"],
            Intent.BUDGET_PLANNING: ["destination_city", "trip_duration", "budget_range"]
        }

        if intent not in required_info:
            return []

        return [info for info in required_info[intent] if info not in collected_info]

    def process_message(self, user_id: str, content: str) -> str:
        """
        Processes a user message through the workflow graph to generate a response.
        This method handles the entire conversation flow, from message creation to
        response generation.
        
        Args:
            user_id (str): The unique identifier for the user
            content (str): The message content from the user
            
        Returns:
            str: The assistant's response, or an error message if processing fails
        """
        try:
            # Ensure user state exists
            self._create_user_if_not_exists(user_id)
            
            # Create message object
            message = Message(
                user_id=user_id,
                content=content,
                timestamp=datetime.now()
            )
            
            # Prepare the initial state
            initial_state: TravelState = {
                "message": message,
                "intent": None,
                "extracted_info": None,
                "user_state": None,
                "response": None
            }
            
            # Use chain.invoke() to process the message
            final_state = self.graph.invoke(initial_state)
            
            # Handle the response
            if isinstance(final_state, dict) and "response" in final_state:
                return final_state["response"]
            else:
                return "I apologize, but I couldn't generate a proper response. Please try again."
                
        except Exception as e:
            print(f"Error details: {str(e)}")  # For debugging
            return (
                "I encountered an error while processing your message. "
                "This might be due to a temporary issue. Please try again or "
                "rephrase your request."
            )


def main():
    """
    Example usage demonstrating the enhanced capabilities of the travel assistant.
    """
    assistant = TravelAssistant()
    
    # Example conversation
    conversations = [
        ("user1", "I want to plan a trip to Japan next month"),
        ("user1", "I'm thinking of visiting Tokyo and Kyoto"),
        ("user2", "Looking for a budget-friendly hotel in Paris"),
        ("user2", "I can spend about 150 euros per night"),
    ]
    
    for user_id, message in conversations:
        response = assistant.process_message(user_id, message)
        print(f"\nUser {user_id}: {message}")
        print(f"Assistant: {response}")

if __name__ == "__main__":
    main()
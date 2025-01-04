from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from dotenv import load_dotenv

_ = load_dotenv()


# Define our state structure using TypedDict
class AgentState(TypedDict):
    # Current conversation history
    messages: list[str]
    # Counter for tracking conversation turns
    turns: int
    # Customer satisfaction score (0-10)
    satisfaction_score: float
    # Whether the conversation should end
    should_end: bool

# Initialize the LLM
llm = ChatOpenAI(temperature=0)

# Create a prompt template for the customer service agent
agent_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful customer service agent. Use the context provided to help the customer."),
    ("system", "Current satisfaction score: {satisfaction_score}/10"),
    ("system", "Conversation history: {messages}"),
    ("human", "{input}")
])

def process_message(state: AgentState, input_message: str) -> AgentState:
    """Process an incoming message and update the state."""
    
    # Add the new message to history
    state["messages"].append(f"Customer: {input_message}")
    
    # Get agent response using context
    response = llm.invoke(
        agent_prompt.format(
            satisfaction_score=state["satisfaction_score"],
            messages="\n".join(state["messages"]),
            input=input_message
        )
    )
    
    # Update state
    state["messages"].append(f"Agent: {response.content}")
    state["turns"] += 1
    
    # Update satisfaction score based on conversation length
    # (This is a simple example - in practice, you'd want more sophisticated scoring)
    state["satisfaction_score"] = max(0, 10 - (state["turns"] * 0.5))
    
    # Check if we should end the conversation
    state["should_end"] = "goodbye" in input_message.lower() or state["turns"] >= 10
    
    return state

def should_continue(state: AgentState) -> bool:
    """Determine if the conversation should continue."""
    return not state["should_end"]

# Create and configure the graph
workflow = StateGraph(AgentState)

# Add nodes to the graph
workflow.add_node("process", process_message)

# Add edges
workflow.add_conditional_edges("process", "process", should_continue)
workflow.add_edge("process", END, lambda x: not should_continue(x))

# Compile the graph
app = workflow.compile()

# Initialize the state
initial_state = {
    "messages": [],
    "turns": 0,
    "satisfaction_score": 10.0,
    "should_end": False
}

# Example usage
def run_conversation(messages: list[str]) -> None:
    state = initial_state
    
    for message in messages:
        print(f"\nCustomer: {message}")
        state = app.invoke({"input_message": message, **state})
        print(f"Agent: {state['messages'][-1].replace('Agent: ', '')}")
        print(f"Satisfaction Score: {state['satisfaction_score']:.1f}/10")
        
        if state["should_end"]:
            print("\nConversation ended.")
            break

# Test the conversation
test_messages = [
    "Hi, I need help with my order",
    "Order #12345 hasn't arrived yet",
    "Thank you for your help, goodbye"
]

run_conversation(test_messages)
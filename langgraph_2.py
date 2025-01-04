from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, List
import operator
from langchain_core.messages import (
    AnyMessage, 
    SystemMessage, 
    HumanMessage, 
    ToolMessage,
    AIMessage
)
from langchain_core.tools import Tool
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

# Load environment variables properly
load_dotenv()

# Define a more structured prompt template with tool descriptions
prompt_template = """Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:
Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!
Question: {input}
Thought: {agent_scratchpad}"""

class AgentState(TypedDict):
    messages: Annotated[List[AnyMessage], operator.add]

class RestaurantTool:
    def __init__(self):
        self.name = "restaurant_rating"
        self.description = "Get rating and review information for a restaurant"
    
    def get_restaurant_rating(self, name: str) -> dict:
        ratings = {
            "Pizza Palace": {"rating": 4.5, "reviews": 230},
            "Burger Barn": {"rating": 4.2, "reviews": 185},
            "Sushi Supreme": {"rating": 4.8, "reviews": 320}
        }
        return ratings.get(name, {"rating": 0, "reviews": 0})

    def __call__(self, name: str) -> str:
        result = self.get_restaurant_rating(name)
        return f"Rating: {result['rating']}/5.0 from {result['reviews']} reviews"

class Agent:
    def __init__(self, model: ChatOpenAI, tools: List[Tool], system: str = ''):
        self.system = system
        self.tools = {t.name: t for t in tools}
        
        # Create tool descriptions for the prompt
        tool_descriptions = "\n".join(f"- {t.name}: {t.description}" for t in tools)
        tool_names = ", ".join(t.name for t in tools)
        
        # Bind tools to the model
        self.model = model.bind_tools(tools)
        
        # Initialize the graph
        graph = StateGraph(AgentState)
        
        # Add nodes and edges
        graph.add_node("llm", self.call_llm)
        graph.add_node("action", self.take_action)
        
        # Add conditional edges
        graph.add_conditional_edges(
            "llm",
            self.should_continue,
            {True: "action", False: END}
        )
        graph.add_edge("action", "llm")
        
        # Set entry point and compile
        graph.set_entry_point("llm")
        self.graph = graph.compile()

    def should_continue(self, state: AgentState) -> bool:
        """Check if there are any tool calls to process"""
        last_message = state["messages"][-1]
        return hasattr(last_message, "tool_calls") and bool(last_message.tool_calls)

    def call_llm(self, state: AgentState) -> AgentState:
        """Process messages through the LLM"""
        messages = state["messages"]
        if self.system and not any(isinstance(m, SystemMessage) for m in messages):
            messages = [SystemMessage(content=self.system)] + messages
        response = self.model.invoke(messages)
        return {"messages": [response]}

    def take_action(self, state: AgentState) -> AgentState:
        """Execute tool calls and return results"""
        last_message = state["messages"][-1]
        results = []
        
        for tool_call in last_message.tool_calls:
            tool_name = tool_call['name']
            if tool_name not in self.tools:
                result = f"Error: Unknown tool '{tool_name}'"
            else:
                try:
                    tool_result = self.tools[tool_name].invoke(tool_call['args'])
                    result = str(tool_result)
                except Exception as e:
                    result = f"Error executing {tool_name}: {str(e)}"
            
            results.append(
                ToolMessage(
                    tool_call_id=tool_call['id'],
                    name=tool_name,
                    content=result
                )
            )
        
        return {"messages": results}

    def invoke(self, message: str) -> List[AnyMessage]:
        """Main entry point for the agent"""
        initial_state = {"messages": [HumanMessage(content=message)]}
        final_state = self.graph.invoke(initial_state)
        return final_state["messages"]

# Create and configure the agent
def create_restaurant_agent() -> Agent:
    model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    
    # Create tool instance
    restaurant_tool = RestaurantTool()
    
    # Convert to LangChain Tool
    tool = Tool(
        name=restaurant_tool.name,
        description=restaurant_tool.description,
        func=restaurant_tool
    )
    
    # Create system prompt
    system_prompt = prompt_template.format(
        tools=tool.description,
        tool_names=tool.name,
        input="{input}",
        agent_scratchpad="{agent_scratchpad}"
    )
    
    # Create and return agent
    return Agent(model, [tool], system=system_prompt)

# Example usage
if __name__ == "__main__":
    agent = create_restaurant_agent()
    response = agent.invoke("""which resturant have better rating, Pizza Palace or Burger Barn?""")
    for message in response:
        print(f"{message.type}: {message.content}")
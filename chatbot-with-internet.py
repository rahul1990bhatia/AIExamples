import os
import json
from typing import Annotated
from langchain_openai import ChatOpenAI
from typing_extensions import TypedDict
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langchain_core.messages import ToolMessage
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.prebuilt import ToolNode, tools_condition

#TAVILY_API_KEY='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'

class State(TypedDict):
    messages: Annotated[list, add_messages]

graph_builder = StateGraph(State)
# define tool
tool = TavilySearchResults(max_results=2)
tools = [tool]

# bind llmt with tool
llm = ChatOpenAI(model="gpt-4o", api_key='')
llm_with_tools = llm.bind_tools(tools)

def chatbot(state: State):
    return {"messages" : [llm_with_tools.invoke(state['messages'])]}


graph_builder.add_node("chatbot", chatbot)

tool_node = ToolNode(tools=[tool])
graph_builder.add_node("tools", tool_node)

graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition
)
graph_builder.add_edge("tools", "chatbot")

graph_builder.set_entry_point("chatbot")
graph = graph_builder.compile()

def stream_graph_updates(user_input: str):
    for event in graph.stream({"messages": [("user", user_input)]}):
        for value in event.values():
            print("Assistant:", value["messages"][-1].content)


while True:
    try:
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break

        stream_graph_updates(user_input)
    except:
        # fallback if input() is not available
        break
import os
from typing import Dict, List, Any
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph
from langchain_openai import ChatOpenAI
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools import WikipediaQueryRun
from langgraph.prebuilt import ToolNode  # Removed ToolInvocation
from dotenv import load_dotenv

# Set OpenAI API key
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Define state schema
class ChatState(BaseModel):
    messages: List[Dict[str, str]] = Field(default_factory=list)
    next_steps: List[str] = Field(default_factory=list)
    
    # For tools
    tool_input: str = None
    tool_output: str = None

# Initialize Wikipedia API Wrapper and Tool
wikipedia_api = WikipediaAPIWrapper()
wikipedia_tool = WikipediaQueryRun(api_wrapper=wikipedia_api)

# Create the graph with proper state schema
graph = StateGraph(ChatState)

# Define a function to extract query from messages
def extract_query(state: ChatState) -> ChatState:
    # Extract the latest user message content as the search query
    latest_message = state.messages[-1]["content"]
    state.tool_input = latest_message
    state.next_steps = ["wikipedia_search"]
    return state

# Create a function that calls the Wikipedia tool
def use_wikipedia_tool(state: ChatState) -> ChatState:
    query = state.tool_input
    result = wikipedia_tool.run(query)
    state.tool_output = result
    state.next_steps = ["prepare_response"]
    return state

# Define function to prepare final response
def prepare_response(state: ChatState) -> ChatState:
    # Add an assistant message with the Wikipedia results
    state.messages.append({
        "role": "assistant", 
        "content": f"Here's what I found: {state.tool_output}"
    })
    state.next_steps = []
    return state

# Add nodes to the graph
graph.add_node("extract_query", extract_query)
graph.add_node("wikipedia_search", use_wikipedia_tool)
graph.add_node("prepare_response", prepare_response)

# Define the conditional edges based on next_steps
def decide_next_step(state: ChatState) -> str:
    if not state.next_steps:
        return "__end__"
    return state.next_steps[0]

# Define the flow
graph.set_entry_point("extract_query")
graph.add_conditional_edges("extract_query", decide_next_step)
graph.add_conditional_edges("wikipedia_search", decide_next_step)
graph.add_conditional_edges("prepare_response", decide_next_step)

# Compile the graph
app = graph.compile()

# Invoke with the proper message format
response = app.invoke({
    "messages": [{"role": "user", "content": "Who is the CEO of OpenAI?"}]
})

# Print response
print(response)


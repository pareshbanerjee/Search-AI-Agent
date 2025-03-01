import os
from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, AgentType
from langchain_community.tools import Tool
from langchain_community.utilities import WikipediaAPIWrapper

from dotenv import load_dotenv
load_dotenv()

# Set your OpenAI API key (Make sure to replace 'your-api-key' with an actual key)

openai_api_key = os.getenv("OPENAI_API_KEY")

print(f"Your OpenAI API key is: {openai_api_key}")

# Set up OpenAI API
llm = ChatOpenAI(model="gpt-4", temperature=0.5)

# Create a Wikipedia Search Tool
wiki = WikipediaAPIWrapper()

# Define tools the agent can use
tools = [
    Tool(
        name="Wikipedia Search",
        func=wiki.run,
        description="Useful for finding factual information from Wikipedia."
    )
]

# Create an Agent
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# Run the agent
#response = agent.run("Who is the CEO of OpenAI?")
response = agent.invoke({"input": "Who is the CEO of OpenAI?"})

print(response)

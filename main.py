from dotenv import load_dotenv

load_dotenv()
# package to create agent
from langchain.agents import create_agent
#package to create tools
from langchain.tools import tool
#package to create a human message
from langchain_core.messages import HumanMessage
#package to create a language model
from langchain_openai import ChatOpenAI


@tool
def search(query: str) -> str:
    """
    Tool that searches over internet
    Args:
        query: The query to search for
    Returns:
        The search results
    """
    print(f"Searching for: {query}")
    return "Tokyo weather is sunny"

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
tools = [search]
agent = create_agent(model=llm, tools=tools)

def main():
    response = agent.invoke({"messages": [HumanMessage(content="What is the weather in Tokyo?")]})
    print(response)

if __name__ == "__main__":
    main()

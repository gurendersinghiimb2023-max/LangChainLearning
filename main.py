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
from langchain_tavily import TavilySearch



llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
tools = [TavilySearch()]
agent = create_agent(model=llm, tools=tools)

def main():
    response = agent.invoke({"messages": [HumanMessage(content="What is the weather in Tokyo?")]})
    print(response)

if __name__ == "__main__":
    main()

from dotenv import load_dotenv

load_dotenv()

from langchain_openai import ChatOpenAI
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langsmith import traceable

MAX_ITERATIONS = 10
MODEL = "qwen3:1.7b"

# ----Tools (Langchain @tool decorator) ----
@tool
def get_product_price(product: str) -> float:
    """Look up the price of a product on the catalog"""
    print(f" >> Executing get_product_price tool for product: '{product}'")
    price = {
        "laptop": 1299.99,
        "headphones": 149.95,
        "keyboard": 89.50
    }   
    return price.get(product, 0.0)

@tool
def apply_discount(price: float, discount_tier: str) -> float:
    """Apply a discount to a product"""
    print(f" >> Executing apply_discount tool for price: '{price}' with discount_tier: {discount_tier}")
    discount = {
        "bronze": 0.05,
        "silver": 0.12,
        "gold": 0.23
    }
    return round(price * (1 - discount.get(discount_tier, 0.0)), 2)

# ---Agent Loop ----

@traceable(name="LangChain Agent Loop")
def run_agent(question: str):
    tools = [get_product_price, apply_discount]
    tool_dict = {tool.name: tool for tool in tools}
    llm = init_chat_model(model=f"ollama:{MODEL}")
    llm_with_tools = llm.bind_tools(tools)
    print(f"Question: {question}")
    print("=" * 50)

    messages = [
        SystemMessage(content= ("You are a helpful shopping assistant."
                     "You have access to a product catalog tool"
                     "and a discount tool.\n\n"
                     " STRICT RULES - you must follow these exactly:\n\n"
                     "1. NEVER guess or assume any product price."
                     "You must call get_product_price tool to get the real price of a product."
                     "2.Only call apply_discount AFTER you have received"
                     "a price from get_product_price. Pass the exact price"
                     "returned  by get_product_price to apply_discount - do not pass a made-up number. \n"
                     "3. NEVER calculate discounts yourself using math."
                     "4. If the user does not specify a discount tier,"
                     "ask them which tier to use - do NOT assume one.")),
        HumanMessage(content=question)
    ]
    for iteration in range(1, MAX_ITERATIONS + 1):
        print (f"\n-- Iteration {iteration} --")
        ai_message = llm_with_tools.invoke(messages)
        tool_calls = ai_message.tool_calls
        if not tool_calls:
            print(f"\nFinal Answer: {ai_message.content}")
            return ai_message.content
       
        # Process only the first tool call
        if tool_calls:
            tool_call = tool_calls[0]
            tool_name = tool_call.get("name")
            tool_input = tool_call.get("args",{})
            print(f" >> Calling tool: {tool_name} with input: {tool_input}")
            messages.append(ai_message)
            messages.append(HumanMessage(content=f"TOOL CALL: {tool_name} with input: {tool_input}"))
            tool_result = tool_dict[tool_name].invoke(tool_input)
            messages.append(ToolMessage(content=str(tool_result), tool_call_id=tool_call.get("id")))
            print(f" >> Tool result: {tool_result}")
            messages.append(ai_message)
            messages.append(HumanMessage(content=f"TOOL RESULT: {tool_result}"))
        else:
            print(f"\nFinal Answer: {ai_message.content}")
            return ai_message.content


if __name__ == "__main__":
    print("Hello Langchain Agent (.bind_tools)!")
    print()
    result = run_agent("What is the price of a laptop after applying a gold discount?")
import os

import httpx
from dotenv import load_dotenv

load_dotenv()

# LangSmith sends runs to the cloud when tracing is on; disable by default for this
# standalone script so local/proxy setups do not fail. Opt in with:
#   LANGCHAIN_ENABLE_LANGSMITH_FOR_AGENT_SCRIPT=true
if os.getenv("LANGCHAIN_ENABLE_LANGSMITH_FOR_AGENT_SCRIPT", "").lower() not in {
    "1",
    "true",
    "yes",
}:
    os.environ["LANGCHAIN_TRACING"] = "false"
    os.environ["LANGCHAIN_TRACING_V2"] = "false"

from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage, AIMessage

MAX_ITERATIONS = 10
MODEL = "qwen3:1.7b"
 
SYSTEM_PROMPT = """You are a helpful shopping assistant.
You have access to a product catalog tool and a discount tool.
 
STRICT RULES - you must follow these exactly:
1. NEVER guess or assume any product price. You must call get_product_price to get the real price.
2. Only call apply_discount AFTER you have received a price from get_product_price.
   Pass the exact price returned by get_product_price — do not pass a made-up number.
3. NEVER calculate discounts yourself using math.
4. If the user does not specify a discount tier, ask them which tier to use — do NOT assume one.
"""
 
# ---- Tools (LangChain @tool decorator) ----
 
@tool
def get_product_price(product: str) -> float:
    """Look up the price of a product in the catalog."""
    print(f"  [Tool] get_product_price(product={product!r})")
    prices = {
        "laptop": 1299.99,
        "headphones": 149.95,
        "keyboard": 89.50,
    }
    return prices.get(product.lower(), 0.0)
 
 
@tool
def apply_discount(price: float, discount_tier: str) -> float:
    """Apply a discount tier to a given price and return the discounted price."""
    print(f"  [Tool] apply_discount(price={price}, discount_tier={discount_tier!r})")
    discounts = {
        "bronze": 0.05,
        "silver": 0.12,
        "gold": 0.23,
    }
    rate = discounts.get(discount_tier.lower(), 0.0)
    return round(price * (1 - rate), 2)
 
 
# ---- Agent Loop (ReAct: Reason → Act → Observe) ----


def run_agent(question: str) -> str:
    tools = [get_product_price, apply_discount]
    tool_registry = {t.name: t for t in tools}
 
    llm = init_chat_model(model=f"ollama:{MODEL}")
    llm_with_tools = llm.bind_tools(tools)
 
    print(f"Question: {question}")
    print("=" * 50)
 
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=question),
    ]
 
    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"\n-- Iteration {iteration} --")
 
        # REASON: ask the LLM what to do next
        try:
            ai_message = llm_with_tools.invoke(messages)
        except httpx.ConnectError as e:
            raise RuntimeError(
                "Cannot reach Ollama. Start the Ollama app or run `ollama serve`, "
                "then pull the model if needed: `ollama pull qwen3:1.7b`."
            ) from e
        if not isinstance(ai_message, AIMessage):
            raise TypeError(f"Expected AIMessage from chat model; got {type(ai_message)!r}")

        messages.append(ai_message)

        tool_calls = ai_message.tool_calls

        # No tool calls → LLM has reached a final answer
        if not tool_calls:
            print(f"\nFinal Answer: {ai_message.content}")
            return ai_message.content
 
        # ACT + OBSERVE: execute each tool call and feed results back
        for idx, tool_call in enumerate(tool_calls):
            tool_name = tool_call["name"]
            tool_args = tool_call.get("args", {})
            raw_id = tool_call.get("id")
            tool_call_id = raw_id if raw_id is not None else f"call_{iteration}_{idx}"

            print(f"  [Agent] Calling '{tool_name}' with args: {tool_args}")
 
            if tool_name not in tool_registry:
                result = f"Error: unknown tool '{tool_name}'"
            else:
                result = tool_registry[tool_name].invoke(tool_args)
 
            print(f"  [Agent] Result: {result}")
 
            # OBSERVE: append the tool result so the LLM sees it next iteration
            messages.append(
                ToolMessage(content=str(result), tool_call_id=tool_call_id)
            )
 
    # Safety net if MAX_ITERATIONS is reached without a final answer
    print("\n[Warning] Max iterations reached without a final answer.")
    return "Sorry, I could not complete the request within the allowed number of steps."
 
 
if __name__ == "__main__":
    result = run_agent("What is the price of a laptop after applying a gold discount?")
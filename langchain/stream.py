"""LangChain stream —— 观察 agent 的中间执行过程。

对比 langchain/quickstart.py: 那里用 invoke 只拿最终结果,
这里用 stream 实时观察 agent 的完整流转:
  - messages 流: LLM 逐 token 输出
  - updates 流: 每个步骤完成时的事件 (工具调用请求 / 工具执行结果)
官方文档: https://docs.langchain.com/oss/python/langchain/streaming
"""

import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.messages import AIMessage, AIMessageChunk, ToolMessage
from langchain_anthropic import ChatAnthropic
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv(override=True)

model = ChatAnthropic(
    model=os.environ["MODEL_ID"],
    base_url=os.getenv("ANTHROPIC_BASE_URL") or None,
)


def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"


# stream 需要 checkpointer 维护线程状态
agent = create_agent(
    model=model,
    tools=[get_weather],
    system_prompt="You are a helpful assistant.",
    checkpointer=InMemorySaver(),
)


if __name__ == "__main__":
    config = {"configurable": {"thread_id": "stream-demo"}}
    print("=== agent 中间过程 (stream) ===\n")

    for chunk in agent.stream(
        {"messages": [{"role": "user", "content": "What's the weather in San Francisco?"}]},
        config=config,
        stream_mode=["messages", "updates"],  # 同时开 token 流和步骤流
        version="v2",
    ):
        if chunk["type"] == "messages":
            # LLM 逐 token 输出
            token, _ = chunk["data"]
            if isinstance(token, AIMessageChunk) and token.text:
                print(token.text, end="", flush=True)
        elif chunk["type"] == "updates":
            # 每个步骤完成时的事件: model 步骤产出工具调用, tools 步骤产出工具结果
            for source, update in chunk["data"].items():
                if source in ("model", "tools") and update.get("messages"):
                    msg = update["messages"][-1]
                    if isinstance(msg, AIMessage) and msg.tool_calls:
                        for call in msg.tool_calls:
                            print(f"\n[tool-call] {call['name']}({call['args']})")
                    elif isinstance(msg, ToolMessage):
                        print(f"[tool-result] {msg.content}")
    print("\n\n=== 完成 ===")

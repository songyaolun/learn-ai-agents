"""DeepAgents stream —— 观察 deep agent 的中间执行过程 (含 subagent 委派)。

对比 langchain/stream.py: 那里只有 model/tools 两类步骤,
这里 DeepAgents 的 updates 流里会多出 subagent 委派相关的步骤
(coordinator 调 task 工具委派 → 子 agent 执行 → 结果回传),
能看清 harness 层是如何编排主 agent 与子 agent 的。

注: DeepAgents 还有个实验性的 stream_events(v3) + interleave API, 可区分
coordinator / subagent 的消息流, 但 v3 仍为 beta, 这里用稳定的通用 stream API。
官方文档: https://docs.langchain.com/oss/python/deepagents/event-streaming
"""

import os

from dotenv import load_dotenv
from deepagents import create_deep_agent
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


# checkpointer 让 stream 能维护线程状态 (和 langchain/stream.py 一致)
agent = create_deep_agent(
    model=model,
    tools=[get_weather],
    system_prompt="You are a helpful research assistant.",
    checkpointer=InMemorySaver(),
    subagents=[
        {
            "name": "researcher",
            "description": "Delegate research subtasks to this subagent. Give one topic at a time.",
            "system_prompt": "You are a great researcher. Return a brief summary.",
        }
    ],
)


if __name__ == "__main__":
    config = {"configurable": {"thread_id": "stream-demo"}}
    print("=== deep agent 中间过程 (stream) ===\n")

    for chunk in agent.stream(
        {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "What's the weather in San Francisco? "
                        "Then briefly research why SF has this kind of climate."
                    ),
                }
            ]
        },
        config=config,
        stream_mode=["messages", "updates"],
        version="v2",
    ):
        if chunk["type"] == "messages":
            # LLM 逐 token 输出
            token, _ = chunk["data"]
            if isinstance(token, AIMessageChunk) and token.text:
                print(token.text, end="", flush=True)
        elif chunk["type"] == "updates":
            # 每个步骤完成的事件; source 可能是 model/tools, 也可能是 subagent 名
            for source, update in chunk["data"].items():
                if not update or not update.get("messages"):
                    continue
                msg = update["messages"][-1]
                if isinstance(msg, AIMessage) and msg.tool_calls:
                    for call in msg.tool_calls:
                        # task 工具调用 = coordinator 把子任务委派给 subagent
                        label = "委派子agent" if call["name"] == "task" else "tool-call"
                        print(f"\n[{source}] {label}: {call['name']}({call['args']})")
                elif isinstance(msg, ToolMessage):
                    content = str(msg.content)
                    if len(content) > 200:
                        content = content[:200] + "..."
                    print(f"[{source}] tool-result: {content}")
    print("\n\n=== 完成 ===")

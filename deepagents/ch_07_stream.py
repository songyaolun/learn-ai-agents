"""DeepAgents stream —— 观察 deep agent 的中间执行过程 (含 subagent 委派)。

对比 langchain/ch_07_stream.py: 那里只有 model/tools 两类步骤,
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


# checkpointer 让 stream 能维护线程状态 (和 langchain/ch_07_stream.py 一致): create_deep_agent
# 底层也是一张 LangGraph 图, 只要用 stream 或者要多轮记忆, 就需要配一个 checkpointer。
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
            # LLM 逐 token 输出 (跟 langchain/ch_07_stream.py 里 messages 流的用法一样)
            token, _ = chunk["data"]
            if isinstance(token, AIMessageChunk) and token.text:
                print(token.text, end="", flush=True)
        elif chunk["type"] == "updates":
            # 每个步骤完成的事件; source 实际观察下来还是只有 model/tools 这两种
            # (跟 langchain/ch_07_stream.py 一样), 子 agent 委派并不会在这一层暴露出单独的
            # 节点名字 —— 子 agent 内部完整跑了一轮自己的 model/tools 循环, 但对主 agent
            # 的图来说, 这整个过程只是"调用了一次叫 task 的工具", 子 agent 的执行细节
            # 被封装在这一次 tool-result 里一起返回, 不会拆成逐条事件推送出来。
            # 想看子 agent 内部真正的逐步执行过程, 需要用文件头注释提到的实验性
            # stream_events(v3) + interleave API (目前仍是 beta)。
            for source, update in chunk["data"].items():
                if not update or not update.get("messages"):
                    continue
                msg = update["messages"][-1]
                if isinstance(msg, AIMessage) and msg.tool_calls:
                    for call in msg.tool_calls:
                        # DeepAgents 内置了一个叫 task 的工具, 专门用来"把任务委派给
                        # 某个 subagent"; 模型调用 task(subagent_name, description) 时,
                        # 就相当于把这段描述发给对应的子 agent, 让它独立跑完一整轮对话
                        # 再把最终结果当成这次 task 调用的返回值带回来。
                        # task 工具调用 = coordinator 把子任务委派给 subagent
                        label = "委派子agent" if call["name"] == "task" else "tool-call"
                        print(f"\n[{source}] {label}: {call['name']}({call['args']})")
                elif isinstance(msg, ToolMessage):
                    # 普通工具 (如 get_weather) 的结果很短; 如果是 task 工具的结果,
                    # 这里看到的就是 researcher 子 agent 独立跑完之后给出的最终总结
                    # (子 agent 内部具体调用了几次工具、想了多久, 在这里都看不到)。
                    content = str(msg.content)
                    if len(content) > 200:
                        content = content[:200] + "..."
                    print(f"[{source}] tool-result: {content}")
    print("\n\n=== 完成 ===")

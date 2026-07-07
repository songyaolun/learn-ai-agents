"""DeepAgents quickstart —— 用 create_deep_agent 搭一个带规划 + 子 agent 的深度 agent。

官方上手指南: https://docs.langchain.com/oss/python/deepagents/quickstart

对比 langchain/quickstart.py: 那里是「浅 agent」(一轮工具调用就结束),
这里 DeepAgents 在 create_agent 之上内置了一个 harness 层:
  - planning: 自动生成 to-do list 并跟踪进度 (类似 claude-code/ch03 的 todo)
  - filesystem: 虚拟文件系统, 跨步骤管理上下文
  - subagents: 把子任务委派给专门 agent (下方的 researcher)
适合长时运行、多步骤的复杂任务。
"""

import os

from dotenv import load_dotenv
from deepagents import create_deep_agent
from langchain_anthropic import ChatAnthropic

load_dotenv(override=True)

# 接入方式与 langchain/quickstart.py、ch01-ch03 完全一致
model = ChatAnthropic(
    model=os.environ["MODEL_ID"],
    base_url=os.getenv("ANTHROPIC_BASE_URL") or None,
)


def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"


# create_deep_agent 在 create_agent 之上叠加 harness 层:
# 同样的 model + tools + system_prompt, 但多出 subagents (可委派子任务)
agent = create_deep_agent(
    model=model,
    tools=[get_weather],
    system_prompt="You are a helpful research assistant.",
    subagents=[
        {
            "name": "researcher",
            "description": "Delegate research subtasks to this subagent. Give one topic at a time.",
            "system_prompt": "You are a great researcher. Return a brief summary.",
        }
    ],
)


if __name__ == "__main__":
    # 这个 query 会同时触发: 工具调用 (get_weather) + 子 agent 委派 (researcher)
    # DeepAgents 会先规划, 再逐步执行, 主 agent 上下文保持干净
    result = agent.invoke(
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
        }
    )
    print(result["messages"][-1].content)

"""LangChain quickstart —— 用 create_agent 搭一个最小 agent。

官方上手指南: https://docs.langchain.com/oss/python/langchain/quickstart

对比 claude-code/ch01.py: 那里用原生 anthropic SDK 手写了 agent loop
(消息拼接、tool_use 分发、stop_reason 判断、工具结果回填),
这里用 LangChain 1.0 的 create_agent 一行就把「模型 + 工具 + 系统提示」
组装成一个可调用的 agent —— agent loop 由框架托管。
"""

import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_anthropic import ChatAnthropic

load_dotenv(override=True)

# 沿用 ch01-ch03 的接入方式: ANTHROPIC_BASE_URL + MODEL_ID
# 认证默认从 ANTHROPIC_API_KEY 读取 (anthropic SDK 自动处理)
model = ChatAnthropic(
    model=os.environ["MODEL_ID"],
    base_url=os.getenv("ANTHROPIC_BASE_URL") or None,
)


def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"


# create_agent 是 LangChain 1.0 的核心原语:
# 模型 + 工具 + 系统提示 → 一个带 tool-calling 循环的可调用 agent
agent = create_agent(
    model=model,
    tools=[get_weather],
    system_prompt="You are a helpful assistant.",
)


if __name__ == "__main__":
    result = agent.invoke(
        {
            "messages": [
                {"role": "user", "content": "What's the weather in San Francisco?"}
            ]
        }
    )
    # messages 列表最后一条就是 agent 的最终回答
    print(result["messages"][-1].content)

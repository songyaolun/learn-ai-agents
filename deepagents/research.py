"""DeepAgents research —— 接入 DuckDuckGo 真实搜索, 做一个 deep research agent。

对比 deepagents/quickstart.py: 那里用模拟的 get_weather 工具, 这里接真实搜索引擎,
DeepAgents 的 planning + subagent 在真实多步研究任务上才真正发挥价值。
DuckDuckGo 无需 API key, 适合本地学习。
"""

import os

from dotenv import load_dotenv
from deepagents import create_deep_agent
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_anthropic import ChatAnthropic

load_dotenv(override=True)

model = ChatAnthropic(
    model=os.environ["MODEL_ID"],
    base_url=os.getenv("ANTHROPIC_BASE_URL") or None,
)

# DuckDuckGo 搜索工具, 无需 API key
search = DuckDuckGoSearchRun()

agent = create_deep_agent(
    model=model,
    tools=[search],
    system_prompt=(
        "You are a research assistant. Use the search tool to find current, accurate "
        "information. Plan your research, search multiple queries if needed, then "
        "synthesize a clear answer with sources."
    ),
    subagents=[
        {
            "name": "researcher",
            "description": "Delegate a focused research subtopic to this subagent.",
            "system_prompt": "You are a great researcher. Search and return a brief, accurate summary.",
        }
    ],
)


if __name__ == "__main__":
    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "LangChain 1.0 在 2025 年正式发布了。"
                        "帮我研究一下 LangChain 1.0 和 LangGraph 1.0 的核心变化和主要特性, "
                        "给出简要总结。"
                    ),
                }
            ]
        }
    )
    print(result["messages"][-1].content)

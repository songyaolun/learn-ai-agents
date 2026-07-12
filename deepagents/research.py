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

# DuckDuckGoSearchRun 是 langchain_community 提供的现成工具: 直接调用 DuckDuckGo 的
# 搜索接口, 不需要申请任何 API key, 拿到的是真实的网络搜索结果 (不是模型编出来的)。
# 跟 get_weather 一样, 它之所以能被模型调用, 也是因为自带了描述信息 (工具名/参数/用途)。
search = DuckDuckGoSearchRun()

# 这里没有额外传自定义工具, 只给了 search 一个工具, 但 create_deep_agent 仍然会自动
# 附带 todo 规划工具 + 虚拟文件系统工具 (跟 quickstart.py 一样), 只是这里的例子没有
# 特意在 system_prompt 里要求用文件, 所以不一定每次都会用到 write_file。
agent = create_deep_agent(
    model=model,
    tools=[search],
    system_prompt=(
        "You are a research assistant. Use the search tool to find current, accurate "
        "information. Plan your research, search multiple queries if needed, then "
        "synthesize a clear answer with sources."
    ),
    # researcher 子 agent 也拿不到额外工具, 但因为它是独立跑的一整个 agent, 遇到需要
    # 搜索的问题时同样会调用 search (子 agent 会继承主 agent 提供的工具集)。
    subagents=[
        {
            "name": "researcher",
            "description": "Delegate a focused research subtopic to this subagent.",
            "system_prompt": "You are a great researcher. Search and return a brief, accurate summary.",
        }
    ],
)


if __name__ == "__main__":
    # 这是一个典型的"需要多步骤才能回答好"的问题: 既要搜索最新资料, 又要综合整理成
    # 一份总结。DeepAgents 的价值就体现在这种任务上——先用 todo 规划要查哪几个方面,
    # 再一步步搜索 (可能还会委派给 researcher 子 agent 分头查), 最后再汇总成答案。
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
    # 用 .text 而不是 .content: 开启 extended thinking 的模型返回的 .content 是
    # thinking/text 混合的 block 列表, .text 只取出其中的纯文本部分。
    print(result["messages"][-1].text)

"""DeepAgents quickstart —— 用 create_deep_agent 搭一个带规划 + 子 agent 的深度 agent。

官方上手指南: https://docs.langchain.com/oss/python/deepagents/quickstart

对比 langchain/ch_01_quickstart.py: 那里是「浅 agent」(一轮工具调用就结束),
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

# 接入方式与 langchain/ch_01_quickstart.py、ch01-ch03 完全一致
model = ChatAnthropic(
    model=os.environ["MODEL_ID"],
    base_url=os.getenv("ANTHROPIC_BASE_URL") or None,
)


def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"


# create_deep_agent 和 langchain 的 create_agent 用法几乎一样 (同样是 model + tools +
# system_prompt), 但它在背后自动做了几件 create_agent 不会做的事:
#   1. 自动给 agent 塞一个 todo 规划工具 (类似 claude-code/ch03.py 里手写的 TodoManager),
#      面对复杂任务时先拆解成步骤再执行, 而不是想到哪做到哪
#   2. 自动塞一整套虚拟文件系统工具 (ls/read_file/write_file/edit_file 等), 让 agent
#      能把中间结果存成"文件"而不是全塞进对话历史 (见 deepagents/ch_03_filesystem.py)
#   3. 支持 subagents 参数: 可以把某一类子任务"外包"给一个独立的子 agent 去做
#
# subagents 里配置的 researcher 就是一个子 agent: 主 agent 遇到需要研究的问题时,
# 会调用一个内置的 task 工具, 把任务描述发给 researcher 去独立完成, 只把 researcher
# 的最终总结带回主对话 —— researcher 内部具体调用了几次工具、想了多久, 对主 agent
# 的上下文都是不可见的 (这样主 agent 的上下文不会被子任务的中间过程撑爆)。
agent = create_deep_agent(
    model=model,
    tools=[get_weather],
    system_prompt="You are a helpful research assistant. Please answer questions in the user's language.",
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
                        "San Francisco的天气如何？"
                        "然后简单研究一下为什么旧金山会有这样的天气。"
                    ),
                }
            ]
        }
    )
    # 用 .text 而不是 .content: 开启 extended thinking 的模型返回的 .content 是
    # thinking/text 混合的 block 列表, .text 只取出其中的纯文本部分。
    print(result["messages"][-1].text)

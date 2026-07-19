"""DeepAgents filesystem —— 观察 agent 如何用虚拟文件系统管理长任务的中间状态。

对比 deepagents/ch_01_quickstart.py、ch_02_research.py: 那两个只关心 agent 的最终回答,
没有展示 DeepAgents 三大能力之一的"虚拟文件系统"具体在做什么。
create_deep_agent 会自动给 agent 装上 ls/read_file/write_file/edit_file 等文件工具,
agent 可以把搜集到的信息、写到一半的草稿存成"文件", 跨多个步骤引用/修改, 而不是把
所有中间内容都塞进对话历史 (对话历史越长, 上下文窗口压力越大, 参考
langchain/ch_24_middleware_summarization.py)。这里的"文件"默认并不是真实磁盘文件,
而是存在 LangGraph state 里的一个 dict, 所以叫"虚拟"文件系统。

官方文档: https://docs.langchain.com/oss/python/deepagents/filesystem
"""

import os

from dotenv import load_dotenv
from deepagents import create_deep_agent
from langchain_anthropic import ChatAnthropic

load_dotenv(override=True)

model = ChatAnthropic(
    model=os.environ["MODEL_ID"],
    base_url=os.getenv("ANTHROPIC_BASE_URL") or None,
)

# 注意这里没有传 tools 参数: create_deep_agent 会自动给 agent 装上一整套文件工具
# (ls / read_file / write_file / edit_file / glob / grep), 不需要我们像
# deepagents/ch_01_quickstart.py 里那样自己写 get_weather 这类工具才能演示。
# system_prompt 里明确要求"写文件", 是为了确保 agent 一定会用到这些工具, 方便观察效果
# (实际使用时 agent 会自己判断什么时候该记笔记, 不需要这么明确地要求)。
agent = create_deep_agent(
    model=model,
    system_prompt=(
        "You are a research assistant. When asked to research a topic, use write_file "
        "to save your findings to a file named notes.md before giving your final answer. "
        "Structure the file with markdown headings."
    ),
)


if __name__ == "__main__":
    # 跟 ch_01_quickstart.py 一样用 invoke 拿最终结果; 中间 agent 可能会先调用 write_file
    # (把整理好的内容存进 notes.md), 再基于文件内容组织语言给出最终回复。
    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "简单整理一下 Python 和 JavaScript 的三个主要区别, 存到文件里。",
                }
            ]
        }
    )
    print("=== agent 的最终回复 ===")
    print(result["messages"][-1].text)

    # result["files"] 是这次运行结束后虚拟文件系统里的全部文件: {文件路径: 文件内容}。
    # 默认用的是"内存态"的 backend —— 文件内容就存在 LangGraph 的 state 里, 跟对话历史
    # 一样靠 checkpointer 才能跨进程持久化 (这里没配 checkpointer, 进程一退出就丢);
    # 想让 agent 真正写到本机磁盘上, 需要给 create_deep_agent 传
    # backend=FilesystemBackend(root_dir="...") 换一个落盘的 backend 实现。
    print("\n=== 虚拟文件系统里的文件 ===")
    for path, file_data in result.get("files", {}).items():
        print(f"--- {path} ---")
        print(file_data["content"])

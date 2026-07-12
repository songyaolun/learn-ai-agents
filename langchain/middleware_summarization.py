"""LangChain middleware —— 用 SummarizationMiddleware 自动压缩过长的对话历史。

对比 claude-code/ch02.py 的 normalize_messages(): 那里只做格式清洗 (补 tool_result、
合并连续同角色消息), 历史会无限增长, 迟早超出上下文窗口;
这里 SummarizationMiddleware 挂在 create_agent 上, 在每次调用模型前检查历史长度,
一旦达到 trigger 阈值, 就用一个模型调用把"旧消息"压缩成一段摘要, 只保留摘要 +
最近 keep 条消息 —— 这是长任务/多轮对话里管理上下文窗口的标准做法。

官方文档: https://docs.langchain.com/oss/python/langchain/middleware#summarization
"""

import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langchain_anthropic import ChatAnthropic
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv(override=True)

model = ChatAnthropic(
    model=os.environ["MODEL_ID"],
    base_url=os.getenv("ANTHROPIC_BASE_URL") or None,
)


def remember(note: str) -> str:
    """记一条笔记 (工具本身没什么用, 只是用来制造带 tool_call 的历史消息)."""
    # 每次调用这个工具, 消息历史里就会多出"模型的工具调用请求" + "工具执行结果"
    # 两条消息, 方便快速把历史消息数堆上去, 观察触发摘要的效果。
    return f"已记录: {note}"


# 为什么需要"上下文窗口管理": 每次模型调用都要把完整的消息历史当输入发给模型, 而模型
# 能一次处理的 token 数是有上限的 (上下文窗口)。对话/任务越长, 历史消息越多, 迟早会
# 超出这个上限导致报错或者截断。SummarizationMiddleware 就是解决这个问题的标准做法:
# 定期把"旧的、不再需要逐字保留的历史"压缩成一段摘要, 只保留摘要 + 最近几条原始消息。
#
# trigger=("messages", 8): 历史消息数达到 8 条时就触发一次摘要
#                           (也可以按 token 数 ("tokens", N) 或按 max 输入长度的比例
#                           ("fraction", 0.8) 触发, 这里选消息数是因为最直观、最容易复现)
# keep=("messages", 2):    摘要生成之后, 只保留最近 2 条原始消息, 其余的全部被替换成
#                           前面那一段摘要文字 (摘要本身会被当成一条新消息插入历史开头)
#
# 注意: model 这个参数是"用来生成摘要的模型", 可以跟 agent 主模型不是同一个 (比如摘要
# 用更便宜的小模型), 这里图省事直接复用了同一个 model。
agent = create_agent(
    model=model,
    tools=[remember],
    system_prompt="You are a helpful assistant. Keep replies to one short sentence.",
    middleware=[
        SummarizationMiddleware(
            model=model,
            trigger=("messages", 8),
            keep=("messages", 2),
        )
    ],
    checkpointer=InMemorySaver(),
)


if __name__ == "__main__":
    # 同一个 thread_id 意味着这 5 轮对话共享同一份消息历史 (多轮记忆), 这样历史消息数
    # 才会持续累积, 才有机会触发 SummarizationMiddleware 的自动摘要。
    config = {"configurable": {"thread_id": "summarization-demo"}}
    turns = [
        "我叫小明, 记一下我喜欢喝美式咖啡。",
        "我养了一只猫, 叫豆豆。",
        "我在学习 LangChain, 记一下这件事。",
        "我下周要去杭州出差, 记一下。",
        "我最喜欢的编程语言是 Python。",
    ]

    for i, turn in enumerate(turns, 1):
        # 每一轮只需要传入这一句新的用户消息, 之前的历史 (以及可能已经发生的摘要压缩)
        # 都由 checkpointer 在背后自动维护, 不需要我们手动拼接。
        result = agent.invoke(
            {"messages": [{"role": "user", "content": turn}]}, config=config
        )
        messages = result["messages"]
        print(f"[第 {i} 轮] 用户: {turn}")
        print(f"  回复: {messages[-1].text}")
        # 观察这个数字: 正常情况下每轮至少 +2 (用户消息 + 模型回复, 用了工具还会更多),
        # 一旦触发摘要, 会看到这个数字突然"回落" —— 说明旧消息被压缩掉了。
        print(f"  当前历史消息数: {len(messages)}")

    print("\n=== 摘要后的历史消息 (可以看到旧消息已被压缩成一条摘要) ===")
    # agent.get_state(config) 读取当前 thread 的最新状态快照, .values["messages"]
    # 就是这个 thread 现在真正保存着的消息列表 (已经是摘要压缩之后的版本)。
    for msg in agent.get_state(config).values["messages"]:
        preview = str(msg.text)[:80].replace("\n", " ")
        # msg.type 会是 human/ai/tool 等, 摘要本身通常表现为一条较长的 human 消息,
        # 内容是模型总结出来的"到目前为止发生了什么"。
        print(f"  [{msg.type}] {preview}")

"""trim_messages —— 不生成摘要, 直接"砍掉"旧消息, 管理上下文窗口的另一种思路。

对比 langchain/middleware_summarization.py: 那里的 SummarizationMiddleware 会先花一次
模型调用把旧消息"浓缩成一段摘要"再保留, 好处是旧信息不会完全丢失; 这里的 trim_messages
更简单粗暴——不调用模型, 直接按 token 数从后往前数, 数够了就把更旧的消息整个丢弃,
优点是零成本 (不用额外调模型)、零延迟, 缺点是被丢掉的消息彻底没了, 一点痕迹都不留。
两种思路怎么选取决于场景: 聊天机器人这种"最近几句话最重要, 老话题忘了也无所谓"的场景
用 trim_messages 就够了; 需要长期记得任务背景的场景才值得多花一次模型调用去做摘要。

这里顺便展示 LangChain 1.0 的另一个概念: 用 @before_model 装饰器可以把一个普通函数
直接变成一个 middleware —— SummarizationMiddleware 内部其实就是用同样的 before_model
钩子实现的, 只是官方把"调用模型生成摘要"这部分逻辑也封装好了。

官方文档: https://docs.langchain.com/oss/python/langchain/short-term-memory#trimming-messages
"""

import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import before_model
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import RemoveMessage, trim_messages
from langchain_core.messages.utils import count_tokens_approximately
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.message import REMOVE_ALL_MESSAGES

load_dotenv(override=True)

model = ChatAnthropic(
    model=os.environ["MODEL_ID"],
    base_url=os.getenv("ANTHROPIC_BASE_URL") or None,
)


def remember(note: str) -> str:
    """记一条笔记 (工具本身没什么用, 只是用来制造带 tool_call 的历史消息)."""
    return f"已记录: {note}"


# @before_model 把下面这个普通函数变成一个 middleware: 它会在每次调用模型之前自动执行,
# 函数签名固定是 (state, runtime), 返回 None 表示"这次不用改状态", 返回一个 dict
# 表示"用这个 dict 去更新 state" (跟 SummarizationMiddleware 内部实现的模式一模一样)。
@before_model
def trim_history(state, runtime):
    messages = state["messages"]
    # trim_messages 按 token 数从消息列表尾部往前数, 凑够 max_tokens 就停, 更早的消息
    # 直接丢弃; strategy="last" 表示保留"最后面 (最新)"的部分; start_on="human" 保证
    # 裁剪后第一条消息是用户消息 (不会从模型回复或工具结果中间截断, 那样发给 API 会报错)。
    trimmed = trim_messages(
        messages,
        max_tokens=150,
        token_counter=count_tokens_approximately,  # 粗略估算 token 数, 不用真的调用分词器
        strategy="last",
        start_on="human",
    )
    if len(trimmed) == len(messages):
        return None  # 还没超限, 不用动
    # RemoveMessage(id=REMOVE_ALL_MESSAGES) 是 LangGraph 的约定写法: 表示"清空当前消息
    # 历史", 后面紧跟着的 trimmed 就是清空之后要保留下来的新历史。
    return {"messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES), *trimmed]}


agent = create_agent(
    model=model,
    tools=[remember],
    system_prompt="You are a helpful assistant. Keep replies to one short sentence.",
    middleware=[trim_history],
    checkpointer=InMemorySaver(),
)


if __name__ == "__main__":
    config = {"configurable": {"thread_id": "trim-demo"}}
    turns = [
        "我叫小明, 记一下我喜欢喝美式咖啡。",
        "我养了一只猫, 叫豆豆。",
        "我在学习 LangChain, 记一下这件事。",
        "我下周要去杭州出差, 记一下。",
        "我最喜欢的编程语言是 Python。",
        "我周末喜欢去爬山, 记一下。",
    ]

    for i, turn in enumerate(turns, 1):
        result = agent.invoke(
            {"messages": [{"role": "user", "content": turn}]}, config=config
        )
        messages = result["messages"]
        print(f"[第 {i} 轮] 用户: {turn}")
        print(f"  回复: {messages[-1].text}")
        # 跟 middleware_summarization.py 的效果对比: 那边触发压缩后消息数会回落到
        # "摘要 + keep 条数"; 这里触发裁剪后消息数会回落到"trim_messages 算出来刚好
        # 塞进 max_tokens 的那几条", 而且更早的内容是真的消失了, 不会变成一段摘要文字。
        print(f"  当前历史消息数: {len(messages)}")

"""LangGraph Store —— 跨 thread_id 的长期记忆, 和 checkpointer 是两回事。

对比 langgraph/persistence.py: 那里 SqliteSaver (checkpointer) 存的是"某一个
thread_id 自己的完整历史状态", 换一个 thread_id 就是全新的对话, 什么都不记得;
这里的 Store 存的是"跨 thread_id 共享"的长期记忆, 按你自己定义的 namespace
(比如 (user_id, "memories")) 组织, 只要两次调用用同一个 namespace, 哪怕
thread_id (对话) 完全不同, 也能读到同一份记忆 —— 这才是"记住用户"该用的机制:
thread_id 划分的是"一次对话/一个会话", user_id 划分的是"一个人", 两者维度不同。

关键机制 (用 inspect 验证过): 节点函数只要把参数命名为 `store` 并标注类型
`BaseStore`, LangGraph 在调用节点时会自动把 compile(store=...) 传入的 store
实例注入进来 (不需要手动从 config 里掏), 用法和注入 `config: RunnableConfig`
是同一套机制。store.put(namespace, key, value) 写入, store.search(namespace)
按 namespace 前缀取出该命名空间下的所有条目。

官方文档: https://docs.langchain.com/oss/python/langgraph/persistence#memory-store
"""

import os
import uuid
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph, add_messages
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore

load_dotenv(override=True)

model = ChatAnthropic(
    model=os.environ["MODEL_ID"],
    base_url=os.getenv("ANTHROPIC_BASE_URL") or None,
)


class State(TypedDict):
    messages: Annotated[list, add_messages]


def chat_node(state: State, config: RunnableConfig, *, store: BaseStore) -> dict:
    """每轮对话: 先从 store 里读出这个用户的长期记忆, 拼进 system prompt; 回答完
    再把这轮用户说的话存回 store —— 记忆的写入/读取和 thread_id 完全无关,
    只认 config 里的 user_id, 所以哪怕是全新的一次对话(新 thread_id), 只要
    user_id 一样, 依然能读到之前对话里存下的记忆。
    """
    user_id = config["configurable"]["user_id"]
    namespace = (user_id, "memories")  # namespace 是你自己定义的分组方式, 这里按用户分组

    # search(namespace) 取出该用户名下已存的所有记忆条目 (是个 list[SearchItem])
    remembered = store.search(namespace)
    if remembered:
        memory_text = "\n".join(f"- {item.value['text']}" for item in remembered)
    else:
        memory_text = "(暂无已知信息)"

    system_prompt = (
        f"以下是你对当前用户已经了解到的长期偏好/信息, 回答时请遵循:\n{memory_text}"
    )
    response = model.invoke([{"role": "system", "content": system_prompt}, *state["messages"]])

    # 把这轮用户说的话存成一条新记忆 (演示简化: 每轮都存; 生产环境通常会用一次
    # 额外的模型调用或规则判断"这句话是否值得长期记住", 而不是无脑全存)
    last_human_text = state["messages"][-1].content
    store.put(namespace, str(uuid.uuid4()), {"text": last_human_text})

    return {"messages": [response]}


builder = StateGraph(State)
builder.add_node("chat", chat_node)
builder.add_edge(START, "chat")
builder.add_edge("chat", END)
# checkpointer 管"每个 thread_id 自己的对话历史", store 管"跨 thread_id 共享的长期记忆",
# 两者可以同时配, 互不冲突, 分别解决不同维度的持久化需求。
graph = builder.compile(checkpointer=InMemorySaver(), store=InMemoryStore())


if __name__ == "__main__":
    print("=== 第 1 轮: thread_id=conv-1, user_id=alice, 用户声明偏好 ===")
    config1 = {"configurable": {"thread_id": "conv-1", "user_id": "alice"}}
    result1 = graph.invoke(
        {"messages": [HumanMessage("以后请你都用中文回答我, 我看不懂英文。")]},
        config=config1,
    )
    print(f"  模型回复: {result1['messages'][-1].text}")

    print("\n=== 第 2 轮: 全新 thread_id=conv-2 (模拟另一次全新对话), 但 user_id 还是 alice ===")
    config2 = {"configurable": {"thread_id": "conv-2", "user_id": "alice"}}
    result2 = graph.invoke(
        {"messages": [HumanMessage("What's the capital of France?")]},
        config=config2,
    )
    print(f"  模型回复: {result2['messages'][-1].text}")
    print("  (thread_id 变了, 按 checkpointer 的逻辑这应该是全新对话, 但 store 记得")
    print("   alice 说过'请用中文回答', 所以即使这轮用英文提问, 模型依然按偏好用中文回答)")

    print("\n=== 验证隔离性: 换一个 user_id=bob, 读不到 alice 的记忆 ===")
    config3 = {"configurable": {"thread_id": "conv-3", "user_id": "bob"}}
    result3 = graph.invoke(
        {"messages": [HumanMessage("What's the capital of France?")]},
        config=config3,
    )
    print(f"  模型回复: {result3['messages'][-1].text}")
    print("  (bob 是不同的 user_id, namespace 不同, 读不到 alice 存的偏好, 默认用英文回答)")

    print("\n=== 直接查看 alice 名下 store 里积累的记忆条目 ===")
    store = graph.store
    for item in store.search(("alice", "memories")):
        print(f"  [{item.key[:8]}] {item.value['text']}")

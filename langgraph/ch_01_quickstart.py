"""LangGraph quickstart —— 手动定义一个带条件分支的图, 理解 runtime 层。

对比 langchain/ch_01_quickstart.py: 那里 create_agent 把 agent loop 封装好了,
这里手动用 StateGraph 搭一个图, 看清 runtime 层的三件事:
  - state: 图的共享状态 (TypedDict), 节点读写它
  - conditional_edges: 根据状态做条件路由 (类似 if-else, 但是图结构)
  - checkpointer: 状态持久化, 同一 thread_id 跨调用可恢复

注: 这里节点用纯 Python 逻辑, 聚焦 graph 概念本身; 实际 agent 中节点会调用 LLM
(LangChain 的 create_agent 底层就是一个 LangGraph 图, stream 用的就是这里的 runtime).
官方文档: https://docs.langchain.com/oss/python/langgraph/low-level
"""

from typing import TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph


class State(TypedDict):
    input: str
    category: str  # classify 节点写入: weather / greeting / unknown
    result: str  # handler 节点写入
    steps: list[str]  # 记录执行轨迹


def classify_node(state: State) -> dict:
    """根据输入关键词分类 (实际 agent 里这里会调 LLM 做意图判断)."""
    text = state["input"].lower()
    if "weather" in text or "天气" in text:
        category = "weather"
    elif "hello" in text or "你好" in text:
        category = "greeting"
    else:
        category = "unknown"
    return {
        "category": category,
        "steps": state.get("steps", []) + ["classify"],
    }


def handle_weather(state: State) -> dict:
    return {
        "result": f"☀️ 天气类问题: {state['input']}",
        "steps": state.get("steps", []) + ["handle_weather"],
    }


def handle_greeting(state: State) -> dict:
    return {
        "result": "👋 你好!有什么可以帮你的吗?",
        "steps": state.get("steps", []) + ["handle_greeting"],
    }


def handle_unknown(state: State) -> dict:
    return {
        "result": "🤔 这个我暂时处理不了。",
        "steps": state.get("steps", []) + ["handle_unknown"],
    }


def route(state: State) -> str:
    """条件路由: 根据 classify 写入的 category 决定走哪个分支."""
    return f"handle_{state['category']}"


# -- 构建图 --
builder = StateGraph(State)
builder.add_node("classify", classify_node)
builder.add_node("handle_weather", handle_weather)
builder.add_node("handle_greeting", handle_greeting)
builder.add_node("handle_unknown", handle_unknown)

builder.add_edge(START, "classify")
builder.add_conditional_edges("classify", route)  # 条件分支: classify → handle_*
builder.add_edge("handle_weather", END)
builder.add_edge("handle_greeting", END)
builder.add_edge("handle_unknown", END)

# checkpointer 让图的每一步状态都可持久化 (按 thread_id 保存)
graph = builder.compile(checkpointer=InMemorySaver())


if __name__ == "__main__":
    queries = [
        "What's the weather in SF?",
        "Hello there!",
        "Tell me a joke",
    ]

    # 每个 query 用独立 thread, 互不干扰
    for i, query in enumerate(queries):
        config = {"configurable": {"thread_id": f"demo-{i}"}}
        print(f"\n=== input: {query!r} ===")
        result = graph.invoke({"input": query, "steps": []}, config=config)
        print(f"  category: {result['category']}")
        print(f"  result:   {result['result']}")
        print(f"  steps:    {' → '.join(result['steps'])}")

    # 展示 checkpoint: 同一 thread_id 的状态历史 (每一步都是一个 checkpoint)
    print("\n=== checkpoint 状态历史 (thread_id=demo-0) ===")
    config0 = {"configurable": {"thread_id": "demo-0"}}
    for state in graph.get_state_history(config0):
        steps = state.values.get("steps", [])
        if steps:  # 跳过初始空状态
            print(f"  steps={' → '.join(steps)}, result={state.values.get('result', '(none)')}")

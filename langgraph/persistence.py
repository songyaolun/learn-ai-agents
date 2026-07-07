"""LangGraph persistence —— 用 SqliteSaver 让 checkpoint 落盘, 跨进程持久化。

对比 langgraph/quickstart.py: 那里用 InMemorySaver, 进程一结束 checkpoint 全丢;
这里用 SqliteSaver 把每一步 state 写入 sqlite 文件, 新建 graph 实例 (相当于新进程)
用同一 thread_id 仍能读回之前的 state。这是 runtime 层持久化的关键。

官方文档: https://docs.langchain.com/oss/python/langgraph/checkpointers
"""

import os
import sqlite3
from typing import TypedDict

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph


class State(TypedDict):
    input: str
    result: str
    steps: list[str]


def process_node(state: State) -> dict:
    return {
        "result": f"处理了: {state['input']}",
        "steps": state.get("steps", []) + ["process"],
    }


def build_graph(checkpointer) -> object:
    """用给定 checkpointer 构建图 (同一个图定义, 不同 checkpointer)."""
    builder = StateGraph(State)
    builder.add_node("process", process_node)
    builder.add_edge(START, "process")
    builder.add_edge("process", END)
    return builder.compile(checkpointer=checkpointer)


DB_PATH = "langgraph_checkpoints.db"


def make_checkpointer() -> SqliteSaver:
    """每次新建一个 SqliteSaver 连同一 db 文件 (模拟新进程)."""
    return SqliteSaver(sqlite3.connect(DB_PATH, check_same_thread=False))


if __name__ == "__main__":
    # 清理旧 db, 保证演示干净
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)

    # 第一次: 用 thread_id="session-1" 跑, checkpoint 落盘
    print("=== 第一次运行: 写入 checkpoint 到 sqlite ===")
    graph = build_graph(make_checkpointer())
    config = {"configurable": {"thread_id": "session-1"}}
    result = graph.invoke({"input": "hello", "steps": []}, config=config)
    print(f"  result: {result['result']}, steps: {result['steps']}")

    # 第二次: 全新 graph 实例 (模拟新进程), 同一 db + thread_id, 能读回 state
    print("\n=== 第二次运行: 全新 graph 实例, 从磁盘读回 checkpoint ===")
    graph2 = build_graph(make_checkpointer())  # 新 graph, 内存里没有之前的 state
    snapshot = graph2.get_state(config)
    print(f"  从磁盘读到的 state: {snapshot.values}")
    print(f"  next: {snapshot.next}  (空 = 已跑完, state 已持久化)")

    print(f"\n(checkpoint 已写入 {DB_PATH}, 可用 sqlite3 查看)")

"""LangGraph human-in-the-loop —— 用 interrupt 在节点里暂停等人工审批。

对比 langgraph/ch_01_quickstart.py: 那里的图一路跑到底, 这里在关键节点用 interrupt
暂停执行, 等外部 (人) 决策后用 Command(resume=...) 恢复。这是 runtime 层的能力:
interrupt 时整个 state 被存入 checkpointer, 恢复时从断点继续, thread_id 是恢复指针。

官方文档: https://docs.langchain.com/oss/python/langgraph/interrupts
"""

from typing import TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt


class State(TypedDict):
    task: str
    approved: bool | None
    result: str
    steps: list[str]


def prepare_node(state: State) -> dict:
    """准备任务, 产出待审批的草案."""
    return {
        "result": f"草案: 执行任务 {state['task']!r}",
        "steps": state.get("steps", []) + ["prepare"],
    }


def approve_node(state: State) -> dict:
    """暂停等人工审批.

    interrupt(value) 把 value 暴露给外部 (在 snapshot.tasks[].interrupt 里), 图在此暂停;
    之后 Command(resume=X) 的 X 会作为 interrupt() 的返回值, 节点从这里继续执行.
    """
    approved = interrupt(
        {
            "message": "是否批准执行此任务?",
            "task": state["task"],
            "draft": state["result"],
        }
    )
    return {"approved": approved, "steps": state.get("steps", []) + ["approve"]}


def execute_node(state: State) -> dict:
    return {
        "result": f"✅ 已执行: {state['result']}",
        "steps": state.get("steps", []) + ["execute"],
    }


def cancel_node(state: State) -> dict:
    return {"result": "❌ 已取消", "steps": state.get("steps", []) + ["cancel"]}


def route_after_approve(state: State) -> str:
    """根据审批结果路由: 批准 → execute, 拒绝 → cancel."""
    return "execute" if state["approved"] else "cancel"


builder = StateGraph(State)
builder.add_node("prepare", prepare_node)
builder.add_node("approve", approve_node)
builder.add_node("execute", execute_node)
builder.add_node("cancel", cancel_node)

builder.add_edge(START, "prepare")
builder.add_edge("prepare", "approve")
builder.add_conditional_edges("approve", route_after_approve)
builder.add_edge("execute", END)
builder.add_edge("cancel", END)

# interrupt 必须配 checkpointer: 暂停时 state 落盘, 恢复时从断点继续
graph = builder.compile(checkpointer=InMemorySaver())


if __name__ == "__main__":
    config = {"configurable": {"thread_id": "approval-demo"}}

    print("=== 第 1 次 invoke: 跑到 approve 节点, 触发 interrupt 暂停 ===")
    graph.invoke({"task": "发送营销邮件", "steps": []}, config=config)
    snapshot = graph.get_state(config)
    print(f"  是否暂停: {bool(snapshot.next)}  (next={snapshot.next})")
    for task in snapshot.tasks:
        for intr in task.interrupts or []:
            print(f"  interrupt payload: {intr.value}")

    print("\n=== 第 2 次 invoke: Command(resume=True) 批准, 从断点恢复 ===")
    result = graph.invoke(Command(resume=True), config=config)
    print(f"  result: {result['result']}")
    print(f"  steps:  {' → '.join(result['steps'])}")

    print("\n=== 演示拒绝 (新 thread) ===")
    config2 = {"configurable": {"thread_id": "approval-reject"}}
    graph.invoke({"task": "删除生产数据库", "steps": []}, config=config2)
    result2 = graph.invoke(Command(resume=False), config=config2)
    print(f"  result: {result2['result']}")
    print(f"  steps:  {' → '.join(result2['steps'])}")

"""LangGraph 综合示例 —— subgraph + Send + Store + checkpointer 拼成一个多文档摘要器。

这是本目录的收尾篇, 把前面四个独立概念拼进同一个连贯场景, 而不是分别演示:
  - ch_10_subgraph.py 的能力: "总结一篇文档"本身是个两步子流程(先写初稿、再按用户风格
    精炼), 封装成一个独立的子图 summarize_doc_subgraph, 只暴露 doc/target_style 输入。
  - ch_11_send_map_reduce.py 的能力: 一次要处理几篇文档运行时才知道, 用 Send 把每篇
    文档动态派发给上面那个子图, 并行执行, 而不是写死处理 N 篇。
  - ch_06_store.py 的能力: 用户"喜欢什么风格的摘要"是跨会话的长期偏好——即使这是一次
    全新的 thread_id (新会话), 只要 user_id 一样, 依然记得上次的偏好设置。
  - ch_05_persistence.py / ch_01_quickstart.py 的能力: 每个 thread_id (一次摘要任务) 自己的
    执行进度/结果由 checkpointer 保存, 可以用 get_state 单独查看某次任务的状态。

四者的关系: checkpointer 管"一次任务内部的状态", store 管"跨任务共享的长期记忆",
subgraph 管"把一段多步流程封装成一个节点", Send 管"这个节点要并行跑几份"——
这正是一个可用的生产级 pattern: 多文档摘要服务, 每个用户有自己的风格偏好,
每次提交的一批文档是一个独立任务(thread), 任务内部对每篇文档并行摘要。

时间旅行 (ch_08_time_travel.py) 这里没有强行拼进来: 它是独立的调试类概念(回滚重跑),
硬凑进摘要场景只会让代码变复杂而不会让概念更清楚, 所以单独成篇更合适。

官方文档: https://docs.langchain.com/oss/python/langgraph/graph-api (subgraph / Send)
         https://docs.langchain.com/oss/python/langgraph/persistence (store / checkpointer)
"""

import operator
import os
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore
from langgraph.types import Send

load_dotenv(override=True)

model = ChatAnthropic(
    model=os.environ["MODEL_ID"],
    base_url=os.getenv("ANTHROPIC_BASE_URL") or None,
)


# ============================================================
# 子图: 总结单篇文档 = 先写初稿, 再按用户风格精炼成一句话
# ============================================================
class DocState(TypedDict):
    doc: str
    target_style: str  # 由父图通过 Send 传入, 来自 store 里记住的用户偏好
    draft: str
    summaries: Annotated[list[str], operator.add]  # 这个字段名故意和父图 OverallState 一致, 靠 reducer 把并行结果合并回去


def draft_node(state: DocState) -> dict:
    response = model.invoke([{"role": "user", "content": f"用 2-3 句话概括下面这段内容的要点:\n\n{state['doc']}"}])
    return {"draft": response.text}


def refine_node(state: DocState) -> dict:
    response = model.invoke(
        [
            {
                "role": "user",
                "content": (
                    f"把下面这份摘要初稿, 按这个风格要求精炼成一句话: 「{state['target_style']}」\n\n"
                    f"初稿:\n{state['draft']}"
                ),
            }
        ]
    )
    return {"summaries": [response.text]}


doc_builder = StateGraph(DocState)
doc_builder.add_node("draft", draft_node)
doc_builder.add_node("refine", refine_node)
doc_builder.add_edge(START, "draft")
doc_builder.add_edge("draft", "refine")
doc_builder.add_edge("refine", END)
# 子图不单独传 checkpointer/store —— 父图 compile 时配的 checkpointer/store 会自动
# 沿着"父图节点=这个子图"这条路径继承下去, 子图内部节点一样能注入 store。
summarize_doc_subgraph = doc_builder.compile()


# ============================================================
# 父图: 读取用户偏好 → 按文档数量动态并行摘要(子图) → 合并成最终报告
# ============================================================
class OverallState(TypedDict):
    docs: list[str]
    style: str
    summaries: Annotated[list[str], operator.add]
    final_report: str


def load_preference_node(state: OverallState, config: RunnableConfig, *, store: BaseStore) -> dict:
    """从 store 里读取这个用户长期设置的摘要风格偏好 (跨 thread_id 共享)。"""
    user_id = config["configurable"]["user_id"]
    item = store.get((user_id, "summary_prefs"), "style")
    style = item.value["value"] if item is not None else "简洁中立的中文要点式摘要"
    return {"style": style}


def fan_out_to_docs(state: OverallState) -> list[Send]:
    """处理几篇文档运行时才知道 (state['docs'] 的长度), 动态产出对应数量的并行任务,
    每个任务都送去 summarize_doc_subgraph 这个子图节点, 带上刚读到的用户风格偏好。

    踩坑记录: 这里 Send 传给子图的 key 用的是 target_style 而不是 style —— 起初
    直接叫 style (和父图 OverallState.style 同名), 实测会报
    InvalidUpdateError: At key 'style': Can receive only one value per step。
    原因: ch_10_subgraph.py 里讲过, 子图字段名如果和父图字段名相同, 状态会直接"透传"共用
    同一个 channel; summaries 字段特意设计成 Annotated[list, operator.add], 允许多个
    并行分支同时写入并自动合并, 但 style 是普通字段(没有 reducer), 多个并行 Send
    分支同时"写"这个共享 channel 就会冲突。解法: 除了故意要共享/合并的字段
    (这里是 summaries), 子图的其余字段名都要和父图区分开, 让每个并行分支各自持有
    私有状态, 不要意外共用父图的 channel。
    """
    return [
        Send("summarize_doc", {"doc": doc, "target_style": state["style"], "draft": "", "summaries": []})
        for doc in state["docs"]
    ]


def combine_node(state: OverallState) -> dict:
    # combine 节点也遵循同一份 style 偏好, 不然会出现"每篇摘要都是英文极简风格,
    # 最后合并报告却是中文长句"这种前后风格不一致的怪现象。
    bullet_list = "\n".join(f"- {s}" for s in state["summaries"])
    response = model.invoke(
        [
            {
                "role": "user",
                "content": (
                    f"把下面几条独立摘要合并成一份连贯的最终报告(3句话以内), "
                    f"并遵循这个风格要求: 「{state['style']}」\n\n{bullet_list}"
                ),
            }
        ]
    )
    return {"final_report": response.text}


builder = StateGraph(OverallState)
builder.add_node("load_preference", load_preference_node)
builder.add_node("summarize_doc", summarize_doc_subgraph)  # 子图直接作为并行 fan-out 的目标节点
builder.add_node("combine", combine_node)
builder.add_edge(START, "load_preference")
builder.add_conditional_edges("load_preference", fan_out_to_docs, ["summarize_doc"])
builder.add_edge("summarize_doc", "combine")
builder.add_edge("combine", END)

# checkpointer: 每次提交的一批文档(一个 thread_id)自己的执行进度/结果
# store: 跨 thread_id 共享的用户长期偏好, 两者在 compile 时一起配置
memory_store = InMemoryStore()
graph = builder.compile(checkpointer=InMemorySaver(), store=memory_store)


if __name__ == "__main__":
    docs_batch_1 = [
        "LangGraph 的 checkpointer 会在图执行的每一步之后自动保存完整状态快照, "
        "配合 thread_id 就能让同一个任务在暂停后从断点恢复, 或者跨进程重新连接。",
        "LangGraph 的 Store 和 checkpointer 是两套独立的持久化机制: checkpointer "
        "按 thread_id 隔离, store 按你自定义的 namespace 组织, 可以跨 thread_id 共享。",
    ]

    # 提前把 carol 的偏好写进 store, 模拟"用户之前的某次会话里设置过这个偏好"
    memory_store.put(("carol", "summary_prefs"), "style", {"value": "用英文, 极简一句话, 不超过20个单词"})

    print("=== Session 1: thread_id=session-1, user_id=carol, 2 篇文档并行摘要 ===")
    config1 = {"configurable": {"thread_id": "session-1", "user_id": "carol"}}
    result1 = graph.invoke({"docs": docs_batch_1, "style": "", "summaries": [], "final_report": ""}, config=config1)
    print(f"  读到的风格偏好: {result1['style']}")
    for i, s in enumerate(result1["summaries"]):
        print(f"  文档{i + 1} 摘要: {s}")
    print(f"  最终报告: {result1['final_report']}")

    print("\n=== Session 2: 全新 thread_id=session-2 (新任务), 但 user_id 还是 carol ===")
    docs_batch_2 = [
        "LangGraph 的 Send API 允许一个节点在运行时产出任意数量的并行任务, "
        "适合处理数量不固定的批量工作, 比如逐篇总结一批文档。",
    ]
    config2 = {"configurable": {"thread_id": "session-2", "user_id": "carol"}}
    result2 = graph.invoke({"docs": docs_batch_2, "style": "", "summaries": [], "final_report": ""}, config=config2)
    print(f"  读到的风格偏好: {result2['style']}  (新 thread, 但偏好被 store 记住了, 依然是英文极简风格)")
    print(f"  最终报告: {result2['final_report']}")

    print("\n=== Session 3: 换一个从未设置过偏好的用户 dave, 走默认风格 ===")
    config3 = {"configurable": {"thread_id": "session-3", "user_id": "dave"}}
    result3 = graph.invoke({"docs": docs_batch_2, "style": "", "summaries": [], "final_report": ""}, config=config3)
    print(f"  读到的风格偏好: {result3['style']}  (dave 没有存过偏好, 用的是默认值)")

    print("\n=== checkpointer 验证: session-1 和 session-2 各自的进度独立可查 ===")
    state1 = graph.get_state(config1)
    state2 = graph.get_state(config2)
    print(f"  session-1 最终报告: {state1.values['final_report'][:40]}...")
    print(f"  session-2 最终报告: {state2.values['final_report'][:40]}...")
    print("  (两个 thread_id 的状态互不覆盖, 是 checkpointer 按 thread 隔离的结果)")

"""LangGraph Send —— 运行期动态决定并行分支数量的 map-reduce 模式。

对比 langgraph/ch_12_multi_agent.py: 那里 Command(goto=...) 每次只路由到*一个*确定的
下一个节点, 而且分支有哪些 (researcher/writer/FINISH) 在写代码时就写死了;
这里用 Send 在一个节点里*一次性*产出一组"去 X 节点、带上 Y 参数"的任务, 数量由
运行时的数据决定 (比如要处理几篇文档, 只有拿到 state 才知道), 图会把这些任务
全部并行调度执行 —— 这就是 map (并行处理每篇文档) → reduce (合并结果) 模式。

关键机制 (用 inspect.signature(Send) 验证过): Send(node, arg) 里的 arg 会*完全替代*
目标节点收到的输入 (而不是和主 state 合并), 所以目标节点的输入类型可以是一个和主图
State 不同的小 TypedDict, 只包含这次任务需要的字段。目标节点的输出如果某个 key 在主
State 里被声明成 Annotated[list, operator.add] 之类的 reducer, 多个并行分支的输出会
按 reducer 规则自动合并回主 state (这里用 operator.add 把多份 summaries 列表拼接起来)。

官方文档: https://docs.langchain.com/oss/python/langgraph/graph-api#map-reduce-and-the-send-api
"""

import operator
import os
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

load_dotenv(override=True)

model = ChatAnthropic(
    model=os.environ["MODEL_ID"],
    base_url=os.getenv("ANTHROPIC_BASE_URL") or None,
)


# -- 主图 State: docs 数量运行时才知道, summaries 用 operator.add 做 reducer --
class OverallState(TypedDict):
    docs: list[str]
    summaries: Annotated[list[str], operator.add]  # 并行分支各自 append 一条, 自动拼接
    final_summary: str


# -- 单个并行分支收到的输入: 只有一篇文档, 跟主 State 是不同的类型 --
class DocState(TypedDict):
    doc: str


def fan_out_to_docs(state: OverallState) -> list[Send]:
    """扇出节点: 不是普通节点函数, 而是挂在 add_conditional_edges 上的路由函数。

    返回一个 Send 列表 (而不是单个字符串/节点名), LangGraph 看到列表就知道要
    并行调度这么多个任务, 每个 Send("summarize_doc", {...}) 各自带自己的输入。
    docs 有几篇, 这里就产出几个 Send —— 分支数量完全由运行时的数据决定。
    """
    return [Send("summarize_doc", {"doc": doc}) for doc in state["docs"]]


def summarize_doc_node(state: DocState) -> dict:
    """并行执行的节点: 每个分支各自拿到一篇文档, 互不干扰地调用一次模型。"""
    # 注意: 这里刻意不写"不超过 N 字"这种精确字数限制 —— 实测发现开启 extended
    # thinking 的模型会在 thinking 过程里反复数字数、纠结怎么凑够/不超限, 有时会
    # 把 thinking 预算耗尽导致最终 text block 被截断成空字符串。改成宽松的"简要"
    # 措辞后模型直接给结论, 更稳定。
    response = model.invoke(
        [{"role": "user", "content": f"用一句简洁的话总结下面这段内容:\n\n{state['doc']}"}]
    )
    return {"summaries": [response.text]}  # 返回值会通过 operator.add 合并进主 state


def combine_node(state: OverallState) -> dict:
    """reduce 节点: 等所有并行分支都跑完, 拿到完整的 summaries 列表后再执行。"""
    bullet_list = "\n".join(f"- {s}" for s in state["summaries"])
    response = model.invoke(
        [
            {
                "role": "user",
                "content": (
                    f"下面是若干篇文档各自的一句话摘要, 请把它们综合成一段连贯的总述"
                    f"(2-3句话即可):\n\n{bullet_list}"
                ),
            }
        ]
    )
    return {"final_summary": response.text}


builder = StateGraph(OverallState)
builder.add_node("summarize_doc", summarize_doc_node)
builder.add_node("combine", combine_node)
# 从 START 出发的条件边: path_map 里声明"这个路由函数可能会送去哪些节点",
# 供 LangGraph 做图结构校验/可视化用 (实际每次跑几个由 fan_out_to_docs 返回的列表长度决定)。
builder.add_conditional_edges(START, fan_out_to_docs, ["summarize_doc"])
builder.add_edge("summarize_doc", "combine")
builder.add_edge("combine", END)
graph = builder.compile()


if __name__ == "__main__":
    docs = [
        "LangChain 的 create_agent 把模型调用、工具执行、循环控制都封装进一行代码里,"
        "适合快速搭建标准的 agent loop。",
        "LangGraph 是更底层的图运行时, 开发者手动定义节点和边, 换来对状态流转、"
        "并行、暂停恢复的精细控制。",
        "DeepAgents 在 create_agent 之上叠加了自动任务规划、虚拟文件系统和子任务"
        "委派, 适合处理需要多步骤长程规划的复杂任务。",
        "RAG (检索增强生成) 把外部知识库通过向量检索接入模型上下文, 让模型能回答"
        "训练数据之外的、或者需要精确引用的私有信息。",
    ]

    print(f"=== 输入 {len(docs)} 篇文档, 并行 map 阶段 (每篇一次模型调用) ===")
    result = graph.invoke({"docs": docs, "summaries": [], "final_summary": ""})

    for i, (doc, summary) in enumerate(zip(docs, result["summaries"])):
        print(f"\n  文档{i + 1}: {doc[:24]}...")
        print(f"  摘要{i + 1}: {summary}")

    print("\n=== reduce 阶段: 合并成最终综述 ===")
    print(f"  {result['final_summary']}")

    # 换一批数量不同的文档, 证明并行分支数是运行时决定的, 不是写死在图结构里
    print("\n=== 再跑一次: 只给 2 篇文档, 验证分支数量随输入变化 ===")
    result2 = graph.invoke({"docs": docs[:2], "summaries": [], "final_summary": ""})
    print(f"  这次产出了 {len(result2['summaries'])} 条摘要 (对应 2 篇文档)")

"""LangGraph time travel —— 用 get_state_history + update_state 回到过去、改写分支。

对比 langgraph/ch_01_quickstart.py: 那里只在最后用 get_state_history 展示"看一眼"历史,
从没真正"回去改"过; 这里更进一步——不仅要看历史, 还要真的从某个过去的 checkpoint
"分叉"出一条新的执行路径, 这对调试 agent 很有实用价值: agent 在某一步判断错了
(比如解析错了用户意图/参数), 不用把整个对话推倒重来, 只需要回到出错的那一步之前,
把状态改对, 再从那里继续往下跑。

关键机制 (用 get_state_history/update_state 实测验证过):
  - graph.get_state_history(config) 按时间倒序返回这个 thread_id 的每一个 checkpoint
    (StateSnapshot), 每个 snapshot 都有自己独立的 checkpoint_id, 可以用
    snapshot.config 精确定位"图在那一刻的状态"。
  - graph.update_state(snapshot.config, values) 会基于*那个历史 checkpoint*
    创建一个*新的* checkpoint (在同一个 thread_id 下), 把 values 合并进去,
    并返回这个新 checkpoint 的 config。
  - 用 graph.invoke(None, config=new_config) 就会从这个新 checkpoint 继续往下跑
    ——因为 thread_id 相同, 这个新分支自动成为该 thread 此后的"最新历史",
    原来那条走向错误结果的路径依然保留在历史记录里, 只是不再是当前分支的延续。

官方文档: https://docs.langchain.com/oss/python/langgraph/add-human-in-the-loop#time-travel
"""

from typing import TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph


class State(TypedDict):
    question: str
    quantity: int  # interpret 节点从问题里"解析"出的数量
    unit_price: float
    total: float
    steps: list[str]


def interpret_node(state: State) -> dict:
    """解析用户问题, 提取计算需要的参数。

    这里故意模拟一次解析错误: 用户问的是"3 个单价 19.9 元的商品", 但 interpret
    节点(可以想象成一次 LLM 抽取参数的调用)误判成了 2 个 —— 这是现实中 agent
    经常会犯的错误: 参数抽取错、工具调错、路由判断错, 都属于这一类"某一步走岔了"。
    """
    return {
        "quantity": 2,  # 错误: 应该是 3, 模拟一次解析失误
        "unit_price": 19.9,
        "steps": state.get("steps", []) + ["interpret"],
    }


def compute_node(state: State) -> dict:
    """根据 interpret 节点解析出的参数算总价, 这一步本身没有错——错误来自上游输入。"""
    return {
        "total": round(state["quantity"] * state["unit_price"], 2),
        "steps": state["steps"] + ["compute"],
    }


builder = StateGraph(State)
builder.add_node("interpret", interpret_node)
builder.add_node("compute", compute_node)
builder.add_edge(START, "interpret")
builder.add_edge("interpret", "compute")
builder.add_edge("compute", END)
# time travel 依赖 checkpointer 记录下每一步的历史快照, 没有 checkpointer 就没有历史可回溯
graph = builder.compile(checkpointer=InMemorySaver())


if __name__ == "__main__":
    config = {"configurable": {"thread_id": "shopping-1"}}

    print("=== 第 1 次运行: interpret 解析错了数量, 得到错误结果 ===")
    result = graph.invoke(
        {"question": "3 个单价 19.9 元的商品, 一共多少钱?", "quantity": 0, "unit_price": 0.0, "total": 0.0, "steps": []},
        config=config,
    )
    print(f"  quantity={result['quantity']} (应该是 3, 解析错了)")
    print(f"  total={result['total']}  (错误结果, 少算了一份)")

    print("\n=== 查看完整历史: 每一步的 checkpoint ===")
    history = list(graph.get_state_history(config))
    for snap in history:
        # checkpoint_id 是 UUID7 格式, 开头一段是时间戳, 同一次运行里生成的多个
        # checkpoint 前 8 位大概率长得一样 (毫秒级时间戳分辨率不够), 只看第二段
        # (下划线切分后的第 2 个 UUID 分组) 才能看出彼此是不同的 checkpoint。
        cp_id = snap.config["configurable"]["checkpoint_id"].split("-")[1]
        print(f"  checkpoint=...{cp_id}  next={snap.next}  values={snap.values}")

    print("\n=== 定位到 interpret 刚跑完、compute 还没跑的那个 checkpoint ===")
    # next=('compute',) 说明这个快照是"interpret 已完成, 即将进入 compute"的那一刻
    target_snapshot = next(snap for snap in history if snap.next == ("compute",))
    print(f"  找到目标 checkpoint, 此时 quantity={target_snapshot.values['quantity']}")

    print("\n=== 在这个历史 checkpoint 上打补丁: 把 quantity 改成正确的 3 ===")
    forked_config = graph.update_state(target_snapshot.config, {"quantity": 3})
    new_cp_id = forked_config["configurable"]["checkpoint_id"].split("-")[1]
    print(f"  update_state 返回了一个新的 checkpoint config: ...{new_cp_id}")

    print("\n=== 从这个新 checkpoint 继续往下跑 (invoke(None, ...) 表示不追加新输入, 只是继续执行) ===")
    fixed_result = graph.invoke(None, config=forked_config)
    print(f"  quantity={fixed_result['quantity']}  total={fixed_result['total']}  (修正后的正确结果)")
    print(f"  执行轨迹: {' → '.join(fixed_result['steps'])}")

    print("\n=== 验证: 同一个 thread_id 的“当前状态”现在已经指向修正后的分支 ===")
    latest = graph.get_state(config)
    print(f"  最新状态 total={latest.values['total']}  (原来错误的那条历史依然存在于 get_state_history 里, 只是不再是分支终点)")

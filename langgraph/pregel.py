"""LangGraph 底层执行模型 —— Pregel / BSP 的超步(super-step)与同步屏障机制。

先解释两个缩写(用户偏好:首次出现的英文缩写先给中文全称/含义再展开):
  - BSP = Bulk Synchronous Parallel, 中文"批量同步并行", 是一种并行计算模型:
    计算分成一轮一轮的"超步"(super-step), 每个超步内各计算单元并行独立算,
    算完在一道"同步屏障"(sync barrier)处集合, 统一交换/提交数据, 再进下一超步。
  - Pregel 是 Google 2010 年提出的大规模图计算模型(论文名就叫 Pregel), 它正是
    BSP 在"图"上的实现——每个顶点是一个计算单元, 一轮超步里所有顶点并行计算、
    在屏障处统一收发消息。LangGraph 的运行时就以它命名(源码包 langgraph.pregel),
    一个图节点约等于 Pregel 里的一个顶点。

对比 langgraph/send_map_reduce.py: 那个文件讲的是 Send(动态分发)——在一个路由函数里
按运行时数据产出任意数量的并行任务; 它关心的是"这一批要并行跑几份、每份带什么输入"。
本文件不讲怎么产生并行, 而是往下挖一层, 讲这些并行节点*到底怎么被调度和提交的*:
即无论并行来自静态 fan-out 还是 Send, 底层都跑在同一套 Pregel/BSP 引擎上——按超步
推进, 每个超步内的节点并行执行、读的是同一份"超步开始时的状态快照", 各自算完不立刻
互相可见, 而是全部到达同步屏障后, 由 reducer 把这一超步内所有更新*批量原子地*合并进
state, 再开下一超步。send_map_reduce.py 演示"分发", 本文件演示"分发之后的执行语义"。

关键机制 / 踩坑(用 stream_mode 观测过, 本地 langgraph 1.2.9):
  1. 超步边界怎么划分: 一个节点执行完, 它出边指向的下游节点进入*下一个*超步; 由同一个
     上游节点同时 fan-out 出去的多个下游节点, 会落在*同一个*超步里并行执行。注意一个反直觉点:
     两条独立的 add_edge(START, a) / add_edge(START, b), a 和 b 未必在同一超步——真正落到
     同一超步的是"被同一次提交一起触发"的节点。所以本文件用"单节点 root 扇出到 a/b/c"这种
     结构来稳定地制造出同一超步内的并行。
  2. 同步屏障下的批量原子提交: 同一超步内的并行节点, 读到的都是本超步*开始时*的状态快照,
     谁都看不到同超步其它节点的中间写入(见 __main__ 里 seen 字段的断言: a/b/c 都只看到
     ('root',), 看不到彼此)。它们的输出被暂存, 直到屏障处才由 reducer 一次性合并——这就是
     "并行执行、但提交是批量原子的"。
  3. 为什么同超步多个节点写同一 key 必须靠 reducer: 因为它们的写入是在屏障处*同时*到达的,
     没有先后可言, 框架无法用"后写覆盖先写"来决定谁赢。普通字段(无 reducer)在一个超步内被
     多个节点写就会报 InvalidUpdateError: Can receive only one value per step。带 reducer 的
     字段(如 Annotated[list, operator.add])则把这一超步内的多份写入按 reducer 规则聚合成
     一个值——所以并行写共享 key *必须*声明 reducer, 靠合并而不是靠覆盖。
  4. 怎么观测超步: stream_mode="values" 恰好在*每道同步屏障*处 emit 一次完整 state 快照,
     因此它 emit 的次数 = 超步数, 每次快照就是那个超步屏障提交后的状态; 而 stream_mode=
     "updates" 是按*节点*粒度 emit(同超步的并行节点会分成多条), 不适合用来数超步。本文件
     用 values 模式来切出超步边界。

官方文档: https://docs.langchain.com/oss/python/langgraph/graph-api#runtime
         (Pregel/BSP 执行模型: super-step 与并行执行的语义)
"""

import operator  # operator.add: 作为 list 的 reducer, 把多份并行写入拼接合并
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph

# 说明: 本文件是*纯执行机制*演示, 全程不需要 LLM, 节点都是普通 Python 函数,
# 因此不 import 模型、不 load .env——按交接包要求, 纯机制演示优先用普通函数节点。


# ============================================================
# State: 用带 reducer 的字段来记录"谁在第几超步、看到了什么"
# ============================================================
class State(TypedDict):
    # log: 每个节点执行时 append 自己的名字。声明 operator.add 作为 reducer,
    # 这样同一超步内多个并行节点各写一份, 屏障处按 add 拼接, 而不是互相覆盖。
    log: Annotated[list[str], operator.add]
    # step_of: 每个节点记录"我执行时看到的 log 长度", 用来间接反映自己所处的超步:
    # 同超步的并行节点读到的是同一份超步初始快照, 所以它们记录的长度必然相同。
    step_of: Annotated[list[tuple[str, int]], operator.add]
    # seen: 每个节点记录"我执行时看到的完整 log 内容"(转成 tuple 便于断言),
    # 用来证明同超步并行节点彼此看不到对方的中间写入。
    seen: Annotated[list[tuple[str, tuple]], operator.add]


def _observe(name: str, state: State) -> dict:
    """公共观测逻辑: 节点执行时, 记录自己看到的状态快照, 并把自己名字写进 log。

    关键点: state 是本超步*开始时*的快照——即上一道屏障提交后的状态。所以同一超步内
    并行的几个节点调用 _observe 时, 拿到的 state 完全一样(看不到彼此这一步的写入)。
    """
    return {
        "log": [name],  # 通过 operator.add 合并; 同超步多节点写这里, 靠 reducer 聚合
        "step_of": [(name, len(state["log"]))],  # 记录本超步初始 log 长度
        "seen": [(name, tuple(state["log"]))],  # 记录本超步初始 log 的完整内容
    }


# 五个节点: root 先跑, 然后 root 同时扇出到 a/b/c(它们落在同一超步并行),
# a/b/c 都汇入 join。这样可以清晰地切出 4 个超步: root | a,b,c | join(第 0 步是初始状态)。
def root_node(state: State) -> dict:
    return _observe("root", state)


def a_node(state: State) -> dict:
    return _observe("a", state)


def b_node(state: State) -> dict:
    return _observe("b", state)


def c_node(state: State) -> dict:
    return _observe("c", state)


def join_node(state: State) -> dict:
    return _observe("join", state)


builder = StateGraph(State)
builder.add_node("root", root_node)
builder.add_node("a", a_node)
builder.add_node("b", b_node)
builder.add_node("c", c_node)
builder.add_node("join", join_node)

builder.add_edge(START, "root")
# root 的三条出边: a/b/c 由 root 这*同一次提交*触发, 因此会落在同一个超步里并行执行。
builder.add_edge("root", "a")
builder.add_edge("root", "b")
builder.add_edge("root", "c")
# a/b/c 三条边都指向 join: join 要等 a/b/c *全部*到达屏障后, 才在下一超步执行(汇合)。
builder.add_edge("a", "join")
builder.add_edge("b", "join")
builder.add_edge("c", "join")
builder.add_edge("join", END)

graph = builder.compile()


if __name__ == "__main__":
    init = {"log": [], "step_of": [], "seen": []}

    print("=== 1. 用 stream_mode='values' 切出超步边界(每道同步屏障 emit 一次) ===")
    # values 模式在每个超步的屏障处 emit 一次完整快照, emit 次数 = 超步数。
    barriers = list(graph.stream(dict(init), stream_mode="values"))
    for i, snap in enumerate(barriers):
        print(f"  超步 {i} 屏障后 log = {snap['log']}")

    # 断言超步划分: 第 0 步是初始空状态; 第 1 步只有 root; 第 2 步 root/a/b/c 同批提交;
    # 第 3 步再加上 join。注意 a/b/c 的相对顺序不保证(并行), 所以用 set 比较那一批新增的。
    assert barriers[0]["log"] == [], "第 0 个快照应是初始状态"
    assert barriers[1]["log"] == ["root"], "第 1 超步只应执行 root"
    step2_new = set(barriers[2]["log"]) - set(barriers[1]["log"])
    assert step2_new == {"a", "b", "c"}, f"第 2 超步应并行提交 a/b/c, 实际新增 {step2_new}"
    assert barriers[3]["log"][-1] == "join", "第 3 超步应执行汇合节点 join"
    print("  [断言通过] 超步序列 = [初始] -> [root] -> [a,b,c 并行同批] -> [join]")
    print(f"  [断言通过] a/b/c 三个节点在同一超步被批量合并进 log(共增 {len(step2_new)} 条)")

    print("\n=== 2. 证明同步屏障下的批量原子提交: 并行节点读的是超步初始快照 ===")
    result = graph.invoke(dict(init))
    # step_of: 同超步的并行节点看到的初始 log 长度必然相同。root 在第 1 超步(初始 log 长 0),
    # a/b/c 都在第 2 超步(初始 log 长 1, 即只有 root), join 在第 3 超步(初始 log 长 4)。
    step_of = dict(result["step_of"])
    print(f"  各节点执行时看到的 log 长度: {step_of}")
    assert step_of["root"] == 0, "root 处于第 1 超步, 应只看到空初始 log"
    assert step_of["a"] == step_of["b"] == step_of["c"] == 1, "a/b/c 同超步, 都应只看到 root"
    assert step_of["join"] == 4, "join 在下一超步, 应看到 root+a+b+c 共 4 条"
    print("  [断言通过] a/b/c 看到的初始长度都=1(同超步同快照), join 看到=4(屏障提交后)")

    # seen: 更强的证据——a/b/c 都只看到 ('root',), 谁也看不到同超步另外两个的写入。
    seen = dict(result["seen"])
    print(f"  各并行节点看到的 log 内容: a={seen['a']} b={seen['b']} c={seen['c']}")
    assert seen["a"] == seen["b"] == seen["c"] == ("root",), "同超步并行节点应互相看不到对方的写入"
    print("  [断言通过] a/b/c 都只看到 ('root',), 彼此中间写入不可见 —— 提交是屏障处批量原子的")

    print("\n=== 3. 证明同超步写共享 key 必须靠 reducer(而非覆盖) ===")
    # a/b/c 同超步都往 log 写了一份, 若靠覆盖只会剩一份; 靠 operator.add 合并则三份都在。
    final_log = result["log"]
    print(f"  最终 log = {final_log}")
    assert final_log.count("a") == 1 and final_log.count("b") == 1 and final_log.count("c") == 1, (
        "a/b/c 三份并行写入应都被 reducer 保留, 而不是相互覆盖只剩一份"
    )
    assert len(final_log) == 5, "root + a + b + c + join = 5 份写入应全部合并保留"
    print("  [断言通过] 同超步 3 份并行写入被 operator.add 全部合并保留(len=5), 未发生覆盖")

    # 反证: 如果把 log 换成*没有 reducer*的普通字段, 同超步并行写同一 key 会直接报错。
    print("\n=== 4. 反证: 普通字段(无 reducer)被同超步多节点写会报错 ===")

    class BadState(TypedDict):
        val: str  # 普通字段, 没有 reducer, 一个超步内只允许收到一个值

    def bad_root(s: BadState) -> dict:
        return {"val": "root"}

    def bad_write(s: BadState) -> dict:
        return {"val": "x"}  # a/b 同超步都写 val, 无 reducer 无法决定谁赢

    bb = StateGraph(BadState)
    bb.add_node("root", bad_root)
    bb.add_node("a", bad_write)
    bb.add_node("b", bad_write)
    bb.add_edge(START, "root")
    bb.add_edge("root", "a")  # a/b 由 root 同批触发, 落在同一超步
    bb.add_edge("root", "b")
    bb.add_edge("a", END)
    bb.add_edge("b", END)
    bad_graph = bb.compile()
    try:
        bad_graph.invoke({"val": ""})
        raise SystemExit("预期应报错但没有, 说明环境行为与文档不符")
    except Exception as e:  # 期望 InvalidUpdateError: Can receive only one value per step
        print(f"  [符合预期] 同超步多节点写无 reducer 的 key 报错: {type(e).__name__}: {e}")
    print("  结论: 同超步并行写同一 key, 要么声明 reducer 合并, 要么就是非法状态更新。")

    print("\n=== 全部断言通过: Pregel/BSP 的超步 + 同步屏障 + reducer 批量合并语义被验证 ===")

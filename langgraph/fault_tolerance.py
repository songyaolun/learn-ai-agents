"""LangGraph fault tolerance —— 节点级 RetryPolicy 重试 + checkpointer 崩溃恢复(durable execution)。

对比 langgraph/persistence.py: persistence.py 只讲 checkpoint 的"基本落盘/thread 隔离"——
用 SqliteSaver 把每步 state 写进 sqlite, 换个进程用同一 thread_id 能读回 state。本文件在
"能读回"之上再进一步, 演示 checkpoint 真正的价值: **容错**。两条互补的机制:
  1. 节点级 RetryPolicy: 某个节点抛瞬时异常(网络抖动/限流/临时不可用)时, 框架在同一超步内
     原地自动重试 max_attempts 次, 不必让整张图失败。
  2. durable execution(持久化执行): 节点抛出未被重试兜住的异常导致整图崩溃后, 换新进程用
     同一 thread_id 重新 invoke(None), 会从最近一次成功的 checkpoint 处继续——**已经跑完的
     上游节点不会重跑**(靠 checkpoint 缓存的结果), 只重放崩溃的那个节点及其之后。
persistence.py 关心"state 存没存下来", 本文件关心"崩了之后哪些工作不会丢、哪个节点会被重放"。

关键机制/踩坑记录(均已 inspect + 实跑验证, 基于本地 langgraph 1.2.9):
  - RetryPolicy.retry_on 默认值 default_retry_on 对 ValueError/TypeError/RuntimeError/OSError
    等常见异常一律返回 False(即"不重试", 因为这些通常是代码 bug 而非瞬时故障)。所以想演示
    "瞬时失败自动重试", 要么自定义一个异常类(默认对未知异常返回 True 会重试), 要么显式传
    retry_on=你的异常类型。本文件用自定义 TransientError + 显式 retry_on, 两个都点明。
  - RetryPolicy 的重试发生在**同一个超步(superstep)内部**, 每次重试并不额外生成 checkpoint;
    重试次数无法从 checkpoint 直接读, 观测方式是在节点里用外部计数器(这里用全局 dict)数调用
    次数, 断言"函数被调了 N 次、第 N 次才成功"。initial_interval/backoff_factor/max_interval/
    jitter 控制退避节奏, 演示时设成极小值且 jitter=False, 让重试瞬间完成、次数可断言。
  - durable 恢复时"已完成节点不重跑"是超步边界决定的: checkpoint 是在每个超步**结束**时落盘的,
    所以恢复点 = 崩溃节点所在超步的**入口** state。崩溃前已经跑完并落盘的上游节点结果被当作
    缓存直接复用, 不会重新执行(这里用全局计数器 node_a 的调用次数恒为 1 来断言)。恢复靠
    invoke(None): 不传新输入, 表示"从 checkpoint 续跑"而不是"开一个新任务"。
  - durability 模式('sync'/'async'/'exit')控制 checkpoint 写盘的时机而非"存不存":
    'sync' 每个超步结束同步写完再往下走(最强持久性, 崩溃点前的进度一定不丢);
    'async' 后台异步写(更快, 极端情况下可能丢最后一步); 'exit' 尽量攒到整体退出时再写。
    实测(见 __main__ 第 3 节): 即便用 'exit', StateGraph+SqliteSaver 在崩溃退出时仍会把已完成
    超步的 state 落盘, 恢复时上游节点同样不重跑——所以三种模式的差别是"写多勤/多快", 不是
    "崩了会不会从头再来"。演示统一用 durability='sync', 语义最直观。

官方文档: https://docs.langchain.com/oss/python/langgraph/durable-execution
         https://docs.langchain.com/oss/python/langgraph/graph-api  (RetryPolicy / add_node)
"""

import os  # 仅用于拼临时 db 路径
import shutil  # 结尾清理沙箱目录
import sqlite3  # SqliteSaver 需要一个 sqlite 连接
import tempfile  # 建沙箱目录, 不在仓库留 db

from typing import TypedDict  # 定义图的 state schema

from langgraph.checkpoint.sqlite import SqliteSaver  # 落盘型 checkpointer, 崩溃恢复靠它
from langgraph.graph import END, START, StateGraph  # 构图三件套
from langgraph.types import RetryPolicy  # 节点级重试策略


# ============================================================
# 机制一: RetryPolicy —— 瞬时失败在同一超步内自动重试
# ============================================================
class TransientError(Exception):
    """自定义的"瞬时故障"异常。用自定义类而非 ValueError 是有意为之:
    RetryPolicy 默认的 default_retry_on 对 ValueError/RuntimeError 等返回 False(不重试),
    对未知异常类型返回 True(重试)。本文件同时还显式传 retry_on=TransientError, 双保险。"""


class RetryState(TypedDict):
    value: int  # 节点最终写入的结果, 用来断言"最终成功了"


# 全局计数器: 观测 flaky_node 到底被调了几次(重试次数不落 checkpoint, 只能这样数)
flaky_calls = {"count": 0}


def flaky_node(state: RetryState) -> dict:
    """前 2 次调用抛 TransientError(模拟网络抖动/限流), 第 3 次才成功。
    RetryPolicy(max_attempts=5) 会在同一超步内原地重试, 直到成功或用尽次数。"""
    flaky_calls["count"] += 1  # 每被调用一次就 +1
    if flaky_calls["count"] < 3:  # 前两次(第1、2次)故意失败
        raise TransientError(f"瞬时故障, 第 {flaky_calls['count']} 次调用失败")
    # 第 3 次: 正常返回结果
    return {"value": state["value"] + 100}


def build_retry_graph() -> object:
    """构建一张只有一个 flaky 节点的图, 给该节点挂上 RetryPolicy。"""
    builder = StateGraph(RetryState)
    builder.add_node(
        "flaky",
        flaky_node,
        # 重试策略挂在节点级别: 最多尝试 5 次(含首次);
        # 退避参数设成极小 + 关掉 jitter, 让重试瞬间完成、次数可断言;
        # retry_on 显式指定只对 TransientError 重试(其它异常直接冒泡, 不会被无谓重试)。
        retry_policy=RetryPolicy(
            max_attempts=5,  # 最多 5 次尝试
            initial_interval=0.01,  # 首次重试前等 0.01s
            backoff_factor=1.0,  # 每次重试间隔不再放大(演示用, 生产常用 2.0)
            jitter=False,  # 关掉随机抖动, 让行为确定可断言
            retry_on=TransientError,  # 只有 TransientError 才触发重试
        ),
    )
    builder.add_edge(START, "flaky")
    builder.add_edge("flaky", END)
    return builder.compile()  # 演示重试不需要 checkpointer


# ============================================================
# 机制二: durable execution —— 崩溃后从最近 checkpoint 恢复, 已完成节点不重跑
# ============================================================
class DurableState(TypedDict):
    log: list[str]  # 记录哪些节点跑过, 用来看恢复后的完整轨迹


# 两个节点各自的全局调用计数器: 用来断言恢复时"node_a 不重跑、只有 node_b 重放"
node_calls = {"a": 0, "b": 0}

# 一个开关, 让 node_b 第一次运行时崩溃, 恢复运行时正常通过
crash_switch = {"should_crash": True}


def node_a(state: DurableState) -> dict:
    """上游节点: 无论如何只应该执行一次。恢复时它已在崩溃前跑完并落盘, 会被跳过。"""
    node_calls["a"] += 1  # 记录 a 的执行次数
    return {"log": state.get("log", []) + ["a"]}


def node_b(state: DurableState) -> dict:
    """下游节点: 第一次运行时崩溃(未被重试兜住 → 整图失败),
    恢复运行时开关已关, 正常完成。"""
    node_calls["b"] += 1  # 记录 b 的执行次数
    if crash_switch["should_crash"]:  # 第一次: 制造一次"进程崩溃"
        raise RuntimeError("node_b 崩溃, 模拟进程挂掉")
    return {"log": state["log"] + ["b"]}


def build_durable_graph(checkpointer) -> object:
    """a → b 两步链。b 不挂 RetryPolicy —— 这里要演示的是"重试兜不住/直接崩溃"后,
    靠 checkpointer 做进程级恢复, 而不是节点级重试。"""
    builder = StateGraph(DurableState)
    builder.add_node("a", node_a)
    builder.add_node("b", node_b)
    builder.add_edge(START, "a")
    builder.add_edge("a", "b")
    builder.add_edge("b", END)
    return builder.compile(checkpointer=checkpointer)  # 恢复的关键: 带 checkpointer


def make_saver(db_path: str) -> SqliteSaver:
    """每次新建一个连同一 db 文件的 SqliteSaver(模拟"换了个新进程")。"""
    return SqliteSaver(sqlite3.connect(db_path, check_same_thread=False))


if __name__ == "__main__":
    # ---------- 第 1 节: RetryPolicy 瞬时失败自动重试 ----------
    print("=== 第1节: RetryPolicy —— 前2次抛异常, 第3次成功, 断言重试次数 ===")
    flaky_calls["count"] = 0  # 重置计数器
    retry_graph = build_retry_graph()
    retry_out = retry_graph.invoke({"value": 1})  # 图内部会自动重试, 对外表现为一次成功调用
    print(f"  flaky_node 实际被调用次数: {flaky_calls['count']}  (前2次失败, 第3次成功)")
    print(f"  最终结果 value: {retry_out['value']}  (1 + 100)")
    assert flaky_calls["count"] == 3, f"应重试到第3次成功, 实际调了 {flaky_calls['count']} 次"
    assert retry_out["value"] == 101, "最终应成功返回 101"
    print("  [断言通过] 重试了 2 次、第 3 次成功, 图对外只表现为一次成功 invoke")

    # ---------- 第 2 节: durable execution 崩溃恢复, 已完成节点不重跑 ----------
    print("\n=== 第2节: durable execution —— node_b 崩溃后从 checkpoint 恢复 ===")
    sandbox = tempfile.mkdtemp()  # 沙箱目录, 用完即删, 不在仓库留 db
    db_path = os.path.join(sandbox, "ft_checkpoints.db")
    node_calls["a"] = 0  # 重置计数器
    node_calls["b"] = 0
    crash_switch["should_crash"] = True  # 第一次运行让 b 崩溃
    cfg = {"configurable": {"thread_id": "job-1"}}  # 同一 thread_id 才能续跑

    # 第一次运行: 进程 A, node_a 成功落盘, node_b 崩溃, 整图抛异常
    saver_a = make_saver(db_path)
    graph_a = build_durable_graph(saver_a)
    try:
        # durability='sync': 每个超步结束同步写盘, 保证崩溃点前的进度绝不丢失
        graph_a.invoke({"log": []}, config=cfg, durability="sync")
    except RuntimeError as e:
        print(f"  第一次运行崩溃: {e}")
    print(f"  崩溃后计数器: node_a={node_calls['a']}, node_b={node_calls['b']}")
    snapshot = graph_a.get_state(cfg)  # 看 checkpoint 记到了哪
    print(f"  checkpoint 记录的 state: {snapshot.values}")
    print(f"  下一个待执行节点 next: {snapshot.next}  (指向崩溃的 b, 说明 a 已完成)")
    saver_a.conn.close()  # 关掉"进程 A"的连接, 模拟进程结束

    # 第二次运行: 进程 B(全新 saver/graph), 关掉崩溃开关, 从 checkpoint 恢复续跑
    crash_switch["should_crash"] = False  # 这次 b 不再崩溃
    saver_b = make_saver(db_path)
    graph_b = build_durable_graph(saver_b)
    # invoke(None): 不给新输入 = "从 checkpoint 续跑", 而不是开新任务
    recovered = graph_b.invoke(None, config=cfg, durability="sync")
    print(f"  恢复后最终 state: {recovered}")
    print(f"  最终计数器: node_a={node_calls['a']}, node_b={node_calls['b']}")
    # 核心断言: node_a 全程只跑 1 次(恢复时被 checkpoint 缓存跳过, 不重跑)
    assert node_calls["a"] == 1, f"node_a 应只执行1次(恢复不重跑), 实际 {node_calls['a']}"
    # node_b 跑了 2 次: 第一次崩溃 + 恢复时重放一次
    assert node_calls["b"] == 2, f"node_b 应执行2次(崩溃1次+恢复重放1次), 实际 {node_calls['b']}"
    assert recovered["log"] == ["a", "b"], "恢复后轨迹应是完整的 a→b"
    print("  [断言通过] 恢复时 node_a 未重跑(计数恒为1), 只有崩溃的 node_b 被重放, 轨迹 a→b 完整")
    saver_b.conn.close()

    # ---------- 第 3 节: durability='exit' 语义补充 ----------
    print("\n=== 第3节: durability 模式('sync'/'async'/'exit')语义 ===")
    print("  'sync' : 每个超步结束同步写盘, 持久性最强, 崩溃点前的进度绝不丢(本文件默认用它)")
    print("  'async': 后台异步写盘, 更快, 极端情况下可能丢最后一步")
    print("  'exit' : 尽量攒到整体退出时再写盘, 吞吐最好")
    # 实测: StateGraph+SqliteSaver 在崩溃退出时, 即便 'exit' 也会把已完成超步的 state 落盘,
    # 恢复时上游节点同样不重跑。所以三种模式差别是"写多勤/多快", 不是"崩了会不会从头再来"。
    node_calls["a"] = 0
    node_calls["b"] = 0
    crash_switch["should_crash"] = True
    cfg_exit = {"configurable": {"thread_id": "job-2"}}
    saver_c = make_saver(db_path)
    graph_c = build_durable_graph(saver_c)
    try:
        graph_c.invoke({"log": []}, config=cfg_exit, durability="exit")  # 用 exit 模式跑并崩溃
    except RuntimeError:
        pass
    snap_exit = graph_c.get_state(cfg_exit)
    print(f"  'exit' 模式崩溃后 checkpoint state: {snap_exit.values}, next: {snap_exit.next}")
    assert snap_exit.values.get("log") == ["a"], "exit 模式下 a 的结果同样已落盘"
    assert snap_exit.next == ("b",), "恢复点同样指向崩溃的 b"
    print("  [断言通过] 即便 exit 模式, 已完成的 node_a 结果照样落盘, 恢复点指向 b(不会从头再来)")
    saver_c.conn.close()

    # 清理沙箱, 不在仓库目录留 db
    shutil.rmtree(sandbox, ignore_errors=True)
    print(f"\n(临时 db 沙箱 {sandbox} 已清理)")

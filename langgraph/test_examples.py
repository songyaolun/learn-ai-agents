"""LangGraph 单元测试 —— 用 pytest + 假模型注入, 对图逻辑做确定性断言(兼作本仓库冒烟自测)。

对比 langgraph/combo.py: combo.py 在 __main__ 里内联 assert, 靠 `python combo.py` 手动跑一遍
看输出, 而且它直连真实 ChatAnthropic(需要 MODEL_ID / API key / 联网, 结果不确定)。本文件把
"验证图行为"这件事规范化成 pytest 风格的 test 函数, 并且**用假模型(FakeChatModel)替换真实
LLM**, 让每个 test 都能离线、确定、可断言地跑通。二者是"手写内联断言脚本" vs "标准测试套件"
的差别: 前者演示概念, 后者是你在真实项目里给 agent/图写单测该有的姿势。

为什么 agent/图要用假模型测试(核心动机):
  - 真实 LLM 输出**不确定**: 同样的 prompt 每次措辞都不同, 没法写 `assert result == "..."`。
  - 真实 LLM **慢**: 一次调用几百毫秒到几秒, CI 里跑几十个 test 会拖成几分钟。
  - 真实 LLM **要钱、要密钥、要联网**: 单测应该在没有 API key 的机器上、断网也能绿。
  测试要验证的是**你的图逻辑**(路由对不对、reducer 合得对不对、checkpoint 落没落盘、Send
  分了几路), 而不是"模型聪不聪明"。所以把模型换成一个"喂什么吐什么"的替身, 图逻辑就变成
  纯确定性函数, 可以像测普通代码一样断言。

关键机制/踩坑记录(均已实跑验证, 基于本地 langgraph 1.2.9 + langchain-core):
  - 两种假模型行为不同, 别用混:
      * GenericFakeChatModel(messages=iter([...])): 内部是**一次性迭代器**, 逐条吐, 吐完再调
        会抛 StopIteration。适合"我精确知道图会调模型几次、想让第 N 次返回什么"的场景; 但
        一个实例只能用一轮, 每个 test 必须**新建实例**, 否则上个 test 耗尽的迭代器会污染这个。
      * FakeListChatModel(responses=[...]): 到末尾会**循环**从头再来(实测第3次调用又回到第1条),
        永不耗尽。适合"图会调几次不确定、只要每次都有个确定返回"的场景。
    本文件对"需要精确计数/顺序"的用例用 GenericFakeChatModel, 对"只要有稳定返回"的用例用
    FakeListChatModel, 并各自新建实例。
  - **每个 test 都要新建图 + 新建替身**: 图 compile 出来会持有节点闭包引用的那个模型实例;
    InMemorySaver 也会累积 thread 状态。共享实例会让 test 之间相互串味, 违背单测隔离原则。
    本文件用 pytest fixture / 每个 test 内部 build_xxx() 来保证每次都是全新的图和模型。
  - 异步图(ainvoke/astream)要么用 `@pytest.mark.asyncio`(需装 pytest-asyncio), 要么在同步
    test 里 `asyncio.run(...)`。本文件为零额外依赖, 统一用 asyncio.run, 不依赖插件。
  - checkpointer 落盘 test 用 SqliteSaver 写真实 sqlite 文件时, 用 tempfile 沙箱 + fixture
    结尾 rmtree 清理, 不在仓库目录留 db(遵循本仓库沙箱红线)。内存态可断言的用 InMemorySaver。

官方文档: https://docs.langchain.com/oss/python/langchain/test  (给 agent 写单测的官方指引)
         https://docs.langchain.com/oss/python/langgraph/graph-api  (conditional_edges / Send / reducer)
         https://docs.langchain.com/oss/python/langgraph/persistence  (checkpointer / get_state)
"""

import asyncio  # 异步图不装 pytest-asyncio, 直接用 asyncio.run 跑
import operator  # reducer 用 operator.add 拼列表
import os  # 拼临时 db 路径
import shutil  # 结尾清理沙箱目录
import sqlite3  # SqliteSaver 需要一个 sqlite 连接
import tempfile  # 建沙箱目录, 不在仓库留 db
from typing import Annotated, TypedDict  # 图 state schema + reducer 注解

import pytest  # 测试框架; 文件末尾也提供无 pytest 时的手动跑法
from langchain_core.language_models.fake_chat_models import (
    FakeListChatModel,  # 循环型假模型: 到末尾从头再来, 永不耗尽
    GenericFakeChatModel,  # 一次性迭代器假模型: 逐条吐, 吐完抛 StopIteration
)
from langchain_core.messages import AIMessage  # GenericFakeChatModel 逐条吐的就是这种消息
from langgraph.checkpoint.memory import InMemorySaver  # 内存 checkpointer, 无需落盘
from langgraph.checkpoint.sqlite import SqliteSaver  # 落盘型 checkpointer, 验证真实读回
from langgraph.graph import END, START, StateGraph  # 构图三件套
from langgraph.types import Send  # map-reduce 动态并行分发


# ============================================================
# 维度一: 节点输出正确 + 假模型注入(演示"图逻辑可确定性断言")
# ============================================================
class ChatState(TypedDict):
    question: str  # 输入
    answer: str  # 节点写入的模型回答


def build_llm_graph(model):
    """构建一张最小"调模型"的图: 一个节点把 question 丢给注入的 model, 写回 answer。

    关键点: model 由外部传入(依赖注入), 生产代码里传真实 ChatAnthropic, 测试里传假模型。
    图逻辑本身不关心模型真假, 所以可以离线确定性测试。
    """
    def answer_node(state: ChatState) -> dict:
        # 图逻辑: 把用户问题发给模型, 拿 .content 写回 state。这一步在真实项目里就是 LLM 调用点。
        resp = model.invoke([{"role": "user", "content": state["question"]}])
        return {"answer": resp.content}

    builder = StateGraph(ChatState)
    builder.add_node("answer", answer_node)
    builder.add_edge(START, "answer")
    builder.add_edge("answer", END)
    return builder.compile()


def test_node_output_with_fake_model():
    """节点输出正确: 假模型固定返回 "42", 断言 answer 就是 "42"(真实 LLM 做不到这种精确断言)。"""
    # 每个 test 新建假模型实例, 避免和别的 test 共享导致迭代器污染
    fake = GenericFakeChatModel(messages=iter([AIMessage(content="42")]))
    graph = build_llm_graph(fake)  # 每个 test 新建图
    result = graph.invoke({"question": "生命的意义是什么?", "answer": ""})
    assert result["answer"] == "42"  # 确定性断言: 输入无论是什么, 假模型都吐 "42"


def test_generic_fake_model_is_one_shot_iterator():
    """踩坑验证: GenericFakeChatModel 是一次性迭代器, 吐完再调会耗尽报错。

    这解释了"为什么每个 test 要新建替身"——用完就废, 不能跨 test 复用。
    踩坑细节: 迭代器耗尽本应抛 StopIteration, 但它在 Pregel runner 的生成器内部触发,
    受 PEP 479 影响会被转成 `RuntimeError: generator raised StopIteration` 冒出来,
    所以这里断言 RuntimeError 而不是 StopIteration。
    """
    fake = GenericFakeChatModel(messages=iter([AIMessage(content="第1条"), AIMessage(content="第2条")]))
    graph = build_llm_graph(fake)
    assert graph.invoke({"question": "q1", "answer": ""})["answer"] == "第1条"  # 第1次拿第1条
    assert graph.invoke({"question": "q2", "answer": ""})["answer"] == "第2条"  # 第2次拿第2条
    with pytest.raises(RuntimeError):  # 第3次: 迭代器已耗尽, 经 Pregel runner 转成 RuntimeError
        graph.invoke({"question": "q3", "answer": ""})


def test_fake_list_model_cycles():
    """对照: FakeListChatModel 到末尾会循环, 永不耗尽(适合调用次数不确定的场景)。"""
    fake = FakeListChatModel(responses=["A", "B"])  # 只有两条
    graph = build_llm_graph(fake)
    assert graph.invoke({"question": "q", "answer": ""})["answer"] == "A"  # 第1次 A
    assert graph.invoke({"question": "q", "answer": ""})["answer"] == "B"  # 第2次 B
    assert graph.invoke({"question": "q", "answer": ""})["answer"] == "A"  # 第3次又回到 A(循环)


# ============================================================
# 维度二: conditional_edges 路由正确(不需要模型, 纯图结构断言)
# ============================================================
class RouteState(TypedDict):
    text: str  # 输入
    branch: str  # 实际走到的分支节点会写入自己的名字, 用来断言路由结果


def build_router_graph():
    """一张带条件路由的图: classify 决定走 weather / greeting / fallback 哪个分支。"""
    def classify(state: RouteState) -> dict:
        return {}  # classify 本身不改 state, 路由决策放在 route 函数里

    def route(state: RouteState) -> str:
        # 条件路由逻辑: 这就是要被测试的核心。真实 agent 里这里可能读模型输出决定路由。
        t = state["text"].lower()
        if "weather" in t:
            return "weather"
        if "hello" in t:
            return "greeting"
        return "fallback"

    def weather(state: RouteState) -> dict:
        return {"branch": "weather"}  # 每个分支写下自己的名字

    def greeting(state: RouteState) -> dict:
        return {"branch": "greeting"}

    def fallback(state: RouteState) -> dict:
        return {"branch": "fallback"}

    builder = StateGraph(RouteState)
    builder.add_node("classify", classify)
    builder.add_node("weather", weather)
    builder.add_node("greeting", greeting)
    builder.add_node("fallback", fallback)
    builder.add_edge(START, "classify")
    # path_map 声明所有可能的目标, 供图结构校验; 实际走哪个由 route 返回值决定
    builder.add_conditional_edges("classify", route, ["weather", "greeting", "fallback"])
    builder.add_edge("weather", END)
    builder.add_edge("greeting", END)
    builder.add_edge("fallback", END)
    return builder.compile()


@pytest.mark.parametrize(
    "text, expected",  # 参数化: 一个 test 覆盖三条路由分支, 表驱动更清晰
    [
        ("what is the weather today", "weather"),
        ("hello there", "greeting"),
        ("random gibberish", "fallback"),
    ],
)
def test_conditional_edges_routing(text, expected):
    """conditional_edges 路由正确: 不同输入应命中不同分支。"""
    graph = build_router_graph()  # 每个参数组合新建图
    result = graph.invoke({"text": text, "branch": ""})
    assert result["branch"] == expected  # 断言真的走到了预期分支


# ============================================================
# 维度三: reducer 合并正确 + Send 分发数量(map-reduce)
# ============================================================
class FanState(TypedDict):
    items: list[str]  # 运行时才知道有几个, 决定 Send 分发几路
    # reducer: 并行分支各写一条, operator.add 自动把列表拼接合并(没有 reducer 会报冲突)
    results: Annotated[list[str], operator.add]


class ItemState(TypedDict):
    item: str  # 单个并行分支收到的输入(和主 State 是不同类型)


def build_fanout_graph():
    """map-reduce 图: fan_out 按 items 数量动态产出 Send, 每个分支处理一个 item。"""
    def fan_out(state: FanState) -> list[Send]:
        # Send 数量 = items 长度, 完全由运行时数据决定(map 阶段的并行度)
        return [Send("process", {"item": x}) for x in state["items"]]

    def process(state: ItemState) -> dict:
        # 每个并行分支各写一条; 靠主 State 的 operator.add reducer 合并回 results
        return {"results": [state["item"].upper()]}

    builder = StateGraph(FanState)
    builder.add_node("process", process)
    builder.add_conditional_edges(START, fan_out, ["process"])
    builder.add_edge("process", END)
    return builder.compile()


def test_send_fan_out_count_and_reducer_merge():
    """Send 分发数量 = 输入长度; reducer 把并行结果正确合并(不丢不重)。"""
    graph = build_fanout_graph()
    result = graph.invoke({"items": ["a", "b", "c"], "results": []})
    # Send 分了 3 路 → reducer 合并出 3 条结果, 说明分发数量正确且没丢分支
    assert len(result["results"]) == 3
    # reducer 用 operator.add 拼接, 内容是每个分支各自的输出(顺序不保证, 用集合断言)
    assert set(result["results"]) == {"A", "B", "C"}


def test_send_count_varies_with_input():
    """分支数量是运行时决定的: 换个输入长度, Send 路数随之变化(不是写死在图里)。"""
    graph = build_fanout_graph()  # 新建图, 和上一个 test 隔离
    result = graph.invoke({"items": ["x", "y"], "results": []})
    assert len(result["results"]) == 2  # 这次只给 2 个, 就只分 2 路


def test_reducer_missing_would_conflict():
    """反例验证 reducer 的必要性: 普通字段(无 reducer)被多个并行分支同时写会冲突报错。

    这个 test 断言"没有 reducer 就会 InvalidUpdateError", 从反面证明 reducer 不是可有可无的。
    """
    class BadState(TypedDict):
        items: list[str]
        results: list[str]  # 故意不加 Annotated[..., operator.add], 没有 reducer

    def fan_out(state: BadState) -> list[Send]:
        return [Send("process", {"item": x}) for x in state["items"]]

    def process(state: ItemState) -> dict:
        return {"results": [state["item"]]}  # 多个分支同时写无 reducer 的 results → 冲突

    builder = StateGraph(BadState)
    builder.add_node("process", process)
    builder.add_conditional_edges(START, fan_out, ["process"])
    builder.add_edge("process", END)
    bad_graph = builder.compile()
    # 2 个并行分支同时写同一个无 reducer 的 channel, LangGraph 抛 InvalidUpdateError
    with pytest.raises(Exception):  # 具体是 InvalidUpdateError, 用宽泛 Exception 稳妥断言
        bad_graph.invoke({"items": ["a", "b"], "results": []})


# ============================================================
# 维度四: checkpointer 落盘读回(内存 + sqlite 真实文件)
# ============================================================
class CounterState(TypedDict):
    count: int  # 每步 +1, 用来验证状态在 checkpoint 里被记住


def build_counter_graph(checkpointer):
    """一张会累加计数的两步图, 挂上传入的 checkpointer, 用来验证状态持久化。"""
    def step_a(state: CounterState) -> dict:
        return {"count": state["count"] + 1}

    def step_b(state: CounterState) -> dict:
        return {"count": state["count"] + 1}

    builder = StateGraph(CounterState)
    builder.add_node("step_a", step_a)
    builder.add_node("step_b", step_b)
    builder.add_edge(START, "step_a")
    builder.add_edge("step_a", "step_b")
    builder.add_edge("step_b", END)
    return builder.compile(checkpointer=checkpointer)


def test_checkpointer_get_state_in_memory():
    """InMemorySaver: 跑完后 get_state 能读回最终状态, 且不同 thread_id 互相隔离。"""
    graph = build_counter_graph(InMemorySaver())
    cfg1 = {"configurable": {"thread_id": "t1"}}
    cfg2 = {"configurable": {"thread_id": "t2"}}
    graph.invoke({"count": 0}, config=cfg1)  # t1: 0 → 1 → 2
    graph.invoke({"count": 10}, config=cfg2)  # t2: 10 → 11 → 12
    # get_state 读回各自 thread 的最终状态, 证明 checkpoint 落了盘且按 thread 隔离
    assert graph.get_state(cfg1).values["count"] == 2
    assert graph.get_state(cfg2).values["count"] == 12  # t2 不受 t1 影响


@pytest.fixture
def sqlite_sandbox():
    """fixture: 建 tempfile 沙箱目录给 sqlite 用, test 结束后自动 rmtree 清理, 不留产物。"""
    sandbox = tempfile.mkdtemp()  # 建临时目录
    yield os.path.join(sandbox, "test_ckpt.db")  # 把 db 路径交给 test
    shutil.rmtree(sandbox, ignore_errors=True)  # test 跑完(无论成败)清理沙箱


def test_checkpointer_sqlite_persist_and_reload(sqlite_sandbox):
    """SqliteSaver 真实落盘: 一个"进程"写入后关闭, 另一个"进程"用同 thread_id 读回状态。

    这是 checkpointer 最核心的价值——跨进程/跨调用恢复。用两个独立 saver 连同一个 db 文件模拟。
    """
    db_path = sqlite_sandbox
    cfg = {"configurable": {"thread_id": "job-1"}}

    # "进程 A": 建 saver 跑完图, 写盘后关闭连接(模拟进程结束)
    conn_a = sqlite3.connect(db_path, check_same_thread=False)
    saver_a = SqliteSaver(conn_a)
    graph_a = build_counter_graph(saver_a)
    graph_a.invoke({"count": 0}, config=cfg, durability="sync")  # sync: 每步同步落盘
    conn_a.close()  # 关掉进程 A 的连接

    # "进程 B": 全新 saver 连同一个 db 文件, 用同 thread_id 读回——状态还在
    conn_b = sqlite3.connect(db_path, check_same_thread=False)
    saver_b = SqliteSaver(conn_b)
    graph_b = build_counter_graph(saver_b)
    reloaded = graph_b.get_state(cfg)  # 不重新跑, 直接读回上个进程落盘的状态
    assert reloaded.values["count"] == 2  # 0→1→2, 状态被 sqlite 持久化并读回
    conn_b.close()


# ============================================================
# 维度五: 循环上限(recursion_limit)—— 防止图无限打转
# ============================================================
class LoopState(TypedDict):
    n: int  # 递增计数


def build_infinite_loop_graph():
    """一张故意会无限自环的图: tick 节点永远路由回自己。靠 recursion_limit 兜底截断。"""
    def tick(state: LoopState) -> dict:
        return {"n": state["n"] + 1}

    def loop_back(state: LoopState) -> str:
        return "tick"  # 永远回到 tick, 制造无限循环

    builder = StateGraph(LoopState)
    builder.add_node("tick", tick)
    builder.add_edge(START, "tick")
    builder.add_conditional_edges("tick", loop_back, ["tick"])
    return builder.compile()


def test_recursion_limit_stops_infinite_loop():
    """循环上限: 无限自环的图会在 recursion_limit 步后抛 GraphRecursionError, 不会真跑到死。

    真实 agent 里"模型一直要求调工具、停不下来"就是这种情况, recursion_limit 是安全阀。
    """
    from langgraph.errors import GraphRecursionError  # 循环超限时抛的专用异常

    graph = build_infinite_loop_graph()
    with pytest.raises(GraphRecursionError):  # 断言确实被上限截断了
        graph.invoke({"n": 0}, config={"recursion_limit": 5})  # 限制最多 5 个超步


# ============================================================
# 维度六: 异步图 —— 不装 pytest-asyncio, 用 asyncio.run 跑
# ============================================================
def build_async_graph(model):
    """带异步节点的图: 节点用 async def + ainvoke 调模型。"""
    async def anode(state: ChatState) -> dict:
        resp = await model.ainvoke([{"role": "user", "content": state["question"]}])
        return {"answer": resp.content}

    builder = StateGraph(ChatState)
    builder.add_node("anode", anode)
    builder.add_edge(START, "anode")
    builder.add_edge("anode", END)
    return builder.compile()


def test_async_graph_with_asyncio_run():
    """异步图: 用 asyncio.run 在同步 test 里驱动 ainvoke, 无需 pytest-asyncio 插件。"""
    fake = GenericFakeChatModel(messages=iter([AIMessage(content="async-ok")]))
    graph = build_async_graph(fake)
    # asyncio.run 起一个事件循环把协程跑完, 拿到确定性结果
    result = asyncio.run(graph.ainvoke({"question": "hi", "answer": ""}))
    assert result["answer"] == "async-ok"


# ============================================================
# 无 pytest 时的手动跑法: python test_examples.py 也能全绿
# ============================================================
if __name__ == "__main__":
    # 收集本文件里所有 test_ 开头的函数, 依次调用, 用 try/except 汇总通过/失败情况。
    # 带 pytest 特性的用例(parametrize / fixture)在这里手动喂参数, 保证脱离 pytest 也能跑。
    passed, failed = 0, 0  # 计数器

    def run(name, fn, *args):
        """跑一个 test 函数, 打印结果并累计计数(手动模式的 mini test runner)。"""
        global passed, failed
        try:
            fn(*args)
            print(f"  [PASS] {name}")
            passed += 1
        except Exception as e:  # 任何异常都算失败, 打印类型+信息
            print(f"  [FAIL] {name}: {type(e).__name__}: {e}")
            failed += 1

    print("=== 维度一: 节点输出 + 假模型注入 ===")
    run("test_node_output_with_fake_model", test_node_output_with_fake_model)
    run("test_generic_fake_model_is_one_shot_iterator", test_generic_fake_model_is_one_shot_iterator)
    run("test_fake_list_model_cycles", test_fake_list_model_cycles)

    print("=== 维度二: conditional_edges 路由 (手动喂 parametrize 参数) ===")
    for text, expected in [
        ("what is the weather today", "weather"),
        ("hello there", "greeting"),
        ("random gibberish", "fallback"),
    ]:
        run(f"test_conditional_edges_routing[{expected}]", test_conditional_edges_routing, text, expected)

    print("=== 维度三: Send 分发数量 + reducer 合并 ===")
    run("test_send_fan_out_count_and_reducer_merge", test_send_fan_out_count_and_reducer_merge)
    run("test_send_count_varies_with_input", test_send_count_varies_with_input)
    run("test_reducer_missing_would_conflict", test_reducer_missing_would_conflict)

    print("=== 维度四: checkpointer 落盘读回 ===")
    run("test_checkpointer_get_state_in_memory", test_checkpointer_get_state_in_memory)
    # sqlite 用例需要沙箱, 手动模式里自己建/清临时目录(等价于 fixture)
    _sandbox = tempfile.mkdtemp()
    try:
        run("test_checkpointer_sqlite_persist_and_reload", test_checkpointer_sqlite_persist_and_reload,
            os.path.join(_sandbox, "test_ckpt.db"))
    finally:
        shutil.rmtree(_sandbox, ignore_errors=True)  # 清理沙箱, 不留 db

    print("=== 维度五: 循环上限 ===")
    run("test_recursion_limit_stops_infinite_loop", test_recursion_limit_stops_infinite_loop)

    print("=== 维度六: 异步图 ===")
    run("test_async_graph_with_asyncio_run", test_async_graph_with_asyncio_run)

    print(f"\n=== 汇总: {passed} passed, {failed} failed ===")
    if failed:  # 有失败则以非零码退出, 方便脚本/CI 判断
        raise SystemExit(1)

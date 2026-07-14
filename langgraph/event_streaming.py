"""LangGraph astream_events (v2) —— 异步遍历图执行过程中的细粒度全链路事件。

对比 langgraph/combo.py: combo.py 用 graph.invoke(...) 一把拿到最终结果, 只关心
"跑完之后的输出"; 那属于顶层输出流的思路(graph.stream/astream 的 stream_mode
= "values"/"updates"/"messages" 也都是围绕"图的输出"在推)。本文件换一个视角:
用 astream_events(version="v2") 把图执行内部拆成一连串带类型的事件, 能看到
每一层组件的开始/结束/中间产物——不只是最终 state, 而是:
  - on_chain_start / on_chain_end : 图本身(name="LangGraph")和每个节点的进出
  - on_chat_model_start / on_chat_model_stream / on_chat_model_end : LLM 逐 token 流
  - on_tool_start / on_tool_end : 工具调用的进出
每个事件都带 event(类型) / name(哪个组件) / data(输入输出或增量 chunk) /
tags(可自定义打标) / metadata(含 langgraph_node 等上下文), 这就是它比
stream_mode="messages" 更"全链路": messages 模式只给你消息级增量, 而
astream_events 连"进了哪个节点、哪个 LLM 开始吐 token、工具何时返回"都给你。

关键机制 / 踩坑记录:
  - 必须用 async: astream_events 是异步生成器, 只能 `async for ... in
    graph.astream_events(...)`, 用 asyncio.run 驱动; 没有同步版本。
  - version 参数是必填的定型选择: 这里固定传 version="v2"(v1 是旧结构, 字段布局
    不同, 官方推荐 v2)。不传或传错会拿到不一致的事件结构。
  - 事件量非常大: 一次简单执行就可能几十上百个事件(LLM 每个 token 都是一个
    on_chat_model_stream)。实用时几乎总要过滤——按 event 类型(只要
    on_chat_model_stream)、按 name(只要某个节点/某个模型)、或按 tags(给某个
    LLM .with_config(tags=[...]) 打标, 只收这个标签的 token)。本文件都演示了。
  - 无真实 LLM 时 on_chat_model_stream 未必有 token: 事件流里 LLM 的 token 事件
    取决于底层模型是否真的流式产出。本文件用 GenericFakeChatModel 作替身——它
    会把给定文本按空白切成 chunk 逐个吐出, 因此即使没有 API key 也能真实产生
    on_chat_model_stream 事件, 主逻辑不是空壳。换成真实 ChatAnthropic(走 .env)
    时事件结构完全一致, 只是 token 内容变成真模型输出。

官方文档: https://docs.langchain.com/oss/python/langgraph/streaming
         https://python.langchain.com/docs/how_to/streaming/#using-stream-events
"""

import asyncio  # astream_events 是异步生成器, 需要 asyncio.run 驱动
import operator  # 给 messages 字段做 list 累加的 reducer
import os  # 判断有没有 MODEL_ID, 决定用真实模型还是替身
from typing import Annotated, TypedDict  # 图状态用 TypedDict + Annotated 声明

from dotenv import load_dotenv  # 统一从 .env 读模型配置, 不硬编码
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel  # LLM 替身: 会逐 token 流式吐出
from langchain_core.messages import AIMessage, HumanMessage  # 构造给替身模型的消息
from langchain_core.tools import tool  # 用 @tool 定义一个可被 astream_events 观测的工具
from langgraph.graph import END, START, StateGraph  # 手动搭图

load_dotenv(override=True)  # 有 .env 就加载; 当前环境没有密钥, 下面会走替身分支


# ============================================================
# 选择 LLM: 有 MODEL_ID 走真实 ChatAnthropic(.env), 否则用逐 token 的替身
# ============================================================
def make_model(script: str):
    """返回一个 chat model。

    有 MODEL_ID 时用真实模型(走 .env, 不硬编码); 否则用 GenericFakeChatModel,
    它把 script 按空白切成 token 逐个流式吐出, 因此无密钥也能产生真实的
    on_chat_model_stream 事件——这是本文件"LLM 事件"演示不依赖网络的关键。
    """
    if os.getenv("MODEL_ID"):  # 只有配置了模型才走真实分支
        from langchain_anthropic import ChatAnthropic  # 延迟导入, 无密钥时不强依赖

        return ChatAnthropic(
            model=os.environ["MODEL_ID"],  # 模型 id 走环境变量
            base_url=os.getenv("ANTHROPIC_BASE_URL") or None,  # base_url 也走环境变量
        )
    # 无密钥: 替身。messages 是一个可迭代的"待返回消息队列", 每次 ainvoke 取一条
    return GenericFakeChatModel(messages=iter([AIMessage(content=script)]))


# ============================================================
# 一个工具, 让事件流里出现 on_tool_start / on_tool_end
# ============================================================
@tool
def word_count(text: str) -> int:
    """统计文本里的词数(按空白切分)。这是个纯函数工具, 无外部依赖。"""
    return len(text.split())  # 返回词数, 会作为 on_tool_end 事件的 data.output


# ============================================================
# 图状态: 一条"起草 → 润色 → 统计词数"的链路
# ============================================================
class State(TypedDict):
    topic: str  # 输入主题
    draft: str  # writer 节点写入的初稿
    polished: str  # polish 节点写入的润色稿
    count: int  # count 节点写入的词数
    messages: Annotated[list, operator.add]  # 累积 LLM 消息, reducer 合并


def build_graph():
    """每次调用都编译一张全新的图。

    踩坑: 替身 GenericFakeChatModel 用 iter([...]) 装填待返回消息, 迭代器只能消费
    一次; 本文件有 4 个 demo, 每个都重新跑一遍图, 若共用同一个模型实例, 第二个
    demo 就会 StopIteration。所以把模型和图的构建放进工厂, 每个 demo 拿一张带
    "新鲜替身"的图。真实模型无此限制, 但用工厂对两者都安全。
    """
    # 给每个模型打 tags, 便于后面"只收某个 LLM 的 token"; 真实模型下 tags 同样生效
    model_writer = make_model("初稿 完成 了").with_config(tags=["llm_writer"])  # 扮演"起草"的 LLM
    model_polish = make_model("润色 后 更 精炼").with_config(tags=["llm_polish"])  # 扮演"润色"的 LLM

    # -- 节点: 全部是 async, 因为 astream_events 要在异步上下文里驱动整张图 --
    async def writer_node(state: State) -> dict:
        """调用 writer LLM 起草。await 异步调用, 内部会触发 on_chat_model_* 事件。"""
        resp = await model_writer.ainvoke([HumanMessage(content=f"就「{state['topic']}」写一句初稿")])
        return {"draft": resp.content, "messages": [resp]}  # 写回初稿, 并累加消息

    async def polish_node(state: State) -> dict:
        """调用 polish LLM 润色初稿, 同样异步, 会产生第二组 on_chat_model_* 事件。"""
        resp = await model_polish.ainvoke([HumanMessage(content=f"润色: {state['draft']}")])
        return {"polished": resp.content, "messages": [resp]}  # 写回润色稿

    async def count_node(state: State) -> dict:
        """调用 word_count 工具统计词数, 触发 on_tool_start / on_tool_end 事件。"""
        n = await word_count.ainvoke({"text": state["polished"]})  # 异步调用工具
        return {"count": n}  # 写回词数

    # -- 构建并编译图: START → writer → polish → count → END 一条直链 --
    builder = StateGraph(State)
    builder.add_node("writer", writer_node)  # 起草节点
    builder.add_node("polish", polish_node)  # 润色节点
    builder.add_node("count", count_node)  # 统计节点
    builder.add_edge(START, "writer")  # 入口 → 起草
    builder.add_edge("writer", "polish")  # 起草 → 润色
    builder.add_edge("polish", "count")  # 润色 → 统计
    builder.add_edge("count", END)  # 统计 → 结束
    return builder.compile()  # 本文件聚焦事件流, 不需要 checkpointer


# ============================================================
# 演示 1: 收集完整事件序列, 断言"图开始 → 节点 → 图结束"的结构
# ============================================================
async def demo_full_sequence() -> list[tuple[str, str]]:
    """遍历 astream_events(version='v2'), 收集 (event, name) 序列并返回。"""
    graph = build_graph()  # 每个 demo 拿一张带新鲜替身的图
    seq: list[tuple[str, str]] = []  # 累积 (事件类型, 组件名)
    init = {"topic": "事件流", "draft": "", "polished": "", "count": 0, "messages": []}
    # version="v2" 是必填的结构版本, 这里固定用官方推荐的 v2
    async for ev in graph.astream_events(init, version="v2"):
        seq.append((ev["event"], ev.get("name", "")))  # 每个事件都取类型和名字
    return seq


# ============================================================
# 演示 2: 只按 event 类型过滤——收 LLM 逐 token 的 on_chat_model_stream
# ============================================================
async def demo_filter_by_event() -> list[str]:
    """只保留 on_chat_model_stream 事件, 把所有 LLM token 收集成列表。"""
    graph = build_graph()  # 新鲜替身
    tokens: list[str] = []  # 累积每个 token 文本
    init = {"topic": "过滤", "draft": "", "polished": "", "count": 0, "messages": []}
    async for ev in graph.astream_events(init, version="v2"):
        if ev["event"] == "on_chat_model_stream":  # 只关心 LLM 的 token 增量
            chunk = ev["data"]["chunk"]  # data.chunk 是本次增量的 AIMessageChunk
            if chunk.content:  # 空白 chunk 也会来, 有内容才收
                tokens.append(chunk.content)
    return tokens


# ============================================================
# 演示 3: 按 tags 过滤——只收 writer 这个 LLM 的 token, 不要 polish 的
# ============================================================
async def demo_filter_by_tag(want_tag: str) -> list[str]:
    """只收带指定 tag 的 LLM 产生的 token。tag 是 .with_config(tags=[...]) 打上的。"""
    graph = build_graph()  # 新鲜替身
    tokens: list[str] = []
    init = {"topic": "打标", "draft": "", "polished": "", "count": 0, "messages": []}
    async for ev in graph.astream_events(init, version="v2"):
        # 同时判类型和标签: 只要 on_chat_model_stream 且 tags 里含目标标签
        if ev["event"] == "on_chat_model_stream" and want_tag in ev.get("tags", []):
            chunk = ev["data"]["chunk"]
            if chunk.content:
                tokens.append(chunk.content)
    return tokens


# ============================================================
# 演示 4: 按 name 过滤——单独观测某个节点的进出(on_chain_start/end)
# ============================================================
async def demo_filter_by_name(node_name: str) -> list[str]:
    """只收 name 等于指定节点名的 on_chain_start / on_chain_end 事件类型。"""
    graph = build_graph()  # 新鲜替身
    marks: list[str] = []
    init = {"topic": "定位节点", "draft": "", "polished": "", "count": 0, "messages": []}
    async for ev in graph.astream_events(init, version="v2"):
        # name 精确匹配某个节点, 只留下它的 start/end 这两类进出事件
        if ev.get("name") == node_name and ev["event"] in ("on_chain_start", "on_chain_end"):
            marks.append(ev["event"])
    return marks


async def main() -> None:
    # -- 演示 1: 全链路事件序列, 断言结构 --
    print("=== 演示1: astream_events(v2) 全链路事件序列 ===")
    seq = await demo_full_sequence()
    types = [e for e, _ in seq]  # 只看事件类型
    names = {n for _, n in seq}  # 出现过的组件名集合
    print(f"  事件总数: {len(seq)}")
    print(f"  出现的事件类型: {sorted(set(types))}")
    print(f"  出现的组件名(节选): {sorted(n for n in names if n)[:8]}")
    # 断言: 最外层一定是图(name=LangGraph)的 on_chain_start 打头、on_chain_end 收尾
    assert seq[0] == ("on_chain_start", "LangGraph"), "第一个事件应是图的 on_chain_start"
    assert seq[-1] == ("on_chain_end", "LangGraph"), "最后一个事件应是图的 on_chain_end"
    # 断言: 全链路事件里既有节点(chain)进出, 又有 LLM 和工具的进出
    assert "on_chat_model_start" in types and "on_chat_model_end" in types, "应观测到 LLM 的进出"
    assert "on_chat_model_stream" in types, "替身模型应逐 token 产生 stream 事件"
    assert "on_tool_start" in types and "on_tool_end" in types, "应观测到工具 word_count 的进出"
    # 断言: 三个节点名都作为 chain 出现过(这就是比 messages 模式更全的地方)
    for node in ("writer", "polish", "count"):
        assert node in names, f"节点 {node} 应作为一个 chain 事件出现"
    print("  断言通过: 图 on_chain_start 打头、on_chain_end 收尾; 链/LLM/工具事件齐全")

    # -- 演示 2: 按 event 类型过滤 LLM token --
    print("\n=== 演示2: 只按 event 类型收 on_chat_model_stream 的 token ===")
    tokens = await demo_filter_by_event()
    joined = "".join(tokens)  # 拼回完整输出
    print(f"  收到 token 数: {len(tokens)}, 拼接结果: {joined!r}")
    # 替身给 writer 是「初稿 完成 了」、polish 是「润色 后 更 精炼」, 拼起来应含这两段
    assert "初稿" in joined and "润色" in joined, "两个 LLM 的 token 都应被收到"
    print("  断言通过: on_chat_model_stream 覆盖了 writer + polish 两个 LLM 的全部 token")

    # -- 演示 3: 按 tag 过滤, 只要 writer 那个 LLM --
    print("\n=== 演示3: 按 tags 过滤, 只收 llm_writer 的 token(不要 polish) ===")
    writer_tokens = await demo_filter_by_tag("llm_writer")
    writer_joined = "".join(writer_tokens)
    print(f"  llm_writer 拼接结果: {writer_joined!r}")
    assert "初稿" in writer_joined, "应收到 writer 的 token"
    assert "润色" not in writer_joined, "不应混入 polish 的 token(tag 过滤生效)"
    print("  断言通过: tag 精确隔离了两个 LLM, 只拿到打了 llm_writer 标签的那一路")

    # -- 演示 4: 按 name 过滤单个节点的进出 --
    print("\n=== 演示4: 按 name 过滤, 单独观测 polish 节点的 on_chain_start/end ===")
    polish_marks = await demo_filter_by_name("polish")
    print(f"  polish 节点的进出事件: {polish_marks}")
    # 一个节点执行一次, 应恰好一进一出
    assert polish_marks == ["on_chain_start", "on_chain_end"], "polish 应恰好一次 start + 一次 end"
    print("  断言通过: name 过滤能精确定位到单个节点的一进一出")

    # -- 对比小结 --
    print("\n=== 对比小结: astream_events vs stream_mode ===")
    print("  combo.py 的 invoke/stream 关注'图的输出'(顶层 state/updates/messages);")
    print("  astream_events(v2) 关注'执行过程的每一层事件'(链/LLM/工具的 start/stream/end),")
    print("  因此能按 event 类型、name、tags 精细过滤, 拿到全链路而非仅消息流。")


if __name__ == "__main__":
    asyncio.run(main())  # astream_events 是异步的, 统一用 asyncio.run 驱动

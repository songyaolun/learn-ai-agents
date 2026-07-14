"""LangGraph streaming —— 系统演示四种 stream_mode(values/updates/messages/custom)及组合。

对比 langgraph/quickstart.py: quickstart 只用 graph.invoke() 一把梭拿最终结果(以及
简单地展示 checkpoint 历史), 看不到"图执行过程中的中间产物"。本文件专门系统演示
graph.stream() 的四种 stream_mode, 讲清同一个图在不同模式下"流出来的东西"结构完全不同:
  - values:   每一步(super-step)执行后, 流出**完整的最新 state**(一个 dict, 含所有字段)。
              适合: 想看每步之后整体状态长什么样 / 做进度快照。
  - updates:  只流出**该步各节点产生的增量更新**, 结构是 {节点名: 该节点返回的 dict}。
              适合: 只关心"哪个节点动了、改了啥", 数据量小。这是最常用的模式。
  - messages: 流出 **LLM 逐 token 的 AIMessageChunk + 元数据**, 结构是 (chunk, metadata) 元组。
              适合: 做打字机效果 / 前端实时吐字。注意这个模式要节点里真的调了 chat model 才有东西。
  - custom:   流出**节点内主动用 get_stream_writer() 推送的任意自定义数据**(进度条、
              工具调用中间态等)。结构就是你 writer(...) 传进去的那个对象本身。

组合模式: stream_mode=["updates", "custom"] 时, 每个 chunk 变成 (mode, chunk) 二元组,
第一个元素告诉你这条数据来自哪个模式, 便于在同一个流里区分处理不同类型的事件。

关键机制/踩坑记录:
  1. get_stream_writer 的导入路径是 `langgraph.config`(不是 langgraph.types); 只能在
     节点函数**执行期间**调用, 拿到的 writer 会把数据挂到 custom 流上; 若当前没有活跃的
     stream(比如 invoke 而非 stream), writer 是个 no-op, 不会报错。
  2. messages 模式**依赖节点内真的执行了一次 chat model 调用**才会产出 token; 纯 Python
     函数节点在 messages 模式下不产出任何 chunk。本文件用 FakeListChatModel /
     GenericFakeChatModel 做替身来演示 messages 流的**结构**(无需真实 LLM 密钥); 但要看到
     "真正按模型输出节奏逐 token 流出", 仍需接真实 LLM(见文件尾 real_llm_messages_demo)。
  3. 单模式 stream 直接产出对应结构; 一旦传 list 组合, 所有 chunk 统一包成 (mode, chunk),
     单模式下则**没有**这层 mode 前缀 —— 处理代码要按是否组合区分, 否则会解包出错。

模型接入统一走 .env(ChatAnthropic), 禁止硬编码; 当前环境无密钥时, 非模型逻辑
(values/updates/custom 及组合)全部用普通函数节点实跑并断言, 完全不依赖真实 LLM。

官方文档: https://docs.langchain.com/oss/python/langgraph/streaming
"""

import os
from typing import TypedDict

from langgraph.config import get_stream_writer  # 只能在节点执行期间调用, 拿到 custom 流的 writer
from langgraph.graph import END, START, StateGraph


# ============================================================
# 一个不依赖 LLM 的普通两步图: 用来演示 values/updates/custom
# ============================================================
class State(TypedDict):
    topic: str  # 输入主题
    outline: str  # step1 写入: 大纲
    article: str  # step2 写入: 正文


def make_outline(state: State) -> dict:
    """第一步: 生成大纲, 并用 get_stream_writer() 往 custom 流推一条进度事件。"""
    writer = get_stream_writer()  # 取得当前 stream 的自定义 writer(无活跃 stream 时为 no-op)
    writer({"stage": "outline", "progress": 0.5, "msg": "正在拟大纲"})  # 推送自定义进度, 只会出现在 custom 流
    outline = f"# {state['topic']}\n1. 背景\n2. 要点\n3. 结论"  # 纯 Python 生成, 不调模型
    return {"outline": outline}  # 返回增量, 会体现在 updates 流的 {'make_outline': {...}} 里


def write_article(state: State) -> dict:
    """第二步: 基于大纲写正文, 同样推一条 custom 进度事件。"""
    writer = get_stream_writer()  # 同一个 writer 机制, 换个节点再推一次
    writer({"stage": "article", "progress": 1.0, "msg": "正在填正文"})  # 第二条自定义进度
    article = f"围绕「{state['topic']}」展开: 依据大纲逐段说明。"  # 纯 Python, 不调模型
    return {"article": article}  # 增量更新, updates 流里 key 是 'write_article'


# -- 构建这个不依赖 LLM 的图 --
builder = StateGraph(State)
builder.add_node("make_outline", make_outline)  # 节点名会成为 updates 流里的 key
builder.add_node("write_article", write_article)
builder.add_edge(START, "make_outline")
builder.add_edge("make_outline", "write_article")
builder.add_edge("write_article", END)
graph = builder.compile()  # 无需 checkpointer, streaming 本身不依赖持久化


# ============================================================
# messages 模式演示图: 节点里真的调一次 chat model 才会有 token 流
# 无密钥时用 fake 模型做替身, 只为展示 (chunk, metadata) 的结构
# ============================================================
class ChatState(TypedDict):
    question: str  # 输入问题
    answer: str  # 节点写入: 模型回答


def _build_messages_graph(model):
    """用给定的 chat model(真实或替身)构建一个"调一次模型"的图。"""

    def call_model(state: ChatState) -> dict:
        # 节点内调用 chat model —— 只有这一步产生的 token 才会出现在 messages 流里
        response = model.invoke(state["question"])
        return {"answer": response.content}

    b = StateGraph(ChatState)
    b.add_node("call_model", call_model)
    b.add_edge(START, "call_model")
    b.add_edge("call_model", END)
    return b.compile()


def real_llm_messages_demo() -> None:
    """接真实 LLM 时的 messages 流用法(需 .env 里有 MODEL_ID)。

    这里演示官方推荐的模型接入方式, 逐 token 流式打印。无密钥时本函数不会被调用,
    仅作为"真实 LLM 才有 token 流"的可运行范例保留。
    """
    from dotenv import load_dotenv
    from langchain_anthropic import ChatAnthropic

    load_dotenv(override=True)  # 从 .env 读取密钥/模型 id, 禁止硬编码
    model = ChatAnthropic(
        model=os.environ["MODEL_ID"],  # 模型 id 走环境变量
        base_url=os.getenv("ANTHROPIC_BASE_URL") or None,  # base_url 可选, 也走环境变量
    )
    g = _build_messages_graph(model)  # 用真实模型构图
    # messages 模式流出 (AIMessageChunk, metadata) 元组, chunk.content 是这一小段 token
    for chunk, metadata in g.stream({"question": "用一句话解释流式输出"}, stream_mode="messages"):
        print(chunk.content, end="", flush=True)  # 逐 token 拼成打字机效果
    print()


if __name__ == "__main__":
    inputs = {"topic": "LangGraph 流式输出", "outline": "", "article": ""}  # 三种模式共用同一份输入

    # ---------------------------------------------------------
    print("=== stream_mode='values': 每步后流出完整 state ===")
    values_chunks = list(graph.stream(inputs, stream_mode="values"))  # 收集所有 values chunk
    for i, chunk in enumerate(values_chunks):
        # 每个 chunk 都是一个完整 state dict, 字段随执行逐步被填满
        print(f"  step{i}: keys={sorted(chunk.keys())}, outline空={chunk['outline'] == ''}, article空={chunk['article'] == ''}")
    # 断言: values 每个 chunk 都是含全部字段的完整 state
    assert all(isinstance(c, dict) and {"topic", "outline", "article"} <= set(c) for c in values_chunks)
    # 断言: 最后一个 chunk 里 outline 和 article 都被填上了(完整最终态)
    assert values_chunks[-1]["outline"] and values_chunks[-1]["article"]
    print("  [OK] values: 每个 chunk 都是完整 state dict, 末帧字段填满")

    # ---------------------------------------------------------
    print("\n=== stream_mode='updates': 每个节点的增量更新 {节点名: 更新} ===")
    updates_chunks = list(graph.stream(inputs, stream_mode="updates"))  # 收集所有 updates chunk
    for chunk in updates_chunks:
        # 每个 chunk 的 key 是节点名, value 是该节点这一步返回的增量 dict
        (node_name, delta), = chunk.items()
        print(f"  node={node_name}, 更新字段={sorted(delta.keys())}")
    # 断言: updates 产出以节点名为 key 的 dict, 且正是我们定义的两个节点
    node_names = [next(iter(c.keys())) for c in updates_chunks]
    assert node_names == ["make_outline", "write_article"], node_names
    # 断言: updates 的 value 只含"该节点改动的字段", 不含未改字段(区别于 values 的完整 state)
    assert set(updates_chunks[0]["make_outline"].keys()) == {"outline"}
    assert set(updates_chunks[1]["write_article"].keys()) == {"article"}
    print("  [OK] updates: key 是节点名, value 只含该节点改的增量字段")

    # ---------------------------------------------------------
    print("\n=== stream_mode='custom': 节点内 get_stream_writer() 推的自定义事件 ===")
    custom_chunks = list(graph.stream(inputs, stream_mode="custom"))  # 收集所有 custom chunk
    for chunk in custom_chunks:
        # 每个 chunk 就是节点里 writer(...) 传进去的那个对象本身
        print(f"  收到自定义事件: {chunk}")
    # 断言: 收到了两条自定义进度(两个节点各推一条), 内容正是 writer 推送的对象
    assert len(custom_chunks) == 2, custom_chunks
    assert [c["stage"] for c in custom_chunks] == ["outline", "article"]
    assert custom_chunks[-1]["progress"] == 1.0
    print("  [OK] custom: 收到 writer 主动推送的 2 条进度事件")

    # ---------------------------------------------------------
    print("\n=== 组合模式 stream_mode=['updates','custom']: 每个 chunk 变成 (mode, chunk) ===")
    combo_chunks = list(graph.stream(inputs, stream_mode=["updates", "custom"]))  # 组合两种模式
    for mode, payload in combo_chunks:
        # 组合模式下每条数据带上来源 mode 前缀, 便于在同一个流里区分处理
        print(f"  mode={mode:<8} payload={payload}")
    # 断言: 组合模式每个 chunk 都是 (mode, chunk) 二元组, 且 mode 只会是这两种之一
    assert all(isinstance(c, tuple) and len(c) == 2 for c in combo_chunks)
    modes = {mode for mode, _ in combo_chunks}
    assert modes == {"updates", "custom"}, modes
    # 断言: 组合流里 updates 和 custom 两类事件都齐了(各自数量与单模式一致)
    assert sum(1 for m, _ in combo_chunks if m == "updates") == 2
    assert sum(1 for m, _ in combo_chunks if m == "custom") == 2
    print("  [OK] 组合: 每条是 (mode, chunk), updates/custom 事件都带 mode 前缀区分")

    # ---------------------------------------------------------
    print("\n=== stream_mode='messages': LLM token 级流 (需真的调 chat model) ===")
    if os.getenv("MODEL_ID"):
        # 有真实模型密钥时, 走真实 LLM 逐 token 打字机输出
        print("  检测到 MODEL_ID, 用真实 LLM 演示逐 token 流:")
        print("  ", end="")
        real_llm_messages_demo()
    else:
        # 无密钥: 用 fake chat model 做替身, 演示 messages 流的 (chunk, metadata) 结构
        print("  未检测到 MODEL_ID, 用 FakeListChatModel 替身演示 messages 流的结构(非真实 token 流):")
        from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
        from langchain_core.messages import AIMessage, AIMessageChunk

        # 替身模型: 固定回一句话, GenericFakeChatModel 会把 content 按词切成多个 chunk 模拟流式
        fake = GenericFakeChatModel(messages=iter([AIMessage(content="流式 输出 让 前端 实时 吐字")]))
        g = _build_messages_graph(fake)  # 用替身模型构图
        msg_chunks = list(g.stream({"question": "什么是流式输出", "answer": ""}, stream_mode="messages"))
        for chunk, metadata in msg_chunks:
            # messages 流的每个元素是 (AIMessageChunk, metadata) 二元组
            print(f"  chunk.content={chunk.content!r:<8} metadata含节点名={'langgraph_node' in metadata}")
        # 断言: messages 产出 (AIMessageChunk, dict) 元组, 与 values/updates/custom 结构都不同
        assert all(isinstance(c, tuple) and len(c) == 2 for c in msg_chunks)
        assert all(isinstance(c, AIMessageChunk) for c, _ in msg_chunks)
        assert all(isinstance(m, dict) for _, m in msg_chunks)
        # 断言: 把所有 chunk 的 content 拼起来 == 替身模型的完整回答(逐段流出后可还原)
        assert "".join(c.content for c, _ in msg_chunks) == "流式 输出 让 前端 实时 吐字"
        print("  [OK] messages: 每条是 (AIMessageChunk, metadata) 元组, 拼接可还原完整回答")

    print("\n=== 小结: 同一个图, 四种 stream_mode 产出结构各不相同 ===")
    print("  values  -> 完整 state dict (每步全量快照)")
    print("  updates -> {节点名: 增量更新} (只含改动)")
    print("  custom  -> writer 推送的自定义对象 (进度/中间态)")
    print("  messages-> (AIMessageChunk, metadata) (LLM token 级, 需真调模型)")
    print("  组合    -> (mode, chunk) 二元组 (带来源标签)")

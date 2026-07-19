"""LangGraph 前端对接 —— 把 stream()/astream_events 的 chunk 序列化成前端可消费的协议帧(SSE / NDJSON)。

对比 langgraph/streaming.py 与 langgraph/event_streaming.py: 那两个文件讲的是"图怎么产出流"——
streaming.py 系统演示 graph.stream() 的四种 stream_mode(values/updates/messages/custom)各自流出
什么结构; event_streaming.py 讲 astream_events(v2) 怎么把执行拆成全链路细粒度事件。它们的落点
都在"拿到 chunk"。本文件接着往下走一层: chunk 已经在手, 后端要怎么把它序列化成前端浏览器能直接
吃的协议帧, 也就是"数据转换层"。核心是两个纯函数/生成器:
  - chunk 序列化成 SSE(服务器发送事件, Server-Sent Events)文本帧: 每帧形如 `data: {...}\n\n`,
    浏览器原生 EventSource 会自动按帧拆分、逐帧触发 onmessage。
  - chunk 序列化成 NDJSON(换行分隔的 JSON, Newline-Delimited JSON): 每行一个独立 JSON 对象,
    前端按 `\n` 切行、逐行 JSON.parse。fetch + ReadableStream 读取时最常用这种。

前端如何按 stream_mode 增量渲染(本文件把三种模式都转成帧并演示对应的前端消费思路):
  - messages 模式 -> 取每个 AIMessageChunk.content **累加**成一个字符串, 实现打字机逐字渲染。
  - updates 模式  -> 按节点名把 payload 更新到对应的 UI 区块(哪个节点动了就刷哪块)。
  - custom 模式   -> writer 推的是进度对象 {progress: 0~1}, 前端直接驱动进度条。

关键机制 / 踩坑记录(数据转换层特有):
  1. SSE 帧格式是硬规范: 每帧以 `data: ` 开头, 以**两个换行** `\n\n` 结尾(单换行只是同一事件内的
     多行 data, 双换行才代表"一帧结束")。事件负载里若含真实换行(如多行 JSON), 必须每行都加
     `data: ` 前缀, 否则浏览器会提前断帧。本文件统一把 JSON 压成单行(json.dumps 不加缩进)来规避。
  2. NDJSON 每行必须是**紧凑单行 JSON**且行尾一个 `\n`; JSON 内部绝不能含裸换行, 否则按行切分会碎。
  3. token 增量拼接: messages 模式每个 chunk 只是"一小段", 前端要自己累加; 后端序列化时只负责把
     这一小段原样发出去, 不做累加(累加是前端职责), 这点容易搞反。
  4. **红线**: 本文件是纯数据转换 + 打印演示, **绝不启动任何监听端口的进程**。FastAPI 的
     StreamingResponse / sse-starlette 的 EventSourceResponse 接法只以**字符串常量模板**给出
     (见 FASTAPI_SSE_TEMPLATE / FASTAPI_NDJSON_TEMPLATE), 供你复制到真实服务里用; 本文件既不
     import fastapi/uvicorn, 更不 uvicorn.run / app.run / bind 端口。

模型接入统一走 .env; 无 MODEL_ID 时用 GenericFakeChatModel 替身产出可流式的 AIMessageChunk,
所以 messages 模式的序列化在无密钥环境下也能真实跑通并断言, 主逻辑不是空壳。

官方文档: https://docs.langchain.com/oss/python/langgraph/streaming
"""

import json  # 序列化 chunk 为 JSON 字符串, SSE/NDJSON 的负载都是 JSON
import os  # 判断有无 MODEL_ID, 决定 messages 模式用真实模型还是替身
from typing import Any, Iterator, TypedDict  # 类型标注; 生成器返回 Iterator[str]

from dotenv import load_dotenv  # 统一从 .env 读模型配置, 不硬编码
from langgraph.config import get_stream_writer  # custom 流的 writer, 只能在节点执行期间调用
from langgraph.graph import END, START, StateGraph  # 手动搭图

load_dotenv(override=True)  # 有 .env 就加载; 当前环境无密钥, messages 分支会走替身


# ============================================================
# 一个不依赖 LLM 的两步小图: 演示 updates / custom 两种流的序列化
# (结构刻意与 streaming.py 保持一致, 便于对照)
# ============================================================
class State(TypedDict):
    topic: str  # 输入主题
    outline: str  # step1 写入: 大纲
    article: str  # step2 写入: 正文


def make_outline(state: State) -> dict:
    """第一步: 生成大纲, 并往 custom 流推一条进度事件(给前端进度条用)。"""
    writer = get_stream_writer()  # 取得当前 stream 的自定义 writer(无活跃 stream 时为 no-op)
    writer({"stage": "outline", "progress": 0.5, "msg": "正在拟大纲"})  # 推自定义进度, 只进 custom 流
    outline = f"# {state['topic']}\n1. 背景\n2. 要点\n3. 结论"  # 纯 Python 生成, 不调模型
    return {"outline": outline}  # 增量更新, 体现在 updates 流的 {'make_outline': {...}}


def write_article(state: State) -> dict:
    """第二步: 基于大纲写正文, 再推一条 custom 进度事件(进度到 1.0)。"""
    writer = get_stream_writer()  # 同一个 writer 机制, 换个节点再推一次
    writer({"stage": "article", "progress": 1.0, "msg": "正在填正文"})  # 第二条进度
    article = f"围绕「{state['topic']}」展开: 依据大纲逐段说明。"  # 纯 Python, 不调模型
    return {"article": article}  # 增量更新, updates 流里 key 是 'write_article'


# -- 构建这张不依赖 LLM 的图 --
_builder = StateGraph(State)
_builder.add_node("make_outline", make_outline)  # 节点名会成为 updates 流里的 key
_builder.add_node("write_article", write_article)
_builder.add_edge(START, "make_outline")
_builder.add_edge("make_outline", "write_article")
_builder.add_edge("write_article", END)
graph = _builder.compile()  # streaming 本身不依赖持久化, 无需 checkpointer


# ============================================================
# messages 模式演示图: 节点里真的调一次 chat model 才有 token 流
# 无密钥时用 GenericFakeChatModel 替身, 它会把文本按词切成多个可流式 chunk
# ============================================================
class ChatState(TypedDict):
    question: str  # 输入问题
    answer: str  # 节点写入: 模型回答


def _build_messages_graph(model):
    """用给定 chat model(真实或替身)构建一个"调一次模型"的图。"""

    def call_model(state: ChatState) -> dict:
        # 节点内调用 chat model —— 只有这一步产生的 token 才会出现在 messages 流里
        response = model.invoke(state["question"])
        return {"answer": response.content}

    b = StateGraph(ChatState)
    b.add_node("call_model", call_model)
    b.add_edge(START, "call_model")
    b.add_edge("call_model", END)
    return b.compile()


def _make_chat_model():
    """有 MODEL_ID 走真实 ChatAnthropic(.env), 否则用逐词流式的替身。"""
    if os.getenv("MODEL_ID"):  # 只有配置了模型才走真实分支
        from langchain_anthropic import ChatAnthropic  # 延迟导入, 无密钥时不强依赖

        return ChatAnthropic(
            model=os.environ["MODEL_ID"],  # 模型 id 走环境变量
            base_url=os.getenv("ANTHROPIC_BASE_URL") or None,  # base_url 也走环境变量
        )
    # 无密钥: 替身。GenericFakeChatModel 会把 content 按空白切成多个 AIMessageChunk 逐个吐出
    from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
    from langchain_core.messages import AIMessage

    return GenericFakeChatModel(messages=iter([AIMessage(content="流式 输出 让 前端 实时 吐字")]))


# ============================================================
# 核心: 把一个 LangGraph chunk 规整成"可 JSON 序列化的普通 dict"
# 不同 stream_mode 的 chunk 结构完全不同, 这里统一成 {event, data} 形态,
# 前端拿到后按 event 字段分流处理
# ============================================================
def normalize_chunk(mode: str, chunk: Any) -> dict:
    """把一条原始 chunk 归一化成 {"event": <类型>, "data": <可序列化负载>}。

    - updates: chunk 是 {节点名: 增量dict}, 归一化成 {"event":"update","node":..,"data":..}
    - custom : chunk 是 writer 推的任意对象(这里是进度 dict), 归一化成 {"event":"custom",...}
    - messages: chunk 是 AIMessageChunk, 取其 .content 作为一小段 token 文本
    这样前端只认 event 字段, 不必关心 LangGraph 内部结构。
    """
    if mode == "updates":
        # updates 的 chunk 只有一个 key(节点名), 拆出节点名和它的增量
        (node_name, delta), = chunk.items()
        return {"event": "update", "node": node_name, "data": delta}
    if mode == "custom":
        # custom 的 chunk 就是 writer(...) 传进去的那个对象本身(这里是进度 dict)
        return {"event": "custom", "data": chunk}
    if mode == "messages":
        # messages 的 chunk 是 (AIMessageChunk, metadata); 前端只需要这一小段 token 文本
        msg_chunk, _metadata = chunk
        return {"event": "token", "data": {"content": msg_chunk.content}}
    # 兜底: 其它模式(如 values)直接原样塞进 data
    return {"event": mode, "data": chunk}


# ============================================================
# 序列化器 1: chunk -> SSE(服务器发送事件)文本帧
# ============================================================
def to_sse_frame(payload: dict) -> str:
    """把一个已归一化的 payload 序列化成一帧 SSE 文本。

    SSE 帧硬规范: 以 `data: ` 开头, 以两个换行 `\\n\\n` 结尾。JSON 用紧凑单行(无缩进)、
    ensure_ascii=False 保留中文, 保证负载里不含裸换行, 不会把一帧提前截断。
    """
    body = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))  # 紧凑单行 JSON
    return f"data: {body}\n\n"  # SSE 一帧: data: 前缀 + 双换行结尾


def stream_as_sse(inputs: dict, stream_mode: str) -> Iterator[str]:
    """生成器: 真实跑图并把每个 chunk 转成 SSE 帧逐帧 yield(后端可直接把它喂给响应流)。"""
    if stream_mode == "messages":
        # messages 模式需要真的调模型, 用独立的 chat 图和输入
        g = _build_messages_graph(_make_chat_model())
        for chunk in g.stream(inputs, stream_mode="messages"):  # 逐 token 的 (chunk, metadata)
            yield to_sse_frame(normalize_chunk("messages", chunk))  # 归一化后包成 SSE 帧
    else:
        # updates / custom 用不依赖模型的 graph
        for chunk in graph.stream(inputs, stream_mode=stream_mode):  # 逐 chunk 流出
            yield to_sse_frame(normalize_chunk(stream_mode, chunk))
    # SSE 惯例: 流结束时发一帧哨兵事件, 前端收到就关闭 EventSource
    yield to_sse_frame({"event": "done", "data": None})


# ============================================================
# 序列化器 2: chunk -> NDJSON(换行分隔的 JSON), 每行一个 JSON 对象
# ============================================================
def to_ndjson_line(payload: dict) -> str:
    """把一个已归一化的 payload 序列化成一行 NDJSON(紧凑单行 JSON + 一个换行结尾)。"""
    body = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))  # 紧凑单行 JSON
    return body + "\n"  # NDJSON 一行: JSON + 单个换行


def stream_as_ndjson(inputs: dict, stream_mode: str) -> Iterator[str]:
    """生成器: 真实跑图并把每个 chunk 转成一行 NDJSON 逐行 yield。"""
    if stream_mode == "messages":
        g = _build_messages_graph(_make_chat_model())
        for chunk in g.stream(inputs, stream_mode="messages"):
            yield to_ndjson_line(normalize_chunk("messages", chunk))
    else:
        for chunk in graph.stream(inputs, stream_mode=stream_mode):
            yield to_ndjson_line(normalize_chunk(stream_mode, chunk))
    yield to_ndjson_line({"event": "done", "data": None})  # 结束哨兵行


# ============================================================
# 后端框架接入模板(仅字符串常量, 绝不实际运行 —— 红线: 禁止监听端口)
# 下面两段是"复制到真实服务里"用的骨架, 本文件不 import fastapi/uvicorn, 更不启动服务。
# ============================================================
FASTAPI_SSE_TEMPLATE = r'''
# ==== FastAPI + SSE(服务器发送事件)骨架 —— 仅示范写法, 本文件不实际启动服务 ====
# 依赖: pip install fastapi sse-starlette uvicorn
from fastapi import FastAPI
from sse_starlette.sse import EventSourceResponse   # 专门发 SSE 的响应类, 自动补 data:/\n\n

app = FastAPI()

@app.get("/chat/stream")
async def chat_stream(q: str):
    # 复用本文件的 stream_as_sse 思路: 但注意 EventSourceResponse 只要事件"内容",
    # 帧格式(data:/\n\n)由它自己补, 所以这里 yield 归一化后的 JSON 字符串即可。
    async def event_gen():
        chat_graph = _build_messages_graph(_make_chat_model())
        for chunk in chat_graph.stream({"question": q}, stream_mode="messages"):
            payload = normalize_chunk("messages", chunk)
            yield json.dumps(payload, ensure_ascii=False)   # EventSourceResponse 会包成一帧
    return EventSourceResponse(event_gen())

# 前端(浏览器原生 EventSource):
#   const es = new EventSource("/chat/stream?q=你好");
#   let text = "";
#   es.onmessage = (e) => {                       // 每帧触发一次
#       const p = JSON.parse(e.data);
#       if (p.event === "token") text += p.data.content;   // token 增量累加成打字机
#       render(text);
#   };
#
# 启动方式(仅写在文档里, 本文件绝不执行): uvicorn 模块名:app --port 8000
'''

FASTAPI_NDJSON_TEMPLATE = r'''
# ==== FastAPI + NDJSON(换行分隔的 JSON)骨架 —— 仅示范写法, 本文件不实际启动服务 ====
# 依赖: pip install fastapi uvicorn
from fastapi import FastAPI
from fastapi.responses import StreamingResponse    # 通用流式响应, 自己控制每行字节

app = FastAPI()

@app.get("/chat/ndjson")
async def chat_ndjson(q: str):
    def line_gen():
        chat_graph = _build_messages_graph(_make_chat_model())
        for chunk in chat_graph.stream({"question": q}, stream_mode="messages"):
            payload = normalize_chunk("messages", chunk)
            yield json.dumps(payload, ensure_ascii=False) + "\n"   # 每行一个 JSON + 换行
    # media_type 用 application/x-ndjson, 前端 fetch + ReadableStream 按 \n 切行 JSON.parse
    return StreamingResponse(line_gen(), media_type="application/x-ndjson")

# 前端(fetch 流式读取):
#   const resp = await fetch("/chat/ndjson?q=你好");
#   const reader = resp.body.getReader();
#   const dec = new TextDecoder(); let buf = "", text = "";
#   while (true) {
#       const { value, done } = await reader.read();
#       if (done) break;
#       buf += dec.decode(value, { stream: true });
#       let idx;
#       while ((idx = buf.indexOf("\n")) >= 0) {    // 逐行切分
#           const line = buf.slice(0, idx); buf = buf.slice(idx + 1);
#           if (!line) continue;
#           const p = JSON.parse(line);
#           if (p.event === "token") text += p.data.content;   // 增量累加
#       }
#   }
#
# 启动方式(仅写在文档里, 本文件绝不执行): uvicorn 模块名:app --port 8000
'''


if __name__ == "__main__":
    # ---------------------------------------------------------
    print("=== updates 模式 -> SSE(服务器发送事件)帧 ===")
    up_inputs = {"topic": "LangGraph 前端对接", "outline": "", "article": ""}  # updates/custom 共用
    up_sse = list(stream_as_sse(up_inputs, "updates"))  # 真实跑图并转成 SSE 帧
    for frame in up_sse:
        print("  " + repr(frame))  # 用 repr 让 \n\n 可见, 便于人工核对帧边界
    # 断言: 每一帧都以 `data: ` 开头、以 `\n\n` 结尾(SSE 硬规范)
    assert all(f.startswith("data: ") and f.endswith("\n\n") for f in up_sse), "SSE 帧格式不合规"
    # 断言: 剥掉 data: 前缀和末尾双换行后, 帧体是合法 JSON, 且 event 字段可识别
    parsed = [json.loads(f[len("data: "):].rstrip("\n")) for f in up_sse]
    assert [p["event"] for p in parsed] == ["update", "update", "done"], parsed
    # 断言: 两条 update 分别来自两个节点, 顺序与图执行一致
    assert [p["node"] for p in parsed if p["event"] == "update"] == ["make_outline", "write_article"]
    print("  [OK] SSE: 每帧以 'data: ' 开头、'\\n\\n' 结尾; 帧体是合法 JSON; 节点顺序正确")

    # ---------------------------------------------------------
    print("\n=== custom 模式 -> NDJSON(换行分隔的 JSON)行 ===")
    cu_ndjson = list(stream_as_ndjson(up_inputs, "custom"))  # 真实跑图并转成 NDJSON 行
    for line in cu_ndjson:
        print("  " + repr(line))  # repr 让行尾 \n 可见
    # 断言: 每一行都以单个 `\n` 结尾, 且行内不含裸换行(NDJSON 硬规范)
    assert all(l.endswith("\n") and l.count("\n") == 1 for l in cu_ndjson), "NDJSON 行内含裸换行"
    # 断言: 每一行去掉换行后都能被 json.loads 解析(逐行独立 JSON)
    cu_parsed = [json.loads(l) for l in cu_ndjson]
    assert [p["event"] for p in cu_parsed] == ["custom", "custom", "done"], cu_parsed
    # 断言: 两条 custom 事件带进度, 从 0.5 升到 1.0(前端可据此驱动进度条)
    progresses = [p["data"]["progress"] for p in cu_parsed if p["event"] == "custom"]
    assert progresses == [0.5, 1.0], progresses
    print("  [OK] NDJSON: 每行都能 json.loads; 进度从 0.5 -> 1.0, 可驱动进度条")

    # ---------------------------------------------------------
    print("\n=== messages 模式 -> SSE + NDJSON(token 增量, 前端累加成打字机)===")
    chat_inputs = {"question": "什么是流式输出", "answer": ""}
    if os.getenv("MODEL_ID"):
        print("  检测到 MODEL_ID, 用真实 LLM 产出 token 流并序列化")
    else:
        print("  未检测到 MODEL_ID, 用 GenericFakeChatModel 替身产出可流式 token(非真实模型)")
    # 转成 SSE 帧
    msg_sse = list(stream_as_sse(chat_inputs, "messages"))
    for frame in msg_sse:
        print("  " + repr(frame))
    # 断言: messages 的 SSE 帧同样合规(data: 开头、\n\n 结尾)
    assert all(f.startswith("data: ") and f.endswith("\n\n") for f in msg_sse), "messages SSE 帧不合规"
    msg_parsed = [json.loads(f[len("data: "):].rstrip("\n")) for f in msg_sse]
    # 断言: 除末尾 done 外全是 token 事件
    token_events = [p for p in msg_parsed if p["event"] == "token"]
    assert msg_parsed[-1]["event"] == "done", "messages 流应以 done 收尾"
    assert len(token_events) >= 1, "至少应有一个 token 事件"
    # 断言: 前端把所有 token.content 累加(增量拼接), 应还原出替身模型的完整回答
    joined = "".join(p["data"]["content"] for p in token_events)
    print(f"  前端累加(增量拼接)还原的完整文本: {joined!r}")
    if not os.getenv("MODEL_ID"):
        # 替身回的是固定文本, 可精确断言累加结果
        assert joined == "流式 输出 让 前端 实时 吐字", joined
    else:
        assert joined, "真实模型应产出非空文本"
    # 同一份 token 流也能转成 NDJSON, 逐行可解析
    msg_ndjson = list(stream_as_ndjson(chat_inputs, "messages"))
    assert all(json.loads(l)["event"] in ("token", "done") for l in msg_ndjson), "NDJSON 行不可解析"
    print("  [OK] messages: token 帧合规; 前端累加还原完整回答; 同数据也能转 NDJSON")

    # ---------------------------------------------------------
    print("\n=== 后端框架接入模板(仅字符串常量, 本文件绝不启动服务)===")
    # 只打印模板前几行, 证明模板存在且是"给你复制的写法", 不做任何 import/运行
    print("  FASTAPI_SSE_TEMPLATE 首行:", FASTAPI_SSE_TEMPLATE.strip().splitlines()[0])
    print("  FASTAPI_NDJSON_TEMPLATE 首行:", FASTAPI_NDJSON_TEMPLATE.strip().splitlines()[0])
    # 断言: 模板里含红线提示语, 且本文件全程未 import fastapi / uvicorn
    import sys  # 仅用于检查已加载模块, 不做任何网络/端口操作
    assert "不实际启动服务" in FASTAPI_SSE_TEMPLATE and "不实际启动服务" in FASTAPI_NDJSON_TEMPLATE
    assert "fastapi" not in sys.modules and "uvicorn" not in sys.modules, "红线: 不得加载 web 服务框架"
    print("  [OK] 模板仅为字符串; 未加载 fastapi/uvicorn; 无任何端口监听")

    print("\n=== 小结: 数据转换层把 LangGraph 流转成前端协议帧 ===")
    print("  SSE   -> 每帧 'data: {json}\\n\\n', 浏览器 EventSource 原生消费")
    print("  NDJSON-> 每行一个 JSON + '\\n', fetch+ReadableStream 逐行 JSON.parse")
    print("  messages/updates/custom -> token累加 / 按节点刷区块 / 驱动进度条")
    print("  接入 FastAPI 只给模板, 本文件不起服务(红线: 禁止监听端口)")

# LangGraph 应用接入 UI 的落地路径讲解

> **本文件用途**:讲清楚"把一个已经写好的 LangGraph 图(graph)接到用户界面(UI=User Interface,用户界面)"有哪几种典型落地路径,各自的适用场景、优缺点、关键命令/配置片段,以及对应的官方文档链接。这是一份**纯讲解文档**,不含任何需要起服务、监听端口的可运行代码。
>
> **与 `frontend_streaming.py` 的关系**:`frontend_streaming.py`(若同目录存在,由并行任务生成)是**数据转换代码示范**——演示如何把 `graph.stream()` 产出的 chunk 转换成前端能直接消费的结构(如 SSE 事件帧、JSON 行)。本文件则是**集成路径讲解**——从工程形态角度讲"整套系统怎么搭起来"。两者互补:一个回答"每条数据长什么样、怎么转",一个回答"整个后端↔前端怎么连"。流式部分可交叉参考本仓库的 [`streaming.py`](./streaming.py)(系统演示 `values`/`updates`/`messages`/`custom` 四种 `stream_mode`)。

---

## 重要红线声明(务必先读)

> **本仓库示例环境禁止启动任何监听端口的进程。**
>
> 因此,本文档中出现的 `langgraph dev`、`uvicorn`、`chainlit run`、`streamlit run`、`npm run dev` 等**所有服务类命令,仅作为文档说明展示,不在本环境实际执行**。所有服务类命令都放在代码块中并标注"示例命令,请在自己的环境执行"。
>
> 真实部署时,请在**你自己的环境**中,严格按各自官方文档操作。本文档只负责把"选型思路 + 关键配置形态"讲清楚。

---

## 一、总览:LangGraph 后端 ↔ 前端的三种典型形态

一个 LangGraph 应用的核心是一个编译好的图对象(`graph = builder.compile(...)`,见本仓库 [`combo.py`](./combo.py))。把它接到 UI,本质是解决两件事:**(1)** 通过什么协议把图的执行(尤其是流式输出)送到浏览器;**(2)** 前端用什么形态承接与渲染。业界主要有三条路径:

| 形态 | 后端 | 前端 | 一句话定位 |
| --- | --- | --- | --- |
| **(a) 自建后端 + 自研前端** | 自己写 FastAPI / Flask,用 SSE / WebSocket 推流 | 自己写 React / Vue 等 | 完全可控,工作量最大 |
| **(b) LangGraph Platform / `langgraph dev`** | `langgraph-cli` 起本地 API 服务 | 自带 Studio 调试界面 | 官方托管协议,零手写 API,含调试 UI |
| **(c) 现成聊天 UI** | 复用现成框架的服务进程 | Chainlit / Streamlit / agent-chat-ui 等 | 最快出可用聊天界面 |

其中缩写:**SSE=Server-Sent Events(服务器发送事件)**,一种基于 HTTP 的单向服务端推流;**WebSocket** 是全双工双向长连接;**API=Application Programming Interface(应用程序编程接口)**;**CLI=Command-Line Interface(命令行界面)**。

LangGraph(Python)官方文档基准:[LangGraph OSS 文档](https://docs.langchain.com/oss/python/langgraph)。

---

## 二、形态 (a):自建后端(FastAPI / Flask)+ SSE/WebSocket 推流 + 自研前端

### 适用场景
- 已有自己的 Web 后端与前端技术栈,想把 LangGraph 作为一个内部能力嵌进去。
- 对协议、鉴权、埋点、多租户有强定制需求,不想被托管平台约束。

### 优缺点
| 优点 | 缺点 |
| --- | --- |
| 完全可控:路由、鉴权、数据结构都自己定 | 工作量最大,协议帧格式、断线重连、心跳都要自己处理 |
| 可无缝接入既有系统(网关、监控、灰度) | 流式数据的"最后一公里转换"需自己写(即 `frontend_streaming.py` 那类代码) |
| 无额外框架心智负担 | Studio 那类可视化调试要另找方案 |

### 关键机制:如何把 `graph.stream()` 接到 HTTP 流
后端拿到图后,遍历 `graph.stream(...)` 的产物,逐条 `yield` 成 SSE 帧(`data: {json}\n\n`)或 WebSocket 消息。**产出结构因 `stream_mode` 而异**(详见 [`streaming.py`](./streaming.py)):`messages` 模式产出 `(AIMessageChunk, metadata)` 元组适合打字机效果;`updates` 模式产出 `{节点名: 增量}` 适合展示"哪个节点动了";`custom` 模式产出节点内 `get_stream_writer()` 主动推送的对象,适合进度条等中间态。`frontend_streaming.py` 演示的正是这一层"chunk → 前端可消费帧"的转换。

### 关键代码片段(仅作文档展示,勿在本环境执行)
下面是一个 **FastAPI + SSE** 的骨架,**仅示意结构**——因为它会 `uvicorn` 起监听端口,**不要在本环境运行**:

```python
# 示例代码,请在自己的环境运行;本仓库环境禁止起监听端口进程。
# from fastapi import FastAPI
# from fastapi.responses import StreamingResponse
# from your_graph_module import graph   # 你编译好的 LangGraph 图
#
# app = FastAPI()
#
# @app.get("/chat/stream")
# def chat_stream(q: str):
#     def event_gen():
#         # updates 模式:每步产出 {节点名: 增量},转成 SSE 帧
#         for chunk in graph.stream({"question": q}, stream_mode="updates"):
#             import json
#             yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
#     return StreamingResponse(event_gen(), media_type="text/event-stream")
```

```bash
# 示例命令,请在自己的环境执行(本仓库环境禁止起监听端口进程):
# uvicorn app:app --host 0.0.0.0 --port 8000
```

前端用浏览器原生 `EventSource` 消费 SSE,或用 `WebSocket` 承接双向消息。

### 官方文档
- LangGraph 流式输出:[Streaming](https://docs.langchain.com/oss/python/langgraph/streaming)
- FastAPI 官方:[FastAPI 文档](https://fastapi.tiangolo.com/) · SSE 相关:[StreamingResponse](https://fastapi.tiangolo.com/advanced/custom-response/#streamingresponse)
- Flask 官方:[Flask 文档](https://flask.palletsprojects.com/)
- SSE 规范(MDN):[Server-Sent Events](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events) · WebSocket:[WebSocket API](https://developer.mozilla.org/en-US/docs/Web/API/WebSocket)

---

## 三、形态 (b):LangGraph Platform / `langgraph dev` 本地 API + 自带 Studio 调试 UI

### 适用场景
- 想快速拿到一套**官方标准化的 API 协议**(带线程/thread、检查点/checkpoint、流式、中断恢复等语义),不想自己造轮子。
- 开发调试期需要一个可视化面板看图结构、单步执行、时间旅行(time travel)。
- 打算后续走 LangGraph Platform 托管部署,本地开发与生产协议一致。

### 优缺点
| 优点 | 缺点 |
| --- | --- |
| 零手写 API:线程、流式、断点续跑等语义开箱即用 | 需按其约定组织项目(`langgraph.json` 配置) |
| 自带 **Studio** 可视化调试界面 | 生产托管属于 LangGraph Platform 商业能力,注意计费与部署形态 |
| 本地(`langgraph dev`)与线上协议一致,迁移平滑 | 深度定制协议不如自建后端灵活 |

### 关键配置:`langgraph.json`
LangGraph CLI 靠项目根的 `langgraph.json` 找到你的图。**这是配置文件,不起服务,可以安全存在**:

```json
{
  "dependencies": ["."],
  "graphs": {
    "my_agent": "./combo.py:graph"
  },
  "env": ".env"
}
```
其中 `"my_agent": "./combo.py:graph"` 表示把 `combo.py` 里名为 `graph` 的对象注册为一个可调用图。

### 关键命令(仅作文档展示,勿在本环境执行)
```bash
# 示例命令,请在自己的环境执行(本仓库环境禁止起监听端口进程):
# pip install "langgraph-cli[inmem]"     # 安装本地开发用 CLI(装包本身不起服务)
# langgraph dev                          # 起本地 API 服务 + 打开 Studio 调试 UI(会监听端口,勿在本环境跑)
```
`langgraph dev` 会在本地拉起一个 API 服务并连上 Studio 调试界面。**因其监听端口,本仓库环境严禁执行**——此处仅说明其作用。

### 官方文档
- LangGraph Platform 总览:[LangGraph Platform](https://docs.langchain.com/langgraph-platform)
- 本地开发服务器(`langgraph dev`)与 CLI:[Local server / langgraph-cli](https://docs.langchain.com/langgraph-platform/local-server)
- LangGraph Studio(调试 UI):[LangGraph Studio](https://docs.langchain.com/langgraph-platform/langgraph-studio)
- `langgraph.json` 配置参考:[CLI 配置](https://docs.langchain.com/langgraph-platform/cli)

---

## 四、形态 (c):用现成聊天 UI(Chainlit / Streamlit / agent-chat-ui)

### 适用场景
- 想以最小成本得到一个"能聊天"的界面做 Demo、内部工具或原型验证。
- 不想写前端,只想用 Python(Chainlit / Streamlit)或直接复用官方前端(agent-chat-ui)把图挂上去。

### 三种现成 UI 对比
| 方案 | 语言/形态 | 与 LangGraph 的接法 | 定位 |
| --- | --- | --- | --- |
| **Chainlit** | Python,专为对话式 AI 设计 | 在回调里调用 `graph.stream/astream`,用 Chainlit 的消息/步骤组件渲染流式与中间步骤 | 聊天体验最完整(含步骤展开) |
| **Streamlit** | Python,通用数据应用 | 在脚本里遍历 `graph.stream(...)`,配合 `st.chat_message` / `st.write_stream` 渲染 | 上手最快,适合轻量 Demo |
| **agent-chat-ui** | 官方 React 前端 | 连接形态 (b) 起的 LangGraph API(线程/流式协议) | 官方成品聊天前端,直连 Platform API |

### 关键命令(仅作文档展示,勿在本环境执行)
```bash
# 示例命令,请在自己的环境执行(本仓库环境禁止起监听端口进程):
# chainlit run app.py            # 起 Chainlit 服务(监听端口,勿在本环境跑)
# streamlit run app.py           # 起 Streamlit 服务(监听端口,勿在本环境跑)
# 官方 agent-chat-ui(React):git clone 后 npm install && npm run dev(监听端口,勿在本环境跑)
```

### 关键代码片段思路(仅作文档展示)
以 Streamlit 为例,核心仍是"遍历 `graph.stream()` 并把 `messages` 模式的 token 逐段渲染":

```python
# 示例代码,请在自己的环境运行;本仓库环境禁止起监听端口进程。
# import streamlit as st
# from your_graph_module import graph
#
# if prompt := st.chat_input():
#     with st.chat_message("assistant"):
#         def token_gen():
#             # messages 模式产出 (AIMessageChunk, metadata),取 content 逐段吐字
#             for chunk, _meta in graph.stream({"question": prompt}, stream_mode="messages"):
#                 yield chunk.content
#         st.write_stream(token_gen())   # Streamlit 原生逐段渲染
```

### 官方文档
- Chainlit 官方:[Chainlit 文档](https://docs.chainlit.io/) · 与 LangChain/LangGraph 集成:[Chainlit + LangChain](https://docs.chainlit.io/integrations/langchain)
- Streamlit 官方:[Streamlit 文档](https://docs.streamlit.io/) · 流式渲染:[`st.write_stream`](https://docs.streamlit.io/develop/api-reference/write-magic/st.write_stream)
- 官方 agent-chat-ui:[agent-chat-ui(GitHub)](https://github.com/langchain-ai/agent-chat-ui)

---

## 五、流式渲染要点:三种模式前端如何消费

无论选哪种形态,前端渲染的核心都是**读懂 `graph.stream()` 各模式的产出结构**(完整机制见 [`streaming.py`](./streaming.py),数据转换见 `frontend_streaming.py`)。要点如下:

| `stream_mode` | 后端产出结构 | 前端典型消费方式 |
| --- | --- | --- |
| **`messages`** | `(AIMessageChunk, metadata)` 元组,`chunk.content` 是这一小段 token | 打字机效果:把每段 `content` 追加到当前气泡尾部 |
| **`updates`** | `{节点名: 该节点返回的增量 dict}` | 展示执行进度:哪个节点动了、改了哪些字段(适合"步骤"面板) |
| **`custom`** | 节点内 `get_stream_writer()` 推送的任意对象 | 进度条、工具调用中间态等自定义 UI 事件 |
| **组合** `["updates","custom"]` | `(mode, chunk)` 二元组,首元素标明来源模式 | 同一条流里按 `mode` 分流:token 走气泡、进度走进度条 |

前端交叉参考:
- 打字机气泡 → 消费 `messages` 流,注意 `messages` 模式**要求节点里真的调用了一次 chat model** 才有 token(纯 Python 函数节点不产出 token,见 [`streaming.py`](./streaming.py) 踩坑记录)。
- 步骤/节点面板 → 消费 `updates` 流,`key` 即节点名。
- 自定义进度 → 消费 `custom` 流;`get_stream_writer()` 的导入路径是 `langgraph.config`,只能在节点执行期间调用(见 [`streaming.py`](./streaming.py))。
- 一个流同时要 token + 进度 → 用组合模式,按 `(mode, chunk)` 前缀分流。

官方文档:[Streaming](https://docs.langchain.com/oss/python/langgraph/streaming)。

---

## 六、选型速查

- **要完全可控、嵌进既有系统** → 形态 (a) 自建后端。
- **要标准协议 + 可视化调试,后续可能托管** → 形态 (b) `langgraph dev` / Platform。
- **只要快速出一个能聊天的界面** → 形态 (c) 现成 UI(纯 Python 用 Chainlit/Streamlit,想要官方成品前端用 agent-chat-ui)。

三者并非互斥:常见做法是**开发调试期用 (b) 的 Studio,产品化时按需切到 (a) 或 (c)**。

---

## 附:本文档涉及的所有服务类命令一览(均标注"不在本环境执行")

| 命令 | 出处形态 | 标注 |
| --- | --- | --- |
| `uvicorn app:app ...` | (a) FastAPI | 示例命令,请在自己的环境执行 |
| `langgraph dev` | (b) Platform/CLI | 示例命令,请在自己的环境执行 |
| `chainlit run app.py` | (c) Chainlit | 示例命令,请在自己的环境执行 |
| `streamlit run app.py` | (c) Streamlit | 示例命令,请在自己的环境执行 |
| `npm run dev`(agent-chat-ui) | (c) 官方前端 | 示例命令,请在自己的环境执行 |

上述命令均会监听端口,**本仓库示例环境一律禁止实际执行**;真实部署请在你自己的环境按对应官方文档操作。

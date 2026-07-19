# LangGraph 应用的部署与可观测(observability，可观测性)入门

> 本文档是**纯讲解 + 配置模板**。文中所有服务命令均为**示例命令，请在你自己的环境执行，本仓库环境禁止起服务**(禁止启动任何监听端口的进程)。追踪(tracing，链路追踪)默认关闭，所有密钥/网关地址一律用**占位符**，切勿提交真实密钥或内部地址。

配套文件:同目录下的 [`langgraph.json.template`](./langgraph.json.template)(部署配置模板)。

官方文档基准:[LangGraph(Python)](https://docs.langchain.com/oss/python/langgraph)。

---

## 1. 本地开发服务:`langgraph-cli`

`langgraph-cli`(LangGraph 命令行工具)可以把你在 `.py` 里定义好的图(graph)拉起为一个本地 API 服务，用于开发调试(可配合 LangGraph Studio 可视化调试)。

常用命令(**以下均为示例命令，请在你自己的环境执行；本仓库环境禁止起服务，请勿在本环境运行**):

```bash
# 示例命令,请在自己的环境执行 —— 勿在本仓库环境执行
pip install -U "langgraph-cli[inmem]"   # 安装带本地内存后端的 CLI

# 示例命令,请在自己的环境执行 —— 勿在本仓库环境执行
langgraph dev                            # 启动本地开发服务(热重载),会监听端口

# 示例命令,请在自己的环境执行 —— 勿在本仓库环境执行
langgraph up                             # 用 Docker 在本地拉起一套更接近生产的服务
```

> `langgraph dev` 需要读取当前目录的 `langgraph.json` 配置(见下)。参见 [langgraph-cli 参考](https://docs.langchain.com/langgraph-platform/cli)。

### `langgraph.json` 配置结构

CLI 通过项目根目录的 `langgraph.json` 找到你的图与依赖。核心字段(完整模板见 [`langgraph.json.template`](./langgraph.json.template)):

| 字段 | 含义 |
| --- | --- |
| `dependencies` | 依赖来源。`"."`/`"./"` 表示当前目录(读取本地包或目录内的 `requirements.txt`)；也可指向包目录。不要把 `"./requirements.txt"` 文件名本身写进该字段。CLI/平台据此装依赖。 |
| `graphs` | 把「图对象的导入路径」映射为部署名。格式 `"<文件路径>:<变量名>"`，例如 `"quickstart": "./ch_01_quickstart.py:graph"`。**需要你在对应 `.py` 里导出一个顶层 `graph` 变量**(如本仓库 `ch_01_quickstart.py` 末尾的 `graph = builder.compile(...)`)。可配置多个键，每个键是一个可独立调用的图。 |
| `env` | 指向环境变量文件，如 `".env"`。密钥只写进该 `.env`，不写进 `langgraph.json`。 |
| `python_version` | 可选，声明运行时 Python 版本(本仓库示例要求 >= 3.11)。 |

> JSON 本身不支持注释，模板里用 `_comment*` 字段承载说明(属未知字段，CLI 会忽略)，因此 `langgraph.json.template` 仍是合法 JSON。字段详解见 [应用结构与配置文件](https://docs.langchain.com/langgraph-platform/application-structure)。

---

## 2. 生产部署形态概述

LangGraph 图上线到生产，通常有三种形态:

1. **LangGraph Platform(托管)**:官方托管的运行时与 API，自带持久化、任务队列、并发与扩缩容，配合 LangGraph Studio/LangSmith。
2. **自托管(Docker)**:用 `langgraph build` / `langgraph up` 产出镜像，自己在 Docker/K8s 里运行，掌控数据面。
3. **直接把图嵌入自己的后端**:不引入平台，直接在你已有的 Python 服务里 `import` 编译好的图并 `graph.invoke(...)` / `graph.stream(...)`，把状态持久化交给你自己接的 checkpointer(检查点存储)。

| 形态 | 优点 | 缺点 |
| --- | --- | --- |
| LangGraph Platform(托管) | 开箱即用的持久化/队列/扩缩容;与 Studio、追踪集成好;运维负担最低 | 依赖外部托管服务;数据出域需评估;有平台成本 |
| 自托管(Docker) | 数据留在自己环境;可控性强;贴近生产 | 需自行运维镜像、存储、扩缩容;初始搭建成本高 |
| 嵌入自有后端 | 最轻量、零新增基础设施;完全复用现有服务栈 | 需自己实现持久化/并发/流式接口;缺少平台级能力 |

参考:[LangGraph Platform 概述](https://docs.langchain.com/langgraph-platform)、[部署选项](https://docs.langchain.com/langgraph-platform/deployment-options)。

---

## 3. 可观测 / 追踪(tracing，链路追踪):默认关闭的可选开关

LangGraph 图的运行可以用 **LangSmith**(LangChain 官方的可观测性平台)做链路追踪——查看每个节点、每次模型调用的输入输出、耗时与 token。**本文档把追踪做成默认关闭的开关**:只有同时设置环境变量 `LANGSMITH_TRACING=true` 且提供 `LANGSMITH_API_KEY` 才开启;**不设则完全不追踪**，什么也不上报。

### 开关如何默认关闭

- LangChain/LangSmith 生态默认**不开启**追踪:未显式设置 `LANGSMITH_TRACING=true` 时，不会向任何服务上报数据。
- 凭证一律走环境变量**占位符**，代码里**绝不硬编码密钥**，**绝不连接任何内部推理网关**。

### 安全示例片段(只读 env 决定是否开启)

下面这段只**读取环境变量**来决定是否开启追踪，既不硬编码密钥、也不连任何内部地址。放在应用启动处即可(**示例代码，勿在本仓库环境执行**):

```python
# 示例代码,请在自己的环境执行 —— 追踪默认关闭,仅当 env 显式开启时才启用
import os

def maybe_enable_tracing() -> bool:
    """只读环境变量决定是否开启 LangSmith 追踪; 缺任一条件则保持关闭。"""
    # 默认关闭: 未设或非 "true" 时,直接返回,不做任何上报
    enabled = os.getenv("LANGSMITH_TRACING", "").lower() == "true"
    api_key = os.getenv("LANGSMITH_API_KEY")  # 只读占位符,绝不硬编码
    if not (enabled and api_key):
        return False  # 保持默认关闭
    # 走到这里才算开启; LangChain 会自动读取这些标准环境变量完成上报
    # (不在代码里写任何密钥/网关地址,全部来自 env)
    return True

if maybe_enable_tracing():
    print("LangSmith 追踪已开启(凭证来自环境变量)")
else:
    print("LangSmith 追踪保持关闭(默认)")
```

> 说明:LangChain 在检测到 `LANGSMITH_TRACING=true` 时会自动读取 `LANGSMITH_API_KEY`(及可选的 `LANGSMITH_ENDPOINT` / `LANGSMITH_PROJECT`)完成追踪，你**无需在代码里传入密钥**。上面的函数只是把「是否开启」的判定显式化，方便在关闭时短路。参见 [LangSmith 可观测性快速上手](https://docs.langchain.com/langsmith/observability-quickstart)。

---

## 4. 环境变量清单

以下变量**全部为占位符**，请在 `.env`(由 `langgraph.json` 的 `env` 字段指向)中替换成你自己的值。**切勿提交真实密钥或内部地址**，建议 `.gitignore` 忽略 `.env`。

| 变量 | 用途 | 是否必填 | 占位符示例 |
| --- | --- | --- | --- |
| `MODEL_ID` | 模型标识，传给 `ChatAnthropic(model=...)` | 需要真实模型时必填 | `<你的模型ID占位符>` |
| `ANTHROPIC_BASE_URL` | 模型 API 网关地址(可选)。留空用官方默认;**请勿填任何内部地址** | 可选 | `<你的API地址占位符或留空>` |
| `ANTHROPIC_API_KEY` | 模型访问密钥。请勿提交、请勿分享 | 需要真实模型时必填 | `<你的API密钥占位符>` |
| `LANGSMITH_TRACING` | 追踪总开关。**默认关闭**;设为 `true` 才开启 | 可选(默认关) | `false` |
| `LANGSMITH_API_KEY` | LangSmith 访问密钥。仅在开启追踪时需要;占位符,勿提交 | 仅开启追踪时 | `<你的LangSmith密钥占位符>` |

对应 `.env` 片段(占位符):

```dotenv
MODEL_ID=<你的模型ID占位符>
ANTHROPIC_BASE_URL=<你的API地址占位符或留空>
ANTHROPIC_API_KEY=<你的API密钥占位符>

# 链路追踪默认关闭; 需要时才改为 true 并填 LANGSMITH_API_KEY
LANGSMITH_TRACING=false
LANGSMITH_API_KEY=<你的LangSmith密钥占位符>
```

---

## 5. 官方文档链接

- [LangGraph(Python)总览](https://docs.langchain.com/oss/python/langgraph)
- [LangGraph Platform 概述](https://docs.langchain.com/langgraph-platform)
- [langgraph-cli 命令行参考](https://docs.langchain.com/langgraph-platform/cli)
- [应用结构与 `langgraph.json` 配置](https://docs.langchain.com/langgraph-platform/application-structure)
- [LangSmith 可观测性快速上手](https://docs.langchain.com/langsmith/observability-quickstart)

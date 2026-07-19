# 环境与运行说明

本目录是 LangGraph 学习示例集合。每个 `.py` 文件是一个独立、可 `python 文件名.py` 直接运行的概念演示。本文档说明如何装依赖、配环境变量、以及在无模型密钥时如何验证。

## 1. Python 版本

要求 **Python >= 3.11**(本地实测环境为 3.13)。LangGraph 1.x / LangChain 1.x 需要 3.11+。

## 2. 安装依赖

建议使用虚拟环境隔离:

```bash
cd langgraph
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

`requirements.txt` 覆盖了运行示例所需的全部第三方包(langgraph、langchain、langchain-anthropic、langchain-core、langchain-community、python-dotenv、langgraph-checkpoint-sqlite、pydantic、duckduckgo-search)。

说明:

- **B2 Agentic RAG(向量检索)** 使用 `langchain-core` 自带的 `InMemoryVectorStore` + `DeterministicFakeEmbedding`,零额外依赖(真实 embedding 才需要可选的 `voyageai`,见 requirements.txt 底部注释)。
- **B3 SQL agent** 使用 Python 标准库 `sqlite3`,无需额外安装。

## 3. 环境变量配置

需要访问真实大模型的示例(quickstart / multi_agent / workflows_agents 等)统一通过 `.env` 文件读取配置,**代码中不硬编码任何密钥、model id 或网关地址**。

在 `langgraph/` 目录下新建 `.env` 文件,填入以下变量(以下均为占位符,请替换成你自己的值):

```dotenv
# 模型标识, 传给 ChatAnthropic(model=...)。例如某个 claude 系列模型的 id
MODEL_ID=<你的模型ID占位符>

# 模型 API 网关地址 (可选)。留空则用官方默认地址; 请勿填写任何内部地址
ANTHROPIC_BASE_URL=<你的API地址占位符或留空>

# 模型访问密钥。请勿提交到 git, 请勿分享
ANTHROPIC_API_KEY=<你的API密钥占位符>

# 可选: 若 B2 RAG 使用真实 embedding (voyageai) 才需要
# VOYAGE_API_KEY=<你的embedding密钥占位符>
```

代码里的统一读取方式(参见各示例):

```python
import os
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic

load_dotenv(override=True)
model = ChatAnthropic(
    model=os.environ["MODEL_ID"],
    base_url=os.getenv("ANTHROPIC_BASE_URL") or None,
)
```

> 安全红线:`.env` 只放占位符替换成的你自己的值,**切勿把真实密钥或内部推理网关地址写入代码或提交到仓库**。建议在 `.gitignore` 中加入 `.env`。

## 4. 无 MODEL_ID 时的降级运行(重要)

本次新增并在 `test_examples.py` 覆盖的示例设计为:**即使没有配置 `MODEL_ID` / `ANTHROPIC_API_KEY` 也能跑通结构性逻辑**。较早的独立示例(如 `multi_agent.py`、`send_map_reduce.py`、`combo.py`、`store.py`)仍可能要求真实环境变量。

- 检测到无 `MODEL_ID` 时,示例会自动降级,用**假模型替身**(`FakeListChatModel` / `GenericFakeChatModel` 等)代替真实 LLM。
- 这样可在无密钥、无网络的环境下,验证图的核心机制:Send 分发 / reducer 合并 / checkpointer 落盘读回 / SQL 执行 / 向量检索本地部分 / 重试计数 / 事件序列结构等,均为真实执行并带断言,而非空壳。
- 只有模型的自然语言"内容"是替身产出的;概念本身被真实演示。

因此,你可以先不配 `.env` 直接运行 README/测试列出的 fallback-enabled 示例,观察每个 `=== ... ===` 分节的可判定输出;等配好密钥后再运行同一文件,即可换成真实模型响应。

联网搜索工具(`DuckDuckGoSearchRun`,仅 multi_agent.py 用到)在无网络时会失败,该示例的无密钥路径同样走替身,不依赖真实联网。

## 5. 运行示例

在 `langgraph/` 目录下运行任意示例,例如:

```bash
cd langgraph
python quickstart.py          # 最小可运行图 + create_agent 入门
python persistence.py         # checkpointer 持久化
python send_map_reduce.py     # Send 动态 fan-out / map-reduce
python multi_agent.py         # supervisor + worker 多 agent
python functional_api.py      # @entrypoint / @task 函数式 API
python streaming.py           # 流式输出
python fault_tolerance.py     # RetryPolicy 容错
```

## 6. 官方文档

LangGraph(Python)官方文档:<https://docs.langchain.com/oss/python/langgraph>

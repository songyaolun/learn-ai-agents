# LangGraph 学习示例集

一组**可直接运行**的 LangGraph(Python)学习示例,每个文件聚焦一个概念,配中文逐行注释、与相邻示例的对比说明、以及对应官方文档链接。基准版本:本地实测 **LangGraph 1.2.7** / LangChain 1.x。

> 官方文档基准:<https://docs.langchain.com/oss/python/langgraph>

---

## 快速开始

```bash
# 1. 安装依赖(建议先建 venv,Python >= 3.11)
pip install -r requirements.txt

# 2. 配置模型(可选):复制并填写 .env
#    MODEL_ID / ANTHROPIC_BASE_URL / ANTHROPIC_API_KEY —— 详见 ENVIRONMENT.md

# 3. 跑任意一个示例
python quickstart.py
```

**大部分示例无需模型密钥也能跑**:多数示例在检测到没有 `MODEL_ID` 时,会自动把模型调用点降级为**假模型替身**(`GenericFakeChatModel` / `FakeListChatModel`),从而离线、确定性地把概念的**结构性机制**(路由、reducer 合并、checkpoint 落盘、Send 分发、重试/循环、事件流、流式帧)完整演示并断言通过。填入真实 `MODEL_ID` 后则自动切换到 `ChatAnthropic` 走真实模型(如 `sql_agent.py` 会 `bind_tools` 让真实模型自行调用工具)。注意:少数较早的示例(如 `multi_agent.py` / `combo.py` / `send_map_reduce.py` / `store.py`)直连真实模型,需先配置 `MODEL_ID` 才能运行。

环境搭建与环境变量说明详见 [ENVIRONMENT.md](ENVIRONMENT.md)。

---

## 示例索引

### 基础:图的定义与运行时

| 文件 | 一句话 |
|------|--------|
| [quickstart.py](quickstart.py) | 手动定义一个带条件分支的图,理解 runtime 层 |
| [choosing_apis.py](choosing_apis.py) | 同一业务需求,Graph API 与 Functional API 各写一遍并排对比,讲选型 |
| [functional_api.py](functional_api.py) | 用普通函数 + `@entrypoint`/`@task` 写出等价于图的可持久化流程 |
| [pregel.py](pregel.py) | 底层执行模型:Pregel / BSP(批量同步并行)的超步与同步屏障机制 |

### 状态、持久化与人机协作

| 文件 | 一句话 |
|------|--------|
| [persistence.py](persistence.py) | 用 SqliteSaver 让 checkpoint 落盘,跨进程持久化 |
| [store.py](store.py) | 跨 `thread_id` 的长期记忆,和 checkpointer 是两回事 |
| [human_in_loop.py](human_in_loop.py) | 用 `interrupt` 在节点里暂停等人工审批 |
| [time_travel.py](time_travel.py) | 用 `get_state_history` + `update_state` 回到过去、改写分支 |
| [fault_tolerance.py](fault_tolerance.py) | 节点级 `RetryPolicy` 重试 + checkpointer 崩溃恢复(durable execution) |

### 组合与并行编排

| 文件 | 一句话 |
|------|--------|
| [subgraph.py](subgraph.py) | 把一个编译好的 StateGraph 当成"一个节点"嵌进另一个图 |
| [send_map_reduce.py](send_map_reduce.py) | 运行期动态决定并行分支数量的 map-reduce 模式(Send API) |
| [multi_agent.py](multi_agent.py) | supervisor 模式,用 `Command` 同时做状态更新 + 路由 |
| [workflows_agents.py](workflows_agents.py) | 6 种通用编排范式的最小实现与横向对比(workflow vs agent) |
| [combo.py](combo.py) | 综合示例:subgraph + Send + Store + checkpointer 拼成多文档摘要器 |

### 流式输出与事件

| 文件 | 一句话 |
|------|--------|
| [streaming.py](streaming.py) | 系统演示四种 `stream_mode`(values/updates/messages/custom)及组合 |
| [event_streaming.py](event_streaming.py) | `astream_events` (v2):异步遍历图执行过程中的细粒度全链路事件 |
| [frontend_streaming.py](frontend_streaming.py) | 把流式 chunk 序列化成前端可消费的协议帧(SSE / NDJSON) |

### 应用范式

| 文件 | 一句话 |
|------|--------|
| [agentic_rag.py](agentic_rag.py) | 智能体式检索增强生成(RAG),让图自主决定检索/打分/改写循环 |
| [sql_agent.py](sql_agent.py) | 数据库问答智能体:列表→看 schema→写 SQL→执行→报错自纠→作答 |

### 测试、集成与部署

| 文件 | 一句话 |
|------|--------|
| [test_examples.py](test_examples.py) | 用 pytest + 假模型注入,对图逻辑做确定性断言(兼作冒烟自测) |
| [UI_INTEGRATION.md](UI_INTEGRATION.md) | 把 LangGraph 应用接到 UI 的三种落地路径讲解 |
| [DEPLOYMENT.md](DEPLOYMENT.md) | 部署与可观测入门 + `langgraph.json` 配置模板 |
| [langgraph.json.template](langgraph.json.template) | `langgraph.json` 配置模板(占位符) |

---

## 推荐学习路径

1. **入门**:`quickstart.py` → `choosing_apis.py` → `functional_api.py` → `pregel.py`(理解图/函数两种写法与底层执行模型)
2. **状态与恢复**:`persistence.py` → `store.py` → `human_in_loop.py` → `time_travel.py` → `fault_tolerance.py`
3. **编排进阶**:`subgraph.py` → `send_map_reduce.py` → `multi_agent.py` → `workflows_agents.py` → `combo.py`
4. **流式**:`streaming.py` → `event_streaming.py` → `frontend_streaming.py`
5. **应用**:`agentic_rag.py` → `sql_agent.py`
6. **工程化**:`test_examples.py` → `UI_INTEGRATION.md` → `DEPLOYMENT.md`

---

## 运行与测试

```bash
# 单个示例:直接运行,末尾都有 `if __name__ == "__main__":` 的可判定输出(带断言)
python <文件名>.py

# 单元测试(13 个用例,离线确定性)
pytest -q test_examples.py
```

---

## 约定与边界

- **模型接入统一走 `.env`**:所有需要 LLM 的示例都通过 `ChatAnthropic(model=os.environ["MODEL_ID"], base_url=os.getenv("ANTHROPIC_BASE_URL") or None)` 读取,**不硬编码任何 model id / base_url / 密钥**。
- **副作用沙箱化**:凡涉及 SQLite / 文件写入的示例(如 `fault_tolerance.py` / `sql_agent.py`)一律在 `tempfile` 临时目录里建库,运行结束自动清理,不在仓库目录留产物。注意例外:`persistence.py` 为演示"跨进程从磁盘读回 checkpoint",会在当前目录写下 `langgraph_checkpoints.db` 且运行结束**不自动删除**(仅在下次运行开始时清理),如需清理请手动删除该文件。
- **可观测默认关闭**:链路追踪(LangSmith)做成可选开关,仅当设置了相应环境变量才启用,凭证走环境变量占位符,详见 [DEPLOYMENT.md](DEPLOYMENT.md)。
- **本仓库环境不启动任何监听端口的服务**:`frontend_streaming.py` 只演示"把流转成协议帧"的数据转换,FastAPI 接法仅以模板字符串给出;`UI_INTEGRATION.md` / `DEPLOYMENT.md` 里的 `langgraph dev` / `uvicorn` / `chainlit run` 等命令均为文档说明,请在你自己的环境执行。

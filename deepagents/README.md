# DeepAgents 学习示例包(deepagents 0.6.12)

本包是围绕 [deepagents](https://docs.langchain.com/oss/python/deepagents/quickstart) 0.6.12(基于 LangChain / LangGraph 的 deep-agent 框架)整理的一套**可运行教学示例 + 专题文档**。每个 Python 文件都带中文文件头(含"与既有文件的对比说明 + 官方文档链接")、逐行中文注释、踩坑记录,以及 `__main__` 自检段。

> ⚠️ **模型接入**:所有示例统一通过 `.env` 读取 `MODEL_ID` / `ANTHROPIC_BASE_URL` / `ANTHROPIC_API_KEY`,**不硬编码任何密钥**。先复制**仓库根目录**下的 `.env.example` 为 `.env` 并填入你自己的配置后再运行需要模型的示例。

## 快速开始

> 本仓库约定使用 [uv](https://docs.astral.sh/uv/) 管理依赖与执行。以下命令均在**仓库根目录**执行。

```bash
# 1. 配置模型接入
#    .env.example 位于仓库根目录 (不是 deepagents/ 子目录内)
cp .env.example .env
# 编辑 .env,填入 MODEL_ID / ANTHROPIC_BASE_URL / ANTHROPIC_API_KEY

# 2. 用 uv 跑任意示例 (uv 会按需自动解析并安装依赖到项目虚拟环境)
uv run python deepagents/ch_01_quickstart.py
```

<details>
<summary>不使用 uv 的 pip 回退方式</summary>

```bash
# 需 Python >= 3.11
python3 -m venv .venv && source .venv/bin/activate
pip install -r deepagents/requirements.txt
cp .env.example .env   # .env.example 在仓库根目录
python deepagents/ch_01_quickstart.py
```

</details>

## 运行分级说明

示例按"是否需要真实模型"分两类,文件头均有诚实标注:

- **结构验证通过(无需模型)**:`__main__` 里的断言不调用模型即可跑通(如直接构造 backend、直接调用后端 CRUD、校验配置对象等)。这些文件在**没有** `MODEL_ID` 的环境下也能 `python 文件名.py` 退出码 0。
- **降级-骨架(需要模型 / 外部服务)**:核心行为依赖真实模型或远端服务(如多模态视觉、远程 async 子 agent、云沙箱)。这类文件把"需要模型"的部分用 `MODEL_ID` 是否存在做了守卫:没有配置时会清晰跳过而不是崩溃;文件里用 `# 接入点:` 标出真实端点/凭证应填入的位置。**未做过"已跑通"的虚假声明。**

## 目录索引

> 文件名前缀 `ch_NN_` 即推荐阅读顺序:原始模板(01-10) → A 组(11-15) → B 组(16-22) → C 组(23-24) → D 组(25-29)。

### 原始模板(仓库自带,作为对比基准)

| 文件 | 主题 |
|------|------|
| `deepagents/ch_01_quickstart.py` | 最小可用 deep agent(model + tools + subagents) |
| `deepagents/ch_02_research.py` | 接入 Tavily 真实搜索的研究型 agent |
| `deepagents/ch_03_filesystem.py` | 默认虚拟文件系统(StateBackend) |
| `deepagents/ch_04_backend.py` | 换 FilesystemBackend 落真实磁盘 |
| `deepagents/ch_05_permissions.py` | 文件工具的路径级权限(allow/deny + 新增 interrupt 段) |
| `deepagents/ch_06_hitl.py` | 内置 `interrupt_on` 人工审批(approve/reject) |
| `deepagents/ch_07_stream.py` | 观察中间执行过程(stream v2) |
| `deepagents/ch_08_structured_output.py` | `response_format` 结构化输出 |
| `deepagents/ch_09_skills_memory.py` | skills 技能库 + memory(AGENTS.md) |
| `deepagents/ch_10_combo.py` | 综合示例:委派 + 落盘 + 审批 + 结构化 |

> 📓 [`ch_01_quickstart.qa.md`](ch_01_quickstart.qa.md):精读 `ch_01_quickstart.py` 时深挖 `create_deep_agent`/`create_agent`/LangGraph 源码整理的问答记录(invoke 语义、middleware 生命周期、`task` 工具的任务转交与结果回传、子 agent 的并行与人工审批、middleware 顺序管理等),附源码文件路径与行号。

### A 组 —— 执行环境与上下文管理

| 文件 | 主题 | 运行分级 |
|------|------|----------|
| `deepagents/ch_11_sandbox_backend.py` | 隔离/远程沙箱 backend 概念(vs 同机磁盘) | 结构验证通过(沙箱服务需外部,骨架) |
| `deepagents/ch_12_interpreter.py` | 代码解释器式执行(LocalShellBackend 跑有限计算) | 结构验证通过 |
| `deepagents/ch_13_context_offloading.py` | 把大块中间产物卸载到文件、消息里只留指针 | 降级-骨架(卸载行为由模型驱动) |
| `deepagents/ch_14_summarization.py` | 自动摘要压缩 + `compact_conversation` 工具 | 结构验证通过 |
| `deepagents/ch_15_store_memory.py` | `StoreBackend` 跨线程持久记忆 | 结构验证通过 |

### B 组 —— 委派与定制

| 文件 | 主题 | 运行分级 |
|------|------|----------|
| `deepagents/ch_16_runtime_context.py` | `context_schema` + 工具读运行时上下文 | 结构验证通过 |
| `deepagents/ch_17_prompt_caching.py` | Anthropic 提示词缓存(降本增速) | 结构验证通过(缓存命中不可客户端断言) |
| `deepagents/ch_18_async_subagents.py` | `AsyncSubAgent` 远端 LangGraph 服务子 agent | 降级-骨架(需远端服务) |
| `deepagents/ch_19_compiled_subagent.py` | `CompiledSubAgent` 用预构建 runnable 当子 agent | 结构验证通过 |
| `deepagents/ch_20_custom_backend_middleware.py` | 自定义 `BackendProtocol` 后端 + 自定义中间件 | 结构验证通过 |
| `deepagents/ch_21_multimodal.py` | 向 agent 传图片等多模态内容 | 降级-骨架(需视觉模型) |
| `deepagents/ch_22_profiles.py` | `HarnessProfile` / `ProviderProfile` 定制框架注入 | 结构验证通过 |

### C 组 —— 权限与人工介入

| 文件 | 主题 | 运行分级 |
|------|------|----------|
| `deepagents/ch_05_permissions.py` | (原地扩展)新增 `mode="interrupt"` 路径需人工审批 | 结构验证通过 + 模型场景守卫 |
| `deepagents/ch_23_hitl_edit_respond.py` | HITL 的 `edit` / `respond` 决策(ch_06_hitl.py 只演示了 approve/reject) | 结构验证通过 |
| `deepagents/ch_24_conditional_interrupt.py` | `InterruptOnConfig.when` 条件式审批 + stream v3 事件骨架 | 结构验证通过(v3 部分为 beta 骨架) |

### D 组 —— 生产与生态文档 + shell 后端

| 文件 | 主题 |
|------|------|
| `deepagents/ch_25_local_shell_backend.py` | `LocalShellBackend` 真实 shell `execute` 工具(沙箱在临时目录) | 结构验证通过 |
| `deepagents/ch_26_going_to_production.md` | 上生产:持久化、审批门、权限沙箱、可观测、成本 |
| `deepagents/ch_27_dcode.md` | Deep Agents Code CLI 简介 |
| `deepagents/ch_28_acp.md` | Agent Client Protocol(ACP)与 deepagents 的关系 |
| `deepagents/ch_29_vs_claude_agent_sdk.md` | DeepAgents vs Claude Agent SDK 对比 |

## 推荐学习路径

> 文件名前缀 `ch_NN_` 与本节顺序一一对应。

1. **入门模板**:`ch_01_quickstart.py` → `ch_02_research.py` → `ch_03_filesystem.py` → `ch_04_backend.py` → `ch_05_permissions.py` → `ch_06_hitl.py` → `ch_07_stream.py` → `ch_08_structured_output.py` → `ch_09_skills_memory.py` → `ch_10_combo.py`(最小可用 → 联网研究 → 文件系统 → 落盘 → 权限 → 审批 → 流式 → 结构化 → 技能/记忆 → 综合)
2. **执行环境与上下文(A 组)**:`ch_11_sandbox_backend.py` → `ch_12_interpreter.py` → `ch_13_context_offloading.py` → `ch_14_summarization.py` → `ch_15_store_memory.py`
3. **委派与定制(B 组)**:`ch_16_runtime_context.py` → `ch_17_prompt_caching.py` → `ch_18_async_subagents.py` → `ch_19_compiled_subagent.py` → `ch_20_custom_backend_middleware.py` → `ch_21_multimodal.py` → `ch_22_profiles.py`
4. **权限与人工介入进阶(C 组)**:`ch_23_hitl_edit_respond.py` → `ch_24_conditional_interrupt.py`
5. **生产与生态(D 组)**:`ch_25_local_shell_backend.py` → `ch_26_going_to_production.md` → `ch_27_dcode.md` → `ch_28_acp.md` → `ch_29_vs_claude_agent_sdk.md`

## 安全约定(所有示例遵守)

- **不启动任何监听端口的进程**(无任何形式的 server)。
- **不直连内部推理网关**;**不硬编码模型/密钥**,统一走 `.env`。
- 一切落盘/副作用用 `tempfile.mkdtemp(...)` 沙箱化,结尾 `shutil.rmtree(...)` 清理,不污染工作区。
- 文档类内容仅使用可核实的官方链接;个别无法核实的具体细节标注为「需进一步确认」而非杜撰。

## 官方文档

- DeepAgents: https://docs.langchain.com/oss/python/deepagents/quickstart

# DeepAgents 上生产 (Going to Production)

> 把一个本地跑通的 deep agent 变成"多人、可持久、有护栏、能部署"的生产服务需要考虑什么。
> 官方文档: https://docs.langchain.com/oss/python/deepagents/going-to-production

本仓库里的示例 (ch_04_backend.py / ch_11_sandbox_backend.py / ch_05_permissions.py / ch_14_summarization.py ...)
都是"单进程、本地、给你自己跑"的原型。这份文档讲的是从原型到生产要补齐的那几块:
记忆的作用域、执行环境的隔离、护栏、以及部署方式。这里【不涉及】任何可在本地跑通的
代码 —— 生产化本质上依赖 LangSmith 部署 / Store / checkpointer 等外部基础设施, 本地
环境不具备, 因此本文件是"讲解 + 官方接口指引", 属于纯文档。

---

## 0. 三个贯穿始终的作用域原语

生产里"信息怎么共享、谁能访问"由三个原语决定 (官方 Overview):

- **Thread (线程 / 一次会话)**: 一段对话。消息历史和临时草稿文件默认【只属于这个
  thread】, 不会跨会话带过去。用 `config={"configurable": {"thread_id": ...}}` 标识。
- **User (用户)**: 与 agent 交互的人。记忆和文件可以私有于某个 user, 也可以在多个
  user 间共享。身份和授权来自你的 auth 层。
- **Assistant (助手实例)**: 一个配置好的 agent 实例。记忆 / 文件可以绑在某一个
  assistant 上, 也可以在所有实例间共享。

后面的"记忆作用域"和"执行环境作用域"都是围绕这三者展开的。

---

## 1. 部署方式: LangSmith Deployments

官方推荐路径是 **Managed Deep Agents** (LangSmith 上的 CLI-first 托管运行时,
目前 private preview, 需排队)。需要自定义应用代码 / 路由 / 高级鉴权时, 可以直接配一个
**LangSmith Deployment**。两条路都会帮你把 agent 需要的基础设施 (threads、runs、
一个 store、一个 checkpointer) 直接备好, 不用自己搭; 传统 Deployment 还自带鉴权、
webhooks、cron、可观测性, 并能把 agent 通过 MCP 或 A2A 暴露出去。

所有官方片段基于一个 `langgraph.json` (放在项目根, `langgraph dev` 本地开发和生产
部署都需要):

```json
{
  "dependencies": ["."],
  "graphs": {"agent": "./agent.py:agent"},
  "env": ".env"
}
```

- `dependencies`: 要装的包, `["."]` 表示把当前目录当包装 (读 requirements.txt /
  pyproject.toml / package.json)。
- `graphs`: 图 ID 到代码位置的映射, `"<id>": "./<file>:<variable>"`。
- `env`: `.env` 路径, 构建期写入、运行期可用。

自托管产物: 跑 `langgraph build` 产出一个 standalone Docker 镜像, 部署到任意地方。

---

## 2. 生产调用: thread_id + context 一起传

每次调用带两个 run 级参数 (两者独立, 但几乎总是一起传):

- **`thread_id`** (走 `config={"configurable": {"thread_id": ...}}`): 会话的稳定标识,
  checkpointer 用它持久化 / 恢复消息历史, 让后续轮次接着同一会话。换一个新的
  `thread_id` 就是开一段全新对话。
- **`context`**: 每次 run 的数据 (如 `user_id`、API key、feature flag、session 元数据),
  工具和 middleware 在调用时读取。用 `context_schema` 定义结构, 通过 `runtime.context` 访问。

```python
from dataclasses import dataclass
from deepagents import create_deep_agent
from langchain_core.utils.uuid import uuid7

@dataclass
class Context:
    user_id: str

agent = create_deep_agent(model="anthropic:claude-sonnet-4-6", context_schema=Context)

config = {"configurable": {"thread_id": str(uuid7())}}
agent.invoke(
    {"messages": [{"role": "user", "content": "Plan a 3-day trip to Tokyo"}]},
    config=config,
    context=Context(user_id="user-123"),
)
# 后续轮次: 复用同一个 thread_id, 换新 thread_id 则开新会话
```

要点: `thread_id` 圈的是【会话】(消息历史 / checkpoint); `context` 携带的是【每次 run】
的数据。二者互不影响, 可只传其一, 也可都传。

---

## 3. 多租户 (Multi-tenancy)

对多人服务时要处理三件事: 验证每个用户是谁、控制他们能访问什么、管理 agent 代其行动
时用的凭证。

- **用户身份与访问控制**: LangSmith Deployment 支持 custom authentication (确立用户身份)
  和 authorization handlers (控制对 threads / assistants / store namespace 的访问)。
  授权 handler 可以: 给资源打上归属元数据 (如 `owner: user_id`)、返回过滤器让用户只看到
  自己的资源、对越权操作返回 HTTP 403。
- **团队访问控制 (RBAC)**: LangSmith 的 role-based access control 管的是"你团队里谁能
  部署 / 配置 / 监控 agent", 与上面的终端用户授权是两回事。内置角色: Workspace Admin /
  Editor / Viewer; 企业版可自定义细粒度角色。
- **终端用户凭证**: agent 代用户调外部 API 时:
  - **OAuth via Agent Auth**: 托管的 OAuth 2.0 流程。首次使用时 agent 会 interrupt,
    给出 OAuth 同意链接, 用户认证后 agent 带着有效 token 恢复; token 自动存储与刷新。
  - **沙箱凭证注入**: 若 agent 在沙箱里跑代码并调外部 API, sandbox auth proxy 能把凭证
    自动注入出站请求, 沙箱代码永远拿不到原始 API key。
  - **Workspace secrets**: 所有用户共享的 key (如组织的 LLM provider key) 存为
    workspace secret。

---

## 4. Async 与 Durability

- **Async**: LLM 应用是重 I/O 的, async 让这些操作并发而非阻塞。生产建议: 写 async 工具、
  用 async middleware hook (如 `abefore_agent` 而非 `before_agent`)、外部资源生命周期
  (建沙箱、连 MCP server) 用 async。LangChain 约定异步方法名前缀 `a` (ainvoke / astream)。
- **Durability (持久化执行)**: Deep Agents 跑在 LangGraph 上, 每步 checkpoint state。
  一次 run 因故障 / 超时 / HITL 暂停中断后, 能从上一个 checkpoint 恢复, 不重跑已完成的
  步骤 —— 对会 spawn 大量 subagent 的长跑 agent 尤其重要 (中途崩了不丢已完成的活)。
  还带来: 可无限期 interrupt (HITL 可暂停数天再恢复)、time travel (回到任意 checkpoint)、
  敏感操作的审计与恢复点。LangSmith Deployment 会自动配持久 checkpointer; 自托管见
  persistence 文档。

---

## 5. 记忆 (Memory) 的作用域

Deep Agents 里记忆以"虚拟文件系统里的文件"形式存在。默认文件【只属于单个 thread】,
不跨会话共享。要跨会话共享, 把某个路径 (如 `/memories/`) 路由到一个写 LangGraph Store
的 `StoreBackend`; 用 `CompositeBackend` 让 agent 既有 thread 级草稿空间又有跨会话长期记忆。

作用域选择 (取决于"谁能看到 / 修改这份数据"):

| 作用域 | Namespace | 用途 | 例子 |
|---|---|---|---|
| **User** (推荐默认) | `(user_id)` | 每用户偏好 / 上下文 | "I prefer concise responses" |
| **Assistant** | `(assistant_id)` | 单个 assistant 的共享指令 | "Cap posts at 280 characters" |
| **Global** | `(org_id)` | 全体只读策略 | "Never disclose internal pricing" |

推荐默认按 `user_id` 命名空间 (`namespace=lambda rt: (rt.server_info.assistant_id,
rt.server_info.user.identity)`; 注意 `rt.server_info` 需 `deepagents>=0.5.0`):

```python
from deepagents import create_deep_agent
from deepagents.backends import CompositeBackend, StateBackend, StoreBackend

agent = create_deep_agent(
    model="anthropic:claude-sonnet-4-6",
    backend=CompositeBackend(
        default=StateBackend(),
        routes={
            "/memories/": StoreBackend(
                namespace=lambda rt: (
                    rt.server_info.assistant_id,
                    rt.server_info.user.identity,
                ),
            ),
        },
    ),
    system_prompt="You have persistent memory at /memories/. ...",
)
```

> **安全告警 (官方 Warning)**: 共享记忆 (assistant / user / org 作用域) 是 prompt
> injection 的入口。如果一个用户能写入另一个用户会话会读到的记忆, 恶意用户就能往共享
> state 里注入指令。该只读的地方就强制只读 —— 例如组织级策略只允许通过应用代码写入、
> 不允许 agent 自己写。可用 permissions (见本仓库 ch_05_permissions.py) 声明式地拒绝对共享
> 路径的写, 或用 backend policy hooks 做自定义校验。

---

## 6. 执行环境 (Execution Environment)

本地时 agent 能直接读写磁盘、跑 shell。生产要考虑隔离和持久化, 取决于 agent 是否需要
执行代码:

- **只读写文件 → 用 filesystem backend**: 按持久化需求选:
  - `StateBackend` (默认): thread 级草稿空间, 靠 checkpointer 在同一 thread 内跨轮持久,
    不跨 thread。每步都 checkpoint, 别写大文件。
  - `StoreBackend`: 跨会话存储, 用 namespace factory 划作用域。
  - `CompositeBackend`: 两者混用 (默认 thread 草稿 + 特定路径如 `/memories/` 走跨会话)。
  - `ContextHubBackend`: 存在 LangSmith Hub repo 里的持久文件。
- **需要跑代码 (装包、跑 shell) → 用 sandbox**: sandbox 同时提供文件系统和 `execute`
  工具, 且【隔离】。

> **关键安全告警 (官方 Warning)**: `FilesystemBackend` 和 `LocalShellBackend`
> 【直接访问宿主机】。**不要在部署的 agent 里用它们。** —— 这正是本仓库 ch_04_backend.py /
> ch_25_local_shell_backend.py 明确只圈在临时目录、只做本地演示的原因; 生产要执行代码请换
> 远程沙箱 (见 ch_11_sandbox_backend.py)。

其余生产要点 (官方页面后续章节, 具体接口以官方为准, 部分需进一步确认):
- **Lifecycle**: 沙箱可按 user / assistant 配置生命周期 (何时建、何时销毁)。
- **File transfers / Managing secrets**: 文件进出沙箱、以及用 sandbox auth proxy /
  workspace secrets 管密钥。
- **Guardrails**: rate limiting (限流)、handling errors (错误处理 / resilience
  middleware)、data privacy (数据隐私)。
- **Frontend**: 用 LangGraph SDK (`get_client` → `client.runs.stream(...)`) 把 UI
  接到已部署 agent, SDK 帮你管 thread。

---

## 7. 从本仓库示例过渡到生产的对照清单

| 本地示例做法 | 生产替代 |
|---|---|
| `FilesystemBackend` / `LocalShellBackend` 圈临时目录 | 换远程 **sandbox** (隔离), 绝不在部署里直连宿主机 |
| 无 checkpointer, `result["messages"]` 一次性 | LangSmith Deployment 自动配 **checkpointer**, 传 `thread_id` |
| 记忆存 state, 进程退出即失 | `StoreBackend` / `CompositeBackend` 按 `user_id` 作用域持久 |
| 单人本地跑 | custom auth + authorization handlers + RBAC 做**多租户** |
| 裸 execute | HITL 审批 + permissions 声明式护栏 (见 ch_05_permissions.py) |
| `agent.invoke(...)` 手跑 | `langgraph build` 出镜像自托管, 或 Managed Deep Agents 托管 |

> 说明: 本文件不含可运行代码; 上表片段依赖 LangSmith / Store / checkpointer 等外部
> 基础设施, 本地环境不具备也不接入。所有接口以官方文档为准。

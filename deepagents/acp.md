# Agent Client Protocol (ACP) —— 把 Deep Agent 接进编辑器 / IDE

> 官方文档: https://docs.langchain.com/oss/python/deepagents/acp
> 协议官网: https://agentclientprotocol.com/get-started/introduction

## 1. ACP 是什么

**Agent Client Protocol (ACP)** 是一个开放标准, 用来标准化【编码 agent】和【代码编辑器 /
IDE】之间的通信 —— 类比一下: LSP 之于语言服务器, ACP 之于编码 agent。它由 Zed 发起,
现已被一批编辑器和 agent 采用。核心特征:

- **基于 JSON-RPC 2.0, 走 stdio**: agent 作为一个进程被编辑器拉起, 双方通过标准输入/
  输出通信。
- **富交互**: 文本、图片、文件操作、工具调用、终端、diff、权限请求。
- **会话管理**: 持久对话, 支持完整历史回放。
- **无厂商锁定**: 任意模型、可切换 agent, 全走一个开放协议。

> 边界澄清 (常混): **ACP 是 "agent ↔ 编辑器" 的集成协议**; 如果你要的是"agent 调用
> 外部 server 托管的工具", 那是 **MCP (Model Context Protocol)**, 不是 ACP。二者方向
> 相反、用途不同。

把 Deep Agent 暴露成 ACP server 后, 你的自定义 deep agent 就能在任意 ACP 兼容客户端
(Zed、JetBrains 等) 里用, 编辑器负责提供项目上下文并接收富更新。

---

## 2. Quickstart —— 用 `deepagents-acp` 暴露一个 deep agent

安装集成包:

```bash
pip install deepagents-acp
# 或 uv add deepagents-acp
```

以 stdio 模式启动一个 ACP server (从 stdin 读请求、往 stdout 写响应)。实际使用中,
你通常不是手动跑它, 而是让 ACP 客户端 (你的编辑器) 把它作为一条命令拉起, 再通过
stdio 通信:

```python
import asyncio

from acp import run_agent
from deepagents import create_deep_agent
from langgraph.checkpoint.memory import MemorySaver

from deepagents_acp.server import AgentServerACP


async def main() -> None:
    agent = create_deep_agent(
        model="anthropic:claude-sonnet-4-6",
        # 这里可以自定义你的 deep agent: 换 prompt、加工具、挂 middleware、组合 subagent
        system_prompt="You are a helpful coding assistant",
        checkpointer=MemorySaver(),
    )
    server = AgentServerACP(agent)
    await run_agent(server)


if __name__ == "__main__":
    asyncio.run(main())
```

要点:
- 用的还是本仓库示例里那个 `create_deep_agent` —— ACP 只是在它外面包了一层协议 server,
  agent 本身的定制方式 (prompt / tools / middleware / subagent) 完全不变。
- `checkpointer=MemorySaver()` 给会话提供持久化, ACP 的"历史回放 / 会话管理"依赖它。
- `model=` 官方给了 Google / OpenAI / Anthropic / OpenRouter / Fireworks / Baseten /
  Ollama 多种写法, 体现 Deep Agents 的模型无关性 (本文只保留 Anthropic 一种)。

`deepagents-acp` 包自带一个开箱即用的示例编码 agent (带文件系统和 shell):
https://github.com/langchain-ai/deepagents/blob/main/libs/acp/examples/demo_agent.py

> JavaScript 侧也有一个对应的 `deepagents-acp` npm 包 (`npx deepagents-acp`, 支持
> `--name/--model/--workspace/--skills` 等 flag、`ANTHROPIC_API_KEY` 等环境变量),
> 用于把 JS 版 DeepAgents 暴露成 ACP server。Python 与 JS 两套包接口不同, 以各自
> 官方 reference 为准 (具体 JS CLI flag 需进一步确认)。

---

## 3. 客户端 (Clients)

deep agent 能在任何能跑 ACP agent server 的地方用。常见 ACP 客户端:

- **Zed** — https://zed.dev/docs/ai/external-agents
- **JetBrains IDEs** — https://www.jetbrains.com/help/ai-assistant/acp.html
- **VS Code** — 通过 vscode-acp (https://github.com/formulahendry/vscode-acp)
- **Neovim** — 通过 ACP 兼容插件

### 3.1 在 Zed 里注册 DeepAgents

`deepagents` 仓库带了一个 demo ACP 入口脚本, 可注册进 Zed:

```bash
git clone https://github.com/langchain-ai/deepagents.git
cd deepagents/libs/acp
uv sync --all-groups
chmod +x run_demo_agent.sh
cp .env.example .env      # 然后在 .env 里设置 ANTHROPIC_API_KEY
```

在 Zed 的 `settings.json` 里配置 agent server 命令:

```json
{
  "agent_servers": {
    "DeepAgents": {
      "type": "custom",
      "command": "/your/absolute/path/to/deepagents/libs/acp/run_demo_agent.sh"
    }
  }
}
```

然后打开 Zed 的 Agents 面板, 开一个 DeepAgents thread。

### 3.2 用 Toad 当本地 dev 工具跑

想把 ACP agent server 当本地开发工具管理进程, 可以用 Toad:

```bash
uv tool install -U batrachian-toad
toad acp "python path/to/your_server.py" .
# 或 toad acp "uv run python path/to/your_server.py" .
```

---

## 4. 与本仓库示例的关系

| 本仓库示例 | ACP 场景下的对应 |
|---|---|
| `create_deep_agent(model, system_prompt, tools, middleware, ...)` | 完全复用, 只是外面包一层 `AgentServerACP` |
| 自定义 backend (backend.py / sandbox_backend.py / local_shell_backend.py) | demo agent 带文件系统 + shell, 可换成隔离 backend |
| HITL / permissions (hitl.py / permissions.py) | ACP 协议内置 "permission requests", 编辑器侧弹审批 |
| 会话持久 (going_to_production.md 的 checkpointer) | ACP 用 `MemorySaver()` 等 checkpointer 支撑历史回放 |

> 本文件为纯讲解文档, 不含可在本仓库离线运行的代码: 上述片段需要 `deepagents-acp` /
> `acp` 包、模型凭证, 并且要由一个真实 ACP 客户端 (如 Zed) 通过 stdio 拉起才有意义,
> 本地环境不具备。接口以官方 ACP 页与协议官网为准。

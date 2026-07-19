# Deep Agents Code (`dcode`) —— 终端里的编码 agent

> 官方文档: https://docs.langchain.com/oss/python/deepagents/code/overview
> Quickstart: https://docs.langchain.com/oss/python/deepagents/code/quickstart

Deep Agents Code (命令名 `dcode`) 是构建在 Deep Agents SDK 之上的一个开源终端编码
agent —— 可以粗略理解成 "LangChain 版、模型无关的 Claude Code / Codex CLI"。它:

- **与任意大模型工作**, 支持随时切换 provider / 模型 (不锁死某一家);
- 靠**持久记忆**跨会话携带上下文;
- 用可定制的 **skills** 塑造行为;
- 用**审批控制 (HITL)** 给代码执行 (execute)、文件写入这类敏感操作加人工确认闸门。

> 关系澄清 (容易混): `dcode` = `deepagents-code` 包, 是【面向终端用户的交互式 TUI /
> 本地编码工作流】。它原来是 `deepagents-cli` 的一部分, 后来在 v0.1.0 拆成独立包
> `deepagents-code`, 专注交互式终端 UI (基于 Textual 框架)。而本仓库示例用的
> `deepagents` 库是【SDK/框架】本身 —— `dcode` 是用这个 SDK 搭出来的一个成品应用。
> (`deepagents-cli` / `deepagents-code` 的具体版本与拆分细节需进一步确认, 以官方
> CHANGELOG 为准。)

---

## 1. 安装与启动

官方一行安装 + 启动交互式会话:

```bash
curl -LsSf https://langch.in/dcode | bash
dcode
```

> 早期 LangChain 博客 (Introducing Deep Agents CLI, 2025-10-30) 里给的安装方式是
> `uv tool install deepagents-cli`, 启动命令是 `deepagents`。现在官方 overview 页
> 用的是 `dcode`。两种入口都真实存在过, 以你实际安装到的包 / 官方最新 Quickstart 为准。

装好后进项目目录直接 `dcode` 启动。首个任务示例 (官方博客):

```
You: Add type hints to all functions in src/utils.py
```

agent 会: 读文件 → 分析函数 → 展示 diff → 【等你批准后】再写。也有 "Auto-Accept
Edits" 选项加速开发 (关掉人工闸门, 自行权衡风险)。

---

## 2. 能力清单 (官方 Capabilities)

| 能力 | 说明 |
|---|---|
| **Remote sandboxes** | 把 agent 工具跑在远程沙箱, 而不是你本地机器 (隔离, 类似本仓库 ch_11_sandbox_backend.py 的思路) |
| **Goals and rubrics** | 定义可度量的目标 / 评分标准, 让 agent 能自检"活干完没" |
| **Subagents** | 把活委派给专职 subagent, 并行执行 (对应本仓库 ch_18_async_subagents.py / ch_19_compiled_subagent.py) |
| **Memory** | 跨会话存取信息, 包括项目约定和学到的模式 |
| **Context compaction** | 总结旧消息、把原文 offload 到存储 (对应本仓库 ch_14_summarization.py / ch_13_context_offloading.py) |
| **Human-in-the-loop** | 对敏感工具操作要求人工批准 (对应本仓库 ch_06_hitl.py / ch_05_permissions.py) |
| **Skills** | 用自定义专长和指令扩展 agent (对应本仓库 ch_09_skills_memory.py) |
| **MCP tools** | 从 Model Context Protocol server 加载外部工具 |
| **Tracing** | 在 LangSmith 里追踪 agent 操作, 做可观测 / 调试 |

---

## 3. 记忆是"看得见的文件" (Memory-First 协议)

`dcode` 最有辨识度的一点: 记忆就是一堆真实文件, 默认存在
`~/.deepagents/AGENT_NAME/memories/`。默认 agent 名叫 `agent`; 用 `dcode --agent foo`
(旧 CLI 是 `deepagents --agent foo`) 可切换到另一套记忆。

agent 自动遵循 **Memory-First 协议** (官方博客):
1. **研究时** —— 先查 `/memories/` 里有没有相关知识;
2. **回答前** —— 不确定就搜记忆文件;
3. **学到新东西时** —— 写进 `/memories/`。

因为记忆就是文件, 你可以手动打开 `~/.deepagents/AGENT_NAME/memories/` 检查 / 校验,
也可以让 agent 自己 `ls /memories/` + `read_file`。官方给的最佳实践: 用描述性文件名
(`/memories/deployment-checklist.md` 而不是 `/memories/notes.md`)、按主题分目录、
定期人工核对。

---

## 4. 什么时候用 `dcode`, 什么时候用 SDK

- **用 `dcode`**: 你想要一个开箱即用、在终端里干活的编码助手 (改代码、做研究、带记忆),
  不想自己写 agent loop / UI。
- **用 `deepagents` SDK** (本仓库所有示例): 你要把 agent 能力嵌进自己的应用 / 服务,
  需要自定义 backend、middleware、subagent、部署方式 —— `dcode` 本身就是拿这个 SDK
  搭的, 你也能搭出自己的。

> 本文件为纯讲解文档, 不含可在本仓库运行的代码: `dcode` 是一个需要联网安装、需要模型
> 凭证的终端应用, 与本仓库"离线可跑的 SDK 示例"是两个层面。相关命令 / 路径以官方
> Quickstart 与 Configuration 页为准。

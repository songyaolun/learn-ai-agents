# Deep Agents vs Claude Agent SDK —— 该选哪个 agent harness

> 官方对比页: https://docs.langchain.com/oss/python/deepagents/comparison
> (官方注明该对比初稿写于 2026-04-16, 产品在变, 以官方最新为准)

两者都是"搭自定义 agent 的 harness (骨架)", 但在**执行环境、部署、厂商耦合**三处做了
不同取舍。Deep Agents 已在生产中被 OpenSWE 和 LangSmith Fleet 使用。

---

## 1. 一眼对比 (官方 At a glance)

|  | **Deep Agents** | **Claude Agent SDK** |
|---|---|---|
| **agent 跑在哪** | 沙箱内, 或沙箱外远程执行命令 | 沙箱内 |
| **执行 backend** | 可插拔: local / 虚拟文件系统 / 远程沙箱 / 自定义 | 它所在沙箱的本地文件系统 |
| **模型 provider** | 任意 (Anthropic、OpenAI、Google, 100+) | Claude (Anthropic、Bedrock、Vertex、Azure) |
| **按 provider/模型调优** | Harness profiles (beta): 声明式打包 prompt/工具/middleware/subagent, 按 provider 或具体模型注册 | 在每个 model 调用点用代码配 |
| **部署** | LangSmith 上的 Managed Deep Agents, 或用 `langgraph build` 自托管 standalone 镜像 | 自托管, server/auth/streaming 都自己搭; Claude managed agents 是另一个独立产品 |
| **多租户** | 内置: scoped threads、per-user 沙箱、RBAC | 自己搭 |
| **License** | MIT | MIT (但 Claude Code 本身闭源) |

---

## 2. 主要差异 (逐条)

### 2.1 agent 与执行环境 —— 最核心的分歧

agent 连接沙箱有两种模式: **agent 跑在沙箱【里】**, 或 **agent 跑在沙箱【外】、把沙箱
当成一个工具用** (远程执行命令)。

- **Claude Agent SDK 只支持第一种**: agent 跑在沙箱内, 对沙箱本地文件系统执行工具。
  (Anthropic 自家托管的 Claude managed agents 用的是解耦模型, 官方认为这代表生产架构
  的走向。)
- **Deep Agents 两种都支持**, 并用 backend 把它们接起来。实践上你可以:
  - 让 agent 跑在沙箱内 (和 Claude Agent SDK 一样);
  - 让 agent 跑在长驻容器里, 把远程沙箱当工具, 通过网络执行命令;
  - 测试时换成虚拟文件系统, 或换自定义 backend 接你自己的基础设施。

> 这正好对应本仓库的 backend 谱系: filesystem.py (虚拟 state) → backend.py
> (FilesystemBackend 真实磁盘) → local_shell_backend.py (LocalShellBackend 本地 shell)
> → sandbox_backend.py (远程隔离沙箱)。"可插拔 backend"就是 Deep Agents 相对 Claude
> Agent SDK 最直观的灵活性体现。

### 2.2 多租户

生产要给每个终端用户隔离环境。
- **Claude Agent SDK**: SDK 把 agent 绑死在它的沙箱上。要每用户一个隔离环境, 你得自己
  写一层 API wrapper: 每用户起一个沙箱、记录哪个沙箱属于谁、用完销毁。
- **Deep Agents**: 直接在 harness 里按 user / assistant 配沙箱, 自带 scoped threads、
  run 历史、RBAC。用 LangSmith Sandbox 还能白拿一个 auth proxy, 让终端用户从沙箱调
  第三方 API 而无需你逐用户发凭证。

### 2.3 生产 agent server

- **Claude Agent SDK**: 要把自托管应用暴露给终端用户, 你得自己写 HTTP/WebSocket 或 SSE
  server (调 agent、流式返回 token、管会话线程), 这台 server 你自己建、自己运维、自己
  保障安全。
- **Deep Agents**: 部署自带 agent server —— 流式端点、线程管理、run 历史、webhooks、
  鉴权都开箱即用。

### 2.4 托管 vs 自托管

- **Claude Agent SDK**: 部署是自托管的; SDK 和 Claude managed agents 是两个独立产品,
  针对 SDK 写的代码【不能直接】部署到托管产品。
- **Deep Agents**: 同一套代码两种模式无需改动 —— Managed (LangSmith 上的 Managed Deep
  Agents) 或 Self-hosted (`langgraph build` 出 Docker 镜像随处部署)。

### 2.5 LLM 与生态

- **Claude Agent SDK**: 把模型、backend、部署捆在一起, 三者间做深度优化 (专为 Claude /
  Anthropic 产品面打造)。
- **Deep Agents**: 模型 provider、执行 backend、部署目标三者【独立选择】, 换取最大灵活性;
  接入更广的 LangChain 生态 (LangSmith 做可观测 / 评估 / 部署), 跨任意模型 provider。

---

## 3. 结论 (官方 Summary + 选型建议)

- **选 Deep Agents**: 想要模型和基础设施的灵活性、内置多租户部署、以及"托管 / 自托管
  无需改代码即可切换"。—— 尤其当你要跨模型 provider、或想复用 LangChain/LangSmith 生态。
- **选 Claude Agent SDK**: 已深度投入 Anthropic 生态, 且愿意自托管、自己搭 API / auth /
  多租户层。—— 想要与 Claude 最紧的深度优化时。

一句话记忆: **Deep Agents 换灵活性 (任意模型 + 可插拔 backend + 内置多租户/部署);
Claude Agent SDK 换深度整合 (Claude 专属优化, 但周边要自己搭)。**

---

## 4. 一个有意思的边角料 (社区实测)

有社区文章 (Medium, Lit Phansiri, 2026-03) 指出: LM Studio 同时提供 OpenAI 兼容
(`/v1/chat/completions`) 和 Anthropic 兼容 (`/v1/messages`) 端点, 于是两个框架其实都能
指向【本地模型】跑 —— Claude Agent SDK 只要改 `ANTHROPIC_BASE_URL` / `ANTHROPIC_AUTH_TOKEN`
两个环境变量就能重定向到本地。这说明"Claude Agent SDK 只能连 Anthropic"更多是默认取向
而非硬限制。(此为第三方实测观点, 非官方保证, 具体行为需进一步确认。)

> 本文件为纯对比 / 讲解文档, 不含可运行代码。所有事实以官方对比页为准; 官方明确该页
> 有时效性, 产品若已变化请以最新文档为准。

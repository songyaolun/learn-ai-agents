# pi agent 调研笔记 + 学习示例

## 项目是什么

[pi](https://github.com/badlogic/pi-mono)（earendil-works/pi，原 badlogic/pi-mono，作者 Mario Zechner）是一个
**AI agent 工具箱 + 自扩展 coding agent**，MIT 协议，纯 TypeScript，GitHub 70k+ stars。
它的走红点在于设计哲学与主流框架相反：**不做图 runtime、不做重抽象**——消息就是可
`JSON.stringify` 的普通对象，agent loop 就是一个几百行的透明小类，CLI 本体就是用自家
SDK 搭的，代码可以从上读到下。

对本仓库的学习路线而言，pi 恰好是一个"反方参照"：LangChain/LangGraph/DeepAgents 在
不断加抽象层，pi 在证明一个生产级 coding agent 可以只靠三层薄薄的包。

## 三层包结构（与本仓库目录的对照）

| pi 的包 | 干什么 | 本仓库对应物 | 示例 |
|---|---|---|---|
| `@earendil-works/pi-ai` | 统一多厂商 LLM API（消息/流式事件/工具调用协议跨 Anthropic/OpenAI/Google 通用），**不含 loop** | `claude-code/ch01.py` 里的裸 anthropic SDK 那一层 | `ch01_pi_ai.ts` |
| `@earendil-works/pi-agent-core` | `Agent` 类托管 agent loop：工具执行、事件流、steering/followUp 插话队列 | `langchain/quickstart.py` 的 `create_agent` | `ch02_agent_loop.ts` |
| `@earendil-works/pi-coding-agent` | 完整 coding-agent harness：内置 read/bash/edit/write 工具、会话树持久化/分叉、compaction、skills/extensions/AGENTS.md | `deepagents/` 的 `create_deep_agent`（外加 `langgraph/time_travel.py` 的分叉、`middleware_summarization.py` 的压缩） | `ch03_coding_agent.ts` |

另有 `@earendil-works/pi-tui`（终端 UI 库）和 `pi` CLI 本体，不在示例范围内。

## 运行

```bash
cd pi
npm install          # 独立的 node 工程，不走仓库根部的 uv
npm run ch01         # pi-ai：统一 LLM API + 手写工具循环
npm run ch02         # pi-agent-core：Agent 托管 loop + 自定义工具 + 事件流
npm run ch03         # pi-coding-agent SDK：迷你 Claude Code（内置工具 + 会话管理）
```

环境变量沿用根目录 `.env`（`MODEL_ID` 必需；`ANTHROPIC_BASE_URL` 可选网关 +
`ANTHROPIC_API_KEY`）。pi-ai 读的是 `ANTHROPIC_OAUTH_TOKEN`/`ANTHROPIC_API_KEY`，
前者优先级更高，所以示例在设置了网关时会 `delete` 掉 OAuth token（对应 Python 侧
pop 掉 `ANTHROPIC_AUTH_TOKEN` 的老约定）。

## 踩坑记录

- **请求失败不抛异常**：`completeSimple`/`streamSimple`（以及 `Agent.prompt()` 底层）
  失败时正常返回，错误藏在 `AssistantMessage.stopReason === "error"` + `errorMessage`
  里，不检查就只能看到一片空输出、usage 全 0。三个示例都加了防御性检查。
- **`MODEL_ID` 不在 pi 内置目录**：走网关的自定义模型名 `models.getModel()` 查不到，
  返回 `undefined`。`Model` 就是普通对象，拿一个内置模型当模板改 `id`/`baseUrl` 字段
  即可（`baseUrl` 是 Model 上的字段而非全局配置，这是 pi "数据优先"设计的体现）。
- **npm 包名已迁移**：老文章里的 `@mariozechner/pi-*` 已改为 `@earendil-works/pi-*`
  （仓库从 badlogic/pi-mono 转到 earendil-works 组织，GitHub 会自动跳转）。
- **`tsx -e` 内联脚本不支持顶层 await**（默认按 CJS 转译），调试片段要写成 `.ts`
  文件放进本目录（`package.json` 已声明 `"type": "module"`）再 `npx tsx` 跑。

## 参考

- 仓库：https://github.com/badlogic/pi-mono
- SDK 文档：https://github.com/badlogic/pi-mono/blob/main/packages/coding-agent/docs/sdk.md
- 各包说明：https://github.com/badlogic/pi-mono/blob/main/packages/coding-agent/docs/packages.md

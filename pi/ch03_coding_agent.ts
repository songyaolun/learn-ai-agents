/**
 * ch03 —— pi-coding-agent SDK：完整 coding-agent harness（createAgentSession）
 *
 * 对照 ch02_agent_loop.ts：Agent 只有你手动给的工具；createAgentSession 直接给你一个
 * "迷你 Claude Code"——内置 read/bash/edit/write/grep/find/ls 编码工具、系统提示词自动组装、
 * 会话树持久化（默认 ~/.pi/agent/sessions/，可分叉/回溯任意历史节点，对照
 * langgraph/time_travel.py 的 checkpoint 分叉）、上下文压缩 compaction（对照
 * langchain/middleware_summarization.py）、skills/extensions/AGENTS.md 加载（对照
 * deepagents/skills_memory.py）。
 * 对照 deepagents/quickstart.py：create_deep_agent 是"在 create_agent 上叠 harness"，
 * createAgentSession 是同一思路的 TS 版；关键差别是 pi 的 CLI 本体就是用这个 SDK 搭出来的，
 * 学它约等于直接看一个真实 coding agent 的内部构造（而不是简化教学版）。
 * 另一个差别：deepagents 默认虚拟文件系统（StateBackend），pi 的文件工具直接操作真实磁盘 ——
 * 相当于 deepagents/backend.py 里 FilesystemBackend 是常态而非可选项。
 *
 * 本文件演示：
 * 1. SessionManager.inMemory() —— 不落盘（换成 SessionManager.create(cwd) 即持久化+可续接）
 * 2. tools 白名单 —— 只开只读的 read/ls，不给 bash/write/edit（演示脚本避免意外改文件）
 * 3. defineTool 自定义工具与内置工具并存
 * 4. 事件订阅观察内置工具的真实执行
 *
 * 运行：cd pi && npm install && npm run ch03
 */
import * as dotenv from "dotenv";
import { fileURLToPath } from "node:url";
import { createAgentSession, defineTool, SessionManager } from "@earendil-works/pi-coding-agent";
import { Type, type Api, type Model } from "@earendil-works/pi-ai";
import { builtinModels } from "@earendil-works/pi-ai/providers/all";

// 环境加载与模型解析逻辑同 ch01/ch02（每个示例独立可读，不抽公共模块）
dotenv.config({ path: fileURLToPath(new URL("../.env", import.meta.url)), override: true });
if (process.env.ANTHROPIC_BASE_URL) delete process.env.ANTHROPIC_OAUTH_TOKEN;

const models = builtinModels();

function getModelFromEnv(): Model<Api> {
  const modelId = process.env.MODEL_ID;
  if (!modelId) throw new Error("请在根目录 .env 配置 MODEL_ID（参考 .env.example）");
  const catalog = models.getModel("anthropic", modelId);
  const template: Model<Api> = catalog ?? { ...models.getModels("anthropic")[0], id: modelId, name: modelId };
  const baseUrl = process.env.ANTHROPIC_BASE_URL;
  return baseUrl ? { ...template, baseUrl } : template;
}

// ---- 自定义工具：defineTool 比 ch02 的裸 AgentTool 多了 label/promptSnippet 等 UI/提示词元数据 ----
const getTime = defineTool({
  name: "get_time",
  label: "当前时间",
  description: "获取当前本地时间",
  parameters: Type.Object({}),
  execute: async () => ({
    content: [{ type: "text", text: new Date().toLocaleString("zh-CN", { timeZone: "Asia/Shanghai" }) }],
    details: {},
  }),
});

// ---- 组装 session：这一个调用背后是 模型注册表+设置+会话存储+工具装配 的完整 harness ----
const { session } = await createAgentSession({
  model: getModelFromEnv(),
  thinkingLevel: "off",
  // 白名单模式：只启用列出的工具名。省略该参数则默认开 read/bash/edit/write 全套编码工具
  tools: ["read", "ls", "get_time"],
  customTools: [getTime],
  // 内存态会话：演示脚本不往 ~/.pi/agent/sessions/ 写文件。
  // 换成 SessionManager.create(process.cwd()) 就是 pi CLI 的默认行为：
  // 每条消息追加写入 jsonl，之后可以 continueRecent() 续接、可以 fork 任意历史节点
  sessionManager: SessionManager.inMemory(),
});

// 事件协议与 ch02 完全一致 —— AgentSession 内部包的就是 pi-agent-core 的 Agent
session.subscribe((event) => {
  switch (event.type) {
    case "message_update":
      if (event.assistantMessageEvent.type === "text_delta") {
        process.stdout.write(event.assistantMessageEvent.delta);
      }
      break;
    case "tool_execution_start":
      console.log(`\n[工具开始] ${event.toolName}(${JSON.stringify(event.args)})`);
      break;
    case "tool_execution_end":
      console.log(`[工具结束] ${event.toolName} ${event.isError ? "出错" : "成功"}`);
      break;
    // 踩坑（与 ch01/ch02 相同）：底层请求失败不抛异常，错误在 stopReason === "error" 的消息里
    case "message_end":
      if (event.message.role === "assistant" && event.message.stopReason === "error") {
        console.error(`\n[请求失败] ${event.message.errorMessage}`);
      }
      break;
  }
});

// read/ls 是内置工具（真实读磁盘），get_time 是自定义工具 —— 对模型来说无差别
await session.prompt("看一下当前目录有哪些文件，读一下 package.json 总结这个项目用了哪些 pi 的包，最后告诉我现在几点。");

session.dispose();

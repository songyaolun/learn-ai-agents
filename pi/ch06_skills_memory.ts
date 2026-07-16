/**
 * ch06 —— pi-coding-agent：skills（技能库）+ AGENTS.md（长期记忆/偏好）
 *
 * 对照 deepagents/skills_memory.py：概念一一对应 ——
 * - skills：SKILL.md + YAML frontmatter（name/description），只有"名字+一句话描述"预先进
 *   system prompt，模型判断和当前任务相关时才主动 read 全文（"渐进式展示"，省上下文）；
 * - AGENTS.md：每一轮完整进 system prompt（适合放用户偏好/项目约定，等价 deepagents 的 memory）。
 * pi 的目录约定：项目级 skills 放 <cwd>/.pi/skills/<名字>/SKILL.md，全局放 ~/.pi/agent/skills/；
 * AGENTS.md 直接放项目根目录（也兼容读 CLAUDE.md）。这套约定和 Claude Code 同构 ——
 * 你为 Claude Code 写的 AGENTS.md/skills 在 pi 里原样可用，反之亦然。
 *
 * 本文件演示：
 * 1. 在临时目录里现场生成 AGENTS.md 和一个 haiku 技能
 * 2. DefaultResourceLoader 从该目录发现资源（这一步纯本地，不用 API key 也能跑通验证）
 * 3. 带上这些资源起一个 session，观察模型先 read 技能全文、再按 AGENTS.md 的偏好作答
 *
 * 运行：cd pi && npm install && npm run ch06
 */
import * as dotenv from "dotenv";
import { mkdirSync, mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { createAgentSession, DefaultResourceLoader, SessionManager, SettingsManager } from "@earendil-works/pi-coding-agent";
import type { Api, Model } from "@earendil-works/pi-ai";
import { builtinModels } from "@earendil-works/pi-ai/providers/all";

// 环境加载与模型解析逻辑同 ch01（每个示例独立可读，不抽公共模块）
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

// ---- 1. 现场搭一个"项目目录"：AGENTS.md + .pi/skills/haiku/SKILL.md ----
const workdir = mkdtempSync(join(tmpdir(), "pi-skills-demo-"));
const agentDir = join(workdir, "fake-agent-dir"); // 隔离的全局配置目录，避免碰真实的 ~/.pi/agent

writeFileSync(
  join(workdir, "AGENTS.md"),
  ["# 项目约定", "", "- 用户的名字是「阿松」，回答时请这样称呼", "- 用户偏好：先给结论，再给解释"].join("\n"),
);

// SKILL.md 的 frontmatter 只有 name/description 会预先进 system prompt，正文按需读取
const skillDir = join(workdir, ".pi", "skills", "haiku");
mkdirSync(skillDir, { recursive: true });
writeFileSync(
  join(skillDir, "SKILL.md"),
  [
    "---",
    "name: haiku",
    "description: 当用户要求写俳句（haiku）时，先读取本技能了解格式规则",
    "---",
    "",
    "# 俳句写作规则",
    "",
    "1. 恰好三行，意象取自季节",
    "2. 每行结尾加一个贴合意境的 emoji",
    "3. 最后单独一行标注季节，如「—— 秋」",
  ].join("\n"),
);

// ---- 2. 资源发现：DefaultResourceLoader 扫描 cwd/agentDir 下的约定路径 ----
// 这一步不调用模型，纯本地文件扫描 —— 可以先单独验证 skills/AGENTS.md 是否被正确发现
const settingsManager = SettingsManager.create(workdir, agentDir);
const loader = new DefaultResourceLoader({ cwd: workdir, agentDir, settingsManager });
await loader.reload();

console.log("=== 资源发现结果（本地扫描，无需 API key） ===");
for (const skill of loader.getSkills().skills) {
  console.log(`skill: ${skill.name} —— ${skill.description}（${skill.filePath}）`);
}
for (const f of loader.getAgentsFiles().agentsFiles) {
  console.log(`context file: ${f.path}`);
}

// ---- 3. 带上资源起 session：观察"渐进式展示"真实发生 ----
const { session } = await createAgentSession({
  cwd: workdir,
  agentDir,
  model: getModelFromEnv(),
  thinkingLevel: "off",
  tools: ["read"], // 模型要主动 read 技能全文，所以 read 工具必须在
  resourceLoader: loader,
  sessionManager: SessionManager.inMemory(),
  settingsManager,
});

session.subscribe((event) => {
  switch (event.type) {
    case "message_update":
      if (event.assistantMessageEvent.type === "text_delta") {
        process.stdout.write(event.assistantMessageEvent.delta);
      }
      break;
    case "tool_execution_start":
      // 预期能看到一次 read(.../.pi/skills/haiku/SKILL.md) —— 模型按需取全文，这就是渐进式展示
      console.log(`\n[工具] ${event.toolName}(${JSON.stringify(event.args)})`);
      break;
    case "message_end":
      if (event.message.role === "assistant" && event.message.stopReason === "error") {
        console.error(`\n[请求失败] ${event.message.errorMessage}`);
      }
      break;
  }
});

console.log("\n=== 提问（预期：称呼「阿松」+ 先读技能再按规则写俳句） ===");
await session.prompt("给我写一首关于秋天的俳句");
console.log();
session.dispose();

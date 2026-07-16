/**
 * ch05 —— pi-coding-agent：会话持久化 / 会话树分叉 / compaction 压缩
 *
 * 对照 langgraph/persistence.py（SqliteSaver 落盘 + 新图实例用同一 thread_id 续接）和
 * langgraph/time_travel.py（get_state_history 找到历史 checkpoint → update_state 分叉重跑）：
 * pi 用"一个 jsonl 文件"实现同样的三件事，没有数据库、没有 checkpointer 抽象 ——
 * 每条消息是一个带 id/parentId 的 entry，天然构成一棵树；"分叉"不复制任何历史，
 * 只是把当前叶子指针（leafId）挪到某个历史节点上继续追加，旧分支原样保留在文件里。
 * compaction 对照 langchain/middleware_summarization.py：把旧历史压成一段摘要 entry，
 * 之后组装上下文时用摘要替代被压掉的部分（原文仍在文件里，只是不再进模型）。
 *
 * 本文件演示：
 * 1. SessionManager.create() 持久化会话 → 看 jsonl 文件长什么样（就是普通文本）
 * 2. 关掉再用 SessionManager.open() 打开同一文件 → 模型仍记得之前的对话（跨"进程"续接）
 * 3. navigateTree() 回到第一条用户消息处分叉 → 换一个说法重问 → 树上出现两条分支
 * 4. session.compact() 手动触发压缩，看摘要内容
 *
 * 运行：cd pi && npm install && npm run ch05
 */
import * as dotenv from "dotenv";
import { mkdtempSync, readFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { createAgentSession, SessionManager, type AgentSession } from "@earendil-works/pi-coding-agent";
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

// 只订阅文本输出和错误，让演示聚焦在会话管理上
function printTextEvents(session: AgentSession) {
  session.subscribe((event) => {
    if (event.type === "message_update" && event.assistantMessageEvent.type === "text_delta") {
      process.stdout.write(event.assistantMessageEvent.delta);
    }
    if (event.type === "message_end" && event.message.role === "assistant" && event.message.stopReason === "error") {
      console.error(`\n[请求失败] ${event.message.errorMessage}`);
    }
  });
}

const model = getModelFromEnv();
// 会话文件放系统临时目录，不污染 ~/.pi/agent/sessions/（pi CLI 的默认存放处）
const sessionDir = mkdtempSync(join(tmpdir(), "pi-sessions-"));

// ---- 1. 持久化会话：每条消息实时追加写入 jsonl ----
console.log("=== 1. 持久化会话（SessionManager.create） ===");
const sm1 = SessionManager.create(process.cwd(), sessionDir);
const { session: session1 } = await createAgentSession({
  model,
  thinkingLevel: "off",
  tools: [], // 纯对话即可，不需要任何工具
  sessionManager: sm1,
});
printTextEvents(session1);
await session1.prompt("我最喜欢的颜色是绿色，请记住。回复不超过10个字。");
console.log();
const sessionFile = sm1.getSessionFile()!;
session1.dispose();

// 会话文件就是普通 jsonl：第一行是 header，之后每行一个 entry（带 id/parentId 构成树）
console.log(`\n会话文件：${sessionFile}`);
const lines = readFileSync(sessionFile, "utf-8").trim().split("\n");
console.log(`共 ${lines.length} 行，第一条消息 entry 长这样（截断）：`);
console.log(`  ${lines[1]?.slice(0, 120)}...`);

// ---- 2. 跨"进程"续接：重新 open 同一个文件，历史自动读回 ----
// 对照 langgraph/persistence.py 里"新建图实例 + 同一 thread_id"的验证方式
console.log("\n=== 2. 续接（SessionManager.open 同一文件） ===");
const sm2 = SessionManager.open(sessionFile);
const { session: session2 } = await createAgentSession({
  model,
  thinkingLevel: "off",
  tools: [],
  sessionManager: sm2,
});
printTextEvents(session2);
await session2.prompt("我最喜欢的颜色是什么？回复不超过10个字。"); // 预期答"绿色"——历史生效
console.log();

// ---- 3. 会话树分叉：回到过去，换个说法重来 ----
// 对照 langgraph/time_travel.py：那里要 get_state_history 翻 checkpoint 列表再 update_state；
// 这里更直接 —— 找到想回去的 entry id，把叶子指针挪过去（navigateTree），继续对话即分叉。
console.log("\n=== 3. 分叉（navigateTree 回到第一条用户消息） ===");
const forkPoints = session2.getUserMessagesForForking(); // 所有用户消息 = 天然的候选分叉点
console.log(`候选分叉点：${forkPoints.map((p) => JSON.stringify(p.text.slice(0, 15))).join(" / ")}`);
await session2.navigateTree(forkPoints[0].entryId); // 回到"说自己喜欢绿色"那条消息之后
await session2.prompt("不对，其实我最喜欢蓝色。我最喜欢的颜色是什么？回复不超过10个字。");
console.log();
// 树上现在有两条分支：绿色线（问答各一条）和蓝色线（刚追加的）。旧分支没有被删除：
const entries = sm2.getEntries();
console.log(`\n文件里共 ${entries.length} 个 entry（两条分支的历史都在），当前分支上有 ${sm2.getBranch().length} 个`);

// ---- 4. compaction：把当前分支的旧历史压成摘要 ----
// 平时由阈值自动触发（上下文快满时），这里手动调用观察效果。压缩也是一种 entry：
// 原文还在文件里，只是之后组装模型上下文时用摘要替代
console.log("\n=== 4. compaction（手动触发） ===");
const compaction = await session2.compact();
console.log(`压缩前估算 tokens：${compaction.tokensBefore}`);
console.log(`摘要（截断）：${compaction.summary.slice(0, 150)}...`);

session2.dispose();
console.log(`\n(会话文件保留在 ${sessionDir}，可以用文本编辑器打开观察树结构)`);

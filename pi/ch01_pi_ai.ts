/**
 * ch01 —— pi-ai：统一多厂商 LLM API（pi 三层包的最底层，本身还不是 agent）
 *
 * 对照 claude-code/ch01.py（裸 anthropic SDK）：那里消息格式、流式事件、工具调用协议
 * 全是 Anthropic 一家的私有格式；pi-ai 把这一层统一掉 —— 同一套 Context/Message/事件协议
 * 跨 Anthropic/OpenAI/Google 等所有厂商通用，换模型只是换 getModel() 的参数。
 * 对照 langchain 的 ChatAnthropic：定位类似（统一模型接口），但 pi-ai 刻意保持轻量 ——
 * 消息就是普通 JSON 对象，可以直接 JSON.stringify 落盘、再读回来继续对话，
 * 没有 Runnable/LCEL 那层包装（这也是 pi 整个项目的设计哲学：能少一层抽象就少一层）。
 *
 * 本文件演示（注意：pi-ai 不替你跑 agent loop，工具调用循环要自己写，下一章才托管）：
 * 1. completeSimple —— 非流式一问一答 + usage 统计
 * 2. streamSimple  —— 流式逐 token 输出（text_delta 事件）
 * 3. 手写 while 循环处理 stopReason === "toolUse"，对照 claude-code/ch01.py 的 run_one_turn
 *
 * 运行：cd pi && npm install && npm run ch01
 */
import * as dotenv from "dotenv";
import { fileURLToPath } from "node:url";
import { Type, validateToolCall, type Api, type Context, type Model, type Tool } from "@earendil-works/pi-ai";
import { builtinModels } from "@earendil-works/pi-ai/providers/all";

// 与本仓库 Python 脚本同一约定：加载根目录 .env（override 生效）
dotenv.config({ path: fileURLToPath(new URL("../.env", import.meta.url)), override: true });
// 配了自定义网关就只走 ANTHROPIC_API_KEY —— pi-ai 里 ANTHROPIC_OAUTH_TOKEN 优先级更高，
// 留着会和网关鉴权冲突（对应 Python 侧 pop 掉 ANTHROPIC_AUTH_TOKEN 的逻辑）
if (process.env.ANTHROPIC_BASE_URL) delete process.env.ANTHROPIC_OAUTH_TOKEN;

const models = builtinModels();

// MODEL_ID 可能不在 pi 的内置模型目录里（比如走网关的自定义模型名），
// 目录里查不到就拿一个内置 anthropic 模型当模板、只改 id —— Model 就是个普通对象，改字段即可。
// baseUrl 是 Model 上的字段（不是全局配置），所以网关地址也是覆盖字段实现的。
function getModelFromEnv(): Model<Api> {
  const modelId = process.env.MODEL_ID;
  if (!modelId) throw new Error("请在根目录 .env 配置 MODEL_ID（参考 .env.example）");
  const catalog = models.getModel("anthropic", modelId);
  const template: Model<Api> = catalog ?? { ...models.getModels("anthropic")[0], id: modelId, name: modelId };
  const baseUrl = process.env.ANTHROPIC_BASE_URL;
  return baseUrl ? { ...template, baseUrl } : template;
}

const model = getModelFromEnv();

// ---- 1. 非流式：completeSimple ----
// Context 是纯数据：systemPrompt + messages + tools，没有任何隐藏状态
console.log("=== 1. completeSimple（非流式） ===");
const context: Context = {
  systemPrompt: "你是一个简洁的中文助手，用一两句话回答。",
  messages: [{ role: "user", content: "用一句话解释什么是 agent loop", timestamp: Date.now() }],
};
const reply = await models.completeSimple(model, context);
// 踩坑：completeSimple/streamSimple 请求失败不抛异常，错误静默地放在返回消息的
// stopReason === "error" + errorMessage 里 —— 不检查的话只会看到一片空输出
if (reply.stopReason === "error") throw new Error(`请求失败：${reply.errorMessage}`);
for (const block of reply.content) {
  if (block.type === "text") console.log(block.text);
}
console.log(`[usage] ${JSON.stringify(reply.usage)}`);

// ---- 2. 流式：streamSimple ----
// 事件协议同样是跨厂商统一的：text_delta / toolcall_end / done / error
console.log("\n=== 2. streamSimple（流式） ===");
const streamContext: Context = {
  systemPrompt: "你是一个简洁的中文助手。",
  messages: [{ role: "user", content: "数出 1 到 5，每个数字一行", timestamp: Date.now() }],
};
const stream = models.streamSimple(model, streamContext);
for await (const event of stream) {
  if (event.type === "text_delta") process.stdout.write(event.delta);
  if (event.type === "error") throw new Error(`请求失败：${event.error.errorMessage}`);
}
await stream.result(); // 拿到最终完整的 AssistantMessage（这里只为等流结束）
console.log();

// ---- 3. 手写工具调用循环 ----
// 对照 claude-code/ch01.py：结构一模一样 —— 调模型 → 若要用工具则执行并回填 toolResult → 再调模型
console.log("\n=== 3. 手写 agent loop（stopReason === 'toolUse'） ===");
const tools: Tool[] = [
  {
    name: "get_time",
    description: "获取当前时间",
    parameters: Type.Object({
      timezone: Type.Optional(Type.String({ description: "IANA 时区，如 Asia/Shanghai" })),
    }),
  },
];
const loopContext: Context = {
  systemPrompt: "你是一个中文助手，需要时间信息时必须调用 get_time 工具。",
  messages: [{ role: "user", content: "现在几点了？", timestamp: Date.now() }],
  tools,
};

while (true) {
  const message = await models.completeSimple(model, loopContext);
  if (message.stopReason === "error") throw new Error(`请求失败：${message.errorMessage}`);
  loopContext.messages.push(message); // assistant 消息原样回填历史

  if (message.stopReason !== "toolUse") {
    for (const block of message.content) {
      if (block.type === "text") console.log(block.text);
    }
    break; // 模型不再要求用工具，回合结束
  }

  for (const block of message.content) {
    if (block.type !== "toolCall") continue;
    console.log(`[toolCall] ${block.name}(${JSON.stringify(block.arguments)})`);
    // validateToolCall 用 TypeBox schema 校验模型给的参数（防幻觉参数），校验失败会 throw
    const args = validateToolCall(tools, block) as { timezone?: string };
    const result = new Date().toLocaleString("zh-CN", { timeZone: args.timezone ?? "Asia/Shanghai" });
    loopContext.messages.push({
      role: "toolResult", // 统一的 toolResult 角色，不是 Anthropic 私有的 user+tool_result 块
      toolCallId: block.id,
      toolName: block.name,
      content: [{ type: "text", text: result }],
      isError: false,
      timestamp: Date.now(),
    });
  }
}

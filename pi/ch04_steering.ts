/**
 * ch04 —— pi-agent-core 特色能力：steering / followUp（agent 运行中插话）
 *
 * 对照 ch02_agent_loop.ts：那里 prompt() 发出后只能干等结果；但实际用 coding agent 时
 * 经常想中途纠偏（"等等，方向错了"）或追加任务（"做完顺便把 X 也处理了"）。
 * 对照 langgraph/human_in_loop.py 的 interrupt：那是 agent 主动暂停、等人批准（人是被动的）；
 * steering 反过来 —— agent 不停，人主动往正在跑的循环里塞消息（人是主动的），两者互补。
 * LangChain 的 create_agent 没有等价物；这是 pi 从 Claude Code 的交互习惯
 * （esc 打断输入新指令）里提炼进框架层的能力。
 *
 * 两个队列的语义区别：
 * - steer()    —— 插话：当前工具执行完就生效，模型下一轮立刻看到插话内容并调整方向；
 * - followUp() —— 排队：等当前任务完整跑完，再作为新的用户消息自动开启下一轮。
 *
 * 本文件演示：让 agent 依次查 3 个城市的天气（sequential 强制逐个执行、拉长运行时间），
 * 在第 1 个工具执行时插话"后面的不用查了"——观察模型收到插话后放弃剩余任务；
 * 同时预先排一个 followUp，观察它在主任务结束后才被处理。
 *
 * 运行：cd pi && npm install && npm run ch04
 */
import * as dotenv from "dotenv";
import { fileURLToPath } from "node:url";
import { Agent, type AgentTool } from "@earendil-works/pi-agent-core";
import { Type, type Api, type Model } from "@earendil-works/pi-ai";
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

const weatherParams = Type.Object({
  city: Type.String({ description: "城市名" }),
});
const getWeather: AgentTool<typeof weatherParams> = {
  name: "get_weather",
  label: "查天气",
  description: "查询单个城市的当前天气（一次只能查一个城市，演示用返回模拟数据）",
  parameters: weatherParams,
  execute: async (_toolCallId, params) => {
    await new Promise((r) => setTimeout(r, 300)); // 模拟真实工具的耗时，留出插话的时间窗
    return {
      content: [{ type: "text", text: `${params.city}：晴，28°C` }],
      details: {},
    };
  },
};

const agent = new Agent({
  initialState: {
    systemPrompt: "你是一个简洁的中文助手。查天气必须用 get_weather 工具，一次一个城市。",
    model: getModelFromEnv(),
    thinkingLevel: "off",
    tools: [getWeather],
  },
  toolExecution: "sequential", // 强制工具逐个执行（默认 parallel 会把 3 个查询并发跑完，没机会插话）
});

let steered = false;
agent.subscribe((event) => {
  switch (event.type) {
    case "message_update":
      if (event.assistantMessageEvent.type === "text_delta") {
        process.stdout.write(event.assistantMessageEvent.delta);
      }
      break;
    case "tool_execution_start": {
      console.log(`\n[工具开始] ${event.toolName}(${JSON.stringify(event.args)})`);
      // 第一个工具刚开始跑，就往队列里塞插话 —— steer 不打断正在执行的工具，
      // 而是保证"这个工具跑完、结果和插话一起进入模型下一轮的视野"
      if (!steered) {
        steered = true;
        console.log("[插话] >>> steer：其他城市不用查了，直接汇报已有结果");
        agent.steer({
          role: "user",
          content: "改主意了：其他城市不用查了，就用已查到的结果直接回答。",
          timestamp: Date.now(),
        });
      }
      break;
    }
    case "tool_execution_end":
      console.log(`[工具结束] ${event.toolName}`);
      break;
    case "message_end":
      if (event.message.role === "assistant" && event.message.stopReason === "error") {
        console.error(`\n[请求失败] ${event.message.errorMessage}`);
      }
      break;
    case "agent_end":
      console.log("\n[agent_end] —— 一轮任务（含被 steer 改道的部分）结束");
      break;
  }
});

// followUp 先排上：它不会打扰当前任务，等 agent 空闲后自动作为下一条用户消息触发新一轮
agent.followUp({
  role: "user",
  content: "用一句话总结你刚才实际做了什么、跳过了什么。",
  timestamp: Date.now(),
});

// 主任务：3 个城市。预期轨迹 —— 查北京(此时被插话) → 模型看到插话放弃上海/广州直接作答
// → agent 空闲 → followUp 出队 → 模型再答一轮总结
await agent.prompt("依次查一下北京、上海、广州的天气，然后汇总。");
await agent.waitForIdle(); // followUp 触发的后续轮次也要等完，否则脚本提前退出

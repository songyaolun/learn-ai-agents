/**
 * ch02 —— pi-agent-core：托管 agent loop（Agent 类）
 *
 * 对照 ch01_pi_ai.ts：那里 while 循环、参数校验、工具执行、toolResult 回填全要自己写；
 * Agent 类把这套循环接管 —— 你只提供 systemPrompt/model/tools，agent.prompt() 一句话
 * 跑完整个多轮工具调用，中间过程通过 subscribe 的事件流观察。
 * 对照 langchain/quickstart.py 的 create_agent：定位几乎一样（托管 loop + 自定义工具 + 事件流），
 * 区别在于 pi 没有图 runtime —— Agent 就是一个持有 state 的小类，事件驱动、透明可读；
 * 且自带 steering（运行中插话改方向）/followUp（排队追加任务）队列，这是 create_agent 没有的。
 *
 * 本文件演示：
 * 1. AgentTool 自定义工具 —— TypeBox schema + execute 函数（对照 langchain 的 @tool 装饰器）
 * 2. subscribe 事件流 —— message_update 逐 token / tool_execution_start/end 观察工具执行
 * 3. 一次 prompt 触发多个工具的自动循环（工具默认并行执行，可配 toolExecution: "sequential"）
 *
 * 运行：cd pi && npm install && npm run ch02
 */
import * as dotenv from "dotenv";
import { fileURLToPath } from "node:url";
import { Agent, type AgentTool } from "@earendil-works/pi-agent-core";
import { Type, type Api, type Model } from "@earendil-works/pi-ai";
import { builtinModels } from "@earendil-works/pi-ai/providers/all";

// 环境加载与模型解析逻辑同 ch01（延续仓库惯例：每个示例独立可读，不抽公共模块）
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

// ---- 自定义工具：schema（TypeBox）+ execute，比 ch01 多了 execute —— 循环执行交给 Agent ----
const weatherParams = Type.Object({
  city: Type.String({ description: "城市名" }),
});
const getWeather: AgentTool<typeof weatherParams> = {
  name: "get_weather",
  label: "查天气",
  description: "查询指定城市的当前天气（演示用，返回模拟数据）",
  parameters: weatherParams,
  execute: async (_toolCallId, params) => ({
    content: [{ type: "text", text: `${params.city}：晴，28°C，湿度 40%` }],
    details: { city: params.city }, // details 是给 UI/日志用的结构化数据，不进模型上下文
  }),
};

const rateParams = Type.Object({
  amount: Type.Number({ description: "金额" }),
  from: Type.String({ description: "源货币，如 USD" }),
  to: Type.String({ description: "目标货币，如 CNY" }),
});
const convertCurrency: AgentTool<typeof rateParams> = {
  name: "convert_currency",
  label: "汇率换算",
  description: "按固定汇率换算货币（演示用，USD->CNY 固定 7.2）",
  parameters: rateParams,
  execute: async (_toolCallId, params) => ({
    content: [{ type: "text", text: `${params.amount} ${params.from} ≈ ${(params.amount * 7.2).toFixed(2)} ${params.to}` }],
    details: {},
  }),
};

// ---- 组装 Agent：initialState 就是全部配置，没有隐藏的图结构 ----
const agent = new Agent({
  initialState: {
    systemPrompt: "你是一个简洁的中文助手，回答前先调用需要的工具。",
    model: getModelFromEnv(),
    thinkingLevel: "off",
    tools: [getWeather, convertCurrency],
  },
});

// ---- 订阅事件流：等价于 langchain/stream.py 里 messages+updates 两种流合在一个回调里 ----
agent.subscribe((event) => {
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
    // 踩坑（与 ch01 相同）：底层请求失败不抛异常，prompt() 正常返回但输出为空，
    // 错误在 assistant 消息的 stopReason === "error" + errorMessage 里，要自己盯
    case "message_end":
      if (event.message.role === "assistant" && event.message.stopReason === "error") {
        console.error(`\n[请求失败] ${event.message.errorMessage}`);
      }
      break;
    case "agent_end":
      console.log(`\n[agent_end] 本轮共产生 ${event.messages.length} 条消息`);
      break;
  }
});

// 一句话里两个任务 → 模型会发起两个工具调用，Agent 自动执行并把结果喂回模型直到得出最终答案。
// 运行中还可以 agent.steer("...") 插话纠偏、agent.followUp("...") 排队追加任务（此处不演示）。
await agent.prompt("北京今天天气怎么样？顺便帮我把 100 美元换算成人民币。");

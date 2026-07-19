"""LangChain event streaming —— 节点/工具粒度的类型化事件投影。

对比 langchain/stream.py: 那里用 stream_mode="messages"/"updates" 实现消息级流式,
这里用 astream_events 实现更细粒度的类型化事件流(如 on_chat_model_stream/on_tool_start/on_tool_end 等事件类型),
可订阅并解析 agent 内部每个节点的执行事件。

能力点:
1. 事件类型覆盖: 模型流、工具调用/结果、节点执行等类型化事件
2. 事件解析: 结构化解析不同事件的 payload
3. 多事件订阅: 同时监听多种类型事件

官方文档: https://docs.langchain.com/oss/python/langchain/streaming#event-streaming

踩坑记录:
- astream_events 是异步生成器, 必须 async for 消费并跑在事件循环里 (asyncio.run);
  普通 for 或同步函数里直接调用拿不到事件。
- 要指定 version="v2": v1 与 v2 的事件结构/命名不同, 混用会导致解析字段取不到。
- 事件种类非常多 (on_chain_start/on_chat_model_stream/on_tool_end...), 别全量打印,
  按 event["event"] 过滤出关心的类型再解析, 否则输出会被淹没。
"""

import os
import asyncio

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_anthropic import ChatAnthropic
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv(override=True)

# 模型初始化沿用仓库约定
model = ChatAnthropic(
    model=os.environ.get("MODEL_ID", "claude-3-5-sonnet-20240620"),
    base_url=os.getenv("ANTHROPIC_BASE_URL") or None,
)


# 示例工具: 天气查询
# 首次出现缩写: LLM (Large Language Model, 大语言模型)
def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"


# 构建 agent
# checkpointer 用于追踪事件流状态
agent = create_agent(
    model=model,
    tools=[get_weather],
    system_prompt="You are a helpful assistant.",
    checkpointer=InMemorySaver(),
)


async def stream_events() -> None:
    """异步订阅并解析 agent 事件流"""
    config = {"configurable": {"thread_id": "event-stream-demo"}}
    input_msg = {"messages": [{"role": "user", "content": "What's the weather in San Francisco?"}]}

    # 订阅事件流
    # astream_events 返回 async generator, 需在 async 函数中使用
    async for event in agent.astream_events(
        input_msg,
        config=config,
        version="v2",
        # include_types 匹配的是 runnable 类型 (如 chat_model/tool/chain),
        # 不是完整事件名 (如 on_tool_start)。具体事件名在循环里用 event["event"] 过滤。
        include_types=["chat_model", "tool", "chain"],
    ):
        event_type = event["event"]
        payload = event["data"]

        # 解析不同类型事件
        if event_type == "on_chat_model_stream":
            # 模型流事件: 逐 token 输出
            token = payload["chunk"].content
            if token:
                print(f"[MODEL] {token}", end="", flush=True)
        elif event_type == "on_tool_start":
            # 工具调用开始事件
            tool_name = event["name"]
            tool_args = payload["input"]
            print(f"\n[TOOL-START] {tool_name}({tool_args})")
        elif event_type == "on_tool_end":
            # 工具调用结束事件
            tool_name = event["name"]
            tool_result = payload["output"]
            print(f"[TOOL-END] {tool_name} -> {tool_result}")
        elif event_type == "on_chain_start":
            # 节点执行开始事件
            node_name = event["name"]
            print(f"\n[CHAIN-START] {node_name}")
        elif event_type == "on_chain_end":
            # 节点执行结束事件
            node_name = event["name"]
            print(f"[CHAIN-END] {node_name}")


if __name__ == "__main__":
    # ===== 无网络自测: 验证 agent 组装与事件订阅协程可构造, 不触发真实模型调用 =====
    print("=== 无网络自测 ===")
    assert agent is not None, "agent 应已组装"
    assert asyncio.iscoroutinefunction(stream_events), "stream_events 应为异步协程函数"
    assert get_weather("SF").startswith("It's always sunny"), "get_weather 输出异常"
    print("✓ agent 组装成功, 事件流协程可用, 工具函数可调用")

    # ===== 有网络部分(需配置 .env) =====
    print("\n=== 有网络部分(需配置 .env: MODEL_ID / ANTHROPIC_API_KEY) ===")
    if os.getenv("MODEL_ID") and os.getenv("ANTHROPIC_API_KEY"):
        print("=== 事件流演示 ===\n")
        asyncio.run(stream_events())
        print("\n\n=== 完成 ===")
    else:
        print("跳过: 未检测到 MODEL_ID / ANTHROPIC_API_KEY, 请配置 .env 后运行。")

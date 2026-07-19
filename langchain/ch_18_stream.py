"""LangChain stream —— 观察 agent 的中间执行过程。

对比 langchain/ch_17_quickstart.py: 那里用 invoke 只拿最终结果,
这里用 stream 实时观察 agent 的完整流转:
  - messages 流: LLM 逐 token 输出
  - updates 流: 每个步骤完成时的事件 (工具调用请求 / 工具执行结果)
官方文档: https://docs.langchain.com/oss/python/langchain/streaming
"""

import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.messages import AIMessage, AIMessageChunk, ToolMessage
from langchain_anthropic import ChatAnthropic
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv(override=True)

model = ChatAnthropic(
    model=os.environ["MODEL_ID"],
    base_url=os.getenv("ANTHROPIC_BASE_URL") or None,
)


def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"


# create_agent 底层其实是用 LangGraph 搭的一张图 (可以理解成"模型节点 + 工具节点 +
# 连接它们的循环逻辑"), 只要用到 stream 或者想要多轮记忆, 图就需要一个 checkpointer
# 来记录/追踪运行到哪一步、当前状态是什么。
# InMemorySaver 是最简单的实现: 状态存进程内存里, 进程一退出就丢
# (langgraph/ch_05_persistence.py 会展示换成 SqliteSaver 落盘持久化的版本)。
agent = create_agent(
    model=model,
    tools=[get_weather],
    system_prompt="You are a helpful assistant.",
    checkpointer=InMemorySaver(),
)


if __name__ == "__main__":
    # thread_id 相当于给这次对话起一个"会话编号", checkpointer 按这个编号存取状态:
    # 同一个 thread_id 多次调用 agent.invoke/stream, 历史消息会自动延续 (多轮记忆);
    # 换一个 thread_id 就是全新的对话, 互不干扰。这里只演示单轮, 但 config 仍是必需的。
    config = {"configurable": {"thread_id": "stream-demo"}}
    print("=== agent 中间过程 (stream) ===\n")

    # stream_mode 可以传一个列表, 同时订阅多种事件流; 循环里拿到的每个 chunk 会带一个
    # "type" 字段标明这是哪种流的数据:
    #   "messages" 流: 模型逐 token 生成回复时, 每收到一小段文字就推送一次
    #                  (聊天界面"打字机效果"就是靠这个实现的)
    #   "updates"  流: agent 内部每完成一个"步骤" (比如模型这一轮决定调用了工具、
    #                  或者工具执行完毕拿到了结果) 就推送一次该步骤的增量数据,
    #                  可以看清 agent 内部具体在做什么、走到哪一步了
    for chunk in agent.stream(
        {"messages": [{"role": "user", "content": "What's the weather in San Francisco?"}]},
        config=config,
        stream_mode=["messages", "updates"],  # 同时开 token 流和步骤流
        version="v2",  # LangGraph 流式 API 的版本号, 目前推荐固定用 v2
    ):
        if chunk["type"] == "messages":
            # messages 流的 data 是一个 (token, metadata) 二元组; token 通常是
            # AIMessageChunk (模型输出的一小段增量文字), .text 取出这一小段的纯文本
            token, _ = chunk["data"]
            if isinstance(token, AIMessageChunk) and token.text:
                print(token.text, end="", flush=True)
        elif chunk["type"] == "updates":
            # updates 流的 data 是 {步骤来源: 该步骤新产出的状态}, 例如
            # {"model": {"messages": [...]}} 表示"model"这个节点这一步新产出了消息;
            # source 的取值取决于图里的节点名, create_agent 内置的图固定叫 model/tools。
            # 每个步骤完成时的事件: model 步骤产出工具调用, tools 步骤产出工具结果
            for source, update in chunk["data"].items():
                if source in ("model", "tools") and update.get("messages"):
                    msg = update["messages"][-1]
                    if isinstance(msg, AIMessage) and msg.tool_calls:
                        # AIMessage.tool_calls 非空, 说明模型这一步决定要调用工具
                        # (而不是直接给出文字回复), 列表里每一项是一次具体的调用请求
                        # (工具名 + 参数), 还没真正执行
                        for call in msg.tool_calls:
                            print(f"\n[tool-call] {call['name']}({call['args']})")
                    elif isinstance(msg, ToolMessage):
                        # ToolMessage 是工具真正执行完之后产出的结果, 会被自动加进
                        # 消息历史、喂回给模型让它继续推理 (决定回答还是再调用别的工具)
                        print(f"[tool-result] {msg.content}")
    print("\n\n=== 完成 ===")

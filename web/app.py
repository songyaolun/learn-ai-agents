"""Chainlit 网页交互 —— 把 deepagents/research 的 agent 搬到浏览器。

对比 deepagents/research.py: 那里在终端 invoke 只拿最终结果,
这里用 Chainlit 起一个本地网页, 在浏览器里跟同一个 agent 对话:
  - 流式 token: agent.astream(stream_mode="messages") 逐 token 推到页面
  - 工具步骤可视化: cl.LangchainCallbackHandler() 自动把搜索调用 / subagent 委派
    展示成可折叠步骤 (替代 deepagents/stream.py 里的手动 print)
  - 多轮记忆: InMemorySaver checkpointer + thread_id=会话 id, 同一对话自动延续上下文
  - 多模态: 支持在输入框粘贴/上传图片, 后端读取 -> base64 -> LangChain 图片 block,
    交给底层视觉模型识别 (要求 MODEL_ID 指向支持视觉的模型)

运行: uv run chainlit run web/app.py  (自动开 http://localhost:8000)
官方文档: https://docs.chainlit.com/integrations/langchain
"""

import base64
import os
from pathlib import Path

import chainlit as cl
from deepagents import create_deep_agent
from dotenv import load_dotenv
from langchain.messages import AIMessageChunk
from langchain_anthropic import ChatAnthropic
from langchain_community.tools import DuckDuckGoSearchRun
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv(override=True)


def build_agent() -> object:
    """构建 deep research agent (同 deepagents/research.py, 多了 checkpointer)."""
    # 沿用 ch01-ch03 + deepagents 的接入方式: ANTHROPIC_BASE_URL + MODEL_ID
    model = ChatAnthropic(
        model=os.environ["MODEL_ID"],
        base_url=os.getenv("ANTHROPIC_BASE_URL") or None,
    )
    search = DuckDuckGoSearchRun()  # DuckDuckGo 搜索, 无需 API key

    return create_deep_agent(
        model=model,
        tools=[search],
        system_prompt=(
            "You are a research assistant. Use the search tool to find current, accurate "
            "information. Plan your research, search multiple queries if needed, then "
            "synthesize a clear answer with sources."
        ),
        # checkpointer 是网页版的关键: stream 需要它维护线程状态,
        # 配合 thread_id 还能跨消息保持多轮对话上下文
        checkpointer=InMemorySaver(),
        subagents=[
            {
                "name": "researcher",
                "description": "Delegate a focused research subtopic to this subagent.",
                "system_prompt": "You are a great researcher. Search and return a brief, accurate summary.",
            }
        ],
    )


@cl.on_chat_start
async def on_chat_start() -> None:
    """会话开始时构建 agent 存入 user_session (每个浏览器会话独立一个 agent)."""
    cl.user_session.set("agent", build_agent())
    await cl.Message(
        content="👋 你好!我是 deep research agent,接了 DuckDuckGo 真实搜索。问我任何需要查资料的问题吧。"
    ).send()


@cl.on_message
async def on_message(message: cl.Message) -> None:
    agent = cl.user_session.get("agent")
    # thread_id 绑定当前浏览器会话 → 多轮记忆;
    # callbacks 放 LangchainCallbackHandler → 自动把搜索调用 / subagent 委派展示成可折叠步骤
    config = {
        "configurable": {"thread_id": cl.context.session.id},
        "callbacks": [cl.LangchainCallbackHandler()],
    }
    final_answer = cl.Message(content="")

    # 组装 user message content: 有图片附件时用多模态 block 列表, 否则用普通文本
    user_content = _build_user_content(message)

    async for chunk, _metadata in agent.astream(
        {"messages": [{"role": "user", "content": user_content}]},
        stream_mode="messages",
        config=config,
    ):
        # 只把 LLM 逐 token 输出推到主消息; 工具消息等中间步骤由 callback handler 展示
        if isinstance(chunk, AIMessageChunk) and chunk.text:
            await final_answer.stream_token(chunk.text)

    await final_answer.send()


def _build_user_content(message: cl.Message) -> str | list[dict]:
    """把 chainlit 消息(可能带图片附件)转成 LangChain 消息 content 格式.

    - 无附件: 直接返回文本字符串
    - 有图片附件: 返回 content block 列表 [{type: text}, {type: image, base64, mime_type}, ...]
      使用 LangChain 1.x 标准多模态格式, langchain_anthropic 会自动转成 Anthropic API 格式
    """
    image_elements = [
        el for el in (message.elements or [])
        if getattr(el, "mime", "") and el.mime.startswith("image/") and el.path
    ]
    if not image_elements:
        return message.content

    blocks: list[dict] = [{"type": "text", "text": message.content or "请识别这些图片。"}]
    for el in image_elements:
        image_bytes = Path(el.path).read_bytes()
        blocks.append({
            "type": "image",
            "base64": base64.b64encode(image_bytes).decode("utf-8"),
            "mime_type": el.mime,
        })
    return blocks

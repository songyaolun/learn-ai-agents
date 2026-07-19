"""DeepAgents multimodal —— 给深度 agent 传图片等多模态内容。

对比 deepagents/quickstart.py: 那里 messages 里的 content 是一个纯字符串
("What's the weather..."), 也就是纯文本消息。本文件演示【多模态消息】: content 不再是
字符串, 而是一个"内容块列表", 里面既可以有文本块, 也可以有图片块 (base64 或 URL)。
这样就能让带视觉能力的模型"看图回答"。

LangChain 多模态消息的 content 结构 (列表, 每项是一个 dict 块):
  文本块:   {"type": "text", "text": "这张图里是什么?"}
  图片块(URL):    {"type": "image", "source_type": "url", "url": "https://.../x.png"}
  图片块(base64): {"type": "image", "source_type": "base64",
                   "mime_type": "image/png", "data": "<base64 字符串>"}

【降级说明 (诚实标注)】真正"看图"需要一个【带视觉能力的在线模型】。本地没有可用模型/
密钥, 无法把图片喂给模型跑出结果 —— 因此本文件属于【降级-骨架】: 只离线断言"多模态
消息 payload 结构良构 + agent 能构建", 真正的 invoke 用 MODEL_ID 守卫、按需运行。
图片来源处用 "# 接入点" 标注, 换成你自己的图片 URL 或 base64。

踩坑记录:
  1. content 的形态从"字符串"变成"块列表"是最容易错的点: 一旦要放图片, 整个
     content 就必须是列表, 文本也要包成 {"type": "text", "text": ...} 块, 不能
     文本还是裸字符串、图片单独塞。
  2. 不是所有模型/供应商都收图: 纯文本模型收到图片块会直接报错或忽略。传图前要确认
     用的是 vision 模型。参数(图片块)能构造 != 模型一定能处理。
  3. extended thinking 下取结果依旧用 result["messages"][-1].text (不要用 .content):
     .content 可能是 thinking/text 混合块列表, .text 才是纯文本答案 —— 多模态并不
     改变这一点。

官方文档: https://docs.langchain.com/oss/python/deepagents/quickstart
"""

import os

from dotenv import load_dotenv
from deepagents import create_deep_agent
from langchain_anthropic import ChatAnthropic

load_dotenv(override=True)


# ---- 构造一条多模态用户消息 (文本 + 一张图片) ----
# 接入点: 把 IMAGE_URL 换成你要让模型看的真实图片地址; 或改用 base64 块。
IMAGE_URL = "https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/PNG_transparency_demonstration_1.png/280px-PNG_transparency_demonstration_1.png"


def build_multimodal_message() -> dict:
    """返回一条 content 为"块列表"的多模态用户消息。"""
    return {
        "role": "user",
        "content": [
            # 文本块: 注意即使是文本, 在多模态列表里也要包成 type=text 的块
            {"type": "text", "text": "这张图片里画的是什么? 用一句话描述。"},
            # 图片块 (URL 形式)。接入点: 换成你自己的图片 URL。
            {"type": "image", "source_type": "url", "url": IMAGE_URL},
        ],
    }


def build_base64_image_block(b64_data: str, mime: str = "image/png") -> dict:
    """另一种图片来源: base64 内联 (适合本地图片读出来再编码)。接入点: b64_data。"""
    return {"type": "image", "source_type": "base64", "mime_type": mime, "data": b64_data}


def build_agent(model):
    """构建一个用于多模态问答的 deep agent (结构上与文本 agent 无差别)。"""
    return create_deep_agent(
        model=model,
        tools=[],
        system_prompt="You are a helpful visual assistant. Describe images concisely.",
    )


if __name__ == "__main__":
    # --- 不依赖模型的结构验证: 断言多模态 payload 良构 ---
    msg = build_multimodal_message()
    assert isinstance(msg["content"], list), "多模态消息 content 必须是块列表, 而非字符串"
    types = [block["type"] for block in msg["content"]]
    assert "text" in types and "image" in types, "应同时包含文本块与图片块"
    image_block = next(b for b in msg["content"] if b["type"] == "image")
    assert image_block["source_type"] == "url" and image_block["url"].startswith("http")

    # base64 块也能正确构造
    b64_block = build_base64_image_block("aGVsbG8=")  # "hello" 的 base64, 仅作结构演示
    assert b64_block["type"] == "image" and b64_block["source_type"] == "base64"

    # agent 能在 model=None 下构建
    agent_struct = build_agent(model=None)
    assert agent_struct is not None
    print("结构验证通过: 多模态消息(文本块+图片块)良构, base64 块良构, agent 构建 OK")
    print("(降级-骨架: 真正看图需在线 vision 模型, 本地无法跑出识图结果。)")

    # --- 需要真实 vision 模型的部分 ---
    if os.getenv("MODEL_ID"):
        model = ChatAnthropic(
            model=os.environ["MODEL_ID"],
            base_url=os.getenv("ANTHROPIC_BASE_URL") or None,
        )
        agent = build_agent(model=model)
        # 只有配了具备视觉能力的模型, 下面才可能真正"看图"
        result = agent.invoke({"messages": [build_multimodal_message()]})
        # 依旧用 .text 取纯文本答案 (extended thinking 下 .content 是混合块)
        print(result["messages"][-1].text)
    else:
        print("未配置 MODEL_ID: 跳过真实模型调用 (仅结构验证)。"
              " 接入点: 配好 vision 模型 + 真实图片 URL/base64 后可跑通识图。")

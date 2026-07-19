"""LangChain 消息对象 —— 四类消息对象、内容块、多模态消息构造、消息分片、token 用量元数据。

对比 langchain/ch_17_quickstart.py: 那里只使用了最基础的消息格式；这里深入讲解 LangChain 中的消息系统，包括不同类型的消息对象、如何构造多模态消息、如何从消息中提取信息等。

官方文档: https://docs.langchain.com/oss/python/langchain/messages

本文件覆盖的能力点：
1. 四类消息对象（SystemMessage/HumanMessage/AIMessage/ToolMessage）
2. 内容块（content blocks）：结构化消息内容
3. 多模态消息构造：包含文本、图片、文件等
4. 消息分片：从 AIMessage 中提取文本和工具调用
5. token 用量元数据：获取消息的 token 消耗

踩坑记录:
- 打印 AIMessage 时用 .text 而不是 .content: 开启 extended thinking 的模型, .content 是
  thinking/text 混合的 block 列表, 直接打印会看到一堆结构而非纯文本; .text 只取纯文本部分。
- token 用量在 .usage_metadata (input_tokens/output_tokens/total_tokens), 但只有"真实调用过
  模型"返回的 AIMessage 才有; 自己手工构造的 AIMessage 这个字段是 None, 不要据此断言。
- 多模态消息 (图片/文件) 的 content 必须写成 block 列表 (每个 block 带 type), 不能塞成
  纯字符串, 否则模型收不到图片。
"""

import os
import tempfile
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    AIMessage,
    ToolMessage,
    ToolCall,
)
from langchain_core.messages.tool import ToolCall

# 统一模型初始化（仓库约定）
load_dotenv(override=True)
# 从环境变量获取模型配置，若不存在则使用默认值
model_id = os.getenv("MODEL_ID", "claude-3-sonnet-20240229")
base_url = os.getenv("ANTHROPIC_BASE_URL") or None
model = ChatAnthropic(
    model=model_id,
    base_url=base_url,
)


# 1. 四类消息对象
# SystemMessage: 系统提示，设定模型行为
# HumanMessage: 用户输入
# AIMessage: 模型输出
# ToolMessage: 工具调用结果

# 构造系统消息
system_message = SystemMessage(
    content="You are a helpful assistant that can call tools."
)

# 构造用户消息
user_message = HumanMessage(
    content="What's the weather in Beijing?"
)

# 构造模型消息（模拟）
ai_message = AIMessage(
    content="I need to call the get_weather tool to answer this question.",
    tool_calls=[
        ToolCall(
            name="get_weather",
            args={"city": "Beijing"},
            id="call_1",
        )
    ]
)

# 构造工具消息
tool_message = ToolMessage(
    content="It's sunny in Beijing.",
    tool_call_id="call_1",
)


# 2. 内容块（content blocks）
# 消息内容可以是结构化的内容块列表
structured_message = HumanMessage(
    content=[
        {"type": "text", "text": "Please analyze this data:"},
        {"type": "text", "text": "Sales: 100, 200, 300"},
        {"type": "text", "text": "What's the average?"}
    ]
)


# 3. 多模态消息构造
# 创建临时文件模拟图片
with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
    f.write(b"Dummy image data")
    image_path = f.name

# 构造包含图片的消息
multimodal_message = HumanMessage(
    content=[
        {"type": "text", "text": "What's in this image?"},
        {"type": "image", "path": image_path},
    ]
)


# 4. 消息分片
# 从 AIMessage 中提取文本和工具调用
# 踩坑记录: 工具调用不在 content 里, 而在 AIMessage 的独立属性 .tool_calls 上;
# 且 ToolCall 是 TypedDict, 不能用 isinstance(x, ToolCall) 判断 (会 TypeError)。
# content 本身可能是字符串 (纯文本) 或 block 列表 (多模态/thinking), 要分别处理。
def extract_message_parts(message: AIMessage):
    """Extract text and tool calls from AIMessage."""
    # 文本: content 是字符串就直接用; 是 block 列表就拼接其中的 text block
    if isinstance(message.content, str):
        text = message.content
    else:
        text = "".join(
            part["text"]
            for part in message.content
            if isinstance(part, dict) and part.get("type") == "text"
        )
    # 工具调用: 统一从 .tool_calls 取 (这是规范化后的工具调用列表)
    tool_calls = list(message.tool_calls or [])
    return {
        "text": text,
        "tool_calls": tool_calls,
    }


# 5. token 用量元数据
# 获取消息的 token 消耗（需要模型支持）
def get_token_usage(message):
    """Get token usage metadata from message."""
    if hasattr(message, "usage_metadata") and message.usage_metadata:
        return message.usage_metadata
    return None


if __name__ == "__main__":
    # 无网络验证部分
    print("=== 无网络验证 ===")
    # 验证消息对象类型
    assert isinstance(system_message, SystemMessage), "Should be SystemMessage"
    assert isinstance(user_message, HumanMessage), "Should be HumanMessage"
    assert isinstance(ai_message, AIMessage), "Should be AIMessage"
    assert isinstance(tool_message, ToolMessage), "Should be ToolMessage"
    # 验证内容块
    assert len(structured_message.content) == 3, "Structured message should have 3 parts"
    # 验证多模态消息
    assert len(multimodal_message.content) == 2, "Multimodal message should have 2 parts"
    # 验证消息分片
    parts = extract_message_parts(ai_message)
    assert parts["text"] == "I need to call the get_weather tool to answer this question.", "Text extraction failed"
    assert len(parts["tool_calls"]) == 1, "Tool call extraction failed"
    print("无网络验证通过！")

    # 需要模型调用的部分（需本地配置 .env）
    print("\n=== 需要模型调用的部分 ===")
    print("请本地配置 .env 文件（MODEL_ID/ANTHROPIC_API_KEY）后运行")
    # 示例：获取 token 用量
    # result = model.invoke([user_message])
    # print(get_token_usage(result))

    # 清理临时文件
    os.unlink(image_path)

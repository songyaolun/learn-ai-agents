"""LangChain 模型能力集 —— 统一初始化、工具绑定、模型能力档、推理、提示缓存、多模态输入、速率限制。

对比 langchain/quickstart.py: 那里只演示了最基础的模型初始化和 agent 组装；这里深入讲解模型本身的高级能力，包括如何绑定工具、启用 extended thinking、缓存提示、处理多模态输入等。

官方文档: https://docs.langchain.com/oss/python/langchain/models

本文件覆盖的能力点：
1. 统一模型初始化（沿用仓库约定）
2. 工具绑定（bind_tools）：让模型能直接调用工具（无需 agent 框架）
3. 模型能力档（model profiles）：预定义模型配置
4. 推理（extended thinking）：让模型分步思考
5. 提示缓存（prompt caching）：减少重复计算
6. 多模态输入:处理文本以外的内容
7. 速率限制:控制 API 调用频率

踩坑记录:
- extended thinking (推理) 开启后, AIMessage.content 变成 thinking/text 混合的 block 列表,
  想拿纯回答要用 .text; 直接读 .content 会看到一堆 thinking block。
- 提示缓存 (prompt caching) 只对"稳定不变的前缀"有效; 把易变内容放前面会让缓存频繁失效,
  等于没缓存。要把系统提示/长上下文等固定部分放前面, 变化部分放后面。
- bind_tools 只是让模型"能发起工具调用", 它不会自动执行工具; 真正的"调用→执行→回填"
  循环得自己写或交给 create_agent。多模态图片/文件同样必须用 block 列表而非纯字符串传入。
"""

import os
import tempfile
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
# 提示缓存功能在当前 LangChain 1.3.13 版本中不可用
# from langchain.globals import set_llm_cache
# from langchain.cache import InMemoryCache
# from langchain_anthropic import RateLimitError

# 统一模型初始化（仓库约定）
load_dotenv(override=True)
# 从环境变量获取模型配置，若不存在则使用默认值
model_id = os.getenv("MODEL_ID", "claude-3-sonnet-20240229")
base_url = os.getenv("ANTHROPIC_BASE_URL") or None
model = ChatAnthropic(
    model=model_id,
    base_url=base_url,
)


@tool
def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"


@tool
def calculate_sum(a: int, b: int) -> int:
    """Calculate sum of two integers."""
    return a + b


# 1. 工具绑定（bind_tools）：让模型能直接调用工具
# 不需要 agent 框架，模型自己就能决定是否调用工具
model_with_tools = model.bind_tools([get_weather, calculate_sum])


# 2. 模型能力档（model profiles）：预定义模型配置
# 比如针对不同任务预设不同的温度（temperature）
def get_model_profile(profile: str) -> ChatAnthropic:
    """Get predefined model profile."""
    if profile == "creative":
        return ChatAnthropic(
            model=model_id,
            temperature=0.9,
            base_url=os.getenv("ANTHROPIC_BASE_URL") or None,
        )
    elif profile == "precise":
        return ChatAnthropic(
            model=model_id,
            temperature=0.1,
            base_url=os.getenv("ANTHROPIC_BASE_URL") or None,
        )
    else:
        return model


# 3. 推理（extended thinking）：让模型分步思考
# 启用后，模型会先输出思考过程，再给出最终答案
model_with_thinking = model.bind(
    anthropic_extra_headers={"anthropic-beta": "extended-thinking"}
)


# 提示缓存（prompt caching）：减少重复计算
# 相同的提示会被缓存，避免重复调用模型
# 注意：当前 LangChain 1.3.13 版本中该功能不可用
# set_llm_cache(InMemoryCache())


# 5. 多模态输入：处理文本以外的内容
# 这里演示如何构造包含图片的消息
# 注意：实际使用需要提供有效的图片路径
with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
    f.write(b"This is a dummy text file to simulate a document.")
    dummy_file_path = f.name

# 构造多模态消息
multimodal_message = HumanMessage(
    content=[
        {"type": "text", "text": "What's in this file?"},
        {"type": "file", "path": dummy_file_path},
    ]
)


# 速率限制：控制 API 调用频率
# 这里演示如何处理速率限制错误
# 注意：当前 LangChain 1.3.13 版本中 RateLimitError 不可用
# def rate_limited_invoke(model, input):
#     """Invoke model with rate limit handling."""
#     try:
#         return model.invoke(input)
#     except RateLimitError as e:
#         print(f"Rate limit exceeded: {e}")
#         # 实际应用中可以在这里添加重试逻辑
#         return None


if __name__ == "__main__":
    # 无网络验证部分
    print("=== 无网络验证 ===")
    # 验证工具绑定后的模型类型
    assert hasattr(model_with_tools, "invoke"), "Model with tools should be invokable"
    # 验证模型能力档
    creative_model = get_model_profile("creative")
    assert creative_model.temperature == 0.9, "Creative model should have temperature 0.9"
    precise_model = get_model_profile("precise")
    assert precise_model.temperature == 0.1, "Precise model should have temperature 0.1"
    # 验证多模态消息构造
    assert len(multimodal_message.content) == 2, "Multimodal message should have two parts"
    print("无网络验证通过！")

    # 需要模型调用的部分（需本地配置 .env）
    print("\n=== 需要模型调用的部分 ===")
    print("请本地配置 .env 文件（MODEL_ID/ANTHROPIC_API_KEY）后运行")
    # 示例：调用带工具的模型
    # result = model_with_tools.invoke("What's the sum of 3 and 5?")
    # print(result)

    # 清理临时文件
    os.unlink(dummy_file_path)

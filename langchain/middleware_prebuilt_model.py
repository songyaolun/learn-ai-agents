"""LangChain 1.x 预置中间件套件 —— 模型相关中间件合集(模型降级、工具选择器、工具重试、模型重试、工具模拟)。

对比既有 middleware 文件:
- 与 middleware_summarization.py 相比: 前者处理对话历史压缩, 本文件聚焦模型调用生命周期(降级、重试、工具选择/模拟)
- 与 middleware_guardrails.py 相比: 前者是调用次数限制, 本文件是模型/工具的容错与路由

官方文档: https://docs.langchain.com/oss/python/langchain/middleware

本文件覆盖的中间件及适用场景:
1. ModelFallbackMiddleware: 主模型调用失败时自动降级到备选模型(适用: 高可用要求场景)
2. LLMToolSelectorMiddleware: 用模型智能筛选相关工具(适用: 多工具场景下的工具路由)
3. ToolRetryMiddleware: 工具调用失败时自动重试(适用: 网络不稳定或工具偶尔异常的场景)
4. ModelRetryMiddleware: 模型调用失败时自动重试(适用: 模型服务偶尔波动的场景)
5. LLMToolEmulator: 用模型模拟工具调用结果(适用: 测试 agent 逻辑而不想真的调用工具时)

踩坑记录:
- 这些中间件的构造签名在 1.3.13 里和很多旧教程/博客不一样, 是经 inspect.signature 实测得到的:
  * ModelFallbackMiddleware(first_model, *additional_models): 只收位置参数的模型列表,
    没有 fallback_model= / should_fallback= 这些关键字参数; 降级顺序就是参数顺序。
  * LLMToolSelectorMiddleware(*, model, system_prompt, max_tools, always_include): 用 max_tools
    限制最多保留几个工具、always_include 强制保留某些工具, 没有 tool_descriptions= 这种参数;
    想让模型选得准, 要把描述写在工具函数的 docstring 里。
  * ToolRetryMiddleware(*, max_retries, tools, retry_on, on_failure, ...): 用 tools= 指定作用的
    工具(可传工具名字符串)、retry_on= 指定重试的异常类型, 没有 tool_name= / retry_on_exceptions=。
  * ModelRetryMiddleware(*, max_retries, retry_on, ...): 同理用 retry_on=, 不是 retry_on_exceptions=。
  * LLMToolEmulator(*, tools, model): 用 tools= 指定要模拟哪些工具, 没有 tool_name= / simulation_prompt=;
    模拟行为由模型根据工具签名自行推断。
- 照旧教程硬写参数名会直接 TypeError: got an unexpected keyword argument。升级后建议先 inspect 再用。
- model 参数一定要传"已初始化的模型对象", 不要传模型 ID 字符串:
  * LLMToolSelectorMiddleware(model=...) / LLMToolEmulator(model=...) 若收到字符串, 内部会调用
    init_chat_model(<str>) 去推断 provider。对 claude-3-* 这类标准 ID 能推断成功, 但对
    MiniMax-M... 这类走 Anthropic 兼容端点的自定义 ID 会抛
    ValueError: Unable to infer model provider for model='MiniMax-M...'。
  * 正确做法: 先 ChatAnthropic(model=..., base_url=...) 构造好对象(仅构造不发请求, 无网络也能成),
    再把该对象传给中间件; 这样能绕开 provider 推断。也可以显式 init_chat_model(id, model_provider="anthropic")。
"""

import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import (
    ModelFallbackMiddleware,
    LLMToolSelectorMiddleware,
    ToolRetryMiddleware,
    ModelRetryMiddleware,
    LLMToolEmulator,
)
from langchain_anthropic import ChatAnthropic
from langgraph.checkpoint.memory import InMemorySaver

# 加载环境变量
load_dotenv(override=True)

# 模型 ID(沿用仓库约定, 从 .env 读取, 缺省给占位值以便无网络自测)
MAIN_MODEL_ID = os.environ.get("MODEL_ID", "claude-3-sonnet-20240229")
FALLBACK_MODEL_ID = os.environ.get("FALLBACK_MODEL_ID", "claude-3-haiku-20240307")


# 示例工具1: 天气查询(模拟偶尔失败)
def get_weather(city: str) -> str:
    """查询指定城市的天气。输入是城市名称字符串(模拟 10% 概率失败)。"""
    import random
    if random.random() < 0.1:
        raise RuntimeError("Weather service temporarily unavailable")
    return f"It's sunny in {city} today!"


# 示例工具2: 股票查询
def get_stock_price(ticker: str) -> str:
    """查询指定股票代码的最新价格。输入是股票代码字符串(如 AAPL)。"""
    return f"{ticker} is trading at $123.45"


def build_middlewares(main_model, fallback_model):
    """构造 5 个模型相关中间件。传入已初始化的模型对象。"""
    # 1. ModelFallbackMiddleware: 主模型失败时按顺序降级到备选模型
    #    构造签名: (first_model, *additional_models) —— 只收位置参数
    fallback_middleware = ModelFallbackMiddleware(fallback_model)

    # 2. LLMToolSelectorMiddleware: 用模型从多个工具里筛选最相关的
    #    max_tools 限制最多保留几个; 工具描述写在 docstring 里(没有 tool_descriptions 参数)
    tool_selector = LLMToolSelectorMiddleware(
        model=main_model,
        max_tools=1,
    )

    # 3. ToolRetryMiddleware: 指定工具失败时重试
    #    tools= 指定作用工具(可用工具名), retry_on= 指定重试的异常
    weather_retry = ToolRetryMiddleware(
        tools=["get_weather"],
        max_retries=2,
        retry_on=(RuntimeError,),
    )

    # 4. ModelRetryMiddleware: 模型调用失败时重试
    model_retry = ModelRetryMiddleware(
        max_retries=2,
        retry_on=(RuntimeError, TimeoutError),
    )

    # 5. LLMToolEmulator: 用模型模拟指定工具的输出(测试用)
    tool_emulator = LLMToolEmulator(
        tools=["get_weather"],
        model=main_model,
    )

    return {
        "fallback": fallback_middleware,
        "tool_selector": tool_selector,
        "weather_retry": weather_retry,
        "model_retry": model_retry,
        "tool_emulator": tool_emulator,
    }


def build_model(model_id):
    """构造一个 ChatAnthropic 对象。仅构造(不发请求), 无网络也能成功。

    关键: 传给中间件的必须是"已初始化的模型对象", 而不是模型 ID 字符串。
    因为 LLMToolSelectorMiddleware/LLMToolEmulator 若收到字符串, 内部会调用
    init_chat_model(<str>) 去推断 provider; 对 claude-3-* 这类标准 ID 能推断成功,
    但对 MiniMax-M... 这类走 Anthropic 兼容端点的自定义 ID 会直接抛
    ValueError: Unable to infer model provider。构造好的 ChatAnthropic 对象则能绕开推断。
    """
    return ChatAnthropic(
        model=model_id,
        base_url=os.getenv("ANTHROPIC_BASE_URL") or None,
    )


def build_agent():
    """惰性构造 agent, 避免无网络/无 .env 时在导入阶段就崩。"""
    main_model = build_model(MAIN_MODEL_ID)
    fallback_model = build_model(FALLBACK_MODEL_ID)
    mw = build_middlewares(main_model, fallback_model)
    return create_agent(
        model=main_model,
        tools=[get_weather, get_stock_price],
        system_prompt="You are a helpful assistant. Use the appropriate tool for the user's query.",
        middleware=[
            mw["model_retry"],    # 模型调用重试
            mw["fallback"],       # 模型降级
            mw["tool_selector"],  # 工具选择
            mw["weather_retry"],  # 天气工具重试
            # mw["tool_emulator"],  # 工具模拟(测试时启用)
        ],
        checkpointer=InMemorySaver(),
    )


if __name__ == "__main__":
    # ===== 无网络自测: 只验证中间件能按正确签名构造出来, 不触发模型调用 =====
    print("=== 无网络自测 ===")

    # 用"已初始化的模型对象"构造中间件, 不发请求, 无网络也能成功。
    # 注意: 这里必须传 ChatAnthropic 对象而非模型 ID 字符串, 否则
    # LLMToolSelectorMiddleware/LLMToolEmulator 内部会对字符串做 provider 推断,
    # 对 MiniMax-M... 这类自定义 ID 会抛 ValueError: Unable to infer model provider。
    main_model = build_model(MAIN_MODEL_ID)
    fallback_model = build_model(FALLBACK_MODEL_ID)
    mw = build_middlewares(main_model, fallback_model)
    assert isinstance(mw["fallback"], ModelFallbackMiddleware), "ModelFallbackMiddleware 构造失败"
    assert isinstance(mw["tool_selector"], LLMToolSelectorMiddleware), "LLMToolSelectorMiddleware 构造失败"
    assert isinstance(mw["weather_retry"], ToolRetryMiddleware), "ToolRetryMiddleware 构造失败"
    assert isinstance(mw["model_retry"], ModelRetryMiddleware), "ModelRetryMiddleware 构造失败"
    assert isinstance(mw["tool_emulator"], LLMToolEmulator), "LLMToolEmulator 构造失败"
    print("✓ 5 个模型相关中间件均按正确签名构造成功")

    # 验证工具函数本身可直接调用
    assert get_stock_price("AAPL").startswith("AAPL"), "get_stock_price 输出异常"
    print("✓ 工具函数可直接调用")

    # ===== 有网络部分(需配置 .env) =====
    print("\n=== 有网络部分(需配置 .env: MODEL_ID / ANTHROPIC_API_KEY) ===")
    if os.getenv("MODEL_ID") and os.getenv("ANTHROPIC_API_KEY"):
        agent = build_agent()
        result = agent.invoke(
            {"messages": [{"role": "user", "content": "What's the weather in Beijing?"}]},
            config={"configurable": {"thread_id": "prebuilt-model-demo"}},
        )
        print(f"天气查询结果: {result['messages'][-1].text}")
    else:
        print("跳过: 未检测到 MODEL_ID / ANTHROPIC_API_KEY, 请配置 .env 后运行。")

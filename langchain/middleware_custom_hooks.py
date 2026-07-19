"""LangChain middleware —— 自定义中间件钩子(模型调用前后/包装/动态提示)。

对比 middleware_prebuilt_model.py 等"预置中间件": 那些是官方封装好、开箱即用的成品件
(降级、重试、工具选择...); 本文件讲的是"自己写中间件"——用钩子装饰器/基类在 agent 生命周期
的关键节点插入自定义逻辑, 当预置件满足不了需求时用它兜底。

能力点:
1. 模型调用前后钩子: before_model / after_model(继承 AgentMiddleware 基类实现)
2. 包装模型调用钩子: wrap_model_call(一个函数同时管到调用前后, 可改写请求/响应)
3. 动态提示钩子: dynamic_prompt(每轮按当前状态/上下文动态生成 system prompt)

官方文档: https://docs.langchain.com/oss/python/langchain/middleware

踩坑记录:
- 这些钩子的真实签名(经 inspect.signature + 源码 Examples 实测)和很多旧教程不一样:
  * @dynamic_prompt 装饰的函数签名是 def fn(request: ModelRequest) -> str, 入参是 request
    (从 request.state / request.runtime.context 取状态和上下文), 返回的是"字符串", 不是 SystemMessage;
    照旧写成 def fn(state) -> SystemMessage 会拿不到数据或类型不符。
  * @wrap_model_call 装饰的函数签名是 def fn(request, handler), 内部调用 handler(request) 触发真正
    的模型调用, 返回其结果; 不是 def fn(model_call, state, config) —— 少写 handler 会报
    "missing 1 required positional argument"。
  * 类式钩子 before_model(self, state, runtime) / after_model(self, state, runtime) 都带 runtime 参数,
    返回 dict(部分状态更新)或 None(不改状态); 只写 (self, state) 会因参数不匹配出错。
- wrap_model_call 与 before_model/after_model 有重叠: 前者一个函数就能同时管到调用前后并改写
  请求/响应; 后两者是分开的两个切点。简单前后处理用 before/after 更直观, 需要改写请求/响应再上 wrap。
"""

import os
import time

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import (
    AgentMiddleware,
    wrap_model_call,
    dynamic_prompt,
)
from langchain_anthropic import ChatAnthropic

load_dotenv(override=True)

MODEL_ID = os.environ.get("MODEL_ID", "claude-3-sonnet-20240229")


# 示例工具: 天气查询
def get_weather(city: str) -> str:
    """查询指定城市的天气。输入是城市名称字符串。"""
    return f"It's always sunny in {city}!"


# 1. 模型调用前后钩子: 继承 AgentMiddleware, 实现 before_model / after_model
class TimingMiddleware(AgentMiddleware):
    """前后钩子示例: 记录并打印模型调用耗时。"""

    def __init__(self):
        super().__init__()
        self.start_time = None

    def before_model(self, state, runtime):
        """模型调用前记录开始时间。返回 None 表示不修改状态。"""
        self.start_time = time.time()
        return None

    def after_model(self, state, runtime):
        """模型调用后计算耗时。"""
        if self.start_time is not None:
            elapsed = time.time() - self.start_time
            print(f"[Timing] 模型调用耗时: {elapsed:.2f} 秒")
        return None


# 2. 动态提示钩子: 入参是 request, 返回字符串
@dynamic_prompt
def dynamic_system_prompt(request) -> str:
    """动态提示示例: 根据当前时间调整问候语。"""
    current_hour = time.localtime().tm_hour
    if current_hour < 12:
        greeting = "早上好"
    elif current_hour < 18:
        greeting = "下午好"
    else:
        greeting = "晚上好"
    return f"{greeting}! 你是一个乐于助人的助手, 请用中文回答。"


# 3. 包装模型调用钩子: 入参是 (request, handler), 内部调用 handler(request)
@wrap_model_call
def logging_wrap(request, handler):
    """包装模型调用示例: 打印模型调用前后, 并在此可改写请求/响应。"""
    print("[WrapModelCall] 模型调用前")
    response = handler(request)
    print("[WrapModelCall] 模型调用后")
    return response


def build_agent():
    """惰性构造带自定义钩子的 agent。"""
    model = ChatAnthropic(
        model=MODEL_ID,
        base_url=os.getenv("ANTHROPIC_BASE_URL") or None,
    )
    return create_agent(
        model=model,
        tools=[get_weather],
        system_prompt="You are a helpful assistant.",
        middleware=[
            TimingMiddleware(),
            dynamic_system_prompt,
            logging_wrap,
        ],
    )


if __name__ == "__main__":
    # ===== 无网络自测: 验证三种钩子都能按正确签名构造成中间件 =====
    print("=== 无网络自测 ===")

    timing = TimingMiddleware()
    assert isinstance(timing, AgentMiddleware), "TimingMiddleware 应继承 AgentMiddleware"
    # 被装饰器包装后, 会变成 AgentMiddleware 实例
    assert isinstance(dynamic_system_prompt, AgentMiddleware), "dynamic_prompt 装饰结果应为中间件"
    assert isinstance(logging_wrap, AgentMiddleware), "wrap_model_call 装饰结果应为中间件"
    print("✓ before/after 钩子、dynamic_prompt、wrap_model_call 三种自定义钩子均构造成功")

    # 验证工具函数
    assert get_weather("Beijing").startswith("It's always sunny"), "get_weather 输出异常"
    print("✓ 工具函数可直接调用")

    # ===== 有网络部分(需配置 .env) =====
    print("\n=== 有网络部分(需配置 .env: MODEL_ID / ANTHROPIC_API_KEY) ===")
    if os.getenv("MODEL_ID") and os.getenv("ANTHROPIC_API_KEY"):
        agent = build_agent()
        result = agent.invoke(
            {"messages": [{"role": "user", "content": "What's the weather in San Francisco?"}]}
        )
        print(f"最终回复: {result['messages'][-1].text}")
    else:
        print("跳过: 未检测到 MODEL_ID / ANTHROPIC_API_KEY, 请配置 .env 后运行。")

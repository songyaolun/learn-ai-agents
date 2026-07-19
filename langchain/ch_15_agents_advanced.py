"""LangChain Agents 高级配置 —— 异步调用、检查点、存储、状态与上下文 schema 详解

对比 langchain/ch_17_quickstart.py: 那里是最小 agent 示例, 这里聚焦 agent 高级配置能力。
官方文档: https://docs.langchain.com/oss/python/langchain/agents/advanced

能力点:
1. 异步调用: ainvoke/astream, 适用于高并发场景
2. 检查点: InMemorySaver 持久化 agent 状态, 支持断点续跑
3. 存储: 管理 agent 运行时状态存储
4. 状态 schema: 定义 agent 状态结构
5. 上下文 schema: 定义 agent 上下文结构

踩坑记录:
- 检查点 (checkpoint) 要生效, invoke 时必须传 config={"configurable": {"thread_id": "..."}}:
  thread_id 相同才算"同一个会话", 状态才能续上; 不传 thread_id 则每次都是全新对话, 检查点形同虚设。
- InMemorySaver 是纯内存的, 进程退出即丢失, 只适合演示/测试; 生产要持久化续跑请换
  SqliteSaver / PostgresSaver 之类的落盘实现。
- 异步 (ainvoke/astream) 必须在事件循环里跑 (asyncio.run(...)), 不能在普通同步函数里直接 await。
"""

import os
import asyncio
import tempfile
import shutil
from typing import Annotated, NotRequired, TypedDict

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver
# 踩坑记录: LangChain 1.x 已移除 langchain_core.pydantic_v1 兼容垫片, 直接从 pydantic 导入即可
# (现在生态统一用 pydantic v2)。照旧教程写 from langchain_core.pydantic_v1 import ... 会 ImportError。
from pydantic import BaseModel, Field

load_dotenv(override=True)

# 模型初始化沿用仓库统一约定
model = ChatAnthropic(
    model=os.environ.get("MODEL_ID", "claude-3-sonnet-20240229"),
    base_url=os.getenv("ANTHROPIC_BASE_URL") or None,
)

# 工具定义: 简单的天气查询工具

def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"

# 1. 状态 schema 定义: 描述 agent 状态结构
class AgentState(TypedDict):
    """Agent state schema."""
    # 自定义 state_schema 会替换默认 AgentState；messages 必须保留 add_messages reducer,
    # 否则同一 thread_id 的多轮 invoke 会覆盖而不是追加消息历史。
    messages: Annotated[list[AnyMessage], add_messages]
    intermediate_steps: NotRequired[list]

# 2. 上下文 schema 定义: 描述 agent 上下文结构
class AgentContext(BaseModel):
    """Agent context schema."""
    user_id: str = Field(description="用户 ID")
    session_id: str = Field(description="会话 ID")
    timestamp: str = Field(description="时间戳")

# 3. 检查点初始化: InMemorySaver 用于持久化 agent 状态
checkpointer = InMemorySaver()

# 4. 创建 agent 时配置高级参数
def create_advanced_agent():
    """创建带高级配置的 agent。"""
    return create_agent(
        model=model,
        tools=[get_weather],
        system_prompt="You are a helpful assistant that can get weather information.",
        checkpointer=checkpointer,
        state_schema=AgentState,
        context_schema=AgentContext,
    )

# 5. 异步调用示例
async def async_agent_demo():
    """异步调用 agent 示例。"""
    agent = create_advanced_agent()
    config = {
        "configurable": {
            "thread_id": "advanced-demo-1",
        }
    }
    context = AgentContext(
        user_id="user-123",
        session_id="session-456",
        timestamp="2026-07-14T12:00:00Z"
    )
    result = await agent.ainvoke(
        {
            "messages": [
                {"role": "user", "content": "What's the weather in New York?"}
            ]
        },
        config=config,
        context=context,
    )
    return result

# 6. 检查点与状态管理示例
def checkpoint_demo():
    """检查点与状态管理示例。"""
    agent = create_advanced_agent()
    config = {"configurable": {"thread_id": "advanced-demo-2"}}

    # 第一次调用
    result1 = agent.invoke(
        {"messages": [{"role": "user", "content": "What's the weather in London?"}]},
        config=config
    )

    # 获取当前状态
    state = agent.get_state(config)
    print(f"当前状态: {state}")

    # 第二次调用 (基于同一 thread_id, 会恢复之前的状态)
    result2 = agent.invoke(
        {"messages": [{"role": "user", "content": "What about Paris?"}]},
        config=config
    )

    return result1, result2

# 7. 沙箱化存储示例
def sandbox_storage_demo():
    """沙箱化存储示例, 使用 tempfile 确保无副作用。"""
    temp_dir = tempfile.mkdtemp()
    try:
        # 模拟存储 agent 状态到沙箱目录
        state_file = os.path.join(temp_dir, "agent_state.json")
        with open(state_file, "w") as f:
            f.write("{\"messages\": [], \"intermediate_steps\": []}")
        print(f"状态文件已保存到: {state_file}")
        return state_file
    finally:
        # 清理沙箱目录
        shutil.rmtree(temp_dir)

if __name__ == "__main__":
    # 无网络自测部分
    print("=== 无网络自测 ===")

    # 测试状态 schema 定义
    state = AgentState(messages=[], intermediate_steps=[])
    assert isinstance(state, BaseModel), "AgentState 应继承自 BaseModel"
    print("✓ AgentState 定义正确")

    # 测试上下文 schema 定义
    context = AgentContext(user_id="test-user", session_id="test-session", timestamp="2026-07-14")
    assert context.user_id == "test-user", "AgentContext 字段应正确初始化"
    print("✓ AgentContext 定义正确")

    # 测试检查点初始化
    assert isinstance(checkpointer, InMemorySaver), "checkpointer 应为 InMemorySaver 实例"
    print("✓ InMemorySaver 初始化正确")

    # 测试沙箱化存储
    state_file = sandbox_storage_demo()
    assert not os.path.exists(state_file), "沙箱目录应已被清理"
    print("✓ 沙箱化存储测试通过")

    # 有网络部分 (需配置 .env)
    print("\n=== 有网络部分 (需配置 .env) ===")
    print("1. 异步调用示例: asyncio.run(async_agent_demo())")
    print("2. 检查点示例: checkpoint_demo()")
    print("\n请配置 .env 文件 (MODEL_ID/ANTHROPIC_API_KEY) 后运行。")

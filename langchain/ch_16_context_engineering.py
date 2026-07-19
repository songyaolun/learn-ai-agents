#!/usr/bin/env python3
"""LangChain 上下文工程 —— 系统性组织「喂给模型的信息」的方法，通过三类上下文（模型/工具/生命周期）和三种数据来源（运行时/状态/存储）构建完整的上下文体系。

对比 langchain/ch_17_quickstart.py: 那里用 create_agent 封装了上下文管理；这里显式展示上下文的组装与传递机制，核心区别是抽象层级（黑盒封装 vs 显式控制）和可扩展性（固定流程 vs 自定义注入）。

官方文档: https://docs.langchain.com/oss/python/langchain/context-engineering

本文件覆盖的能力点：
1. 三类上下文：模型上下文（system prompt/模型配置）、工具上下文（工具能看到/注入的信息）、生命周期上下文（agent 运行各阶段的数据）
2. 三种数据来源：运行时上下文（runtime context，呼应 ch_05_runtime.py）、状态（state）、存储（store，呼应 ch_04_long_term_memory.py）
3. 上下文组装与传递：从三种来源取数据并注入到模型/工具中
4. 踩坑记录：参数存在但模型不一定按预期使用

本文件串联了 ch_01_models.py（模型初始化）、ch_02_messages.py（消息构造）、ch_04_long_term_memory.py（存储）、ch_05_runtime.py（运行时）四个已有示例，展示了如何从这四个模块中获取数据并构建上下文。
"""

import os
import tempfile
import shutil
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.runtime import get_runtime
from langgraph.store.memory import InMemoryStore

# 统一模型初始化（仓库约定）
load_dotenv(override=True)
model_id = os.getenv("MODEL_ID", "claude-3-sonnet-20240229")
base_url = os.getenv("ANTHROPIC_BASE_URL") or None
model = ChatAnthropic(
    model=model_id,
    base_url=base_url,
)


# 1. 定义上下文 schema（运行时的核心数据结构）
# 上下文 schema 规定了 agent 执行过程中可访问的所有数据字段
class AgentContext(BaseModel):
    user_id: str = Field(description="用户唯一标识")
    session_id: str = Field(description="会话唯一标识")
    memory_store: InMemoryStore = Field(description="长期记忆存储实例")
    temp_dir: str = Field(description="沙箱临时目录路径")
    runtime_config: dict = Field(description="运行时配置")
    model_context: dict = Field(description="模型上下文")
    tool_context: dict = Field(description="工具上下文")
    lifecycle_context: dict = Field(description="生命周期上下文")

    model_config = {
        "arbitrary_types_allowed": True
    }


# 2. 上下文组装函数
# 从三种数据来源（运行时/状态/存储）组装完整的上下文
# 呼应 ch_05_runtime.py 的运行时管理和 ch_04_long_term_memory.py 的存储机制
def assemble_context(user_id: str, session_id: str) -> AgentContext:
    """从三种数据来源组装完整的上下文"""
    # 运行时上下文（runtime context）：来自 ch_05_runtime.py 的运行时配置
    runtime_config = {
        "max_steps": 5,
        "timeout": 30,
        "sandbox_enabled": True
    }

    # 状态（state）：会话级临时状态
    temp_dir = tempfile.mkdtemp()

    # 存储（store）：来自 ch_04_long_term_memory.py 的长期记忆
    memory_store = InMemoryStore()
    namespace = f"user_memory:{user_id}"
    # 写入测试数据（模拟用户长期记忆）
    memory_store.put(namespace, "coffee_preference", "美式咖啡,不加糖")
    memory_store.put(namespace, "work_city", "北京")

    # 模型上下文：来自 ch_01_models.py 的模型配置和 system prompt
    model_context = {
        "system_prompt": "You are a helpful assistant that can access user preferences and runtime information.",
        "temperature": 0.7,
        "max_tokens": 1024
    }

    # 工具上下文：工具能看到/注入的信息
    tool_context = {
        "available_tools": ["get_weather", "calculate_sum"],
        "tool_namespace": "user_tools"
    }

    # 生命周期上下文：agent 运行各阶段的数据
    lifecycle_context = {
        "current_step": 0,
        "execution_history": [],
        "error_count": 0
    }

    # 组装完整上下文
    return AgentContext(
        user_id=user_id,
        session_id=session_id,
        memory_store=memory_store,
        temp_dir=temp_dir,
        runtime_config=runtime_config,
        model_context=model_context,
        tool_context=tool_context,
        lifecycle_context=lifecycle_context
    )


# 3. 上下文传递演示
# 演示如何将上下文注入到模型和工具中
def inject_context_to_model(context: AgentContext) -> ChatAnthropic:
    """将模型上下文注入到模型中"""
    # 从上下文获取模型配置
    model_config = context.model_context
    # 呼应 ch_01_models.py 的模型初始化方式
    return ChatAnthropic(
        model=model_id,
        base_url=base_url,
        temperature=model_config.get("temperature", 0.7),
        max_tokens=model_config.get("max_tokens", 1024)
    )


# 4. 工具如何访问上下文
# 呼应 ch_05_runtime.py 中的工具依赖注入
def get_user_preference(context: AgentContext, preference_key: str) -> str:
    """从上下文中获取用户长期记忆中的偏好设置"""
    # 从上下文中获取存储实例和用户ID
    store = context.memory_store
    namespace = f"user_memory:{context.user_id}"

    # 读取长期记忆（呼应 ch_04_long_term_memory.py 的存储操作）
    result = store.get(namespace, preference_key)
    return result.value if result else "未设置"


# 5. 上下文使用示例
# 展示如何在 agent 执行过程中使用上下文
def agent_execution_example(context: AgentContext):
    """agent 执行过程中使用上下文的示例"""
    # 注入模型上下文
    model_with_context = inject_context_to_model(context)

    # 构造系统消息（呼应 ch_02_messages.py 的消息构造）
    system_message = SystemMessage(content=context.model_context["system_prompt"])

    # 构造用户消息
    user_message = HumanMessage(content="我的咖啡偏好是什么？")

    # 从上下文中获取用户偏好
    coffee_pref = get_user_preference(context, "coffee_preference")

    # 构造包含上下文信息的提示
    prompt = [
        system_message,
        user_message,
        HumanMessage(content=f"用户长期记忆：咖啡偏好={coffee_pref}")
    ]

    return prompt


if __name__ == "__main__":
    # 无网络验证部分
    print("=== 无网络验证 ===")
    # 验证上下文组装
    context = assemble_context("user_123", "session_456")
    assert context.user_id == "user_123", "用户ID验证失败"
    assert context.session_id == "session_456", "会话ID验证失败"
    assert isinstance(context.memory_store, InMemoryStore), "存储实例验证失败"
    print("✅ 上下文组装验证通过")

    # 验证模型上下文注入
    model_with_context = inject_context_to_model(context)
    assert model_with_context.temperature == 0.7, "模型温度验证失败"
    assert model_with_context.max_tokens == 1024, "模型最大 tokens 验证失败"
    print("✅ 模型上下文注入验证通过")

    # 验证工具访问上下文
    coffee_pref = get_user_preference(context, "coffee_preference")
    work_city = get_user_preference(context, "work_city")
    unknown_pref = get_user_preference(context, "unknown_key")
    assert coffee_pref == "美式咖啡,不加糖", "咖啡偏好读取错误"
    assert work_city == "北京", "工作城市读取错误"
    assert unknown_pref == "未设置", "未知偏好处理错误"
    print("✅ 工具访问上下文验证通过")

    # 验证 agent 执行示例
    prompt = agent_execution_example(context)
    assert len(prompt) == 3, "提示长度验证失败"
    assert "用户长期记忆：咖啡偏好=美式咖啡,不加糖" in prompt[2].content, "提示内容验证失败"
    print("✅ agent 执行示例验证通过")

    # 清理沙箱
    shutil.rmtree(context.temp_dir)
    print("🧹 沙箱清理完成")

    # 需要模型调用的部分（需本地配置 .env）
    print("\n=== 需要模型调用的部分 ===")
    print("请本地配置 .env 文件（MODEL_ID/ANTHROPIC_API_KEY）后运行")
    # 示例：调用带上下文的模型
    # result = model_with_context.invoke(prompt)
    # print(result)

    # 踩坑记录
    print("\n=== 踩坑记录 ===")
    print("1. 参数存在但模型不一定按预期使用：即使上下文包含用户偏好，模型可能仍会忽略，需要在 system prompt 中明确提示使用这些信息")
    print("2. 上下文数据过多可能导致 token 超限：需要合理控制上下文大小，避免包含不必要的信息")
    print("3. 不同模型对上下文的处理方式不同：部分模型可能不支持某些上下文字段，需要测试验证")

    print("\n🎉 上下文工程演示完成! 可配置 .env 后扩展为真实 agent 场景。")

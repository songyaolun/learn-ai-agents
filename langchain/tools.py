"""LangChain 工具系统 —— 工具装饰器、自动 schema 生成、状态/运行时注入、工具选型。

对比 langchain/quickstart.py: 那里只把一个普通函数丢进 create_agent 的 tools 就当工具用了,
没有细讲"这个函数是怎么变成模型能理解的工具描述的"、"工具怎么读到对话上下文而不是只靠
入参"。本文件深入讲 LangChain 工具系统的进阶能力: 自动 JSON Schema 生成、状态注入
(InjectedState)、运行时注入 (ToolRuntime / InjectedToolCallId), 以及几类工具的选型。

官方文档: https://docs.langchain.com/oss/python/langchain/tools

本文件覆盖的能力点:
1. 工具装饰器 (@tool): 把普通函数标注成工具
2. 工具 schema: 从类型标注自动生成 JSON Schema (无需手写)
3. 状态注入 (InjectedState): 工具读取 agent 的对话状态, 且该参数对模型隐藏
4. 运行时注入 (ToolRuntime / InjectedToolCallId): 工具拿到本次调用的上下文 (如 tool_call_id)
5. 动态 / 无头 / 预置 / 服务端工具的区别与选型

踩坑记录 (版本相关, 重要):
- InjectedState、InjectedToolCallId、ToolRuntime 这些符号在 **langchain.tools** 里 (顶层包),
  不在 langchain_core.tools 里。早期误以为 1.3.13 "不支持"这些能力, 其实只是 import 路径找错了。
  统一从 `from langchain.tools import tool, ToolRuntime, InjectedState, InjectedToolCallId` 导入。
- 被注入的参数 (runtime / state / call_id) 不会出现在发给模型的 schema 里 —— 框架会自动把它们
  从"模型需要填的参数"中剔除, 模型只看得到真正的业务参数 (如 city / name)。这点很关键:
  注入参数是"框架在执行时帮你填的", 不是"让模型填的"。
"""

import os
from typing import Annotated

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain.tools import tool, ToolRuntime, InjectedState, InjectedToolCallId
from langchain_core.utils.json_schema import dereference_refs

# 统一模型初始化 (仓库约定: 走 .env, 不硬编码 key)。这里放在顶层是为了和其他文件风格一致;
# 本文件的无网络验证部分并不真正调用模型, 只演示"工具本身"的能力, 因此 MODEL_ID 用
# getenv + 占位默认值兜底, 保证没有 .env 时也能跑通下面的 schema / 注入 / 动态工具验证。
load_dotenv(override=True)
model = ChatAnthropic(
    model=os.getenv("MODEL_ID", "claude-sonnet-4-20250514"),
    base_url=os.getenv("ANTHROPIC_BASE_URL") or None,
)


# 1. 工具装饰器 (@tool): 把普通函数变成工具
@tool
def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"


# 2. 工具 schema: 从类型标注 + docstring 自动生成 JSON Schema
# @tool 装饰后的对象自带 .args_schema (一个 Pydantic 模型) 和 .tool_call_schema。
# 调用 .model_json_schema() 就能拿到发给模型的那份 JSON Schema —— 不需要手写。
# dereference_refs 用来把 schema 里可能出现的 $ref 引用展开成内联结构 (嵌套模型时会用到),
# 简单工具通常没有 $ref, 展开后和原来一样, 但保留这一步是通用写法。
tool_schema = get_weather.tool_call_schema.model_json_schema()
dereferenced_schema = dereference_refs(tool_schema)


# 3. 状态注入 (InjectedState): 工具读取 agent 的对话状态
# state 参数用 Annotated[..., InjectedState] 标注后, 框架会在执行时自动把当前 agent 状态
# (含 messages 等) 填进来, 且这个参数对模型隐藏 —— 模型只需要提供 name。
@tool
def count_messages(name: str, state: Annotated[dict, InjectedState]) -> str:
    """Greet the user and report how many messages are in the conversation so far."""
    n = len(state.get("messages", []))
    return f"Hello {name}! There are {n} messages in this conversation."


# 4. 运行时注入 (ToolRuntime / InjectedToolCallId): 工具拿到本次调用的上下文
# 方式一: 声明一个 ToolRuntime 类型的参数, 框架自动注入, 可从中读 tool_call_id 等。
@tool
def echo_with_runtime(text: str, runtime: ToolRuntime) -> str:
    """Echo text back, tagged with the current tool call id."""
    return f"[call={runtime.tool_call_id}] {text}"


# 方式二: 只想要 tool_call_id 时, 用 Annotated[str, InjectedToolCallId] 更轻量。
@tool
def echo_with_call_id(text: str, call_id: Annotated[str, InjectedToolCallId]) -> str:
    """Echo text back with just the injected tool call id."""
    return f"[call={call_id}] {text}"


# 5. 动态 / 无头 / 预置 / 服务端工具的区别与选型
# - 动态工具: 运行时根据配置/元数据现场生成的工具 (见下方 create_dynamic_tool)
# - 无头工具: 只有 schema、没有真实实现的占位工具, 常用于测试或"由外部系统执行"的场景
# - 预置工具: LangChain / 生态内置的现成工具 (检索、文件、shell 等)
# - 服务端工具: 通过 MCP / API 接入的远程工具 (见 langchain/mcp.py)

# 动态工具示例: 用一个工厂函数, 按名字/描述批量生成工具
def create_dynamic_tool(tool_name: str, description: str):
    """Create a tool at runtime from a name + description."""
    def dynamic_tool() -> str:
        return f"This is a dynamic tool: {tool_name}"
    dynamic_tool.__name__ = tool_name
    dynamic_tool.__doc__ = description
    return tool(dynamic_tool)


# 无头工具示例: 有 schema 但故意不给实现 (真正调用会报错), 适合先定契约再补实现
@tool
def dummy_tool() -> str:
    """A headless/placeholder tool (schema only, no real implementation)."""
    raise NotImplementedError("This is a dummy tool")


if __name__ == "__main__":
    print("=== 无网络验证 ===")

    # 验证自动生成的 schema 结构正确
    assert "properties" in dereferenced_schema, "schema 应包含 properties"
    assert "city" in dereferenced_schema["properties"], "schema 应包含 city 字段"
    assert dereferenced_schema["required"] == ["city"], "city 应为必填"
    print("工具 schema 自动生成通过:", dereferenced_schema["properties"])

    # 验证注入参数对模型隐藏: state / runtime / call_id 不应出现在发给模型的 schema 里
    assert "state" not in count_messages.args, "InjectedState 参数应对模型隐藏"
    assert "runtime" not in echo_with_runtime.args, "ToolRuntime 参数应对模型隐藏"
    assert "call_id" not in echo_with_call_id.args, "InjectedToolCallId 参数应对模型隐藏"
    print("注入参数隐藏通过: count_messages 对模型可见的参数 =", list(count_messages.args))

    # 验证动态工具
    dyn = create_dynamic_tool("test_tool", "Test dynamic tool")
    assert dyn.name == "test_tool", "动态工具名应为 test_tool"
    print("动态工具生成通过:", dyn.name)

    # 验证工具可被调用 (@tool 对象自带 .invoke)
    assert hasattr(get_weather, "invoke"), "工具应可被 invoke"
    assert get_weather.invoke({"city": "北京"}) == "It's always sunny in 北京!"
    print("工具 invoke 通过:", get_weather.invoke({"city": "北京"}))

    print("无网络验证全部通过!")

    print("\n=== 需要模型调用的部分 ===")
    print("请本地配置 .env (MODEL_ID / ANTHROPIC_API_KEY, 可选 ANTHROPIC_BASE_URL) 后,")
    print("把 count_messages / echo_with_runtime 等挂到 create_agent 的 tools 上即可看到注入生效。")

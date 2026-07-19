"""LangChain MCP 接入演示 —— 让智能体通过 MCP 协议调用外部工具。

对比 langchain/ch_17_quickstart.py: 那里的工具是直接写在代码里的 Python 函数,
这里的工具是运行在 MCP 服务端的外部服务 (可以是其他语言写的工具、第三方 API 封装等),
agent 通过 MCP 协议与服务端通信来调用这些工具。

MCP = Model Context Protocol, 模型上下文协议 —— 一个让智能体接入外部工具/数据的开放协议,
它定义了客户端与服务端之间的通信规范, 支持多服务端连接和两种传输方式:
1. 标准输入输出 (stdio): 适合本地开发调试
2. 可流式 HTTP: 适合生产环境跨网络调用

官方文档: https://docs.langchain.com/oss/python/langchain/mcp

前置依赖与降级方案:
- 需安装 langchain-mcp-adapters: pip install langchain-mcp-adapters
- 需运行一个 MCP server (如: python -m langchain_mcp_adapters.server)
- 若未安装 langchain-mcp-adapters, 代码骨架仍可通过语法检查, 但实际运行会报错

踩坑记录:
- MCP 工具的加载多为异步 (load_mcp_tools / client 是 async), 要在事件循环里 await,
  不能在同步函数里直接拿到工具列表。
- stdio 传输适合本地调试 (拉起子进程作 server), 可流式 HTTP 适合生产跨网络; 两种传输的
  连接配置不同, 别把本地 stdio 那套直接搬到生产。
- server 没起来 / 端口不通时, 报错通常发生在"连接/列工具"阶段而非 agent 推理阶段;
  排查时先单独验证 client 能否连上并列出工具, 再谈让 agent 调用。
"""

import os
import asyncio
from typing import List, Dict

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_anthropic import ChatAnthropic

# 加载环境变量
load_dotenv(override=True)

# 模型初始化沿用既有约定
# 踩坑记录: 顶层直接 os.environ["MODEL_ID"] 会让无 .env 环境在"导入阶段"就 KeyError 崩掉,
# 这里改成缺省占位值, 真正调用模型的部分放到 __main__ 里并用 os.getenv 门控。
model = ChatAnthropic(
    model=os.environ.get("MODEL_ID", "claude-3-sonnet-20240229"),
    base_url=os.getenv("ANTHROPIC_BASE_URL") or None,
)

# MCP 客户端配置 (骨架)
# 实际使用前需: 1. 安装 langchain-mcp-adapters 2. 启动 MCP server
mcp_config = {
    "weather": {
        # stdio 传输方式 (本地调试)
        "transport": "stdio",
        "command": "python",
        "args": ["-m", "langchain_mcp_adapters.server"],
        # HTTP 传输方式 (生产环境)
        # "transport": "streamable_http",
        # "url": "http://localhost:8000/mcp",
    }
}

# MCP 工具调用代理 (骨架)
async def _call_mcp_tool(tool_name: str, tool_args: Dict) -> str:
    """使用 langchain-mcp-adapters 当前客户端 API 异步加载并调用 MCP 工具。"""
    from langchain_mcp_adapters.client import MultiServerMCPClient

    client = MultiServerMCPClient(mcp_config)
    tools = await client.get_tools()
    for tool in tools:
        if tool.name == tool_name:
            return await tool.ainvoke(tool_args)
    available = ", ".join(tool.name for tool in tools) or "无"
    return f"[MCP 调用失败] 未找到工具 {tool_name}; 当前可用工具: {available}"


def mcp_tool_proxy(tool_name: str, tool_args: Dict) -> str:
    """通过 MCP 协议调用外部工具的代理函数"""
    try:
        return asyncio.run(_call_mcp_tool(tool_name, tool_args))
    except ImportError:
        return f"[MCP 未配置] 需安装 langchain-mcp-adapters 并启动 MCP server 才能调用 {tool_name}"
    except Exception as e:
        return f"[MCP 调用失败] {str(e)}"

# 包装 MCP 工具为 agent 可用的函数
def get_weather_mcp(city: str) -> str:
    """通过 MCP 调用外部天气查询工具"""
    return mcp_tool_proxy("get_weather", {"city": city})

# 创建 agent
agent = create_agent(
    model=model,
    tools=[get_weather_mcp],
    system_prompt=("You are a helpful assistant that uses MCP tools to answer questions."
                   "If MCP is not configured, inform the user of the setup requirements."),
)

if __name__ == "__main__":
    # ===== 无网络自测: 验证工具包装与 agent 组装, 不触发模型调用 =====
    print("=== 无网络自测 ===")
    # MCP 代理在未安装 langchain-mcp-adapters 时应返回友好提示而非崩溃
    proxy_result = mcp_tool_proxy("get_weather", {"city": "San Francisco"})
    assert "MCP" in proxy_result, "MCP 代理应返回降级提示"
    print(f"✓ MCP 代理降级提示: {proxy_result}")
    assert callable(get_weather_mcp), "get_weather_mcp 应可调用"
    assert agent is not None, "agent 应已组装"
    print("✓ MCP 工具包装与 agent 组装成功")

    # ===== 有网络部分(需配置 .env + 启动 MCP server) =====
    print("\n=== 有网络部分(需配置 .env 且启动 MCP server) ===")
    print("注意: 需安装 langchain-mcp-adapters 并启动 MCP server 才能看到真实结果")
    if os.getenv("MODEL_ID") and os.getenv("ANTHROPIC_API_KEY"):
        result = agent.invoke({
            "messages": [
                {"role": "user", "content": "What's the weather in San Francisco?"}
            ]
        })
        print("\n最终回答:")
        print(result["messages"][-1].text)
    else:
        print("跳过: 未检测到 MODEL_ID / ANTHROPIC_API_KEY, 请配置 .env 后运行。")

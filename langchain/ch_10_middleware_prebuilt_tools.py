"""LangChain 1.x 预置中间件套件 —— 工具/文件相关中间件合集(上下文编辑、服务端工具检索、Shell 工具、文件检索)。

对比既有 middleware 文件:
- 与 ch_23_middleware_hitl.py 相比: 前者是人工审批(human-in-the-loop), 本文件是工具/文件的操作与检索
- 与 ch_24_middleware_summarization.py 相比: 前者压缩对话历史, 本文件处理外部工具与文件系统

官方文档: https://docs.langchain.com/oss/python/langchain/middleware

本文件覆盖的中间件及适用场景:
1. ContextEditingMiddleware: 按规则自动裁剪上下文(如清理过多的历史工具调用), 控制上下文长度
2. ProviderToolSearchMiddleware: 让模型服务端(provider)从一批可搜索工具里检索(适用: 工具数量多的场景)
3. ShellToolMiddleware: 给 agent 一个受控的 shell 执行工具(适用: 需要与系统交互的场景)
4. FilesystemFileSearchMiddleware: 给 agent 文件检索能力(适用: 需要查找本地文件的场景)

踩坑记录:
- 这几个中间件的真实签名(经 inspect.signature 实测)和常见旧教程差别很大:
  * ContextEditingMiddleware(*, edits=..., token_count_method=...): 它不是传一个 edit_fn 回调,
    而是传一组"编辑规则对象"(如 ClearToolUsesEdit); 规则决定何时/如何裁剪上下文, 不是任意改 state。
  * ClearToolUsesEdit(trigger=..., keep=..., ...): 当 token 数超过 trigger 时, 只保留最近 keep 组
    工具调用, 其余替换成占位符, 用来防止上下文无限膨胀。
  * ProviderToolSearchMiddleware(*, searchable_tools=...): 传的是"可被服务端搜索的工具标识列表",
    没有 search_fn 这种自定义检索回调; 检索动作发生在模型服务端。
  * ShellToolMiddleware(workspace_root, *, execution_policy=..., ...): 用 workspace_root 限定工作目录、
    execution_policy 控制命令策略, 没有 allowed_commands= / working_directory= 这些参数。
  * FilesystemFileSearchMiddleware(*, root_path, use_ripgrep=..., max_file_size_mb=...): 用 root_path
    限定检索根目录, 没有 root_dir= / allowed_extensions= 参数。
- 照旧教程硬写会 TypeError: got an unexpected keyword argument。升级后务必先 inspect 再用。
"""

import os
import tempfile
import shutil

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import (
    ContextEditingMiddleware,
    ClearToolUsesEdit,
    ShellToolMiddleware,
    FilesystemFileSearchMiddleware,
)
from langchain_anthropic import ChatAnthropic
from langgraph.checkpoint.memory import InMemorySaver

# 加载环境变量
load_dotenv(override=True)

MODEL_ID = os.environ.get("MODEL_ID", "claude-3-sonnet-20240229")


def build_middlewares(workspace_root: str, search_root: str):
    """构造工具/文件相关中间件。传入沙箱目录, 保证无副作用。"""
    # 1. ContextEditingMiddleware: 用 ClearToolUsesEdit 规则自动裁剪历史工具调用
    #    trigger: 超过多少 token 触发裁剪; keep: 保留最近几组工具调用
    context_editor = ContextEditingMiddleware(
        edits=[ClearToolUsesEdit(trigger=4000, keep=2)],
    )

    # 2. ShellToolMiddleware: 受控 shell, 工作目录限定在沙箱内
    shell_tool = ShellToolMiddleware(
        workspace_root=workspace_root,
    )

    # 3. FilesystemFileSearchMiddleware: 文件检索, 根目录限定在沙箱内
    #    use_ripgrep=False 避免环境无 rg 时报错
    file_search = FilesystemFileSearchMiddleware(
        root_path=search_root,
        use_ripgrep=False,
    )

    return {
        "context_editor": context_editor,
        "shell_tool": shell_tool,
        "file_search": file_search,
    }


def build_agent(workspace_root: str, search_root: str):
    """惰性构造 agent, 避免无网络/无 .env 时导入即崩。"""
    model = ChatAnthropic(
        model=MODEL_ID,
        base_url=os.getenv("ANTHROPIC_BASE_URL") or None,
    )
    mw = build_middlewares(workspace_root, search_root)
    return create_agent(
        model=model,
        tools=[],
        system_prompt="You are a helpful assistant. Use tools to answer user queries.",
        middleware=[
            mw["context_editor"],
            mw["shell_tool"],
            mw["file_search"],
        ],
        checkpointer=InMemorySaver(),
    )


if __name__ == "__main__":
    # ===== 无网络自测: 沙箱目录内验证中间件按正确签名构造 =====
    print("=== 无网络自测 ===")
    ws = tempfile.mkdtemp()
    search = tempfile.mkdtemp()
    try:
        # 准备测试文件
        with open(os.path.join(search, "test.txt"), "w") as f:
            f.write("Hello, LangChain!")
        with open(os.path.join(search, "notes.md"), "w") as f:
            f.write("# Test Notes\nThis is a test file.")

        mw = build_middlewares(ws, search)
        assert isinstance(mw["context_editor"], ContextEditingMiddleware), "ContextEditingMiddleware 构造失败"
        assert isinstance(mw["shell_tool"], ShellToolMiddleware), "ShellToolMiddleware 构造失败"
        assert isinstance(mw["file_search"], FilesystemFileSearchMiddleware), "FilesystemFileSearchMiddleware 构造失败"
        print("✓ 3 个工具/文件中间件均按正确签名构造成功")
        print(f"✓ 沙箱工作目录: {ws}")
        print(f"✓ 沙箱检索目录: {search}(含 2 个测试文件)")
    finally:
        shutil.rmtree(ws, ignore_errors=True)
        shutil.rmtree(search, ignore_errors=True)
    print("✓ 沙箱目录已清理")

    # ===== 有网络部分(需配置 .env) =====
    print("\n=== 有网络部分(需配置 .env: MODEL_ID / ANTHROPIC_API_KEY) ===")
    if os.getenv("MODEL_ID") and os.getenv("ANTHROPIC_API_KEY"):
        ws2 = tempfile.mkdtemp()
        search2 = tempfile.mkdtemp()
        try:
            agent = build_agent(ws2, search2)
            result = agent.invoke(
                {"messages": [{"role": "user", "content": "List files in the workspace."}]},
                config={"configurable": {"thread_id": "prebuilt-tools-demo"}},
            )
            print(f"结果: {result['messages'][-1].text}")
        finally:
            shutil.rmtree(ws2, ignore_errors=True)
            shutil.rmtree(search2, ignore_errors=True)
    else:
        print("跳过: 未检测到 MODEL_ID / ANTHROPIC_API_KEY, 请配置 .env 后运行。")

"""DeepAgents sandbox backend —— 把 agent 的文件 / 命令执行隔离到远程沙箱里。

对比 deepagents/ch_04_backend.py: 那里用的 FilesystemBackend(root_dir=...) 是"同一台
宿主机的真实磁盘", agent 的 write_file / read_file 直接落到本机文件系统上 —— 只是
用 virtual_mode + root_dir 圈了一圈护栏, 但 agent 真要执行代码的话仍然跑在你的机器
进程里, 风险很高 (读到 .env、耗光宿主机资源、甚至逃逸)。本文件讲的是另一档隔离级别:
把 agent 的读写和执行统统丢到一个"独立的远程沙箱 / 容器"里 (deepagents 里对应
deepagents.backends.sandbox 定义的 BaseSandbox / SandboxBackendProtocol, 官方现成
实现是 LangSmithSandbox —— 由 LangSmith 托管的云端沙箱)。这样 agent 就算跑了不受
信任的代码, 炸的也只是那个一次性沙箱, 碰不到你的宿主机磁盘和网络。

什么时候该用沙箱 backend:
- agent 需要真正"执行"外部生成的、不受信任的代码 (code interpreter 类场景);
- 你不想让 agent 有能力读到宿主机上的敏感文件 / 环境变量;
- 需要资源隔离 (CPU/内存/超时) 和"用完即毁"的干净环境。
不用沙箱就够的场景: agent 只需要读写自己产出的中间笔记 —— 用默认 StateBackend
(deepagents/ch_03_filesystem.py) 或 FilesystemBackend(deepagents/ch_04_backend.py) 更省事。

【本文件的诚实说明 / 降级声明】
远程沙箱 (如 LangSmith 托管沙箱) 需要一个外部沙箱服务 + 凭证, 本地环境里并不具备,
也不允许连接。所以本文件是"讲解 + 可运行骨架 + 明确标注接入点", 里面涉及真实沙箱
服务的部分【没有、也无法在本地跑通】。__main__ 只做结构性断言 (类 / 协议存在、
骨架能 import、create_deep_agent 能接受一个 backend 参数), 不会真的去连沙箱。

踩坑记录: LangSmithSandbox(sandbox=...) 的构造参数是一个"已经建好的 Sandbox 客户端
对象" (来自 LangSmith 的 SDK), 不是 URL / token 字符串 —— 也就是说创建和鉴权是在
外部 SDK 里完成的, deepagents 只是把它包成一个符合 BackendProtocol 的 backend。
另外注意: 就算 backend 具备 execute 能力, 模型也不一定会主动去执行代码 (参数存在不等于
模型按预期使用), 通常要在 system_prompt 里明确"可以用执行工具跑代码验证"才会用。

官方文档: https://docs.langchain.com/oss/python/deepagents/backends
"""

import os  # 读取 MODEL_ID / ANTHROPIC_BASE_URL 环境变量
import inspect  # 用于结构性断言时查看类的方法签名

from dotenv import load_dotenv  # 从 .env 加载模型配置 (本地可能没有, 属正常)
from deepagents import create_deep_agent  # 主入口: 支持传 backend= 切换执行环境
# 从 deepagents.backends 顶层拿到沙箱相关的名字: LangSmithSandbox 是官方现成实现,
# BaseSandbox / SandboxBackendProtocol 是沙箱 backend 需要遵循的抽象基类 / 协议。
from deepagents.backends import LangSmithSandbox
from deepagents.backends.sandbox import BaseSandbox, SandboxBackendProtocol
from langchain_anthropic import ChatAnthropic  # Anthropic 模型封装

load_dotenv(override=True)  # override=True: .env 里的值覆盖已有同名环境变量


def build_model():
    """把模型构造单独包一层: 本地没有 MODEL_ID 时不至于一 import 就崩。"""
    # 只有真的要跑模型时才需要这个; 结构性验证阶段可以完全不碰它。
    return ChatAnthropic(
        model=os.environ["MODEL_ID"],  # 模型名从环境变量取, 绝不硬编码
        base_url=os.getenv("ANTHROPIC_BASE_URL") or None,  # 没配就传 None 走默认
    )


def build_sandbox_agent(sandbox_client):
    """把一个"已建好的沙箱客户端"包成 backend, 再装进 create_deep_agent。

    这是本文件的核心骨架: 展示真实沙箱是怎么接进 DeepAgents 的。
    sandbox_client 就是 LangSmith SDK 建出来的 Sandbox 对象 (见下方接入点)。
    """
    # 接入点: 下面这行把外部沙箱客户端包成一个 deepagents backend。
    # LangSmithSandbox(sandbox=...) 收的是"沙箱客户端对象", 不是 URL / token。
    sandbox_backend = LangSmithSandbox(sandbox=sandbox_client)
    # 把 backend 交给 create_deep_agent: 之后 agent 的所有 ls/read/write/execute
    # 都会走这个远程沙箱, 而不是宿主机磁盘 —— 这正是它区别于 ch_04_backend.py 的地方。
    model = build_model()  # 只有真正要 invoke 时才需要模型
    agent = create_deep_agent(
        model=model,
        backend=sandbox_backend,  # ← 换 backend 即换执行环境, 调用方式完全不变
        system_prompt=(
            "You are a coding assistant. You may run finite compute in the sandbox "
            "to verify results. Never start any long-running server or listener."
        ),
    )
    return agent


# ---------------------------------------------------------------------------
# 接入点: 真实沙箱客户端应该在这里创建 (需要外部服务 + 凭证, 本地不可用)。
# 伪代码示意 (请勿在本地运行, 会因缺少服务 / 凭证而失败):
#
#     from langsmith import Client                 # 接入点: LangSmith SDK
#     ls_client = Client(api_key=os.environ["LANGSMITH_API_KEY"])  # 接入点: 凭证
#     sandbox_client = ls_client.create_sandbox(...)               # 接入点: 起沙箱
#     agent = build_sandbox_agent(sandbox_client)
#     result = agent.invoke({"messages": [{"role": "user",
#                            "content": "用 Python 算一下 2**20 是多少"}]})
#     print(result["messages"][-1].text)  # 注意: 用 .text 读回复, 不是 .content
#
# 以上整段涉及真实沙箱服务, 【未在本地跑通】。
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    # 这里只做"不需要模型、也不需要连沙箱"的结构性验证。

    # 1) 沙箱抽象类 / 协议确实存在, 且 LangSmithSandbox 继承自它们 ——
    #    这证明"沙箱 backend"是 deepagents 里真实存在的一档 backend, 不是杜撰。
    assert BaseSandbox is not None, "BaseSandbox 应存在"
    assert SandboxBackendProtocol is not None, "SandboxBackendProtocol 应存在"
    assert issubclass(LangSmithSandbox, BaseSandbox), "LangSmithSandbox 应继承 BaseSandbox"
    print("结构断言 1 通过: 沙箱协议 / 基类存在, LangSmithSandbox 继承自 BaseSandbox")

    # 2) 沙箱 backend 该有的一组能力方法都在 (execute 是它区别于纯文件 backend 的关键)。
    for method in ("execute", "read", "write", "ls"):
        assert hasattr(LangSmithSandbox, method), f"沙箱 backend 应有 {method} 方法"
    print("结构断言 2 通过: execute/read/write/ls 等沙箱能力方法齐全")

    # 3) LangSmithSandbox 的构造签名收的是一个 sandbox 对象 (印证'接入点是客户端而非URL')。
    sig = inspect.signature(LangSmithSandbox.__init__)
    assert "sandbox" in sig.parameters, "LangSmithSandbox.__init__ 应接受 sandbox 参数"
    print(f"结构断言 3 通过: LangSmithSandbox.__init__ 签名为 {sig}")

    # 4) 骨架函数 build_sandbox_agent 可被引用 (import 无误)。
    assert callable(build_sandbox_agent), "build_sandbox_agent 应可调用"
    print("结构断言 4 通过: 接入骨架 build_sandbox_agent 可用")

    print("\n[降级说明] 真实远程沙箱需要外部服务 + 凭证, 本地不可用, 上述 invoke 未运行。")

"""DeepAgents interpreter —— 给 agent 一个能真正"执行代码"的 LocalShellBackend。

对比 deepagents/filesystem.py: 那里 agent 只有 ls/read_file/write_file/edit_file,
本质是"读写文件", 不能运行任何东西。本文件换成 LocalShellBackend, agent 会额外拿到
一个 execute (shell) 能力 —— 可以像 code interpreter 那样跑一段有限的计算命令
(例如 python3 -c "print(2**10)"), 拿到 stdout 再继续推理。这是"代码解释器"风格
最贴近的真实机制。

需要澄清一个常见误解: deepagents 并没有一个字面叫 CodeInterpreter 的中间件。真正能
"执行"的现成机制有两条: (1) 本文件用的 deepagents.backends.LocalShellBackend
(直接在宿主机上跑 shell, 用 root_dir + virtual_mode 圈护栏); (2) LangChain 层的
langchain.agents.middleware.ShellToolMiddleware (通过 execution_policy 决定在
Host / Docker / Codex 沙箱里执行)。想要更强隔离时用 (2) 的 Docker 策略或本仓库
sandbox_backend.py 讲的远程沙箱, 而不是本文件的 LocalShellBackend。

【红线】本文件的 system_prompt 和任何示例命令都【绝不】启动任何监听端口 / 服务器
(no server, no listener), 只跑"有限时间内结束的计算" (finite compute)。

踩坑记录:
- 参数存在不等于模型会用: 即便 backend 提供了 execute, 模型也常常懒得真去执行,
  除非 system_prompt 明确"可以用 shell 执行代码来验证结果"。
- 输出有字节上限: LocalShellBackend(max_output_bytes=...), 默认 100000 字节, 超出
  会被截断 (ExecuteResponse.truncated=True), 别指望它回传一个巨大的输出。
- env 隔离的坑: LocalShellBackend 默认 inherit_env=False, 子进程拿不到宿主机的
  环境变量。不要为了省事 inherit_env=True, 否则模型发起的 execute 命令会继承
  .env 里的 API key / LangSmith token。需要跑 python 时, 只显式补 PATH 这类最小变量。

【本文件实际跑通了什么 (诚实说明)】
- backend.execute(...) 这个"不经过模型"的直接调用, 在 __main__ 里【真实运行并断言】
  通过 (跑 python3 -c "print(2**10)" 拿到 1024)。
- 走"模型 -> 决定调用 execute 工具"的完整链路需要 MODEL_ID, 本地没有, 所以那段是
  model-dependent, 有 MODEL_ID 才跑, 否则清晰跳过。

官方文档: https://docs.langchain.com/oss/python/deepagents/backends
"""

import os  # 环境变量 (MODEL_ID / ANTHROPIC_BASE_URL)
import shutil  # 结束时清理临时目录
import tempfile  # 生成沙箱根目录, 隔离到系统临时目录

from dotenv import load_dotenv  # 从 .env 读模型配置
from deepagents import create_deep_agent  # 主入口, 支持 backend=
from deepagents.backends import LocalShellBackend  # 带 execute 能力的本地 shell backend
from langchain_anthropic import ChatAnthropic  # Anthropic 模型封装

load_dotenv(override=True)  # .env 覆盖同名环境变量

# 用系统临时目录当执行根目录: agent 的 shell 命令只能在这里活动, 结束即删,
# 不会碰到本仓库的真实文件 (跟 backend.py 用 tempfile 隔离是同一个思路)。
scratch_dir = tempfile.mkdtemp(prefix="deepagents_interpreter_demo_")
print(f"shell 执行沙箱根目录: {scratch_dir}")

# 构造 LocalShellBackend:
# - root_dir: 把命令的工作目录限制在临时沙箱;
# - virtual_mode=True: agent 看到的是虚拟绝对路径, 挡住 ".."/"~" 穿越;
# - timeout=10: 单条命令最多跑 10 秒, 防止卡死 (也是"不许起长驻服务"的兜底);
# - max_output_bytes: 限制回传输出大小, 见上文踩坑记录;
# - inherit_env=False + env={"PATH": ...}: 不继承 API key 等敏感变量, 只补命令查找路径。
backend = LocalShellBackend(
    root_dir=scratch_dir,
    virtual_mode=True,
    timeout=10,
    max_output_bytes=100000,
    inherit_env=False,
    env={"PATH": os.environ.get("PATH", "")},
)


def build_agent():
    """构造带执行能力的 agent (需要模型, 单独包一层避免 import 即崩)。"""
    model = ChatAnthropic(
        model=os.environ["MODEL_ID"],  # 模型名只从环境变量取
        base_url=os.getenv("ANTHROPIC_BASE_URL") or None,
    )
    # 把 LocalShellBackend 交给 create_deep_agent: agent 因此获得 execute/shell 能力。
    # system_prompt 明确"可以用 execute 跑有限计算来验证", 并强调"绝不启动服务器",
    # 既提高模型真去执行的概率, 又守住不起端口的红线。
    return create_deep_agent(
        model=model,
        backend=backend,
        system_prompt=(
            "You are a Python code interpreter assistant. You may use the execute/shell "
            "capability to run SHORT, finite Python snippets (e.g. python3 -c \"...\") "
            "to compute and verify answers. NEVER start any server, web service, or "
            "long-running listener; only run commands that finish quickly."
        ),
    )


if __name__ == "__main__":
    # ---- 结构性验证 (不需要模型): 直接调用 backend.execute 证明执行能力是真的 ----
    # LocalShellBackend 没有 root_dir 属性, 用 id / virtual_mode 这些公开属性做存在性断言即可。
    assert backend.id, "backend 应已构造 (带一个 id)"
    assert backend.virtual_mode is True, "应开启 virtual_mode 护栏 (限制在沙箱根目录内)"
    print(f"结构断言 1 通过: LocalShellBackend 已构造 (id={backend.id}), virtual_mode 护栏已开, 沙箱={scratch_dir}")

    # 直接 (绕过模型) 调用 execute 跑一段有限计算 —— 这正是 code interpreter 的底层动作。
    resp = backend.execute('python3 -c "print(2**10)"')
    print(f"直接 execute 的返回对象: {resp!r}")
    assert resp.exit_code == 0, f"命令应成功退出, 实际 exit_code={resp.exit_code}"
    assert "1024" in resp.output, f"输出里应包含 1024, 实际: {resp.output!r}"
    print("结构断言 2 通过: backend.execute 真的执行了 python 并返回 1024 (未经过模型)")

    # 再确认 execute 的输出是有上限的 (印证 max_output_bytes 踩坑记录)。
    assert hasattr(resp, "truncated"), "ExecuteResponse 应带 truncated 字段"
    print("结构断言 3 通过: 执行结果带 truncated 字段 (输出会被字节上限截断)")

    # ---- 模型相关部分: 只有拿到 MODEL_ID 才跑完整的 '模型决定调用 execute' 链路 ----
    if os.getenv("MODEL_ID"):
        agent = build_agent()
        assert agent is not None, "带执行能力的 agent 应构造成功"
        print("结构断言 4 通过: 带 execute 能力的 agent 构造成功, 开始 invoke...")
        result = agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": "帮我用 Python 算一下 2 的 10 次方是多少, 执行验证后告诉我。",
                    }
                ]
            }
        )
        # 读回复用 .text (不是 .content), 这是本仓库统一约定。
        print("=== agent 最终回复 ===")
        print(result["messages"][-1].text)
    else:
        print("[跳过] 未检测到 MODEL_ID, 跳过 '模型调用 execute' 的完整链路 (model-dependent)。")

    # 清理沙箱, 不留垃圾。
    shutil.rmtree(scratch_dir, ignore_errors=True)
    print(f"\n已清理临时目录: {scratch_dir}")

"""DeepAgents LocalShellBackend —— 让 agent 在"宿主机本地 shell"里真跑命令。

对比一览 (三档"能不能跑命令 / 跑在哪"):
- ch_03_filesystem.py / ch_04_backend.py 的 StateBackend / FilesystemBackend: 只有 ls/read/
  write/edit/glob/grep 这几个文件工具, 【没有 execute】—— agent 能读写文件, 但没法
  "执行"一条 shell 命令。
- 本文件的 LocalShellBackend: 在 FilesystemBackend 的基础上多了一个 execute() ——
  agent 能通过它在【当前这台宿主机】上直接跑 shell 命令 (python、ls、grep...),
  跑的进程就是你本机的进程, 权限就是你本机用户的权限。
- ch_11_sandbox_backend.py 的 LangSmithSandbox: 同样有 execute, 但命令跑在【远程一次性
  沙箱】里, 炸了也碰不到宿主机 —— 那是"隔离档", 本文件是"本地档"。

LocalShellBackend 的 MRO 实测是:
    LocalShellBackend -> FilesystemBackend -> SandboxBackendProtocol
                      -> BackendProtocol -> ABC
所以它既是一个能读写真实磁盘的文件 backend (继承自 FilesystemBackend), 又实现了
SandboxBackendProtocol 里那个关键的 execute() —— 只不过"沙箱"就是你本机, 一点也不隔离。

【安全告警 (来自源码 docstring, 原文用了 !!! danger "Unrestricted Execution")】
execute() 底层就是 subprocess.run(command, shell=True, cwd=root_dir),
【没有任何沙箱 / 隔离 / 安全限制】。它能:
- 读到文件系统上任意文件 (跟 virtual_mode 无关! virtual_mode 只圈住 read/write
  这类文件工具, 挡不住 execute 里 `cat /etc/passwd` 这种命令);
- 执行任意程序、发起网络连接、改系统配置、装包、起子进程。
官方源码 docstring 明确写着: **Always use Human-in-the-Loop (HITL) middleware when
using this method.** —— 生产里给 agent 开 execute, 必须配合人工审批
(见 deepagents/ch_06_hitl.py、deepagents/ch_05_permissions.py), 否则等于把 shell 直接交给模型。

execute() 的返回是 ExecuteResponse 数据类 (来自 deepagents.backends.protocol),
字段实测为:
- output:   stdout 和 stderr 合并后的文本 (stderr 每行会加 "[stderr] " 前缀);
            非零退出时结尾还会追加一行 "Exit code: N"; 无输出时是 "<no output>"。
- exit_code: 进程退出码 (0 成功, 非 0 失败; 超时固定返回 124)。
- truncated: 输出是否因为超过 max_output_bytes(默认 100000) 被截断。
注意: 读结果要用 r.output / r.exit_code / r.truncated, 【不是】 .stdout/.returncode。

构造参数 (实测签名):
    LocalShellBackend(root_dir=None, *, virtual_mode=None, timeout=120,
                      max_output_bytes=100000, env=None, inherit_env=False)
- root_dir:         命令的工作目录 (cwd); 本文件用系统临时目录, 不碰仓库。
- virtual_mode:     0.5.0 起建议显式指定 (影响 read/write 文件工具的路径护栏);
                    再次强调它【管不住 execute】里跑的命令能访问哪些路径。
- timeout:          单条命令默认超时秒数 (execute 里可用 timeout= 单独覆盖);
                    <=0 会 raise ValueError, 超时命令返回 exit_code=124。
- env / inherit_env: 命令的环境变量。inherit_env=False (默认) 时【不继承】宿主机
                    环境变量 —— 这是个好默认, 能避免把宿主机的密钥 / token 泄露给
                    agent 跑的命令; 但实测有个坑: 剥光后连 PATH 都没有, `python3`
                    会因找不到运行时而崩 (No module named 'encodings')。所以真要跑
                    python / 外部命令时, 需要通过 env= 至少补一个 PATH (见下方
                    build_local_shell_backend)。

本文件可【完全离线、不需要模型】跑通: __main__ 里直接构造 backend 并 execute 一条
有限的 python 命令, 断言 exit_code == 0 且输出里有预期结果。这条命令是有限的
(算个 2**20 就退出), 不会起任何常驻服务 / 监听。真正把 backend 接进 create_deep_agent
让【模型】去调 execute 的部分需要模型, 本地没有 MODEL_ID 就不会执行 (见文末骨架)。

官方文档: https://docs.langchain.com/oss/python/deepagents/backends
"""

import os  # 读取 MODEL_ID / ANTHROPIC_BASE_URL (仅"接模型"骨架里用到)
import shutil  # 清理临时目录
import tempfile  # 造一个仓库之外的临时工作目录当 root_dir

from deepagents import create_deep_agent  # 主入口: 支持 backend= 切换执行环境
from deepagents.backends import LocalShellBackend  # 本文件主角: 本地 shell backend


def build_local_shell_backend(root_dir: str) -> LocalShellBackend:
    """构造一个"本地 shell" backend, 命令跑在 root_dir 下、只放进最小环境变量。

    inherit_env=False: 命令【不继承】宿主机全部环境变量 (少一条把密钥 / token 泄露给
    agent 命令的路径), 这是安全默认。

    【实测踩坑 (重要)】: 如果 inherit_env=False 且 env 也不给, 那么连 PATH 都没有,
    此时 `python3 -c ...` 会直接崩 —— 报 "Failed to import encodings module /
    No module named 'encodings'", exit_code=1。原因: 剥光环境后子进程找不到 PATH /
    Python 的运行时定位信息, 解释器自身都起不来。所以这里显式只补一个 PATH:
    既保留了"不继承敏感变量"的安全姿态, 又能让 python / 常见命令正常跑起来。
    需要更多变量 (如 HOME、代理设置) 时, 按需往 env 里加, 而不是简单 inherit_env=True 全继承。

    timeout=30: 演示用途给一个短超时, 卡住的命令 30s 后返回 exit_code=124。
    """
    return LocalShellBackend(
        root_dir=root_dir,
        virtual_mode=True,   # 圈住 read/write 文件工具 (对 execute 无效, 见文件头警告)
        inherit_env=False,   # 不全继承宿主机环境变量: 安全默认
        env={"PATH": os.environ.get("PATH", "")},  # 只补最小 PATH, 否则 python 都起不来
        timeout=30,          # 单条命令默认超时 30s
    )


def build_agent_with_execute(root_dir: str):
    """把 LocalShellBackend 接进 create_deep_agent —— 此后 agent 就有了 execute 工具。

    【需要模型 + 建议配 HITL】: 一旦 backend 具备 execute, 模型就能在本机跑 shell,
    生产环境务必配合 interrupt_on / permissions 做人工审批 (见 ch_05_permissions.py)。
    这里只演示"怎么接", 真正 invoke 需要 MODEL_ID, 本地缺就不跑 (见 __main__)。
    """
    from langchain_anthropic import ChatAnthropic  # 延迟导入: 结构验证阶段不碰模型

    model = ChatAnthropic(
        model=os.environ["MODEL_ID"],               # 模型名从环境变量取, 绝不硬编码
        base_url=os.getenv("ANTHROPIC_BASE_URL") or None,
    )
    backend = build_local_shell_backend(root_dir)
    agent = create_deep_agent(
        model=model,
        backend=backend,  # ← 有了它 agent 才多出 execute 能力 (纯文件 backend 没有)
        system_prompt=(
            "You are a coding assistant. You may use the execute tool to run finite "
            "shell commands to verify results. Never start a long-running server or "
            "listener; only run commands that finish quickly."
        ),
    )
    return agent


if __name__ == "__main__":
    # ---- 无模型、离线可跑的结构性 + 真执行验证 ----
    # 用系统临时目录当 root_dir, 命令的 cwd 就在这里, 不会碰到本仓库任何文件。
    scratch_dir = tempfile.mkdtemp(prefix="deepagents_localshell_demo_")
    print(f"本地 shell 工作目录 root_dir: {scratch_dir}")

    try:
        backend = build_local_shell_backend(scratch_dir)

        # 1) execute 确实存在, 且是 LocalShellBackend 相对纯文件 backend 多出来的能力。
        assert hasattr(backend, "execute"), "LocalShellBackend 应有 execute 方法"
        print("结构断言 1 通过: LocalShellBackend 具备 execute (纯文件 backend 没有)")

        # 2) 真的在本机跑一条【有限的】python 命令 (算 2**20 就退出, 不起常驻进程)。
        #    这条命令跑在宿主机本地 shell 里, 属于本文件演示的核心行为。
        resp = backend.execute('python3 -c "print(2**20)"')

        # 3) 断言退出码为 0 (成功), 并检查输出里带上了预期结果 1048576。
        #    注意读的是 ExecuteResponse 的 .output / .exit_code, 不是 .stdout/.returncode。
        assert resp.exit_code == 0, f"命令应成功, 实际 exit_code={resp.exit_code}, 输出={resp.output!r}"
        assert "1048576" in resp.output, f"输出里应含 2**20 的结果, 实际: {resp.output!r}"
        print(f"结构断言 2 通过: execute 有限命令成功, exit_code={resp.exit_code}, output={resp.output!r}")

        # 4) 顺带演示 execute 的错误语义: 跑一条注定失败的命令, exit_code 非 0,
        #    且 stderr 会被合并进 output 并加上 "[stderr] " 前缀。
        bad = backend.execute("cat definitely_no_such_file_here.txt")
        assert bad.exit_code != 0, "读不存在的文件应返回非零 exit_code"
        assert "[stderr]" in bad.output, "stderr 应被合并进 output 并带 [stderr] 前缀"
        print(f"结构断言 3 通过: 失败命令 exit_code={bad.exit_code}, output 含 [stderr]")

        # 5) 接入骨架可被引用 (import 无误); 真正 invoke 需要模型, 这里不跑。
        assert callable(build_agent_with_execute), "build_agent_with_execute 应可调用"
        print("结构断言 4 通过: 接模型骨架 build_agent_with_execute 可用 (需 MODEL_ID 才 invoke)")

        print("\n[安全提醒] execute 直接在宿主机跑 shell, 无隔离; 生产给 agent 开 execute "
              "必须配 HITL 人工审批 (见 deepagents/ch_05_permissions.py、ch_06_hitl.py)。")
    finally:
        shutil.rmtree(scratch_dir, ignore_errors=True)
        print(f"已清理临时目录: {scratch_dir}")

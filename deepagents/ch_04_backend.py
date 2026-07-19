"""DeepAgents backend —— 把虚拟文件系统换成真实磁盘的 FilesystemBackend。

对比 deepagents/ch_03_filesystem.py: 那里用的是 create_deep_agent 的默认 backend
(StateBackend) —— agent 写的"文件"其实是存在 LangGraph state 里的一个 dict,
进程一退出就没了 (没配 checkpointer 的话), 从外部 Python 代码完全看不到。
这里给 create_deep_agent 传 backend=FilesystemBackend(root_dir=...), 同样是
write_file/read_file 这几个工具、同样的调用方式, 但工具背后的实现换成了直接读写
本机磁盘上的真实文件 —— agent 跑完之后, 哪怕不用 result["files"], 直接用标准库
pathlib 也能读到 agent 写的文件, 这才是本文件要证明的点。

FilesystemBackend 有个 virtual_mode 参数 (0.5.0 起必须显式指定, 否则会有
DeprecationWarning): virtual_mode=True 时, 所有路径都被当成相对于 root_dir 的
"虚拟绝对路径"处理 (agent 眼里看到的还是 "/notes.md" 这种以 / 开头的路径), 并且会
挡住 ".."/"~" 这类路径穿越, 确保 agent 写不出 root_dir 之外 —— 相当于给真实磁盘
访问加了一圈"虚拟根目录"的护栏。官方文档也强调 FilesystemBackend 让 agent 有真实
读写权限, 存在安全风险 (可能读到 .env、密钥等), 生产环境要么配合人工审批
(interrupt_on/permissions, 见 deepagents/ch_06_hitl.py、deepagents/ch_05_permissions.py),
要么限制在一个专门的沙箱目录里 —— 这里就是把 root_dir 限制在一个系统临时目录中,
不会碰到本仓库任何真实文件。

官方文档: https://docs.langchain.com/oss/python/deepagents/backends
"""

import os
import shutil
import tempfile
from pathlib import Path

from dotenv import load_dotenv
from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend
from langchain_anthropic import ChatAnthropic

load_dotenv(override=True)

model = ChatAnthropic(
    model=os.environ["MODEL_ID"],
    base_url=os.getenv("ANTHROPIC_BASE_URL") or None,
)

# 用系统临时目录当 root_dir, 而不是仓库内的某个子目录: 这样即使 agent 写坏了什么,
# 影响也仅限于这个进程结束后自动被系统清理的临时目录, 不会污染 git 工作区
# (不需要额外改 .gitignore, tempfile.mkdtemp() 产生的路径天然在仓库之外)。
scratch_dir = tempfile.mkdtemp(prefix="deepagents_backend_demo_")
print(f"真实磁盘 root_dir: {scratch_dir}")

# virtual_mode=True: agent 用 "/xxx.md" 这种虚拟绝对路径操作文件, 底层会被映射到
# {scratch_dir}/xxx.md; 同时禁止 ".."/绝对路径穿越出 scratch_dir, 这是官方推荐的
# 用法 (virtual_mode=False 时绝对路径会直接落到真实文件系统根目录, 风险更大)。
backend = FilesystemBackend(root_dir=scratch_dir, virtual_mode=True)

agent = create_deep_agent(
    model=model,
    backend=backend,
    system_prompt=(
        "You are a research assistant. When asked to save findings, use write_file "
        "to save them to a file named notes.md before giving your final answer."
    ),
)


if __name__ == "__main__":
    print("=== agent 通过 write_file 工具写文件 ===")
    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "简单整理一下 Python 装饰器的作用, 一两句话, 存到文件里。",
                }
            ]
        }
    )
    print(result["messages"][-1].text)

    # 关键验证: 完全不用 deepagents 提供的 result["files"], 而是用标准库
    # pathlib 直接从磁盘上把文件读出来 —— 如果这里能读到内容, 就证明 write_file
    # 工具是真的写到了本机文件系统, 而不是像 ch_03_filesystem.py 里那样只存在于
    # LangGraph 的 state (graph 运行结束、进程退出后就查无此文件)。
    notes_path = Path(scratch_dir) / "notes.md"
    print("\n=== 用 pathlib 直接从磁盘读取 (证明是真实文件, 不是虚拟state) ===")
    print(f"文件是否存在于磁盘: {notes_path.exists()}")
    if notes_path.exists():
        print("磁盘上的原始内容:")
        print(notes_path.read_text(encoding="utf-8"))

    # 顺手对照一下 result["files"] —— 实测发现它在这里是空字典, 跟
    # ch_03_filesystem.py 里默认的 StateBackend 不一样 (那边写完文件 result["files"]
    # 会带上完整内容)。原因: result["files"] 反映的是"写进 LangGraph state 里的
    # 文件快照", 而 FilesystemBackend 的 write_file 直接操作磁盘, 并不会把内容
    # 顺带同步一份进 state —— 所以换成真实磁盘 backend 之后, 要拿到文件内容就得
    # 像上面那样直接读磁盘 (或者调用 backend/agent 的 read_file 工具), 不能再指望
    # result["files"] 了。这是切换 backend 时容易踩的一个坑, 值得记住。
    files_snapshot = result.get("files", {})
    print(f"\n=== 对照: result['files'] 在 FilesystemBackend 下是: {files_snapshot!r} ===")

    # 清理临时目录, 不在系统里留垃圾 (演示用途, 生产环境通常会保留 root_dir)
    shutil.rmtree(scratch_dir, ignore_errors=True)
    print(f"\n已清理临时目录: {scratch_dir}")

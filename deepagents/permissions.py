"""DeepAgents permissions —— 用规则限制文件工具能碰哪些路径、能做哪些操作。

对比 deepagents/backend.py: 那里换了 backend 让 agent 能碰真实磁盘, 但只要给了
backend, agent 理论上就能读写 root_dir 下的任何文件——如果 agent 判断错了 (比如
误把密钥文件当成"需要的配置去读"), 后果是真实的。permissions 参数就是用来在
"给不给 backend"这个粗粒度开关之外, 再加一层细粒度的路径级访问控制: 哪些路径
禁止读、哪些路径只能读不能写、哪些路径的操作需要人工审批, 全部在 create_deep_agent
层面用声明式规则表达, 不需要自己在每个工具里手写 if 判断。

FilesystemPermission 是一个 dataclass (deepagents.middleware.filesystem 模块),
实测跑出来的字段形状是:
    FilesystemPermission(
        operations=["read", "write"],   # 只能是 "read"/"write" 这两种取值的列表
        paths=["/secrets/**"],          # glob 模式列表, 必须以 "/" 开头
        mode="deny",                    # "allow" (默认) / "deny" / "interrupt" 三选一
    )
create_deep_agent 的 permissions 参数接收一个 FilesystemPermission 列表, 规则按
声明顺序匹配, 命中第一条就生效, 都不命中则默认放行 (allow)。三种 mode:
- "allow": 放行 (默认值, 一般不需要显式写)
- "deny": 工具直接返回 "permission denied" 错误, agent 拿不到任何数据
- "interrupt": 走 deepagents/hitl.py 里那套 HumanInTheLoopMiddleware, 暂停等人工
  审批 (deepagents 会自动帮你把这条规则转换成对应的 interrupt_on 配置, 不需要
  自己再手写一遍)

下面用两条规则模拟一个真实场景: 一个"只读密钥区"(完全禁止读写) + 一个"已发布
报告区"(只允许读、不允许写, 防止 agent 手滑覆盖历史报告)。

官方文档: https://docs.langchain.com/oss/python/deepagents/permissions
"""

import os
import shutil
import tempfile
from pathlib import Path

from dotenv import load_dotenv
from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend
from deepagents.middleware.filesystem import FilesystemPermission
from langchain_anthropic import ChatAnthropic

load_dotenv(override=True)

model = ChatAnthropic(
    model=os.environ["MODEL_ID"],
    base_url=os.getenv("ANTHROPIC_BASE_URL") or None,
)

scratch_dir = Path(tempfile.mkdtemp(prefix="deepagents_permissions_demo_"))
print(f"真实磁盘 root_dir: {scratch_dir}")

# 预先放两类"敏感文件", 模拟一个真实项目里既有密钥、又有历史已发布产物的场景。
(scratch_dir / "secrets").mkdir()
(scratch_dir / "secrets" / "api_key.txt").write_text("SUPER-SECRET-KEY-12345", encoding="utf-8")

(scratch_dir / "published").mkdir()
(scratch_dir / "published" / "report.md").write_text(
    "# 已发布的旧报告\n\n内容不应该被覆盖。", encoding="utf-8"
)

backend = FilesystemBackend(root_dir=str(scratch_dir), virtual_mode=True)

agent = create_deep_agent(
    model=model,
    backend=backend,
    permissions=[
        # /secrets/** 下的任何文件: 读和写都禁止, agent 连内容都看不到。
        FilesystemPermission(operations=["read", "write"], paths=["/secrets/**"], mode="deny"),
        # /published/** 下的文件: 只禁止写 (不在 operations 里出现的 "read" 默认
        # 仍走 "allow" 分支), 相当于只读区——可以查阅历史报告, 但不能改写它们。
        FilesystemPermission(operations=["write"], paths=["/published/**"], mode="deny"),
    ],
    system_prompt="You are a helpful assistant with filesystem access.",
)


if __name__ == "__main__":
    print("=== 场景 1: 尝试读取完全禁止访问的 /secrets/api_key.txt ===")
    r1 = agent.invoke(
        {
            "messages": [
                {"role": "user", "content": "读取 /secrets/api_key.txt 的内容并告诉我。"}
            ]
        }
    )
    print(r1["messages"][-1].text)
    # 从消息历史里把 read_file 工具的返回结果单独挑出来看, 直接证明请求在工具
    # 这一层就被拦截了 (而不是模型自己决定不告诉我们)。
    tool_msgs = [m for m in r1["messages"] if type(m).__name__ == "ToolMessage"]
    print(f"[验证] read_file 工具的原始返回: {tool_msgs[-1].content!r}")

    print("\n=== 场景 2: 尝试覆盖只读区 /published/report.md (应该被拒绝) ===")
    r2 = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "把 /published/report.md 的内容改成 '新版报告内容'。",
                }
            ]
        }
    )
    print(r2["messages"][-1].text)

    print("\n=== 场景 3: 同一份只读文件, 读取应该被允许 (只挡写, 不挡读) ===")
    r3 = agent.invoke(
        {
            "messages": [
                {"role": "user", "content": "读一下 /published/report.md 里写了什么。"}
            ]
        }
    )
    print(r3["messages"][-1].text)

    print("\n=== 场景 4: 没有匹配任何规则的普通路径, 应该正常允许读写 ===")
    r4 = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "在 /draft.md 里写一句 '这是一份草稿'。",
                }
            ]
        }
    )
    print(r4["messages"][-1].text)
    print(f"[验证] /draft.md 是否真的落到磁盘: {(scratch_dir / 'draft.md').exists()}")

    shutil.rmtree(scratch_dir, ignore_errors=True)
    print(f"\n已清理临时目录: {scratch_dir}")

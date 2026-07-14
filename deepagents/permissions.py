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

本文件后半段 (见下方 "追加: mode=\"interrupt\"" 分节) 额外演示第三种 mode:
把某个敏感写路径设成 mode="interrupt", 让写操作不是被"硬拒绝", 而是"暂停等人工
审批"——它复用了 deepagents/hitl.py 里那套 HumanInTheLoopMiddleware 机制, 但完全
不用自己手写 interrupt_on, deepagents 会自动把这条 permission 规则接上 HITL。

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
# 下面两个 import 是给后半段 mode="interrupt" 演示用的: interrupt 需要 checkpointer
# 保存"暂停时的完整状态", 并用 Command(resume=...) 恢复执行 (跟 hitl.py 一模一样)。
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

load_dotenv(override=True)

# MODEL_ID 存在才构造真实模型; 否则置为 None, 让"结构化断言"(不依赖模型) 依然能跑,
# 只把"需要真实模型的场景"跳过。这样本文件在没有 API key 的机器上也能做基本自检。
_HAS_MODEL = "MODEL_ID" in os.environ

model = (
    ChatAnthropic(
        model=os.environ["MODEL_ID"],
        base_url=os.getenv("ANTHROPIC_BASE_URL") or None,
    )
    if _HAS_MODEL
    else None
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

# 把 deny/allow 规则单独拎成一个模块级变量, 这样即使没有模型 (无法构造 agent),
# 结构化断言依然能直接检查这些 FilesystemPermission 对象本身。
deny_allow_permissions = [
    # /secrets/** 下的任何文件: 读和写都禁止, agent 连内容都看不到。
    FilesystemPermission(operations=["read", "write"], paths=["/secrets/**"], mode="deny"),
    # /published/** 下的文件: 只禁止写 (不在 operations 里出现的 "read" 默认
    # 仍走 "allow" 分支), 相当于只读区——可以查阅历史报告, 但不能改写它们。
    FilesystemPermission(operations=["write"], paths=["/published/**"], mode="deny"),
]

# 没有模型时 agent=None, 原始 deny/allow 演示会被 __main__ 里的 _HAS_MODEL 守卫跳过,
# 但不影响本文件后半段的"结构化断言"照常运行。
agent = (
    create_deep_agent(
        model=model,
        backend=backend,
        permissions=deny_allow_permissions,
        system_prompt="You are a helpful assistant with filesystem access.",
    )
    if _HAS_MODEL
    else None
)


# ========== 追加: mode="interrupt" 需要人工审批的路径 ==========
#
# 前面两条规则用的是 mode="deny"——工具层直接返回 "permission denied", agent 什么
# 都拿不到, 这是一种"硬拦截"。但很多真实场景里, 我们不想彻底禁止某个操作, 而是想
# "让一个人来拍板": 写到 /reports/** 这种对外产物目录时, 先暂停, 交给人工 approve
# 或 reject。这正是第三种 mode="interrupt" 的用途。
#
# mode="deny" vs mode="interrupt" 的核心区别:
#   - "deny":       硬阻断。工具立刻返回错误, agent 拿不到任何结果, 无法挽回,
#                   也没有"再商量一下"的余地。
#   - "interrupt":  暂停并询问人工。执行流在写操作这一步挂起, 人可以 approve
#                   (放行, 真正执行写) 或 reject (否决)。相当于给敏感操作加了
#                   一道"人工签字"关卡, 而不是一刀切禁止。
#
# 关键点: mode="interrupt" 底层复用的就是 deepagents/hitl.py 里那套
# HumanInTheLoopMiddleware 机制 —— deepagents 会自动把这条 permission 规则转换成
# 等价的 interrupt_on 配置接上去, 你完全不需要自己再手写一遍 interrupt_on。
# 换句话说: hitl.py 是"按工具名"审批 (send_email 这个工具需要审批), 而这里是
# "按文件路径 + 操作类型"审批 (往 /reports/** 写这件事需要审批), 两者殊途同归。
#
# 踩坑记录: mode="interrupt" 隐式依赖 checkpointer!
#   interrupt 的本质是"把执行暂停下来、把状态存起来、等 Command(resume=...) 再恢复"。
#   如果构造 agent 时没传 checkpointer, 暂停时的状态无处可存, 后续也就没法 resume,
#   这个"暂停"会变成一个无法恢复的死局 (跟 hitl.py 必须配 InMemorySaver 是同一个
#   道理)。所以下面这个 agent 一定要带上 checkpointer。

interrupt_permissions = [
    # 往 /reports/** 写文件: 不是禁止, 而是暂停等人工审批 (approve 才真正落盘)。
    FilesystemPermission(operations=["write"], paths=["/reports/**"], mode="interrupt"),
]

# 这个 agent 专门演示 interrupt 模式, 因此必须带 checkpointer (见上方踩坑记录)。
interrupt_agent = (
    create_deep_agent(
        model=model,
        backend=backend,
        permissions=interrupt_permissions,
        system_prompt="You are a helpful assistant with filesystem access.",
        checkpointer=InMemorySaver(),
    )
    if _HAS_MODEL
    else None
)


if __name__ == "__main__":
    # ---------- 结构化断言: 不依赖模型, 任何机器上都会跑, 保证规则对象本身合法 ----------
    # deny 规则: /secrets/** 读写全禁。
    assert deny_allow_permissions[0].mode == "deny"
    assert deny_allow_permissions[0].operations == ["read", "write"]
    assert deny_allow_permissions[0].paths == ["/secrets/**"]
    # 只读区规则: 只挡写。
    assert deny_allow_permissions[1].mode == "deny"
    assert deny_allow_permissions[1].operations == ["write"]
    # interrupt 规则: 写到 /reports/** 时暂停等人工审批 (而非硬拒绝)。
    assert interrupt_permissions[0].mode == "interrupt"
    assert interrupt_permissions[0].operations == ["write"]
    assert interrupt_permissions[0].paths == ["/reports/**"]
    # paths 里的 glob 必须以 "/" 开头 (deepagents 的硬性约束)。
    for perm in deny_allow_permissions + interrupt_permissions:
        for p in perm.paths:
            assert p.startswith("/"), f"permission 路径必须以 / 开头: {p!r}"
    print("[结构化断言] deny/allow/interrupt 三种 mode 的 permission 对象均合法。\n")

    if not _HAS_MODEL:
        # 降级说明: 没有 MODEL_ID 就无法构造/运行 agent, 只做上面的结构自检。
        print("(未检测到 MODEL_ID: 已完成不依赖模型的结构化断言; 以下需要真实模型的")
        print(" 演示场景全部跳过。设置 MODEL_ID 后可看到 deny/interrupt 的真实行为。)")
        shutil.rmtree(scratch_dir, ignore_errors=True)
        raise SystemExit(0)

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

    # ---------- 场景 5: mode="interrupt"——写敏感路径时暂停等人工审批 (需模型) ----------
    # 注意用的是 interrupt_agent (带 checkpointer 的那个), 且必须给 thread_id,
    # 否则 Command(resume=...) 找不到"暂停时的那次执行"。
    print("\n=== 场景 5: 写 /reports/** 触发 interrupt, 人工 approve 后才真正落盘 ===")
    icfg = {"configurable": {"thread_id": "permissions-interrupt-demo"}}
    interrupt_agent.invoke(
        {
            "messages": [
                {"role": "user", "content": "在 /reports/q3.md 里写一句 '第三季度报告'。"}
            ]
        },
        config=icfg,
    )
    snap = interrupt_agent.get_state(icfg)
    interrupted = next((t for t in snap.tasks if t.interrupts), None)
    if interrupted is None:
        # 模型偶尔不去调用写工具, 就不会触发 interrupt; 打印线索而非静默。
        print("(本次模型没有触发写操作, 未命中 interrupt; 可重跑一次。)")
    else:
        req = interrupted.interrupts[0].value
        for action in req["action_requests"]:
            print(f"  待审批的写操作: {action['name']}({action['args']})")
        # 人工 approve → 恢复执行, 这时写才真正发生 (对比 deny: 永远写不成)。
        interrupt_agent.invoke(
            Command(resume={"decisions": [{"type": "approve"}]}), config=icfg
        )
        print(f"[验证] approve 后 /reports/q3.md 是否落盘: {(scratch_dir / 'reports' / 'q3.md').exists()}")

    shutil.rmtree(scratch_dir, ignore_errors=True)
    print(f"\n已清理临时目录: {scratch_dir}")

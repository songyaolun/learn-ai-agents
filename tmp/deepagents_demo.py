import os
import tempfile
from pathlib import Path

from deepagents.backends import FilesystemBackend
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_tavily import TavilySearch
from langgraph.checkpoint.memory import InMemorySaver

from deepagents import FilesystemPermission, create_deep_agent

load_dotenv(override=True)


model = ChatAnthropic(
    model=os.environ["MODEL_ID"],
    base_url=os.getenv("ANTHROPIC_BASE_URL") or None,
)


scratch_dir = Path(tempfile.mkdtemp(prefix="deepagents_permissions_demo_"))
print(f"真实磁盘 root_dir: {scratch_dir}")

(scratch_dir / "secrets").mkdir()
(scratch_dir / "secrets" / "api_key.txt").write_text(
    "SUPER-SECRET-KEY-12345", encoding="utf-8"
)


(scratch_dir / "published").mkdir()
(scratch_dir / "published" / "report.md").write_text(
    "# 已发布的旧报告\n\n内容不应该被覆盖。", encoding="utf-8"
)

search = TavilySearch(max_results=5)

backend = FilesystemBackend(root_dir=str(scratch_dir), virtual_mode=True)

deny_allow_permissions = [
    FilesystemPermission(
        operations=["read", "write"], paths=["/secrets/**"], mode="deny"
    ),
    FilesystemPermission(operations=["write"], paths=["/published/**"], mode="deny"),
]


agent = create_deep_agent(
    model=model,
    tools=[search],
    backend=backend,
    system_prompt=(
        "你是一名研究助手。请使用搜索工具查找当前、准确的信息。如果需要调研一个主题，"
        "必须用write_file工具把中间调研结果保存到文件名为notes.md的文件中（不要用其他文件名），"
        "包括搜索结果、引用、对比分析等，然后综合整理并附上来源，给出清晰的答案。"
    ),
    permissions=deny_allow_permissions,
)

interrupt_permissions = [
    FilesystemPermission(operations=["write"], paths=["/reports/**"], mode="interrupt")
]

interrupt_agent = create_deep_agent(
    model=model,
    tools=[search],
    backend=backend,
    system_prompt=(
        "你是一名研究助手。请使用搜索工具查找当前、准确的信息。如果需要调研一个主题，"
        "必须用write_file工具把中间调研结果保存到文件名为notes.md的文件中（不要用其他文件名），"
        "包括搜索结果、引用、对比分析等，然后综合整理并附上来源，给出清晰的答案。"
    ),
    permissions=interrupt_permissions,
    checkpointer=InMemorySaver(),
)

if __name__ == "__main__":
    icfg = {"configurable": {"thread_id": "permissions-interrupt-demo"}}
    interrupt_agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "在 /reports/report.md 里写一句 '这是一份报告'。",
                }
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
            
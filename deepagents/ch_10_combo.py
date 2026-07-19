"""DeepAgents combo —— 把本目录学过的能力拼成一个"发布前需要人工签字"的研究助手。

这是 deepagents/ 目录的收尾文件, 对比前面几个专题文件, 这里不再演示单一能力,
而是把它们组合进一个连贯的真实场景: 一个深度研究 agent, 调研某个主题、把笔记
真正落到磁盘、遇到"对外发布"这种有真实影响的动作时暂停等人工审批、审批通过后
把最终结果整理成定死结构的报告。四块能力分别来自:

- subagents (deepagents/ch_01_quickstart.py): 主 agent 把某个细分子话题的调研工作
  委派给 researcher 子 agent, 委派过程和中间细节对主 agent 的上下文不可见。
- backend=FilesystemBackend (deepagents/ch_04_backend.py): agent 写的调研笔记是真实
  落在磁盘上的文件, 不是只存在于这一次进程的 state 里, 关掉进程也还在。
- interrupt_on (deepagents/ch_06_hitl.py): "发布报告"这个动作定义成一个有真实副作用的
  自定义工具 publish_report, 配置成需要人工 approve/reject 才能真正执行,
  跟 ch_06_hitl.py 里发邮件的例子是同一套机制。
- response_format (deepagents/ch_08_structured_output.py): 拿到人工批准之后, 最终结果
  仍然要求模型调用 ResearchReport 工具收尾, 变成一个字段确定的结构化对象,
  而不是自由文本——即使流程里插了一次人工审批打断, 这个约束依然要在
  system_prompt 里显式强调 (原因见 ch_08_structured_output.py 文件头的踩坑说明)。

这四块能力互不冲突: interrupt_on 依赖 checkpointer 保存"暂停时的完整状态"
(跟 ch_06_hitl.py 一样), backend 只影响文件工具的存储位置, subagents 和
response_format 分别作用在"要不要委派"和"最后一步交什么"上, 彼此独立组合。

官方文档: https://docs.langchain.com/oss/python/deepagents/quickstart
"""

import os
import shutil
import tempfile
from pathlib import Path

from dotenv import load_dotenv
from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend
from langchain.agents.structured_output import ToolStrategy
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_anthropic import ChatAnthropic
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from pydantic import BaseModel, Field

load_dotenv(override=True)

model = ChatAnthropic(
    model=os.environ["MODEL_ID"],
    base_url=os.getenv("ANTHROPIC_BASE_URL") or None,
)

# 跟 ch_04_backend.py 一样, 用系统临时目录当研究笔记的真实落地位置, 不污染仓库。
scratch_dir = Path(tempfile.mkdtemp(prefix="deepagents_combo_demo_"))
print(f"研究笔记的真实磁盘目录: {scratch_dir}")

search = DuckDuckGoSearchRun()


def publish_report(markdown: str) -> str:
    """Publish the final research report externally. Has real side effects (simulated);
    requires human approval before executing."""
    return "(模拟) 报告已正式对外发布。"


# 结构化最终产出的形状, 跟 ch_08_structured_output.py 完全一样。
class ResearchReport(BaseModel):
    topic: str = Field(description="研究主题")
    findings: list[str] = Field(description="关键发现列表, 每条一句话")
    sources: list[str] = Field(description="信息来源 URL 列表")


backend = FilesystemBackend(root_dir=str(scratch_dir), virtual_mode=True)

agent = create_deep_agent(
    model=model,
    tools=[search, publish_report],
    backend=backend,
    system_prompt=(
        "You are a deep research assistant. Follow these steps in order, and do "
        "NOT skip any of them: "
        "1) delegate focused subtopics to the researcher subagent when useful; "
        "2) write your consolidated notes to /notes.md using write_file; "
        "3) you MUST call publish_report with the final markdown — this step is "
        "mandatory and always required once notes are written, never skip it "
        "and never ask the user for confirmation first (that is what the "
        "approval step right after this call is for); "
        "4) IMPORTANT: after publish_report succeeds, you MUST call the "
        "`ResearchReport` tool to submit your final structured answer — do NOT "
        "reply with plain text as your final message."
    ),
    # 能力 1: 子 agent 委派——把细分子话题外包给 researcher, 主 agent 的上下文
    # 不会被子 agent 内部的搜索细节撑爆。
    subagents=[
        {
            "name": "researcher",
            "description": "Delegate a focused research subtopic to this subagent.",
            "system_prompt": "You are a great researcher. Search and return a brief, accurate summary.",
        }
    ],
    # 能力 2: 人工审批——只有 publish_report 这个"对外发布"的动作需要审批,
    # 前面的搜索、写笔记都不受影响, 可以自由执行。
    interrupt_on={"publish_report": {"allowed_decisions": ["approve", "reject"]}},
    # 能力 3: 结构化输出——最终答案定死成 ResearchReport 这个形状。
    response_format=ToolStrategy(ResearchReport),
    # interrupt 机制依赖 checkpointer 保存暂停时的状态, 跟 ch_06_hitl.py 用法一致。
    checkpointer=InMemorySaver(),
)


def run_until_interrupt(payload, config) -> dict:
    """跑 agent 直到结束或命中 interrupt; 命中则打印待审批内容, 结果始终返回 (方便
    没命中时也能诊断模型到底做了什么, 而不是什么线索都拿不到)。

    踩坑记录: 不能想当然假设"snapshot.next 非空就等于 snapshot.tasks[0].interrupts
    一定有内容"——实测在 deep agent 这种更复杂的图里 (带 subagent 委派), snapshot.next
    非空但 tasks[0].interrupts 是空的情况是真实发生过的 (直接按下标 0 取会
    IndexError)，所以这里改成遍历所有 task, 只挑出真的带 interrupts 的那个。
    """
    result = agent.invoke(payload, config=config)
    snapshot = agent.get_state(config)
    if not snapshot.next:
        return result
    interrupted_task = next((t for t in snapshot.tasks if t.interrupts), None)
    if interrupted_task is None:
        return result  # next 非空但没有真正的 interrupt, 当成正常跑完处理
    request = interrupted_task.interrupts[0].value
    for action in request["action_requests"]:
        print(f"  待审批工具调用: {action['name']}(markdown 长度={len(action['args'].get('markdown', ''))})")
    return result


if __name__ == "__main__":
    config = {"configurable": {"thread_id": "combo-demo"}}
    query = {
        "messages": [
            {
                "role": "user",
                "content": "研究一下 LangGraph 的核心特性, 写好笔记后发布报告。",
            }
        ]
    }

    print("=== 第一阶段: 调研 + 写笔记 + 尝试发布 (命中 interrupt) ===")
    first_result = run_until_interrupt(query, config)

    snapshot = agent.get_state(config)
    interrupted_task = next((t for t in snapshot.tasks if t.interrupts), None)
    if interrupted_task is None:
        # system_prompt 已经明确要求"必须调用 publish_report", 正常情况下这里
        # 应该总能命中 interrupt; 万一某次没命中 (模型偶尔不严格遵循指令, 复杂
        # agent 尤其容易发生), 打印出模型最后实际说了什么, 方便判断是哪一步偏离了
        # 预期, 而不是干看着一句"没命中"摸不着头脑。
        print("(未命中 interrupt, 模型没有调用 publish_report。模型最后的回复:")
        print(f"  {first_result['messages'][-1].text}")
        print("  可以直接重跑一次; 复杂 agent 偶尔不严格遵循多步指令是正常现象。)")
    else:
        print("\n=== 第二阶段: 人工批准发布 ===")
        result = agent.invoke(
            Command(resume={"decisions": [{"type": "approve"}]}), config=config
        )

        print("\n=== 结构化最终结果 (审批通过后, 模型仍要调用 ResearchReport 收尾) ===")
        report = result.get("structured_response")
        if report is None:
            print(f"(这次模型没有触发结构化输出, 最后回复: {result['messages'][-1].text})")
        else:
            print(f"类型: {type(report)}")
            print(f"主题: {report.topic}")
            print("关键发现:")
            for i, finding in enumerate(report.findings, 1):
                print(f"  {i}. {finding}")
            print("来源:", report.sources)

        print("\n=== 验证: 研究笔记是否真的落在磁盘上 (backend=FilesystemBackend) ===")
        notes_path = scratch_dir / "notes.md"
        print(f"/notes.md 是否存在于磁盘: {notes_path.exists()}")
        if notes_path.exists():
            print("磁盘上的笔记内容 (前 200 字):")
            print(notes_path.read_text(encoding="utf-8")[:200])

    shutil.rmtree(scratch_dir, ignore_errors=True)
    print(f"\n已清理临时目录: {scratch_dir}")

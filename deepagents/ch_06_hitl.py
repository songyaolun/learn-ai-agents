"""DeepAgents 内置人工审批 —— create_deep_agent 自带的 interrupt_on 参数。

对比 langchain/ch_23_middleware_hitl.py: 那里要自己 import HumanInTheLoopMiddleware,
手动塞进 create_agent(middleware=[...]); create_deep_agent 把这个常用能力直接做成了
一个参数 interrupt_on (底层其实还是同一个 HumanInTheLoopMiddleware, 只是不用自己 import
和拼 middleware 列表了), 这是 DeepAgents 这层"harness"帮你省掉的又一处样板代码。

subagents 里的每个子 agent 也可以单独配 interrupt_on (见 deepagents/ch_01_quickstart.py 的
SubAgent 定义), 互不影响 —— 主 agent 和子 agent 各自决定哪些工具需要审批。

官方文档: https://docs.langchain.com/oss/python/deepagents/human-in-the-loop
"""

import os

from dotenv import load_dotenv
from deepagents import create_deep_agent
from langchain_anthropic import ChatAnthropic
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

load_dotenv(override=True)

model = ChatAnthropic(
    model=os.environ["MODEL_ID"],
    base_url=os.getenv("ANTHROPIC_BASE_URL") or None,
)


def send_email(to: str, subject: str, body: str) -> str:
    """Send an email. 有真实副作用 (模拟), 需要人工审批."""
    return f"(模拟) 已发送邮件给 {to}, 主题: {subject}"


# interrupt_on 的写法和用法跟 langchain/ch_23_middleware_hitl.py 里的
# HumanInTheLoopMiddleware(interrupt_on={...}) 完全一样, 只是不需要自己包一层
# middleware 对象、也不用单独 import HumanInTheLoopMiddleware 了。
agent = create_deep_agent(
    model=model,
    tools=[send_email],
    system_prompt="You are a helpful assistant.",
    interrupt_on={"send_email": {"allowed_decisions": ["approve", "reject"]}},
    # interrupt 机制依赖 checkpointer 保存"暂停时的完整状态", 同一个 thread_id
    # 才能保证 Command(resume=...) 恢复的是同一次暂停的执行。
    checkpointer=InMemorySaver(),
)


def run_until_interrupt(payload, config) -> dict | None:
    """跑 agent 直到结束或命中 interrupt; 返回 interrupt 的 HITLRequest (没有则返回 None)."""
    result = agent.invoke(payload, config=config)

    # snapshot.next 非空说明 agent 被暂停了 (还有节点没跑完); 空则说明这一轮已跑完。
    snapshot = agent.get_state(config)
    if not snapshot.next:
        print(f"  最终回复: {result['messages'][-1].text}")
        return None

    request = snapshot.tasks[0].interrupts[0].value
    for action in request["action_requests"]:
        print(f"  待审批工具调用: {action['name']}({action['args']})")
    return request


if __name__ == "__main__":
    query = {
        "messages": [
            {
                "role": "user",
                "content": "帮我给 alice@example.com 发一封邮件, 主题是'周报', 内容是'本周进展顺利'。",
            }
        ]
    }

    print("=== 批准 (approve) ===")
    config1 = {"configurable": {"thread_id": "deepagents-hitl-approve"}}
    if run_until_interrupt(query, config1):
        run_until_interrupt(
            Command(resume={"decisions": [{"type": "approve"}]}), config1
        )

    print("\n=== 拒绝 (reject) ===")
    config2 = {"configurable": {"thread_id": "deepagents-hitl-reject"}}
    if run_until_interrupt(query, config2):
        rejected = {"type": "reject", "message": "先不发, 我要再检查一下内容。"}
        run_until_interrupt(Command(resume={"decisions": [rejected]}), config2)

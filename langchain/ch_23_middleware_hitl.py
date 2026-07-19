"""LangChain middleware —— 用 HumanInTheLoopMiddleware 做工具审批。

对比 claude-code/ch01.py: 那里用硬编码的危险命令黑名单 (dangerous 列表) 拦截 bash 工具,
对比 langgraph/ch_07_human_in_loop.py: 那里在图节点里手写 interrupt()/Command(resume=...),
这里用 LangChain 1.0 的 middleware 机制 —— create_agent(middleware=[...]) 声明式地指定
"哪些工具需要人工审批", 底层仍是 langgraph 的 interrupt, 但不用手写节点和路由。

HumanInTheLoopMiddleware 支持 4 种人工决策 (allowed_decisions):
  - approve: 照原样放行工具调用
  - edit:    人工修改工具参数后再执行
  - reject:  拒绝执行, 把拒绝理由当工具结果塞回给模型
  - respond: 人工直接代answer, 完全跳过工具执行

官方文档: https://docs.langchain.com/oss/python/langchain/middleware#human-in-the-loop
"""

import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain_anthropic import ChatAnthropic
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

load_dotenv(override=True)

model = ChatAnthropic(
    model=os.environ["MODEL_ID"],
    base_url=os.getenv("ANTHROPIC_BASE_URL") or None,
)


def get_weather(city: str) -> str:
    """Get weather for a given city. 无副作用, 无需审批."""
    return f"It's always sunny in {city}!"


def send_email(to: str, subject: str, body: str) -> str:
    """Send an email. 有真实副作用 (模拟), 需要人工审批."""
    # 现实中这里应该是真的调用邮件服务; 用一句模拟返回值代替, 重点是演示"发邮件前
    # 要先经过人工确认"这件事, 而不是真的发信。
    return f"(模拟) 已发送邮件给 {to}, 主题: {subject}"


# middleware 可以理解成"插在 agent 执行循环里的钩子": 在模型每次给出回复之后、
# 真正执行工具之前 (或者模型调用前后等其他时机) 插入一段自定义逻辑。
# HumanInTheLoopMiddleware 就是官方内置的一个 middleware, 作用是: 只要模型想调用
# interrupt_on 里列出的工具, 就先暂停整个 agent 执行 (底层调用了 langgraph 的
# interrupt()), 等外部人工给出决策后再恢复。
#
# interrupt_on 只声明 send_email 需要审批; get_weather 不在其中 → 自动放行, 不会暂停。
# allowed_decisions 限定人工可以做出的决策类型 (这里开放批准/修改/拒绝三种)。
agent = create_agent(
    model=model,
    tools=[get_weather, send_email],
    system_prompt="You are a helpful assistant.",
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={
                "send_email": {
                    "allowed_decisions": ["approve", "edit", "reject"],
                }
            },
        )
    ],
    # interrupt 机制依赖 checkpointer 把"暂停时的完整状态"存起来, 不然恢复的时候
    # 就不知道该从哪里继续了; 同一个 thread_id 才能保证恢复的是同一次暂停的执行。
    checkpointer=InMemorySaver(),
)


def run_until_interrupt(payload, config) -> dict | None:
    """跑 agent 直到结束或命中 interrupt; 返回 interrupt 的 HITLRequest (没有则返回 None)."""
    # payload 第一次是普通的用户消息 {"messages": [...]}, 恢复执行时则是
    # Command(resume=...) —— 两种都可以直接传给 agent.invoke, LangGraph 会自动识别。
    result = agent.invoke(payload, config=config)

    # agent.get_state(config) 读取这个 thread_id 当前的执行快照。snapshot.next 是
    # "接下来还要跑的节点列表": 如果非空, 说明 agent 被 interrupt 暂停了, 还没跑完;
    # 如果是空的, 说明这一轮已经完整结束 (模型给出了最终回复)。
    snapshot = agent.get_state(config)
    if not snapshot.next:
        print(f"  最终回复: {result['messages'][-1].text}")
        return None

    # snapshot.tasks[0].interrupts[0] 就是这次暂停携带的信息 (HITLRequest): 里面列出了
    # 所有待人工审批的工具调用请求 (action_requests), 包含工具名和调用参数。
    interrupt_obj = snapshot.tasks[0].interrupts[0]
    request = interrupt_obj.value
    for action in request["action_requests"]:
        print(f"  待审批工具调用: {action['name']}({action['args']})")
    return request


if __name__ == "__main__":
    query = {
        "messages": [
            {
                "role": "user",
                "content": "帮我给 alice@example.com 发一封邮件, 主题是'会议改期', 内容是'明天的会议改到下周一'。",
            }
        ]
    }

    print("=== 场景一: 批准 (approve) ===")
    config1 = {"configurable": {"thread_id": "hitl-approve"}}
    if run_until_interrupt(query, config1):
        # Command(resume={"decisions": [...]}) 是"恢复执行"的标准写法: decisions 是一个
        # 列表, 顺序要跟 request["action_requests"] 一一对应 (这里只有一个待审批调用,
        # 所以列表里只放一个决策)。type="approve" 表示人工同意, 工具会照原参数执行。
        run_until_interrupt(
            Command(resume={"decisions": [{"type": "approve"}]}), config1
        )

    print("\n=== 场景二: 修改后执行 (edit) ===")
    config2 = {"configurable": {"thread_id": "hitl-edit"}}
    if run_until_interrupt(query, config2):
        # type="edit" 表示人工不满意原参数, 用 edited_action 提供一份修改后的调用
        # (工具名 + 新参数), agent 会用这份新参数去真正执行工具, 而不是模型原来给的那份。
        edited = {
            "type": "edit",
            "edited_action": {
                "name": "send_email",
                "args": {
                    "to": "bob@example.com",  # 人工把收件人改掉了
                    "subject": "会议改期",
                    "body": "明天的会议改到下周一",
                },
            },
        }
        run_until_interrupt(Command(resume={"decisions": [edited]}), config2)

    print("\n=== 场景三: 拒绝 (reject) ===")
    config3 = {"configurable": {"thread_id": "hitl-reject"}}
    if run_until_interrupt(query, config3):
        # type="reject" 表示人工拒绝执行, 工具根本不会被真正调用; message 会被当成
        # "工具执行结果" 的替代内容喂回给模型, 模型会看到这句话并据此调整回复
        # (例如告诉用户"已取消发送")。
        rejected = {"type": "reject", "message": "先别发, 我要再确认一下收件人。"}
        run_until_interrupt(Command(resume={"decisions": [rejected]}), config3)

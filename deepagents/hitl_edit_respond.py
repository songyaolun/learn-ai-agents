"""DeepAgents HITL 进阶 —— edit / respond 两种人工决策 (approve/reject 之外)。

对比 deepagents/hitl.py: 那里只演示了 approve (放行) 和 reject (否决) 两种最基础的
人工决策。但 HumanInTheLoopMiddleware 实际支持四种决策类型
(DecisionType = "approve" | "edit" | "reject" | "respond"), 本文件补齐剩下的两种:

- edit    人工"改参数再执行": 模型想调用工具, 但参数不太对; 人工把参数改一改
          再放行, 工具用"改过的参数"执行。典型场景: 模型把收款人写错了、把金额
          写多了一个零, 人工顺手纠正而不是打回重来。
- respond 人工"替工具作答, 跳过执行": 工具压根不执行, 而是由人直接给出一段回答,
          这段回答会作为一条 status="success" 的 ToolMessage 塞回给模型。典型场景:
          某个工具本质上就是"问用户一句话", 那"用户的回答"就是工具的真实实现。

要用某种决策, 必须把它写进该工具的 allowed_decisions 列表里 (见踩坑记录)。

踩坑记录 1 (决策 payload 形状易错——已从源码核对):
  edit 决策的正确形状不是 {"type":"edit","args":{...}}, 而是:
      {"type":"edit","edited_action":{"name":"<工具名>","args":{...}}}
  也就是说改参数要包在 edited_action 里, 并且要连工具 name 一起给
  (源码 human_in_the_loop.py 里是 name=edited_action["name"], args=edited_action["args"])。
  如果误写成顶层的 "args", 恢复时会因为拿不到 edited_action 而 KeyError。
  edit 后的 args 必须仍然满足工具的入参 schema, 否则工具执行阶段会报参数错误。

踩坑记录 2 (allowed_decisions 必须显式包含要用的决策类型):
  如果你只配了 allowed_decisions=["approve","reject"], 却 resume 一个 edit 决策,
  框架会走到"Unexpected human decision"分支并报错——决策类型没在白名单里就用不了。
  所以本文件把四种全列进 allowed_decisions=["approve","edit","reject","respond"]。

踩坑记录 3 (respond 与 reject 的区别):
  respond 返回的 ToolMessage 是 status="success" (模型会把它当成"工具成功返回了这个结果");
  reject 返回的是 status="error" (模型会知道"这次调用被否决了")。别把两者搞混。

诚实说明: 本机没有 MODEL_ID / API key, 无法把带模型的 invoke 跑到底。因此 __main__
里做的是"不依赖模型"的结构化自检 (agent 能否构造 + 两个 Command payload 形状是否合法),
带模型的真实 edit/respond 演示用 _HAS_MODEL 守卫, 检测不到模型时清晰跳过, 绝不谎称"已跑通"。

官方文档: https://docs.langchain.com/oss/python/deepagents/human-in-the-loop
"""

import os

from dotenv import load_dotenv
from deepagents import create_deep_agent
from langchain_anthropic import ChatAnthropic
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

load_dotenv(override=True)

# 没有 MODEL_ID 时 model=None, 只做结构化自检 (见文件头"诚实说明")。
_HAS_MODEL = "MODEL_ID" in os.environ

model = (
    ChatAnthropic(
        model=os.environ["MODEL_ID"],
        base_url=os.getenv("ANTHROPIC_BASE_URL") or None,
    )
    if _HAS_MODEL
    else None
)


def transfer_funds(to: str, amount: float) -> str:
    """Transfer money to a recipient. 有真实副作用 (模拟), 转账前必须人工过目。"""
    # 真实系统里这里会调支付网关; 这里返回一句话代表"真的执行了"。
    return f"(模拟) 已向 {to} 转账 {amount} 元。"


# 和 hitl.py 唯一的区别: allowed_decisions 从 ["approve","reject"] 扩到四种全开。
# 这样人工不仅能"放行/否决", 还能"改参数再放行"(edit) 或"直接替工具作答"(respond)。
agent = (
    create_deep_agent(
        model=model,
        tools=[transfer_funds],
        system_prompt="You are a careful banking assistant. Use the transfer_funds tool to move money.",
        interrupt_on={
            "transfer_funds": {
                "allowed_decisions": ["approve", "edit", "reject", "respond"]
            }
        },
        # interrupt 依赖 checkpointer 保存暂停状态 (跟 hitl.py 完全一致)。
        checkpointer=InMemorySaver(),
    )
    if _HAS_MODEL
    else None
)


def run_until_interrupt(payload, config) -> dict | None:
    """跑 agent 直到结束或命中 interrupt; 命中则打印待审批调用并返回 HITLRequest, 否则返回 None。

    遍历 snapshot.tasks 找真正带 interrupts 的那个 (不硬取下标 0), 这是 combo.py 里
    验证过更稳妥的写法。
    """
    agent.invoke(payload, config=config)
    snapshot = agent.get_state(config)
    if not snapshot.next:
        return None
    task = next((t for t in snapshot.tasks if t.interrupts), None)
    if task is None:
        return None
    request = task.interrupts[0].value
    for action in request["action_requests"]:
        print(f"  待审批工具调用: {action['name']}({action['args']})")
    return request


# ---- 两个决策 payload 提前定义成常量, 便于 __main__ 里做"不依赖模型"的形状断言 ----

# edit 决策: 人工把转账参数改掉再放行。注意形状 —— 改动包在 edited_action 里,
# 而且要带上工具 name (见文件头踩坑记录 1)。这里演示"把金额从模型给的值改成 1 元、
# 把收款人纠正为 alice@example.com"。
EDIT_DECISION = {
    "type": "edit",
    "edited_action": {
        "name": "transfer_funds",
        "args": {"to": "alice@example.com", "amount": 1},
    },
}

# respond 决策: 完全跳过 transfer_funds 的执行, 由人直接给模型一段回答。
# 这段 message 会变成一条 status="success" 的 ToolMessage 回传给模型。
RESPOND_DECISION = {
    "type": "respond",
    "message": "本次转账已由人工线下处理, 无需系统再执行, 请据此继续。",
}


if __name__ == "__main__":
    # ---------- 不依赖模型的结构化自检 (任何机器都会跑) ----------
    # 1) 两个决策 payload 的形状是否合法。
    assert EDIT_DECISION["type"] == "edit"
    assert set(EDIT_DECISION["edited_action"]) == {"name", "args"}, \
        "edit 决策必须是 edited_action={name, args}, 不能把 args 放在顶层 (踩坑记录 1)"
    assert EDIT_DECISION["edited_action"]["name"] == "transfer_funds"
    assert isinstance(EDIT_DECISION["edited_action"]["args"], dict)
    assert RESPOND_DECISION["type"] == "respond"
    assert isinstance(RESPOND_DECISION["message"], str) and RESPOND_DECISION["message"]
    # 2) 包装成 Command(resume=...) 后结构是否符合 HITLResponse (decisions 列表)。
    edit_cmd = Command(resume={"decisions": [EDIT_DECISION]})
    respond_cmd = Command(resume={"decisions": [RESPOND_DECISION]})
    assert edit_cmd.resume["decisions"][0]["type"] == "edit"
    assert respond_cmd.resume["decisions"][0]["type"] == "respond"
    print("[结构化断言] edit / respond 两个决策 payload 及其 Command 包装均合法。\n")

    if not _HAS_MODEL:
        print("(未检测到 MODEL_ID: 已完成不依赖模型的结构化断言; 以下带模型的 edit/")
        print(" respond 真实演示全部跳过。设置 MODEL_ID 后可看到两种决策的真实行为。)")
        raise SystemExit(0)

    query = {
        "messages": [
            {
                "role": "user",
                "content": "帮我向 bob@example.com 转账 10000 元。",
            }
        ]
    }

    # ---------- 场景 1: edit —— 人工改参数后放行, 工具用改过的参数执行 ----------
    print("=== 场景 1: edit (人工把收款人/金额改掉再放行) ===")
    cfg1 = {"configurable": {"thread_id": "hitl-edit"}}
    if run_until_interrupt(query, cfg1):
        agent.invoke(Command(resume={"decisions": [EDIT_DECISION]}), config=cfg1)
        final1 = agent.get_state(cfg1).values["messages"][-1]
        print(f"  最终回复: {final1.text}")
        # transfer_funds 的返回里应能看到"改后"的收款人 alice / 金额 1, 证明 edit 生效。
        tool_msgs = [m for m in agent.get_state(cfg1).values["messages"]
                     if type(m).__name__ == "ToolMessage"]
        if tool_msgs:
            print(f"  [验证] transfer_funds 实际执行结果: {tool_msgs[-1].content!r}")

    # ---------- 场景 2: respond —— 跳过工具执行, 人工直接替工具作答 ----------
    print("\n=== 场景 2: respond (跳过转账, 人工直接替工具给出结果) ===")
    cfg2 = {"configurable": {"thread_id": "hitl-respond"}}
    if run_until_interrupt(query, cfg2):
        agent.invoke(Command(resume={"decisions": [RESPOND_DECISION]}), config=cfg2)
        msgs = agent.get_state(cfg2).values["messages"]
        tool_msgs = [m for m in msgs if type(m).__name__ == "ToolMessage"]
        if tool_msgs:
            # respond 塞回的是 status="success" 的合成 ToolMessage, 内容就是人工那句话。
            print(f"  [验证] 合成的 ToolMessage(status={tool_msgs[-1].status!r}): {tool_msgs[-1].content!r}")
        print(f"  最终回复: {msgs[-1].text}")

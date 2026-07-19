"""LangChain 组合演示 —— 把这个目录里学过的能力叠到同一个 agent 上。

前面每个文件只演示一个概念 (HITL / 摘要压缩 / PII 脱敏 / 调用次数上限 / 结构化输出),
但实际项目里这些能力往往要叠加着一起用。这里做一个"客服工单"场景把它们串起来,
体会一下 create_agent(middleware=[...], response_format=...) 组合使用的效果:

  1. PIIMiddleware      —— 用户消息里的邮箱先脱敏, 模型全程看不到原始邮箱
  2. HumanInTheLoopMiddleware —— "退款"这个有资金影响的工具调用前必须人工审批
  3. ToolCallLimitMiddleware  —— 防止 agent 对退款工具失控重复调用
  4. SummarizationMiddleware  —— 工单对话如果拖很长, 自动压缩历史, 不撑爆上下文
  5. response_format (ToolStrategy) —— 无论对话过程多复杂, 最终都要产出一份结构化的
     工单小结 (方便直接存进数据库/展示在后台), 而不是一段自由格式的文字

middleware 列表里的执行顺序是有意义的 (跟声明顺序一致), 这里 PII 放最前面, 保证
"脱敏"发生在其余所有处理之前——一个基本的数据安全原则: 敏感信息越早清洗掉越好。

覆盖范围声明 (诚实起见): langchain.agents.middleware 里还有 ModelFallbackMiddleware
(模型调用失败自动切备用模型)、LLMToolSelectorMiddleware (工具太多时先用小模型筛一遍
再调用主模型)、ContextEditingMiddleware、ShellToolMiddleware 等没有在这个仓库里演示,
用法思路是类似的 (都是塞进 create_agent(middleware=[...])), 需要时查官方文档。

官方文档: https://docs.langchain.com/oss/python/langchain/middleware
"""

import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import (
    HumanInTheLoopMiddleware,
    PIIMiddleware,
    SummarizationMiddleware,
    ToolCallLimitMiddleware,
)
from langchain.agents.structured_output import ToolStrategy
from langchain_anthropic import ChatAnthropic
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from pydantic import BaseModel, Field

load_dotenv(override=True)

model = ChatAnthropic(
    model=os.environ["MODEL_ID"],
    base_url=os.getenv("ANTHROPIC_BASE_URL") or None,
)


def check_order_status(order_id: str) -> str:
    """Check the status of an order. 只读查询, 无需审批."""
    return f"订单 {order_id}: 已发货, 预计 3 天内送达。"


def issue_refund(order_id: str, amount: float, reason: str) -> str:
    """Issue a refund for an order. 有真实资金影响, 需要人工审批."""
    return f"(模拟) 已为订单 {order_id} 退款 {amount} 元, 原因: {reason}"


# 最终答案要长成的样子: 一份结构化的工单小结, 而不是随意的文字总结。
class TicketSummary(BaseModel):
    order_id: str = Field(description="涉及的订单号")
    issue: str = Field(description="用户反馈的问题概述")
    action_taken: str = Field(description="实际采取的处理动作")
    resolved: bool = Field(description="问题是否已解决")


agent = create_agent(
    model=model,
    tools=[check_order_status, issue_refund],
    system_prompt=(
        "You are a customer support agent. Check order status when asked, and "
        "issue refunds when appropriate. You MUST end by calling the TicketSummary "
        "tool to submit a structured summary instead of replying in plain text "
        "(see langchain/ch_14_structured_output.py for why this instruction matters)."
    ),
    middleware=[
        PIIMiddleware("email", strategy="redact"),
        HumanInTheLoopMiddleware(
            interrupt_on={"issue_refund": {"allowed_decisions": ["approve", "reject"]}},
        ),
        ToolCallLimitMiddleware(tool_name="issue_refund", run_limit=2, exit_behavior="end"),
        SummarizationMiddleware(model=model, trigger=("messages", 20), keep=("messages", 4)),
    ],
    response_format=ToolStrategy(TicketSummary),
    # 跑起来会看到一条 "Deserializing unregistered type ... This will be blocked in a
    # future version" 的警告: checkpointer 需要把每一步状态 (包括 structured_response
    # 里的 TicketSummary 实例) 序列化存起来, 而自定义 Pydantic 类型默认不在它的白名单里,
    # 目前只是警告、不影响这个demo 运行, 但提示了一个真实的坑——生产环境如果同时用
    # checkpointer + response_format, 未来版本可能需要显式登记这个自定义类型才能用。
    checkpointer=InMemorySaver(),
)


def run_until_interrupt(payload, config) -> dict | None:
    """跟 langchain/ch_23_middleware_hitl.py 里的同名函数一样: 跑到结束或者命中 interrupt。"""
    result = agent.invoke(payload, config=config)
    snapshot = agent.get_state(config)
    if not snapshot.next:
        summary = result.get("structured_response")
        print(f"  最终回复: {result['messages'][-1].text}")
        if summary:
            print(f"  结构化工单小结: {summary!r}")
        return None
    request = snapshot.tasks[0].interrupts[0].value
    for action in request["action_requests"]:
        print(f"  待审批工具调用: {action['name']}({action['args']})")
    return request


if __name__ == "__main__":
    config = {"configurable": {"thread_id": "combo-demo"}}
    query = {
        "messages": [
            {
                "role": "user",
                "content": (
                    "我是 zhangsan@example.com, 我的订单 ORD-1001 商品有质量问题, "
                    "麻烦查一下状态, 然后退款 99.9 元。"
                ),
            }
        ]
    }

    print("=== 第一步: 提交请求 (邮箱会先被 PII 脱敏, 退款前会暂停等审批) ===")
    if run_until_interrupt(query, config):
        print("\n=== 第二步: 人工批准退款, 恢复执行, 拿到结构化工单小结 ===")
        run_until_interrupt(Command(resume={"decisions": [{"type": "approve"}]}), config)

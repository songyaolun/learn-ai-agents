"""LangChain middleware —— 用 ModelCallLimitMiddleware / ToolCallLimitMiddleware 防止 agent 失控空转。

对比 langchain/middleware_hitl.py、middleware_summarization.py: 那两个 middleware
分别管"要不要人工审批"和"历史太长怎么办", 这两个新的 middleware 管的是另一类风险——
agent 陷入死循环 (比如反复调用同一个工具却一直不满足退出条件) 或者跑了太多轮模型调用
(每一轮都要花钱), 需要一个"硬性上限"兜底, 不管模型多"固执"都能被强制拦下来。

  - ModelCallLimitMiddleware: 限制"调用模型"这个动作的总次数 (thread_limit 跨多轮对话
    累计, run_limit 单次 invoke 内累计)
  - ToolCallLimitMiddleware:  限制某个具体工具 (或全部工具) 被调用的次数, 用法同上

exit_behavior 决定超限之后怎么处理:
  - "end":   优雅结束, 往对话里塞一条"已达上限"的说明, 正常返回, 不报错
  - "error": 直接抛异常, 适合"超限就是严重问题, 必须让上层代码知道"的场景

官方文档: https://docs.langchain.com/oss/python/langchain/middleware#call-limits
"""

import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import ModelCallLimitMiddleware, ToolCallLimitMiddleware
from langchain_anthropic import ChatAnthropic

load_dotenv(override=True)

model = ChatAnthropic(
    model=os.environ["MODEL_ID"],
    base_url=os.getenv("ANTHROPIC_BASE_URL") or None,
)


def roll_dice() -> str:
    """Roll a six-sided die and return the result (1-6)."""
    # 故意写死"永远掷不出 6", 不用真随机数——如果用 random.randint(1, 6), demo 有
    # 1/6 的概率一次就掷到 6 提前结束, 观察不到 guardrail 生效的效果; 写死结果能保证
    # 每次跑这个 demo 都稳定复现"模型会不停调用工具, 直到被 ToolCallLimitMiddleware
    # 强制拦下"这个过程, 而不是听天由命。
    return "3"


# system_prompt 写成"不掷到 6 就不准停", 而 roll_dice 又被写死永远掷不出 6, 这个 agent
# 在没有 guardrail 的情况下会真的死循环下去 (或者一直循环到模型自己放弃为止)。
# ToolCallLimitMiddleware/ModelCallLimitMiddleware 就是防止这种情况把 tokens 烧光的兜底。
agent = create_agent(
    model=model,
    tools=[roll_dice],
    system_prompt=(
        "Keep calling roll_dice until you get a 6, then report all the rolls. "
        "Never give up before getting a 6, no matter how many tries it takes."
    ),
    middleware=[
        # run_limit=3: 这一次 invoke 里, roll_dice 最多只能被调用 3 次左右就会被拦下
        # (具体是"超过限制的下一次调用尝试被拦截", 不是精确的第 3 次之后立刻停,
        # 实测是允许到第 4 次调用发生后才触发拦截并结束, 细节以官方文档为准)。
        ToolCallLimitMiddleware(tool_name="roll_dice", run_limit=3, exit_behavior="end"),
        # 再加一层"模型调用总次数"上限做双重保险: 就算工具调用次数没超, 模型本身
        # 也不能无限"思考"下去。
        ModelCallLimitMiddleware(run_limit=8, exit_behavior="end"),
    ],
)


if __name__ == "__main__":
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "开始掷骰子, 记得掷到 6 才能停。"}]}
    )
    # exit_behavior="end" 的效果: 最后一条消息不是模型自然生成的总结, 而是
    # middleware 自己塞进去的"已达上限"提示——这就是"硬性兜底"的意义: 不依赖模型
    # 自觉配合, 无论如何都能让 agent 停下来。
    print(result["messages"][-1].text)
    tool_call_count = sum(
        1 for m in result["messages"] if type(m).__name__ == "ToolMessage"
    )
    print(f"\n实际 roll_dice 调用次数: {tool_call_count}")

"""LangGraph 多 agent 编排 —— supervisor 模式, 用 Command 同时做状态更新 + 路由。

对比 langgraph/quickstart.py: 那里用 add_conditional_edges 做路由, 路由函数和节点是分开的;
这里每个节点直接返回 Command(update=..., goto=...), 状态更新和路由决策合在一起,
不需要额外写路由函数 —— 图的边由节点返回类型注解 Command[Literal[...]] 自动推断出来。

对比 deepagents/*: 那里 subagents 是"主 agent 内部"的委派 (通过 task 工具调用, 主 agent
只看到子 agent 的最终摘要, 中间过程对主 agent 不可见); 这里是"图层面"的多 agent 编排,
supervisor 和 worker 是平级的图节点, 每个 worker 都是一个完整的 create_agent 实例,
supervisor 能看到每个 worker 的完整产出并决定下一步 —— 更适合需要精细控制路由逻辑的场景。

官方文档: https://docs.langchain.com/oss/python/langgraph/multi-agent
"""

import os
from typing import Annotated, Literal, TypedDict

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_anthropic import ChatAnthropic
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, StateGraph, add_messages
from langgraph.types import Command
from pydantic import BaseModel, Field

load_dotenv(override=True)

model = ChatAnthropic(
    model=os.environ["MODEL_ID"],
    base_url=os.getenv("ANTHROPIC_BASE_URL") or None,
)


class State(TypedDict):
    messages: Annotated[list, add_messages]


# -- supervisor: 用结构化输出决定"下一步交给谁 + 具体指令" --
class RouteDecision(BaseModel):
    next: Literal["researcher", "writer", "FINISH"] = Field(
        description="下一步交给谁处理; 所有信息都已齐备、writer 已给出最终答案时输出 FINISH"
    )
    instruction: str = Field(
        default="", description="给下一个 worker 的具体指令 (FINISH 时可以不填)"
    )


router_model = model.with_structured_output(RouteDecision)

SUPERVISOR_PROMPT = (
    "你是任务调度者 (supervisor), 手下有两个 worker:\n"
    "- researcher: 只负责上网搜索、收集事实信息, 不写最终答案\n"
    "- writer: 只负责根据已收集的信息整理成最终答案, 不做搜索\n"
    "根据当前对话历史决定下一步交给谁、给出具体指令。"
    "在 writer 产出最终答案之前不能输出 FINISH —— 即使 researcher 收集的信息已经够用, "
    "也必须先交给 writer 整理成一份完整答案, 再输出 FINISH。"
)


def supervisor_node(state: State) -> Command[Literal["researcher", "writer", "__end__"]]:
    decision = router_model.invoke(
        [{"role": "system", "content": SUPERVISOR_PROMPT}, *state["messages"]]
    )
    print(f"\n[supervisor] → {decision.next}  {decision.instruction}")
    if decision.next == "FINISH":
        return Command(goto=END)
    return Command(
        goto=decision.next,
        update={"messages": [HumanMessage(content=decision.instruction, name="supervisor")]},
    )


# -- researcher worker: 有搜索工具的完整 create_agent --
researcher_agent = create_agent(
    model=model,
    tools=[DuckDuckGoSearchRun()],
    system_prompt="You are a researcher. Search and report concise, factual findings.",
)


def researcher_node(state: State) -> Command[Literal["supervisor"]]:
    instruction = state["messages"][-1].content
    result = researcher_agent.invoke({"messages": [{"role": "user", "content": instruction}]})
    # .text 只取纯文本块, 过滤掉 thinking 等非文本 block (否则塞回 HumanMessage 会污染下一轮上下文)
    findings = str(result["messages"][-1].text)
    print(f"[researcher] {findings[:200]}")
    return Command(
        goto="supervisor",
        update={"messages": [HumanMessage(content=findings, name="researcher")]},
    )


# -- writer worker: 无工具, 只负责整理最终答案 --
writer_agent = create_agent(
    model=model,
    system_prompt="You are a writer. Synthesize a clear final answer from the conversation so far.",
)


def writer_node(state: State) -> Command[Literal["supervisor"]]:
    result = writer_agent.invoke({"messages": state["messages"]})
    draft = str(result["messages"][-1].text)
    print(f"[writer] {draft[:200]}")
    return Command(
        goto="supervisor",
        update={"messages": [HumanMessage(content=draft, name="writer")]},
    )


builder = StateGraph(State)
builder.add_node("supervisor", supervisor_node)
builder.add_node("researcher", researcher_node)
builder.add_node("writer", writer_node)
builder.add_edge(START, "supervisor")
# 注意: 没有 add_conditional_edges, 也没有手写 researcher/writer → supervisor 的边,
# 全部由各节点返回的 Command(goto=...) 在运行时决定, 图定义只需声明节点。
graph = builder.compile()


if __name__ == "__main__":
    query = "LangGraph 和 LangChain 的 create_agent 有什么区别? 帮我查一下并给出简要总结。"
    # supervisor <-> worker 会来回跳转多次, 加个 recursion_limit 兜底防止跑飞
    result = graph.invoke(
        {"messages": [{"role": "user", "content": query}]},
        config={"recursion_limit": 15},
    )
    print("\n=== 最终答案 ===")
    print(result["messages"][-1].text)

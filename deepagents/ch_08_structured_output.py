"""DeepAgents structured output —— 让 deep research agent 的最终产出是定死结构的报告。

对比 langchain/ch_08_structured_output.py: 那里给普通的 create_agent 传
response_format=ToolStrategy(WeatherReport), 模型只有一个 get_weather 工具可选,
几乎每次都会在拿到天气后老老实实调用 WeatherReport 工具收尾。这里换成
create_deep_agent, 用法 (ToolStrategy 包一层 Pydantic 模型)完全一样, 但
deep agent 自带一整套内置工具 (write_todos/ls/read_file/write_file/edit_file/
glob/grep/execute/task), 加上我们自己传的 search 工具, 模型可选的工具变多了很多。

实测踩坑 (这是本文件要重点说明的差异): 同样的 ToolStrategy(ResearchReport),
在 deep agent 上如果只是常规地把 response_format 传进去、system_prompt 里不
特别提, 实测跑出来 result["structured_response"] 经常是 None——模型做完搜索后
直接输出一段自由文本作为最终答案, 根本没有调用 ResearchReport 这个工具
(create_agent 不会在模型"决定不调用"时强行拦下来再逼一次)。可选的工具越多,
模型就越容易"忘记"结构化输出也是待选项之一。解决方式很简单但必须显式做:
在 system_prompt 里明确点名"最后必须调用 `ResearchReport` 工具提交答案, 不要
直接回复纯文本", 这样每次都能稳定拿到 structured_response。这跟
langchain/ch_08_structured_output.py 里"裸 Pydantic 类不稳定, 要用 ToolStrategy 包一层"
是同一类坑——参数/API 存在不代表模型一定会按预期使用它, 复杂 agent (工具越多)
尤其容易出现这种情况, 都需要跑起来看真实输出才能确认。

官方文档: https://docs.langchain.com/oss/python/deepagents/structured-output
"""

import os

from dotenv import load_dotenv
from deepagents import create_deep_agent
from langchain.agents.structured_output import ToolStrategy
from langchain_tavily import TavilySearch
from langchain_anthropic import ChatAnthropic
from pydantic import BaseModel, Field

load_dotenv(override=True)

model = ChatAnthropic(
    model=os.environ["MODEL_ID"],
    base_url=os.getenv("ANTHROPIC_BASE_URL") or None,
)


# 定死"研究报告"的形状: 主题 + 若干条发现 + 若干条来源, 都是明确的字段和类型,
# 不是让模型自由发挥的一段话。下游程序 (比如存数据库、渲染成网页卡片) 可以直接
# 按字段取值, 不用再自己写正则去解析模型的自然语言输出。
class ResearchReport(BaseModel):
    topic: str = Field(description="研究主题")
    findings: list[str] = Field(description="关键发现列表, 每条一句话")
    sources: list[str] = Field(description="信息来源 URL 列表")


search = TavilySearch(max_results=5)

# 和 langchain/ch_08_structured_output.py 一样用 ToolStrategy(ResearchReport) 而不是
# 直接传 ResearchReport 这个类——ToolStrategy 显式声明"通过工具调用产出结构化
# 结果", 比让框架自动挑策略更可靠。
#
# 但光有 ToolStrategy 还不够 (见上面文件头的踩坑说明): system_prompt 里必须
# 明确要求模型"最后一步调用 ResearchReport 工具", 否则在工具选择变多的
# deep agent 上, 模型很容易做完搜索就直接用文本回复, 完全跳过结构化输出这一步。
agent = create_deep_agent(
    model=model,
    tools=[search],
    system_prompt=(
        "You are a research assistant. Use the search tool to gather accurate, "
        "up-to-date information on the user's topic. "
        "IMPORTANT: once your research is complete, you MUST call the "
        "`ResearchReport` tool to submit your final structured answer — do NOT "
        "reply with plain text as your final message."
    ),
    response_format=ToolStrategy(ResearchReport),
)


if __name__ == "__main__":
    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "简要研究一下 LangGraph 是什么, 给出 2-3 个关键发现。",
                }
            ]
        }
    )

    # 除了正常的 messages, 结果里多了 structured_response —— 一个真正的
    # ResearchReport 实例, 不需要再从模型的自由文本里解析。用 .get() 而不是
    # [...]: 前面踩坑记录里说过这一步不是 100% 稳定, 用 .get() 能在偶尔没触发时
    # 给出清楚的提示而不是让程序直接 KeyError 崩溃 (跟 langchain/ch_08_structured_output.py
    # 里的处理方式保持一致)。
    report = result.get("structured_response")
    if report is None:
        print("(这次模型没有触发结构化输出, 可以重跑试试; 见文件头的踩坑记录)")
    else:
        print(f"类型: {type(report)}")
        print(f"主题: {report.topic}")
        print("关键发现:")
        for i, finding in enumerate(report.findings, 1):
            print(f"  {i}. {finding}")
        print("来源:")
        for src in report.sources:
            print(f"  - {src}")

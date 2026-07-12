"""LangChain 结构化输出 —— 让 agent 的最终回答是一个定死结构的对象, 而不是自由文本。

对比 langchain/quickstart.py: 那里 agent 的最终回答是模型自己组织的一段文字,
每次问法不同、格式可能都不一样, 程序很难可靠地从里面提取字段;
这里给 create_agent 传 response_format, 强制模型最后一步把答案套进一个固定的
Pydantic 模型 (字段名、类型都提前定义好), 返回的 result 里会多一个
structured_response 字段, 直接就是一个 Python 对象, 可以像访问普通属性一样用
(result["structured_response"].city), 不需要自己写正则/字符串解析去"猜"模型说了什么。

这在需要把 agent 输出接入下游程序 (存数据库、调用别的 API、渲染 UI 表单) 时非常关键。

官方文档: https://docs.langchain.com/oss/python/langchain/structured-output
"""

import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain_anthropic import ChatAnthropic
from pydantic import BaseModel, Field

load_dotenv(override=True)

model = ChatAnthropic(
    model=os.environ["MODEL_ID"],
    base_url=os.getenv("ANTHROPIC_BASE_URL") or None,
)


def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"


# WeatherReport 就是我们想要的"最终答案的形状": 一个 Pydantic 模型, 字段名 + 类型 +
# 描述都提前写死。模型在结束前必须把答案填进这几个字段里, 而不是随便写一段话。
class WeatherReport(BaseModel):
    city: str = Field(description="查询的城市名")
    summary: str = Field(description="一句话总结天气情况")
    is_sunny: bool = Field(description="是否晴天")


# response_format 也可以直接传 WeatherReport 这个类本身 (框架会自动挑一种策略去实现),
# 但不同模型/场景下"自动挑的策略"不一定生效 (实测发现某些模型直接传类会导致
# structured_response 变成 None, 模型压根没有触发结构化输出这一步)。
# 用 ToolStrategy(WeatherReport) 显式声明"通过工具调用的方式产出结构化结果"更可靠:
# 相当于把 WeatherReport 也包装成一个隐藏的"工具", 模型在收尾时调用它来提交最终答案。
#
# 踩坑记录: 即使用了 ToolStrategy, 实测这里接的模型也不是每次都会主动调用这个隐藏工具
# (多跑几次会看到 structured_response 时有时无) —— 框架不会 100% 强制模型必须走
# 结构化这条路, 在 system_prompt 里明确要求"最后必须调用 WeatherReport 提交答案"之后
# 才稳定触发。这是个值得记住的经验: 结构化输出这类"框架层面的约束"未必对所有模型/
# 服务商都同样可靠, 关键约束最好在 prompt 里再显式强调一遍。
agent = create_agent(
    model=model,
    tools=[get_weather],
    system_prompt=(
        "You are a helpful assistant. After using tools, you MUST call the "
        "WeatherReport tool to submit your final structured answer instead of "
        "replying in plain text."
    ),
    response_format=ToolStrategy(WeatherReport),
)


if __name__ == "__main__":
    result = agent.invoke(
        {
            "messages": [
                {"role": "user", "content": "What's the weather in San Francisco?"}
            ]
        }
    )
    # 除了正常的 messages, 结果里多了 structured_response —— 一个真正的 WeatherReport
    # 实例 (不是字符串, 不需要再解析), 可以直接当数据用。用 .get() 而不是 [...]:
    # 万一模型这次真的没触发结构化输出 (见上面的踩坑记录), 这里能给出清楚的提示,
    # 而不是让程序直接因为 KeyError 崩溃——生产代码里对"模型不一定 100% 听话"要有心理准备。
    report = result.get("structured_response")
    if report is None:
        print("(这次模型没有触发结构化输出, 可以重跑试试; 见文件里的踩坑记录)")
    else:
        print(f"类型: {type(report)}")
        print(f"city={report.city!r}, is_sunny={report.is_sunny}")
        print(f"summary: {report.summary}")

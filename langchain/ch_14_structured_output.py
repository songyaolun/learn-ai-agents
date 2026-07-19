"""LangChain 结构化输出 —— 让 agent 的最终回答是一个定死结构的对象, 而不是自由文本。

对比 langchain/ch_17_quickstart.py: 那里 agent 的最终回答是模型自己组织的一段文字,
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

# 踩坑记录: 顶层直接 os.environ["MODEL_ID"] 会让无 .env 环境在导入阶段就 KeyError 崩掉。
# 这里改成缺省占位值, 真正触发模型调用的 agent 放到 build_agent()/build_agent_provider()
# 里惰性构造, 并在 __main__ 用 os.getenv 门控, 保证无网络自测(对比表/错误处理)可跑。
model_id = os.environ.get("MODEL_ID", "claude-3-sonnet-20240229")


def _build_model():
    return ChatAnthropic(
        model=model_id,
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
def build_agent():
    """惰性构造使用 ToolStrategy 的 agent(需 .env)。"""
    return create_agent(
        model=_build_model(),
        tools=[get_weather],
        system_prompt=(
            "You are a helpful assistant. After using tools, you MUST call the "
            "WeatherReport tool to submit your final structured answer instead of "
            "replying in plain text."
        ),
        response_format=ToolStrategy(WeatherReport),
    )


# ------------------------------
# 新增: 供应商原生策略 (ProviderStrategy)
# ------------------------------
from langchain.agents.structured_output import ProviderStrategy

# 供应商原生策略: 直接使用模型自身的结构化输出能力
# 与 ToolStrategy 不同, ProviderStrategy 依赖模型原生支持的结构化输出格式
# (如 Claude 的 structured_output 参数)
def build_agent_provider():
    """惰性构造使用 ProviderStrategy 的 agent(需 .env)。"""
    return create_agent(
        model=_build_model(),
        tools=[get_weather],
        system_prompt=(
            "You are a helpful assistant. After using tools, output your final answer "
            "as a JSON object matching the WeatherReport schema."
        ),
        response_format=ProviderStrategy(WeatherReport),
    )

# ------------------------------
# 新增: 多种结构定义形式
# ------------------------------
from dataclasses import dataclass
from typing import TypedDict, Union

# 1. 数据类 (dataclass) 结构
@dataclass
class WeatherDataClass:
    city: str
    summary: str
    is_sunny: bool

# 2. 类型字典 (TypedDict) 结构
class WeatherTypedDict(TypedDict):
    city: str
    summary: str
    is_sunny: bool

# 3. JSON Schema (dict) 结构
weather_json_schema = {
    "type": "object",
    "properties": {
        "city": {"type": "string"},
        "summary": {"type": "string"},
        "is_sunny": {"type": "boolean"}
    },
    "required": ["city", "summary", "is_sunny"]
}

# 4. 联合类型 (Union) 结构
class WeatherReportV1(BaseModel):
    city: str
    summary: str
    is_sunny: bool

class WeatherReportV2(BaseModel):
    city: str
    temperature: float
    humidity: float

WeatherUnion = Union[WeatherReportV1, WeatherReportV2]

# ------------------------------
# 新增: 错误处理
# ------------------------------
# 踩坑记录: OutputParserException 在 1.x 里从 langchain_core.exceptions 导入,
# 不在 langchain_core.output_parsers 下; 照旧写会 ImportError。
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.exceptions import OutputParserException

parser = PydanticOutputParser(pydantic_object=WeatherReport)

# 模拟解析失败场景
# 踩坑记录: pydantic v2 会把 "yes"/"true" 之类字符串强制转成 bool, 所以想触发"解析失败",
# 必须给真正非法的内容(这里 is_sunny 给一段无法转成布尔的文本)。
invalid_output = "{\"city\": \"London\", \"summary\": \"Cloudy\", \"is_sunny\": \"not-a-bool\"}"

# 错误处理示例
def handle_parsing_error(output: str) -> WeatherReport:
    """处理结构化解析失败的情况。"""
    try:
        return parser.parse(output)
    except OutputParserException as e:
        print(f"解析失败: {e}")
        # 降级处理: 返回默认值或提示用户
        return WeatherReport(city="Unknown", summary="Parsing failed", is_sunny=False)

# ------------------------------
# 新增: 多策略对比
# ------------------------------
# 策略对比表格
strategy_comparison = """\
策略对比: ToolStrategy vs ProviderStrategy
┌─────────────────┬─────────────────────────────────────────────┬─────────────────────────────────────────────┐
│ 策略类型        │ ToolStrategy                               │ ProviderStrategy                            │
├─────────────────┼─────────────────────────────────────────────┼─────────────────────────────────────────────┤
│ 实现方式        │ 通过工具调用产出结构化结果                 │ 依赖模型原生结构化输出能力                 │
│ 兼容性          │ 所有模型通用                               │ 仅支持原生支持结构化输出的模型             │
│ 可靠性          │ 较高 (显式工具调用)                       │ 依赖模型支持度                             │
│ 适用场景        │ 跨模型兼容需求                             │ 模型原生支持结构化输出的场景               │
└─────────────────┴─────────────────────────────────────────────┴─────────────────────────────────────────────┘
"""

# 结构定义形式对比
structure_comparison = """\
结构定义形式对比:
1. Pydantic 模型: 最常用, 支持类型检查和验证
2. 数据类 (dataclass): 轻量级, 适合简单结构
3. 类型字典 (TypedDict): 适合与外部系统交互
4. JSON Schema: 通用格式, 适合跨语言场景
5. 联合类型 (Union): 支持多种可能的结构
"""


if __name__ == "__main__":
    # ===== 无网络自测: 结构定义 / 错误处理 / 对比表(不触发模型调用) =====
    print("=== 无网络自测 ===")

    # 验证多种结构定义形式都能实例化
    assert WeatherDataClass(city="A", summary="s", is_sunny=True).city == "A", "dataclass 定义异常"
    td: WeatherTypedDict = {"city": "A", "summary": "s", "is_sunny": True}
    assert td["is_sunny"] is True, "TypedDict 定义异常"
    assert weather_json_schema["required"] == ["city", "summary", "is_sunny"], "JSON Schema 定义异常"
    assert WeatherReportV1(city="A", summary="s", is_sunny=True).city == "A", "Union 分支 V1 异常"
    print("✓ Pydantic / dataclass / TypedDict / JSON Schema / Union 五种结构定义均可用")

    # 错误处理: 非法输出应触发降级
    test_report = handle_parsing_error(invalid_output)
    assert test_report.city == "Unknown", "错误处理未正确降级"
    print(f"✓ 错误处理降级正常: {test_report}")

    # 打印对比表
    print("\n=== 策略对比 ===")
    print(strategy_comparison)
    print("=== 结构定义形式对比 ===")
    print(structure_comparison)

    # ===== 有网络部分(需配置 .env) =====
    print("=== 有网络部分(需配置 .env: MODEL_ID / ANTHROPIC_API_KEY) ===")
    if os.getenv("MODEL_ID") and os.getenv("ANTHROPIC_API_KEY"):
        agent = build_agent()
        result = agent.invoke(
            {
                "messages": [
                    {"role": "user", "content": "What's the weather in San Francisco?"}
                ]
            }
        )
        # 除了正常的 messages, 结果里多了 structured_response —— 一个真正的 WeatherReport
        # 实例。用 .get() 而不是 [...]: 万一模型这次没触发结构化输出(见踩坑记录), 能给出提示。
        report = result.get("structured_response")
        if report is None:
            print("(这次模型没有触发结构化输出, 可以重跑试试; 见文件里的踩坑记录)")
        else:
            print(f"类型: {type(report)}")
            print(f"city={report.city!r}, is_sunny={report.is_sunny}")
            print(f"summary: {report.summary}")
    else:
        print("跳过: 未检测到 MODEL_ID / ANTHROPIC_API_KEY, 请配置 .env 后运行。")

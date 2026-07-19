"""LangChain quickstart —— 用 create_agent 搭一个最小 agent。

官方上手指南: https://docs.langchain.com/oss/python/langchain/quickstart

对比 claude-code/ch01.py: 那里用原生 anthropic SDK 手写了 agent loop
(消息拼接、tool_use 分发、stop_reason 判断、工具结果回填),
这里用 LangChain 1.0 的 create_agent 一行就把「模型 + 工具 + 系统提示」
组装成一个可调用的 agent —— agent loop 由框架托管。
"""

import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_anthropic import ChatAnthropic

# load_dotenv 会读取项目根目录的 .env 文件, 把里面写的 KEY=VALUE 注入到当前进程的
# 环境变量里 (相当于帮你提前 export 了), override=True 表示 .env 里的值优先生效。
load_dotenv(override=True)

# ChatAnthropic 是 LangChain 对 Claude 模型的封装, 代表"一个可以对话的模型"这个概念;
# 之后 create_agent 会拿它去做实际的模型推理调用。
# 沿用 ch01-ch03 的接入方式: ANTHROPIC_BASE_URL (可选, 走自定义网关时用) + MODEL_ID。
# 鉴权默认从环境变量 ANTHROPIC_API_KEY 读取, 不需要在代码里显式传 key。
model = ChatAnthropic(
    model=os.environ["MODEL_ID"],
    base_url=os.getenv("ANTHROPIC_BASE_URL") or None,
)


def get_weather(city: str) -> str:
    """Get weather for a given city."""
    # 这只是一个普通的 Python 函数, 但把它放进 create_agent 的 tools 列表后就变成了一个
    # "工具" (tool): LangChain 会读取函数名、参数类型标注 (city: str) 和这段 docstring,
    # 自动生成一份工具描述 (JSON Schema) 发给模型, 相当于告诉模型"有个叫 get_weather 的
    # 工具, 需要一个字符串参数 city, 用途是查天气"。模型看到用户问天气时就会主动决定调用
    # 它, 而不是凭空编答案 —— 这就是"工具调用" (tool calling / function calling)。
    return f"It's always sunny in {city}!"


# create_agent 是 LangChain 1.0 的核心原语: 模型 + 工具 + 系统提示 → 一个带
# tool-calling 循环的可调用 agent。
# 这里的"循环"具体是: 模型先看用户问题 → 决定要不要调用工具 → 如果调用了, 就执行工具、
# 把结果喂回给模型 → 模型再决定是直接回答还是继续调用别的工具 → 如此反复直到模型给出
# 最终的文字回复为止。这整个过程 (对应 claude-code/ch01.py 里手写的 while 循环 + 消息
# 拼接 + tool_use 分发) 在这里全部由 create_agent 内部托管, 我们不用自己写。
agent = create_agent(
    model=model,
    tools=[get_weather],
    # system_prompt 是给模型的"人设"/行为指令, 只在对话开始时生效一次, 不是用户说的话
    system_prompt="You are a helpful assistant.",
)


if __name__ == "__main__":
    # agent.invoke(...) 是同步调用: 把输入喂进去, 等 agent 内部的循环全部跑完 (可能包含
    # 好几轮"模型决策 → 调用工具 → 模型再决策"), 一次性拿到最终结果, 中间过程不可见
    # (想看中间过程用 langchain/ch_18_stream.py)。
    #
    # 输入的标准格式是 {"messages": [...]}, 列表里每条消息是 {"role": ..., "content": ...}。
    # role 目前只需要传 "user" (代表这是用户说的话) 就够了; 模型的回复会被自动记成
    # role="assistant", 工具的执行结果会被自动记成 role="tool" —— 这些都是 agent 内部
    # 运行时自动追加进消息历史的, 我们只需要提供最初的 "user" 消息作为输入。
    result = agent.invoke(
        {
            "messages": [
                {"role": "user", "content": "What's the weather in San Francisco?"}
            ]
        }
    )
    # result["messages"] 是这一轮对话产生的完整消息列表 (可能是: 用户提问 → 模型发起
    # 工具调用 → 工具执行结果 → 模型给出的最终文字回复 这样 4 条), 最后一条就是 agent
    # 的最终回答。用 .text 而不是 .content 打印: 开启 extended thinking 的模型返回的
    # .content 是 thinking/text 混合的 block 列表, .text 只取出其中的纯文本部分。
    print(result["messages"][-1].text)

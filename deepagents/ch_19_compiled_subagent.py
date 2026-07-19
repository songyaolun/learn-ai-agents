"""DeepAgents compiled_subagent —— 用 CompiledSubAgent 把"已编译好的 runnable"当子 agent。

对比 deepagents/ch_01_quickstart.py: 那里 subagents 里的 researcher 是【声明式的 dict】
(只给 name/description/system_prompt, 由框架 create_deep_agent 在背后自动把它编译成一个
子 agent 图, 并自动接上默认中间件栈、继承主 agent 的工具/模型)。写起来最省事, 但你对
子 agent 内部长什么样几乎没有控制权。

本文件演示 CompiledSubAgent: 你【自己】先构建好一个 runnable (可以是另一个
create_deep_agent, 也可以是 langchain 的 create_agent 图, 甚至是任意 LangGraph 图),
再用 CompiledSubAgent(name=..., description=..., runnable=<你构建的图>) 塞进 subagents。
好处: 完全掌控子 agent 的模型/工具/中间件/状态, 还能复用一个"已经存在的图"; 代价:
框架不再替你自动接线 —— 你得自己保证这个 runnable 满足契约。

踩坑记录:
  1. runnable 必须"消息进、消息出": 它的 state schema 必须包含 "messages" 键 ——
     这是子 agent 把结果回传给主 agent 的唯一通道。用 create_deep_agent /
     create_agent 构建的图天然满足; 手写 LangGraph 图时忘了加 messages 就接不通。
  2. CompiledSubAgent 的 runnable 是"拿来即用"的: 它【不会】继承主 agent 的
     state_schema / 中间件 / 工具, 也【不会】被 create_deep_agent 追加任何东西。
     声明式 dict form 会自动给子 agent 加默认中间件栈、继承主 agent 工具 —— 换成
     CompiledSubAgent 后这些"自动接线"全没了, 需要你在构建 runnable 时自己配好。
  3. 结果回传规则: 子 agent 跑完后, 若其 state 里 structured_response 非空, 会被
     JSON 序列化当作 ToolMessage 内容回给主 agent; 否则取最后一条非空 AIMessage 文本。
     想要结构化回传就给内部图配 response_format。

官方文档: https://docs.langchain.com/oss/python/deepagents/subagents
"""

import os

from dotenv import load_dotenv
from deepagents import create_deep_agent, CompiledSubAgent
from langchain_core.tools import tool
from langchain_anthropic import ChatAnthropic

load_dotenv(override=True)


# ---- 一个给内部子 agent 用的小工具 ----
@tool
def word_count(text: str) -> int:
    """统计一段文本里的单词数。"""
    return len(text.split())


def build_inner_agent(model):
    """自己先构建一个"已编译的 runnable"当子 agent —— 这里复用 create_deep_agent。

    关键: 用 create_deep_agent/create_agent 构建的图, state 天然含 messages 键,
    满足 CompiledSubAgent 的"消息进消息出"契约。
    """
    return create_deep_agent(
        model=model,
        tools=[word_count],
        system_prompt=(
            "You are a text analyst subagent. Use word_count when asked about length, "
            "and return a short structured summary."
        ),
    )


def build_main_agent(model):
    """把内部 agent 作为 CompiledSubAgent 挂到主 agent 上。"""
    inner = build_inner_agent(model)
    return create_deep_agent(
        model=model,
        tools=[],
        system_prompt=(
            "You are a coordinator. Delegate any text-analysis subtask to the "
            "'analyst' subagent via the task tool."
        ),
        subagents=[
            # 与 ch_01_quickstart.py 的 dict 形式并列, 但这里给的是 runnable 形式:
            CompiledSubAgent(
                name="analyst",
                description="分析文本 (如统计长度) 并返回简短结论。把文本分析类子任务交给它。",
                runnable=inner,  # 你自己构建、自己掌控的那个图
            )
        ],
    )


if __name__ == "__main__":
    # --- 不依赖模型的结构验证 ---
    # (a) 内部 agent 能在 model=None 下构建
    inner_struct = build_inner_agent(model=None)
    assert inner_struct is not None

    # (b) CompiledSubAgent 规格良构: 三个必填键 + runnable 是可调用/可执行对象
    spec = CompiledSubAgent(
        name="analyst",
        description="分析文本并返回简短结论。",
        runnable=inner_struct,
    )
    assert spec["name"] == "analyst"
    assert spec["description"]
    assert hasattr(spec["runnable"], "invoke"), "runnable 必须是可 invoke 的图/链"

    # (c) 主 agent 能把 CompiledSubAgent 接进 subagents 并构建
    main_struct = build_main_agent(model=None)
    assert main_struct is not None
    print("结构验证通过: 内部 agent 构建 OK, CompiledSubAgent 规格良构, 主 agent 接线 OK")

    # --- 需要真实模型的部分 ---
    if os.getenv("MODEL_ID"):
        model = ChatAnthropic(
            model=os.environ["MODEL_ID"],
            base_url=os.getenv("ANTHROPIC_BASE_URL") or None,
        )
        agent = build_main_agent(model=model)
        result = agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": "帮我数一下这句话有几个单词: the quick brown fox jumps",
                    }
                ]
            }
        )
        print(result["messages"][-1].text)
    else:
        print("未配置 MODEL_ID: 跳过真实模型调用 (仅结构验证)。"
              " 接入点: 配好模型后, 主 agent 会通过 task 工具把子任务派给 analyst。")

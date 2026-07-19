"""DeepAgents runtime_context —— 用 context_schema 给工具注入"每次调用的上下文"。

对比 deepagents/quickstart.py: 那里的工具 get_weather(city: str) 只接收模型从对话里
"猜"出来的普通参数 —— 也就是说, 参数值完全由模型根据用户消息生成。这里演示的是另一
类信息: user_id、locale 这种"和这次调用绑定、但不该出现在对话消息里"的运行时上下文
(比如当前登录用户是谁、用什么语言、租户 ID 等)。我们把它定义成一个 dataclass, 通过
create_deep_agent(context_schema=...) 声明, 在 agent.invoke(payload, context=Ctx(...))
时传入, 工具则通过一个名为 runtime 的特殊参数 (类型标注 ToolRuntime) 读到它 ——
模型看不到、也改不了这份 context, 它是调用方注入的可信数据。

state vs context 的区别 (关键概念):
  - state: 随 graph 运行不断变化的可变状态 (messages、files、todos 等), 会被写回、
    会在多步之间流转; 工具可以读也可以通过返回值更新它。
  - context: 一次 invoke 期间只读的、由调用方一次性注入的配置/身份信息, 不随执行变化,
    也不进对话历史。适合放"谁在调用""什么环境"这类信息。

踩坑记录:
  1. 工具要拿到 context, 参数名必须叫 runtime、类型标注必须是 ToolRuntime
     (来自 langchain.tools), 不要用 Annotated 包装。框架靠"参数名 + 类型"识别并
     自动注入, 名字写错 (比如写成 rt) 就注入不进去。
  2. ToolRuntime 是工具专用的, 和 langgraph.runtime.Runtime 不是一个东西:
     Runtime 注入给 graph 节点/中间件, ToolRuntime 额外带了 state / tool_call_id /
     config 等工具才需要的字段。二者的 .context 指向同一份运行时上下文。
  3. context 里的字段是"模型看不到"的 —— 所以如果你希望模型据此改变行为 (比如按
     locale 切换语言), 通常还得在 system_prompt 里显式说明, 或者由工具把 context
     的值读出来再拼进返回文本。光传 context 不等于模型就会"知道"并遵守。

官方文档: https://docs.langchain.com/oss/python/deepagents/customization
"""

import os
from dataclasses import dataclass

from dotenv import load_dotenv
from deepagents import create_deep_agent
from langchain_core.tools import tool
from langchain.tools import ToolRuntime  # 工具专用运行时对象, 自动注入
from langchain_anthropic import ChatAnthropic

load_dotenv(override=True)


# ---- 1. 定义 context schema (一次调用绑定的只读身份/环境信息) ----
@dataclass
class UserContext:
    """每次 invoke 注入的运行时上下文: 当前用户是谁、用什么语言。"""

    user_id: str  # 当前登录用户 ID —— 不该由模型"编"出来, 而是调用方注入
    locale: str = "zh-CN"  # 语言/区域, 决定工具用什么语言回话


# ---- 2. 定义一个会读取 context 的工具 ----
# 注意: 除了模型可见的普通参数 (topic), 还多了一个 runtime: ToolRuntime 参数。
# runtime 不会出现在模型看到的工具签名里 —— 它由框架在执行工具时自动注入。
@tool
def make_greeting(topic: str, runtime: ToolRuntime) -> str:
    """根据当前用户上下文, 就某个话题生成一句问候语。topic 由模型给出。"""
    # runtime.context 就是这次 invoke 传进来的 UserContext 实例。
    ctx: UserContext = runtime.context
    # 把 context 里的身份/语言信息读出来, 拼进返回结果 ——
    # 这些值全程没进过对话消息, 是调用方可信注入的。
    if ctx.locale.startswith("zh"):
        return f"[用户 {ctx.user_id}] 关于「{topic}」: 你好, 这是为你准备的中文说明。"
    return f"[user {ctx.user_id}] About '{topic}': hello, here is your note in English."


# ---- 3. 把 context_schema 声明给 agent ----
# 接入点: 若无 MODEL_ID 则 ChatAnthropic 构造时读环境变量会失败, 因此 model 相关
# 部分放在 __main__ 里按需构造; 这里先给出不依赖模型也能成立的结构。
def build_agent(model):
    """用给定 model 构建带 context_schema 的 deep agent。"""
    return create_deep_agent(
        model=model,
        tools=[make_greeting],
        system_prompt=(
            "You are a helpful assistant. Use the make_greeting tool to answer. "
            "The user's identity/locale come from runtime context, not the message."
        ),
        # 关键: 声明 context 的类型, 框架据此校验 invoke 时传入的 context。
        context_schema=UserContext,
    )


if __name__ == "__main__":
    # --- 不依赖模型即可验证的结构性事实 ---
    # (a) context dataclass 能正常实例化, 字段就位
    ctx = UserContext(user_id="u_10086", locale="zh-CN")
    assert ctx.user_id == "u_10086"
    assert ctx.locale == "zh-CN"

    # (b) agent 能在 model=None 下构建 —— 证明 context_schema + 带 runtime 的工具
    #     被 create_deep_agent 正确接受 (不真正调用模型)
    agent_struct = build_agent(model=None)
    assert agent_struct is not None
    print("结构验证通过: UserContext 实例化 OK, context_schema 工具接线 OK")

    # --- 需要真实模型的部分: 仅在配置了 MODEL_ID 时运行 ---
    if os.getenv("MODEL_ID"):
        model = ChatAnthropic(
            model=os.environ["MODEL_ID"],
            base_url=os.getenv("ANTHROPIC_BASE_URL") or None,
        )
        agent = build_agent(model=model)
        # invoke 的第二个参数 context= 才是注入运行时上下文的入口 ——
        # payload 里只放对话消息, 身份/语言从 context 走。
        result = agent.invoke(
            {"messages": [{"role": "user", "content": "给我讲讲 Python 的装饰器"}]},
            context=UserContext(user_id="u_10086", locale="zh-CN"),
        )
        print(result["messages"][-1].text)
    else:
        print("未配置 MODEL_ID: 跳过真实模型调用 (仅结构验证)。"
              " 接入点: 设置 MODEL_ID / ANTHROPIC_BASE_URL 后可跑通完整 invoke。")

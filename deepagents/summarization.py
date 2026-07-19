"""DeepAgents summarization —— 上下文自动摘要 / 压缩机制, 及一个关键的"重复中间件"坑。

对比 deepagents/context_offloading.py: 那边是"主动别把大内容放进上下文"(offload 到
文件); 本文件是另一头 —— 当对话历史已经涨得太长时, 把旧消息压缩成一段摘要, 腾出
上下文空间。两者互补: 一个防患于未然, 一个事后收拾。

对比"不做任何摘要的长对话": 如果一个 agent 跑几十上百轮工具调用, 消息历史会无限
膨胀, 迟早撞上模型上下文上限 (报 context length 错误), token 成本 / 延迟也一路飙升。
摘要机制就是为了让长任务 agent 能持续跑下去。

deepagents 里"摘要 / 压缩"有两条机制, 但它们的用法【很不一样】, 这正是本文件的重点:

(1) 自动摘要 SummarizationMiddleware (deepagents 调优版): 当上下文超过 trigger 阈值时
    【自动】把旧消息抽成结构化摘要 (SESSION INTENT / SUMMARY / ARTIFACTS / NEXT STEPS
    四段), 只保留最近 keep=('messages', N) 条原样。关键事实 (实测确认):
    create_deep_agent 【默认就已经内置了一个 SummarizationMiddleware】! 所以你【不能】
    再往 middleware=[...] 里塞一个自己 new 出来的 SummarizationMiddleware —— 那会被
    LangChain 判为 "Please remove duplicate middleware instances" 而直接构造失败。
    想改它的参数, 正确姿势是: 要么用 profile 的 excluded_middleware 按名字
    ("SummarizationMiddleware" 这个 public alias) 把内置那个排除掉再换自己的, 要么把
    自定义 SummarizationMiddleware 放到 subagent 里用 (见 deepagents/subagents 相关示例)。
    直接 append 一个同类实例 = 撞车, 这是本文件要你避开的头号坑。

(2) 主动压缩工具 create_summarization_tool_middleware: 造出一个 SummarizationToolMiddleware,
    它给模型一个可【主动调用】的 compact_conversation 工具, 让模型在"该换任务、旧上下文
    没用了"时自己决定压缩。它是【不同的类】, 不和内置的自动摘要中间件撞车, 所以可以
    安全地 append 进 middleware=[...] —— 这是本文件唯一"能真的挂进 agent"的那个。

踩坑记录 (重要, 逐条):
- 头号坑: create_deep_agent 已内置 SummarizationMiddleware, 再加同类实例 =
  "duplicate middleware instances" 报错。要替换而非叠加 (excluded_middleware / subagent)。
- SummarizationMiddleware(model, backend=..., ...) 里 backend 是【必填关键字参数】,
  漏了直接构造失败。
- trigger / keep 的形状: 是 ContextSize (如 ('tokens', 120000) / ('messages', 100)) 或
  TriggerClause, 不是随便一个数字。
- 只有跨过阈值才触发: 短对话根本达不到 trigger, 一个几句话的 demo 里你【看不到】摘要
  真的发生 —— 这不是 bug, 别以为"挂了中间件每轮都摘要"。
- "参数存在 != 模型按预期使用": compact_conversation 工具挂上了, 模型也不一定主动去调,
  取决于任务和 prompt。

【诚实说明 / 已验证 vs 降级】
- 【已结构验证 (无需真实模型)】: 内置摘要的默认 agent 可构造; 单独 new 一个
  SummarizationMiddleware 也能构造 (仅对象层面); 而"往默认 agent 再 append 同类实例会
  报重复"这个坑, __main__ 里用 try/except 真实复现并断言到了那条错误信息; 追加
  compact_conversation 工具中间件的 agent 能成功构造。
- 【降级 / model-dependent】: "摘要真的触发并压缩历史"需要真实模型 + 足够长对话,
  只在有 MODEL_ID 时才尝试, 否则清晰跳过 —— 绝不谎称跑通。

官方文档: https://docs.langchain.com/oss/python/deepagents/context-engineering
"""

import os  # 环境变量

from dotenv import load_dotenv  # .env 加载
from deepagents import create_deep_agent  # 主入口; 注意它默认已内置 SummarizationMiddleware
from deepagents.backends import StateBackend  # 摘要中间件需要一个 backend
from deepagents.middleware import (
    SummarizationMiddleware,  # (1) 自动摘要中间件类 (deepagents 调优版); 默认已被内置
    create_summarization_tool_middleware,  # (2) 造出可主动调用的 compact 工具中间件
)
from langchain_anthropic import ChatAnthropic  # Anthropic 模型封装

load_dotenv(override=True)  # override=True: .env 覆盖同名环境变量


# 用字符串模型名即可构造中间件与 agent (不发起任何真实调用) —— 这让结构验证得以在
# 没有 MODEL_ID 的环境下也能进行。真正 invoke 时才需要能连通的模型。
STRING_MODEL = "anthropic:claude-sonnet-4-5"


def make_auto_summary_mw(model=STRING_MODEL):
    """新建一个'自动摘要中间件'实例 (机制 1)。

    注意: 这个实例【不能】直接 append 到 create_deep_agent(middleware=[...]),
    因为默认已内置同类中间件 —— 见文件头"头号坑"。它主要用于:
    (a) 演示构造本身; (b) 放进 subagent; (c) 配合 excluded_middleware 做替换。
    """
    return SummarizationMiddleware(
        model,  # 摘要用的模型 (字符串名即可构造)
        backend=StateBackend(),  # 必填关键字: 摘要处理所需的 backend
        trigger=("messages", 40),  # 消息数超过 40 条才自动触发摘要
        keep=("messages", 20),  # 触发后保留最近 20 条原样, 其余压成摘要
    )


def make_compact_tool_mw(model=STRING_MODEL):
    """新建一个'主动 compact 工具中间件'实例 (机制 2), 可安全 append 进 agent。"""
    # 注意这个工厂函数的 backend 是"位置参数" (紧跟在 model 后面), 跟机制 (1) 略有不同。
    return create_summarization_tool_middleware(
        model,  # 压缩用的模型
        StateBackend(),  # backend, 这里是位置参数
    )


def build_default_summary_agent():
    """最省事的用法: 什么都不加, 默认 agent 就已经带自动摘要能力 (机制 1 的内置实例)。"""
    return create_deep_agent(
        model=STRING_MODEL,
        system_prompt="You are a long-running assistant. Keep working across many steps.",
    )


def build_compact_tool_agent():
    """在默认自动摘要之外, 再给模型一个可主动调用的 compact_conversation 工具 (机制 2)。"""
    return create_deep_agent(
        model=STRING_MODEL,
        middleware=[make_compact_tool_mw()],  # 不同类, 不与内置自动摘要撞车, 可安全叠加
        system_prompt="You may call compact_conversation when past context is no longer needed.",
    )


if __name__ == "__main__":
    # ---- 结构性验证 (不需要能连通的真实模型) ----

    # 1) 两种中间件对象都能构造 (仅对象层面)。
    auto_summary_mw = make_auto_summary_mw()
    compact_tool_mw = make_compact_tool_mw()
    assert auto_summary_mw is not None, "自动摘要中间件应构造成功"
    assert compact_tool_mw is not None, "compact 工具中间件应构造成功"
    print(f"结构断言 1 通过: 两种摘要中间件均已构造 "
          f"({type(auto_summary_mw).__name__} / {type(compact_tool_mw).__name__})")

    # 2) 默认 agent 就自带自动摘要 —— 什么都不加也能构造成功。
    agent_default = build_default_summary_agent()
    assert agent_default is not None, "默认 (已内置自动摘要) 的 agent 应构造成功"
    print("结构断言 2 通过: 默认 agent 已内置自动摘要能力, 无需手动挂载")

    # 3) 复现"头号坑": 往默认 agent 再 append 一个同类 SummarizationMiddleware 会报重复。
    #    我们真实地把这个错误捕获并断言到, 以证明这个坑客观存在 (而非臆测)。
    duplicated = False
    try:
        create_deep_agent(model=STRING_MODEL, middleware=[make_auto_summary_mw()])
    except AssertionError as e:
        duplicated = "duplicate" in str(e).lower()
    assert duplicated, "预期追加同类 SummarizationMiddleware 会触发 'duplicate middleware' 报错"
    print("结构断言 3 通过: 复现头号坑 —— 追加同类 SummarizationMiddleware 确会报 'duplicate middleware'")

    # 4) 追加'不同类'的 compact_conversation 工具中间件则安全, agent 构造成功。
    agent_compact = build_compact_tool_agent()
    assert agent_compact is not None, "追加 compact 工具中间件的 agent 应构造成功"
    print("结构断言 4 通过: 追加不同类的 compact_conversation 工具中间件, agent 构造成功")

    # ---- 模型相关: 真正触发摘要需要真实模型 + 足够长对话, 有 MODEL_ID 才尝试 ----
    if os.getenv("MODEL_ID"):
        # 用默认内置摘要的 agent 直接跑一轮 (短对话通常不会跨过阈值, 只是演示链路通)。
        real_model = ChatAnthropic(
            model=os.environ["MODEL_ID"],
            base_url=os.getenv("ANTHROPIC_BASE_URL") or None,
        )
        agent = create_deep_agent(model=real_model)  # 默认即带自动摘要
        result = agent.invoke(
            {"messages": [{"role": "user", "content": "用一句话介绍一下上下文摘要的作用。"}]}
        )
        print("=== agent 最终回复 ===")
        print(result["messages"][-1].text)  # 读回复用 .text 而非 .content
        print("[说明] 是否真的触发压缩取决于是否跨过 trigger 阈值, 短对话通常未触发, 属正常。")
    else:
        print("[跳过] 未检测到 MODEL_ID, 跳过'真实触发摘要'的部分 (model-dependent)。")

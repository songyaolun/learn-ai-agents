"""DeepAgents profiles —— 用 HarnessProfile / ProviderProfile 重塑框架注入的东西。

对比 deepagents/quickstart.py: 那里用的是"默认 harness" —— create_deep_agent 背后自动
注入的那套 system prompt、工具集 (todo/filesystem/task 等)、中间件栈, 你没去动它。
本文件演示 profiles: 一套"按 provider 或 provider:model 注册的定制配置", 让你在【不改
每次 create_deep_agent 调用】的前提下, 全局改变框架给这个模型注入什么。

两类 profile 分工明确 (关键概念):
  - HarnessProfile: 定制"agent 怎么跑" —— 系统提示后缀 (system_prompt_suffix)、
    隐藏哪些工具 (excluded_tools)、改工具描述 (tool_description_overrides)、加/减
    中间件、默认 general-purpose 子 agent 开关等。由 create_deep_agent 在"模型已构建好"
    之后消费。
  - ProviderProfile: 定制"模型怎么建" —— 传给 init_chat_model 的 init_kwargs
    (temperature/max_tokens/base_url...)、构建前的 pre_init 钩子、动态 kwargs 工厂。
    影响的是模型客户端的构造阶段。
  用 register_harness_profile(key, ...) / register_provider_profile(key, ...) 注册,
  key 可以是 provider ("anthropic") 或 provider:model ("anthropic:xxx")。

踩坑记录:
  1. "隐藏了某工具 / 加了提示后缀"这类【行为差异】很难在没有真实模型的情况下断言 ——
     你没法离线证明"模型确实看不到 grep 了"。但 profile【对象本身】和【注册动作】是
     完全可离线验证的 (构造 + register 不报错 + 字段就位), 本文件的自测就断言这些。
  2. profile 对象是 frozen + 只读容器: 构造后再改 tool_description_overrides["ls"]=...
     会抛 TypeError; init_kwargs 同理。想改只能重新 register 或重建一个 profile。
  3. excluded_middleware 有护栏: 不能排除脚手架类 (FilesystemMiddleware /
     SubAgentMiddleware), 构造时就会 ValueError。想去掉 task 工具不要走这里, 而是把
     general_purpose_subagent 设为 disabled 且不传同步 subagents。
  4. tool_description_overrides 覆盖 "task" 工具时, 描述里必须保留 {available_agents}
     占位符, 否则模型看不到有哪些子 agent, task 工具基本作废 —— 参数能设 != 效果符合
     预期, 这是最隐蔽的一个坑。

官方文档: https://docs.langchain.com/oss/python/deepagents/customization
"""

import os

from dotenv import load_dotenv
from deepagents import (
    create_deep_agent,
    HarnessProfile,
    ProviderProfile,
    GeneralPurposeSubagentProfile,
    register_harness_profile,
    register_provider_profile,
)

load_dotenv(override=True)


# ---- 1. 定制一个 HarnessProfile: 重塑"agent 怎么跑" ----
def build_harness_profile() -> HarnessProfile:
    """构建一个自定义 harness profile。"""
    return HarnessProfile(
        # 在框架基础系统提示末尾追加模型专属指令 (最靠近对话历史, 权重高)
        system_prompt_suffix="始终用简体中文回答, 并把答案控制在 100 字以内。",
        # 从工具集里隐藏 grep (built-in 文件系统工具之一) —— 模型将看不到它
        excluded_tools=frozenset({"grep"}),
        # 改写 ls 工具的描述 (不影响 task 工具, 所以不需要 {available_agents} 占位符)
        tool_description_overrides={"ls": "列出当前工作区里的文件 (自定义描述)。"},
    )


# ---- 2. 定制一个 ProviderProfile: 重塑"模型怎么建" ----
def build_provider_profile() -> ProviderProfile:
    """构建一个自定义 provider profile, 设置构建模型时的默认 init kwargs。"""
    init_kwargs = {"temperature": 0.2, "max_tokens": 1024}
    if os.getenv("ANTHROPIC_BASE_URL"):
        init_kwargs["base_url"] = os.environ["ANTHROPIC_BASE_URL"]
    return ProviderProfile(init_kwargs=init_kwargs)


def build_agent(model):
    """用默认方式构建 agent —— profile 通过全局注册生效, 无需在这里显式传入。"""
    return create_deep_agent(
        model=model,
        tools=[],
        system_prompt="You are a helpful assistant.",
    )


if __name__ == "__main__":
    # === 无需模型的真实结构性测试: 构造 profile 对象 + 调 register_* ===
    hp = build_harness_profile()
    # 字段就位
    assert hp.system_prompt_suffix.startswith("始终用简体中文")
    assert "grep" in hp.excluded_tools
    assert hp.tool_description_overrides["ls"].startswith("列出")

    # frozen/只读校验: 构造后不能再改容器内容 (会抛 TypeError)
    try:
        hp.tool_description_overrides["ls"] = "偷偷改"
        raise AssertionError("tool_description_overrides 本应只读")
    except TypeError:
        pass  # 符合预期

    # 护栏校验: 排除脚手架类 SubAgentMiddleware 应在构造时就 ValueError
    try:
        HarnessProfile(excluded_middleware=frozenset({"SubAgentMiddleware"}))
        raise AssertionError("排除脚手架中间件本应报错")
    except ValueError:
        pass  # 符合预期

    # 真正注册 harness profile (key 用 provider:model 形式; 这里用演示 key)
    register_harness_profile("anthropic:profiles-demo-model", hp)

    # provider profile 构造 + 只读校验 + 注册
    pp = build_provider_profile()
    assert pp.init_kwargs["temperature"] == 0.2
    assert pp.init_kwargs["max_tokens"] == 1024
    try:
        pp.init_kwargs["temperature"] = 2.0
        raise AssertionError("init_kwargs 本应只读")
    except TypeError:
        pass
    register_provider_profile("anthropic:profiles-demo-model", pp)
    if os.getenv("MODEL_ID"):
        # 真实模型路径也注册到当前 MODEL_ID, 下面把字符串传给 create_deep_agent,
        # 让 DeepAgents 走 string-model resolution, 从而真正消费 ProviderProfile。
        register_provider_profile(os.environ["MODEL_ID"], pp)

    # general-purpose 子 agent 开关也能构造 (演示禁用默认子 agent 的写法)
    gp_off = HarnessProfile(general_purpose_subagent=GeneralPurposeSubagentProfile(enabled=False))
    assert gp_off.general_purpose_subagent.enabled is False

    # agent 能在 model=None 下构建
    agent_struct = build_agent(model=None)
    assert agent_struct is not None

    print("结构验证通过: HarnessProfile/ProviderProfile 构造 + 只读校验 + 护栏校验 + "
          "register_* 注册全部通过 (无需模型)")

    # === 需要真实模型才能观察"行为差异"的部分 ===
    if os.getenv("MODEL_ID"):
        agent = build_agent(model=os.environ["MODEL_ID"])
        result = agent.invoke({"messages": [{"role": "user", "content": "介绍一下你自己"}]})
        print(result["messages"][-1].text)
        print("(提示: profile 是否真的隐藏了 grep/追加了中文后缀, 属于模型行为,"
              " 需在真实运行中观察, 客户端无法离线断言。)")
    else:
        print("未配置 MODEL_ID: 跳过真实模型调用 (profile 对象与注册已完成无模型验证)。")

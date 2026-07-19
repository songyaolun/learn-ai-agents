"""DeepAgents prompt_caching —— 用 Anthropic 提示缓存降低大 system_prompt 的成本/延迟。

对比 deepagents/ch_01_quickstart.py: 那里的 system_prompt 只有一句
"You are a helpful research assistant.", 短到没有缓存的必要。真实项目里 system_prompt
往往很大 (几千 token 的规范、示例、工具说明), 每一轮对话都把这一大坨重新发给模型 =
每轮都为同样的前缀重复付费 + 重复排队。Anthropic 支持"提示缓存 (prompt caching)":
在 system 内容块上打一个 cache_control ephemeral 标记, 命中缓存的部分按更低价计费、
延迟也更低。本文件演示如何把一段大的静态 system_prompt 结构化成带 cache_control 的
内容块, 并接到 create_deep_agent 上。

【降级说明 (诚实标注)】提示缓存是"供应商侧、尽力而为"的能力:
  - 是否真的命中缓存由 Anthropic 服务端决定, 客户端无法可靠断言"这次命中了缓存"。
  - 命中与否还取决于前缀是否逐字节一致、缓存是否过期 (ephemeral 约 5 分钟) 等。
  - 不同模型/网关暴露的 usage 缓存指标 (cache_creation/cache_read tokens) 不统一,
    甚至可能不返回。
  因此本文件属于【部分降级】: 结构 (带 cache_control 的 prompt + agent 构建) 可离线
  验证; 但"缓存确实命中并省钱"这一点无法在客户端断言, 也不做任何"已验证命中"的声称。

踩坑记录:
  1. 不要指望 create_deep_agent 有个 "enable_cache" 开关 —— 提示缓存没有 deepagents
     专属参数, 它完全是"消息/内容块上的 cache_control"这一 provider 机制。
  2. cache_control 要打在"足够大且稳定不变"的前缀块上才有意义; 打在小块或每轮都变的
     内容上, 既省不了钱还可能白占一次 cache 断点 (Anthropic 每个请求 cache 断点数量
     有限)。所以把"静态大规范"和"每轮动态部分"拆成不同块, 只给静态块打标记。
  3. 即使打了标记, 也可能因为前缀没对齐、缓存过期而不命中 —— 客户端看到的行为可能和
     不加缓存时"看起来一样", 这不代表标记没生效, 只是命中与否你无法从返回里稳定确认。

官方文档: https://docs.langchain.com/oss/python/deepagents/quickstart
"""

import os

from dotenv import load_dotenv
from deepagents import create_deep_agent
from langchain_core.messages import SystemMessage
from langchain_anthropic import ChatAnthropic

load_dotenv(override=True)

# ---- 1. 构造一段"大的、静态的"系统规范 ----
# 真实场景里这可能是几千 token 的编码规范/领域知识; 这里用重复文本模拟"很大且固定"。
LARGE_STATIC_SPEC = (
    "You are a senior research assistant operating under a detailed style guide.\n"
    + "\n".join(f"- Rule {i}: keep answers factual, cite assumptions, be concise." for i in range(1, 60))
    + "\nAlways plan before answering complex questions."
)


# ---- 2. 把系统提示结构化成带 cache_control 的内容块 ----
# LangChain 的 SystemMessage.content 可以是"内容块列表", 每个块是一个 dict。
# 接入点: cache_control 标记就打在这个静态大块上 —— Anthropic 会尝试缓存"到这里为止
# 的前缀"。类型 "ephemeral" 表示短期缓存 (约 5 分钟窗口)。
def build_cached_system_message() -> SystemMessage:
    """返回一个把大规范标记为可缓存的 SystemMessage。"""
    return SystemMessage(
        content=[
            {
                "type": "text",
                "text": LARGE_STATIC_SPEC,
                # 接入点: 这一行是提示缓存的核心 —— 只给"静态大前缀"打标记
                "cache_control": {"type": "ephemeral"},
            },
        ]
    )


def build_agent(model):
    """用带缓存标记的 system 消息构建 deep agent。"""
    # 注意: 这里把带 cache_control 的内容作为 system_prompt 传入。
    # create_deep_agent 的 system_prompt 接受字符串; 若要传结构化内容块, 一种稳妥的
    # 做法是把纯文本给 system_prompt, 并在首轮 messages 里附带带 cache_control 的
    # SystemMessage —— 下面 __main__ 演示后者 (更能体现 cache_control 落点)。
    return create_deep_agent(
        model=model,
        tools=[],
        system_prompt="You are a senior research assistant.",
    )


if __name__ == "__main__":
    # --- 不依赖模型的结构验证 ---
    sys_msg = build_cached_system_message()
    # 断言 content 是"内容块列表"而不是普通字符串, 且第一块带 cache_control ephemeral
    assert isinstance(sys_msg.content, list), "缓存版 system 应为内容块列表"
    first_block = sys_msg.content[0]
    assert first_block["type"] == "text"
    assert first_block["cache_control"] == {"type": "ephemeral"}, "静态块应打上 ephemeral 缓存标记"
    assert len(first_block["text"]) > 1000, "被缓存的前缀应该足够大才划算"

    agent_struct = build_agent(model=None)
    assert agent_struct is not None
    print("结构验证通过: cache_control ephemeral 标记就位, 大前缀 > 1000 字符, agent 构建 OK")

    # --- 需要真实模型的部分 ---
    if os.getenv("MODEL_ID"):
        model = ChatAnthropic(
            model=os.environ["MODEL_ID"],
            base_url=os.getenv("ANTHROPIC_BASE_URL") or None,
        )
        agent = build_agent(model=model)
        # 把带 cache_control 的 SystemMessage 放进 messages 首位, 后面跟用户问题。
        # 是否真正命中缓存由服务端决定 —— 这里不做"命中"断言, 只演示接线方式。
        result = agent.invoke(
            {
                "messages": [
                    build_cached_system_message(),
                    {"role": "user", "content": "用两句话解释什么是提示缓存。"},
                ]
            }
        )
        print(result["messages"][-1].text)
        # 诚实提示: 无法从客户端可靠断言缓存命中; 如需观察, 请查看服务端 usage 里的
        # cache_creation_input_tokens / cache_read_input_tokens (若模型/网关返回的话)。
        print("(注意: 缓存是否命中由 Anthropic 服务端决定, 客户端无法可靠断言。)")
    else:
        print("未配置 MODEL_ID: 跳过真实模型调用 (仅结构验证)。"
              " 接入点: 配好 Anthropic 模型后, cache_control 才会实际参与服务端缓存。")

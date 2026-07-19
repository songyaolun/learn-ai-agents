"""DeepAgents context offloading —— 用虚拟文件系统给 LLM 上下文"减负"。

对比 deepagents/ch_03_filesystem.py: 那里演示的是"虚拟文件系统有哪些工具、怎么读写";
本文件用的是完全一样的默认 StateBackend 和一样的 write_file/read_file 工具, 但
关注点不同 —— 讲的是 context offloading (上下文卸载) 这个动机: 把体积大的中间产物
(长搜索结果、大段原文、爬来的网页正文等) 写进文件, 在对话消息里只留一个"文件路径
指针 + 一句话摘要", 而不是把整坨内容原样粘回消息历史。

为什么要这么做: LLM 的上下文窗口有限, 而且越长越贵、越慢、越容易"迷失在中间"
(lost in the middle)。如果每一步都把大段中间内容塞进 messages, token 会迅速膨胀。
把大内容 offload 到文件后, 模型后续需要时再用 read_file 按需取回 (甚至只读某一段),
messages 里始终保持精简 —— 这是 DeepAgents 管理长任务上下文的核心手法之一, 也和
langchain 的自动摘要 (见 deepagents/ch_14_summarization.py) 互补: offload 是"主动别放进来",
摘要是"太多了再压缩"。

用的是默认 StateBackend (零外部依赖): 文件存在 LangGraph state 里, 不落磁盘、不连
任何服务, 所以本文件不需要 tempfile 沙箱 (没有真实副作用)。

踩坑记录: 模型到底会不会"乖乖 offload"是高度 prompt 敏感的 —— 参数 / 工具存在不等于
模型按预期使用。如果 system_prompt 不明确要求"大内容写文件、回复里只留指针", 模型
往往图省事直接把全部内容 inline 进回复, 达不到卸载效果。所以下面的 system_prompt 写得
比较"硬": 明确要求先 write_file 再只回摘要 + 路径。即便如此也不能 100% 保证, 这是
使用这类"软约束"时要有的心理预期。

【诚实说明】完整链路 (模型真的去 write_file 并只留指针) 需要 MODEL_ID, 本地没有,
属 model-dependent。__main__ 里 agent 构造是可无模型验证的; 真正的 invoke + 检查
result["files"] 里出现被卸载的文件, 只在有 MODEL_ID 时才跑, 否则清晰跳过。

官方文档: https://docs.langchain.com/oss/python/deepagents/filesystem
"""

import os  # 读取 MODEL_ID / ANTHROPIC_BASE_URL

from dotenv import load_dotenv  # 从 .env 加载模型配置
from deepagents import create_deep_agent  # 主入口; 不传 backend 即用默认 StateBackend
from langchain_anthropic import ChatAnthropic  # Anthropic 模型封装

load_dotenv(override=True)  # override=True: .env 覆盖同名环境变量


def build_model():
    """单独包一层模型构造: 本地没 MODEL_ID 时不至于一 import 就抛异常。"""
    return ChatAnthropic(
        model=os.environ["MODEL_ID"],  # 模型名只从环境变量取, 不硬编码
        base_url=os.getenv("ANTHROPIC_BASE_URL") or None,  # 没配就 None
    )


# 不传 backend: 默认就是 StateBackend —— 文件存活在 LangGraph state 里, 零外部依赖。
# system_prompt 是本文件的灵魂: 明确要求"大内容写文件, 回复里只保留短指针",
# 这正是 context offloading 的行为约定 (但注意: 只是软约束, 见踩坑记录)。
OFFLOAD_PROMPT = (
    "You are a research assistant that practices CONTEXT OFFLOADING. "
    "When you produce any large intermediate content (long text, raw findings, "
    "detailed notes), you MUST save it with write_file to a file named findings.md "
    "instead of pasting it into your reply. In your final reply, keep ONLY a short "
    "2-3 sentence summary plus the file path (e.g. 'Full details saved to findings.md'). "
    "Do NOT inline the full content into the message."
)


def build_agent():
    """构造一个会把大内容卸载到文件的 agent (需要模型)。"""
    return create_deep_agent(
        model=build_model(),  # 只有 invoke 时才真正用到模型
        system_prompt=OFFLOAD_PROMPT,  # 硬性要求 offload 行为
    )


if __name__ == "__main__":
    if os.getenv("MODEL_ID"):
        # ---- 有模型: 跑完整链路, 并验证大内容确实被卸载进了文件 ----
        agent = build_agent()
        assert agent is not None, "agent 应构造成功"
        print("结构断言通过: offloading agent 构造成功, 开始 invoke...")
        result = agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            "详细整理一份'Python 与 JavaScript 的十点区别'的资料。"
                            "按要求把完整内容存文件, 回复里只给我摘要和文件名。"
                        ),
                    }
                ]
            }
        )
        # 读回复用 .text (不是 .content)。理想情况下这里应该很短 —— 只有摘要 + 指针。
        print("=== agent 最终回复 (应是精简的摘要 + 文件指针) ===")
        print(result["messages"][-1].text)

        # 关键验证: 大内容被 offload 到了虚拟文件系统 —— result["files"] 里应能看到它。
        files = result.get("files", {})
        print(f"\n=== 虚拟文件系统里现存文件: {list(files.keys())} ===")
        assert files, "期望模型把大内容卸载成了至少一个文件 (若为空, 说明模型没按 prompt offload)"
        print("验证通过: 大内容已卸载到虚拟文件, 消息历史得以保持精简。")
    else:
        # ---- 无模型: 仅做无副作用的结构验证, 并清楚说明为何不跑 invoke ----
        agent_prompt_ok = "OFFLOADING" in OFFLOAD_PROMPT.upper()
        assert agent_prompt_ok, "system_prompt 应包含 offloading 指令"
        print("结构断言通过: offloading system_prompt 已就绪 (含'大内容写文件、只回指针'约定)。")
        print("[跳过] 未检测到 MODEL_ID, 跳过 invoke 与 result['files'] 检查 (model-dependent)。")

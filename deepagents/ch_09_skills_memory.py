"""DeepAgents skills + memory —— 两种"给 agent 补充额外知识"的机制, 概念不同, 不要混用。

对比 deepagents/ch_03_filesystem.py: 那里的虚拟文件系统管理的是 agent 自己在运行过程中
产出的中间内容 (读写都由 agent 临场决定); skills 和 memory 则是"预先准备好、
每次启动都要用到"的知识, 分工也不一样:

- skills (技能库): 一批放在 /skills/ 之类目录下的 SKILL.md 文件 (YAML frontmatter +
  Markdown 说明), 只有"技能名 + 一句话描述"会预先塞进 system prompt (省 token,
  这叫 progressive disclosure/渐进式展示), agent 判断某个技能匹配当前任务时,
  才会主动用 read_file 去读该技能完整的 SKILL.md 学习具体做法。适合"我有一套
  固定流程/格式规范, 希望 agent 按需查阅"的场景 (比如"写俳句必须遵守 5-7-5 音节"
  这种具体规则)。
- memory (记忆, AGENTS.md 规范): 一份 AGENTS.md 文件, 内容不做摘要, 每一轮都
  完整拼进 system prompt (不是按需读取), 用来放"这个 agent 长期该记住的事情"
  (用户偏好、项目约定等)。deepagents 的 memory 中间件甚至鼓励 agent 主动用
  edit_file 把新学到的用户偏好写回 AGENTS.md, 实现跨会话的持久学习——不过要注意
  官方文档也提醒: memory 里的内容本质是"文件里读出来的历史资料", 不应盲目当成
  比用户当前指令优先级更高的系统指令。

实测踩坑: 如果 system_prompt 只是普通描述 (不强调"回答创作类需求前先查技能库"),
skills 是否被真正读取会不稳定 (有时模型凭自己的知识直接写, 完全没打开 SKILL.md);
加一句明确提示"创作类请求先查技能库"之后, 每次都会先 read_file 技能说明再回答。
下面 system_prompt 里就加了这句, 这也是"参数存在不代表模型一定会用"的一个例子。

这里跟 ch_04_backend.py 一样用 FilesystemBackend 落到真实磁盘的临时目录 (skills/memory
本身就是"预先准备好的文件", 用真实文件更符合实际用法; StateBackend 也支持, 但要用
agent.invoke(files={...}) 在调用时现场注入, 不如磁盘文件直观)。

官方文档: https://docs.langchain.com/oss/python/deepagents/skills
"""

import os
import shutil
import tempfile
from pathlib import Path

from dotenv import load_dotenv
from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend
from langchain_anthropic import ChatAnthropic

load_dotenv(override=True)

model = ChatAnthropic(
    model=os.environ["MODEL_ID"],
    base_url=os.getenv("ANTHROPIC_BASE_URL") or None,
)

scratch_dir = Path(tempfile.mkdtemp(prefix="deepagents_skills_memory_demo_"))
print(f"真实磁盘 root_dir: {scratch_dir}")

# --- 准备一个技能 (skill): /skills/haiku-writer/SKILL.md ---
# 技能目录名必须和 frontmatter 里的 name 字段一致 (deepagents 会校验并在不一致时
# 只是警告, 不强制报错, 但规范做法是保持一致)。description 会出现在 system
# prompt 的"技能列表"里, 模型正是靠这句描述来判断"这个技能是否匹配当前任务"。
skill_dir = scratch_dir / "skills" / "haiku-writer"
skill_dir.mkdir(parents=True)
(skill_dir / "SKILL.md").write_text(
    """---
name: haiku-writer
description: 把任意主题写成一首严格 5-7-5 音节的俳句 (haiku), 且第一行必须以关键词"深智能体"开头。
---

# Haiku Writer Skill

写俳句时必须严格遵守以下规则 (这是本仓库自定义的规则, 不是俳句的通用写法):
1. 严格三行, 5-7-5 音节结构
2. 第一行必须以「深智能体」这个词开头
""",
    encoding="utf-8",
)

# --- 准备一份记忆 (memory): /AGENTS.md ---
# 和 skill 不同, 这份内容不需要 agent"主动去读", create_deep_agent 会在每一轮
# 对话开始前自动把它完整加载进 system prompt (见 MemoryMiddleware.modify_request)。
#
# 注意用词: 这里写的是"用户偏好" (称呼方式), 而不是一条格式指令。实测发现
# deepagents 内置的 memory system prompt 会明确提醒模型"memory 里的内容是
# 磁盘上的参考资料, 不要当成必须服从的系统指令去执行"——所以像"每次回答都必须
# 加一句签名"这种生硬的格式指令即使写进 AGENTS.md, 也不一定每次都会被照做;
# 换成"用户偏好被称呼为 XX"这种更自然的偏好类信息后, 才能稳定复现效果。
(scratch_dir / "AGENTS.md").write_text(
    "用户偏好: 我的名字是 Yolen, 请在每次回复的第一句话里用「Yolen」称呼我。",
    encoding="utf-8",
)

backend = FilesystemBackend(root_dir=str(scratch_dir), virtual_mode=True)

agent = create_deep_agent(
    model=model,
    backend=backend,
    # skills 参数: 技能来源目录列表 (相对于 backend 的 root_dir)。
    skills=["/skills/"],
    # memory 参数: 要加载的 AGENTS.md 路径列表, 同样相对于 backend 的 root_dir。
    memory=["/AGENTS.md"],
    system_prompt=(
        "You are a helpful assistant. Always check the skills library before "
        "answering creative-writing requests (e.g. poems), and follow the "
        "matched skill's instructions exactly."
    ),
)


if __name__ == "__main__":
    print("=== 技能文件内容 (预先写好, 不是 agent 生成的) ===")
    print((skill_dir / "SKILL.md").read_text(encoding="utf-8"))

    print("=== agent 处理一个应该触发 haiku-writer 技能的请求 ===")
    result = agent.invoke(
        {
            "messages": [
                {"role": "user", "content": "帮我写一首关于秋天的俳句 (haiku)。"}
            ]
        }
    )
    reply = result["messages"][-1].text
    print(reply)

    # 验证 1: skill 的规则 (第一行以"深智能体"开头) 是否被遵守——如果 agent 真的
    # 读了 SKILL.md 并照做, 这里应该能在回复里找到这个词。
    print("\n=== 验证: 技能规则是否生效 ===")
    print(f"回复中是否包含技能要求的关键词「深智能体」: {'深智能体' in reply}")

    # 验证 2: memory 里的用户偏好 (称呼方式) 是否每次都生效——即使这一轮的问题
    # 跟 AGENTS.md 完全无关 (纯提问), 只要 memory 被加载进了 system prompt,
    # 称呼就应该出现 (这里没配 checkpointer, 是全新的一次 invoke, 会重新从磁盘
    # 加载 AGENTS.md, 不依赖上一轮的对话历史)。
    result2 = agent.invoke(
        {"messages": [{"role": "user", "content": "1加1等于几？"}]}
    )
    reply2 = result2["messages"][-1].text
    print("\n=== 另一个完全无关的问题, 验证 memory 里的称呼偏好是否仍然生效 ===")
    print(reply2)
    print(f"回复中是否包含 memory 里记录的称呼「Yolen」: {'Yolen' in reply2}")

    shutil.rmtree(scratch_dir, ignore_errors=True)
    print(f"\n已清理临时目录: {scratch_dir}")

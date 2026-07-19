"""DeepAgents store memory —— 用 StoreBackend + LangGraph BaseStore 做跨线程持久记忆。

对比 deepagents/ch_04_backend.py: FilesystemBackend 也能"持久化" (写真实磁盘), 但它绑死在
一台宿主机上, 换机器 / 换部署就带不走, 且天然是"目录+文件"模型。本文件的 StoreBackend
把 agent 的文件存进一个 LangGraph BaseStore (键值存储) 里 —— 只要多个会话 (线程)
共享同一个 store, 线程 A 写的文件, 线程 B (不同 thread_id) 也能读到。这才是"跨线程
共享记忆"的正解。

对比 deepagents/ch_09_skills_memory.py: 那里的 AGENTS.md memory 是"每次加载时把一份文件内容
拼进 system prompt", 它是"随 agent 启动读进来的静态资料", 不是一个可增删改查、能跨
会话累积的键值存储。想让 agent"这次记下的东西, 下次别的会话还能用", 要用本文件的
StoreBackend, 而不是 AGENTS.md。

核心对照 (本文件要证明的点): 默认 StateBackend 的文件只活在【单次 graph 运行的 state】
里, 换个 thread_id 就互相看不见 (线程隔离); StoreBackend 因为文件落在【共享的 store】
里, 所以跨 thread_id 可见 (线程共享)。这一"隔离 vs 共享"的差别就是全部意义所在。

wiring 说明 (已从解释器实测确认): StoreBackend 可以两种方式拿到 store ——
(a) 直接 StoreBackend(store=my_store) 显式传入; 或
(b) 只写 backend=StoreBackend(), 由 create_deep_agent(store=my_store) 把 store 通过
    runtime 注入。本文件的结构验证走 (a) (可脱离模型直接读写); agent 装配演示走 (b)。

踩坑记录:
- 没有 store 就没处存: StoreBackend 若既没显式传 store、运行时也拿不到 store, 就没有
  任何持久化载体 —— "记忆"无从谈起。用它就必须配一个 store (本地用 InMemoryStore,
  生产可换 Postgres 等持久化实现)。
- 隔离 vs 共享是个"微妙点": 换 thread_id 时, StateBackend 的文件会消失, StoreBackend
  的文件还在 —— 这不是 bug, 正是两者定位的差别。
- InMemoryStore 顾名思义只活在进程内存里: 进程一退出照样丢, 它只用于演示 store 的
  "跨线程共享"语义; 想跨进程 / 重启还在, 得换成真正落盘的 store 实现。

【诚实说明】本文件的核心对照 (跨"线程"共享) 可以【不经过模型】直接用 backend 的
read/write 结构性验证 —— __main__ 里【已真实运行并断言】: 用同一个 store 建两个
StoreBackend (模拟两个线程), 线程 A 写、线程 B 读到。而"agent 通过模型自动读写记忆"的
完整链路需要 MODEL_ID, 属 model-dependent, 有则尝试、无则清晰跳过, 不谎称跑通。

官方文档: https://docs.langchain.com/oss/python/deepagents/backends
"""

import os  # 环境变量

from dotenv import load_dotenv  # .env 加载
from deepagents import create_deep_agent  # 主入口, 支持 backend= 与 store=
from deepagents.backends import StoreBackend  # 由 LangGraph store 支撑的跨线程 backend
from langgraph.store.memory import InMemoryStore  # 进程内共享的 store 实现 (演示用)
from langchain_anthropic import ChatAnthropic  # Anthropic 模型封装

load_dotenv(override=True)  # override=True: .env 覆盖同名环境变量


def build_agent_with_store(store):
    """把一个共享 store 装配进 agent (wiring 方式 b: create_deep_agent(store=...))。"""
    model = ChatAnthropic(
        model=os.environ["MODEL_ID"],  # 模型名只从环境变量取
        base_url=os.getenv("ANTHROPIC_BASE_URL") or None,
    )
    # backend=StoreBackend() 不显式传 store, 由 create_deep_agent(store=store) 注入。
    return create_deep_agent(
        model=model,
        backend=StoreBackend(),  # 运行时从 runtime 拿到下面传入的 store
        store=store,  # ← 共享的 store, 是"跨线程记忆"的载体
        system_prompt=(
            "You are an assistant with persistent memory. Use write_file to remember "
            "facts the user tells you, and read_file to recall them later."
        ),
    )


if __name__ == "__main__":
    # ---- 核心结构验证 (不需要模型): 用同一个 store 模拟两个线程, 证明"共享可见" ----
    shared_store = InMemoryStore()  # 两个"线程"共用的同一个 store

    # 线程 A: 建一个绑定该 store 的 StoreBackend, 写入一条记忆。
    backend_thread_a = StoreBackend(store=shared_store)
    write_res = backend_thread_a.write("memory.txt", "用户偏好: 回答尽量简洁, 用中文。")
    assert write_res.error is None, f"线程 A 写入应成功, 实际: {write_res.error}"
    print("结构断言 1 通过: 线程 A 已把一条记忆写入共享 store")

    # 线程 B: 另建一个 StoreBackend, 但绑定【同一个】 shared_store (模拟不同 thread_id)。
    backend_thread_b = StoreBackend(store=shared_store)
    read_res = backend_thread_b.read("memory.txt")
    # 关键点: 线程 B 没写过这个文件, 却能读到线程 A 写的内容 —— 因为 store 是共享的。
    assert read_res.error is None, f"线程 B 读取应成功, 实际: {read_res.error}"
    assert "简洁" in read_res.file_data["content"], "线程 B 应读到线程 A 写的记忆内容"
    print("结构断言 2 通过: 线程 B (不同实例/共享 store) 读到了线程 A 写的记忆 —— 跨线程共享成立")
    print(f"    线程 B 读到的内容: {read_res.file_data['content']!r}")

    # 反面对照 (概念说明, 不实跑): 若换成默认 StateBackend, 文件只活在单次运行 state 里,
    # 换个 thread_id 就读不到 —— 那是"线程隔离", 而这里是"线程共享", 差别就在 store。
    print("对照说明: 换成 StateBackend 时文件随 thread 隔离, 换 thread_id 即不可见 (故需 store)。")

    # ---- 模型相关: agent 自动读写记忆的完整链路需要模型 ----
    if os.getenv("MODEL_ID"):
        store = InMemoryStore()
        agent = build_agent_with_store(store)
        assert agent is not None, "带 store 的 agent 应构造成功"
        print("\n结构断言 3 通过: 带共享 store 的 agent 构造成功, 开始 invoke...")
        # 线程 A 的会话: 通过 config 指定 thread_id, 让模型记住一件事。
        cfg_a = {"configurable": {"thread_id": "thread-A"}}
        agent.invoke(
            {"messages": [{"role": "user", "content": "请记住: 我最喜欢的语言是 Python。存到记忆里。"}]},
            config=cfg_a,
        )
        # 线程 B 的会话: 换一个 thread_id, 看模型能否从共享 store 里回忆起来。
        cfg_b = {"configurable": {"thread_id": "thread-B"}}
        result_b = agent.invoke(
            {"messages": [{"role": "user", "content": "读一下记忆, 我最喜欢的语言是什么?"}]},
            config=cfg_b,
        )
        print("=== 线程 B 的回复 (应能从共享 store 回忆出 Python) ===")
        print(result_b["messages"][-1].text)  # 读回复用 .text 而非 .content
    else:
        print("\n[跳过] 未检测到 MODEL_ID, 跳过'agent 通过模型自动读写记忆'的链路 (model-dependent)。")

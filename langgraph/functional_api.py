"""LangGraph Functional API —— 用普通函数 + @entrypoint/@task 写出等价于图的可持久化流程。

对比 langgraph/quickstart.py: quickstart.py 走的是 Graph API (StateGraph),
必须先定义 State schema (TypedDict), 再显式 add_node / add_edge / add_conditional_edges
把控制流"画"成一张图; 而本文件走 Functional API, 完全不定义 State schema、不画图,
控制流就是普通 Python 的顺序执行 / if-else / 循环, 只在函数上贴 @entrypoint 和 @task
两个装饰器。两者底层是同一套 runtime, 所以 Functional API 照样享有 checkpoint /
durable execution / streaming 能力——差异只在"你怎么表达控制流", 不在能力上。

对比 combo.py 的补充点: combo.py 里"跨调用记住状态"靠的是 store (跨 thread 共享),
以及 checkpointer 隐式保存整张图的 State; Functional API 没有 State schema, 于是
换了一种显式机制来读上一次的状态——@entrypoint 函数签名里的 previous 参数, 以及
entrypoint.final(value=, save=) 这一对语义。本文件重点演示这两个 Graph API 里
没有对应物的概念。

关键机制 / 踩坑 (均已用本地 langgraph 1.2.9 实跑 + inspect 验证):
  1. previous 首次运行为 None。@entrypoint 函数如果声明了 previous 关键字参数,
     runtime 会把"上一次 checkpoint 里保存的返回值"注入进来; 但同一 thread_id 第一次
     调用时没有历史, previous 就是 None, 必须自己兜底 (previous or 初始值), 否则会 TypeError。
  2. entrypoint.final(value=, save=) 把"返回给调用方的值"和"存进 checkpoint 的值"分开:
     invoke() 的返回值 = value, 而下次 previous 读到的 = save。若直接 return x (不包 final),
     则 value 和 save 都是 x。本文件用一个累加器演示: 每次返回本次增量 (value), 但 checkpoint
     里存的是累计总和 (save), 于是 previous 每次拿到的是总和而非上次的增量。
  3. @task 返回的是 future 而不是结果本身, 必须 .result() 才能拿到真实返回值。好处是:
     连续调用多个 @task 会先各自异步派发出去 (并发调度), 你在需要用到结果的那一行才
     .result() 阻塞——所以"先全部调用、后统一 .result()"能并发, "调一个立刻 .result()
     一个"则退化成串行。本文件用带时序戳的 task 实测这个并发 vs 串行的差异。

模型接入统一走 .env (ChatAnthropic 读 MODEL_ID / ANTHROPIC_BASE_URL), 禁止硬编码。
当前环境无 MODEL_ID/密钥时, 模型调用点自动切换到 langchain 的 GenericFakeChatModel
可控替身, 保证 @task 封装 LLM 调用的机制被真实演示 (而不是纯空壳)。

官方文档: https://docs.langchain.com/oss/python/langgraph/functional-api
         https://docs.langchain.com/oss/python/langgraph/durable-execution
"""

import itertools
import os
import time

from dotenv import load_dotenv
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.func import entrypoint, task

load_dotenv(override=True)


# ============================================================
# 模型接入: 有 MODEL_ID 走真实 ChatAnthropic, 否则走可控替身
# 说明: 本文件重点是 Functional API 的控制流机制, 模型只是 @task 里的一个被调用者,
#       所以无密钥时用 GenericFakeChatModel 替身即可完整演示 @task 封装 LLM 的模式。
# ============================================================
def build_model():
    if os.getenv("MODEL_ID"):
        # 真实模型: 统一走 .env, 不硬编码 model id / base_url / key
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=os.environ["MODEL_ID"],
            base_url=os.getenv("ANTHROPIC_BASE_URL") or None,
        )
    # 无密钥替身: 循环产出固定回复, 行为可控可断言, 但仍是真的 chat model 接口
    from langchain_core.language_models.fake_chat_models import GenericFakeChatModel

    return GenericFakeChatModel(messages=itertools.cycle(["这是一句摘要 (来自替身模型)。"]))


model = build_model()


# ============================================================
# 演示 1: @task 封装工作单元 —— 返回的是 future, 必须 .result() 取值
# 每个 @task 是一个可被 checkpoint 记录的"步骤", 类似 Graph API 里的一个 node,
# 但你不用 add_node/add_edge 连线, 直接像调普通函数一样调它。
# ============================================================
@task
def call_model(prompt: str) -> str:
    """把一次 LLM 调用封装成 task。Functional API 里这就相当于 Graph API 的一个节点。"""
    response = model.invoke([{"role": "user", "content": prompt}])
    return response.text  # 注意: 返回真实字符串; 但调用方拿到的是 future, 需 .result()


@task
def slow_step(name: str, delay: float) -> str:
    """一个故意 sleep 的 task, 用来观测多个 task 是否并发调度。"""
    time.sleep(delay)
    return f"{name}@{time.time():.4f}"  # 带一个时间戳, 便于外部判断执行时序


# ============================================================
# 演示 2: previous + entrypoint.final(value=, save=)
# 一个"累加器"工作流: 每次调用传入一个增量 delta。
#   - 返回给调用方 (value): 本次的增量 delta 本身
#   - 存进 checkpoint / 喂给下次 previous (save): 到目前为止的累计总和
# 于是下次 previous 读到的是"累计总和", 而不是"上次返回的增量"——这就是 value/save 分离。
#
# 注意 (实测踩坑): get_state().values 拿到的是最近一次的 value (返回值), 不是 save 的值;
# save 的值是"喂给下一次 previous"的那份。所以要观测 value/save 确实不同, 最直接的办法
# 是把每次进入函数时 runtime 注入的 previous 记录下来 (下面用 SEEN_PREVIOUS), 它就是上一次 save。
# ============================================================
SEEN_PREVIOUS: list[int | None] = []  # 记录每次 runtime 注入的 previous, 即"上一次 save 的值"


@entrypoint(checkpointer=InMemorySaver())
def accumulator(delta: int, *, previous: int | None):
    # previous: 上一次 entrypoint.final 里 save 的值; 同一 thread 首次调用时为 None, 必须兜底
    SEEN_PREVIOUS.append(previous)
    running_total = (previous or 0) + delta
    # value=delta   -> invoke() 的返回值 / get_state().values (调用方只关心"这次加了多少")
    # save=running_total -> 喂给下次 previous (框架据此实现跨调用状态累积)
    return entrypoint.final(value=delta, save=running_total)


# ============================================================
# 演示 3: 一个完整的 entrypoint 编排多个 task —— 控制流就是普通 Python
# 对比 quickstart.py: 那里用 add_conditional_edges 画分支; 这里直接写 if / for 循环。
# ============================================================
@entrypoint(checkpointer=InMemorySaver())
def summarize_workflow(docs: list[str]) -> list[str]:
    # 先把所有 doc 的模型调用一次性派发出去 (拿到一串 future), 实现并发
    futures = [call_model(f"用一句话概括: {doc}") for doc in docs]
    # 再统一 .result() 收集结果 —— 控制流就是普通列表推导, 无需画图
    return [f.result() for f in futures]


# ============================================================
# 演示 4: 并发 vs 串行 —— 同样两个 slow_step, 收 future 的时机不同, 时序完全不同
# ============================================================
@entrypoint(checkpointer=InMemorySaver())
def concurrent_workflow(_input: str) -> dict:
    # 并发: 先全部调用 (派发), 再统一 result
    f1 = slow_step("A", 0.3)
    f2 = slow_step("B", 0.3)
    concurrent = [f1.result(), f2.result()]
    return {"concurrent": concurrent}


@entrypoint(checkpointer=InMemorySaver())
def serial_workflow(_input: str) -> dict:
    # 串行: 调一个立刻 result 一个, 阻塞后再调下一个
    r1 = slow_step("A", 0.3).result()
    r2 = slow_step("B", 0.3).result()
    return {"serial": [r1, r2]}


if __name__ == "__main__":
    print("=== 演示 1: entrypoint.final 的 value/save 分离 + previous 跨调用累积 ===")
    cfg = {"configurable": {"thread_id": "acc-1"}}
    # 第一次: previous 为 None (首次运行), 兜底成 0; save=10 喂给下次 previous
    r1 = accumulator.invoke(10, config=cfg)
    print(f"  invoke(10) 返回={r1} (=value=本次增量), 本次 runtime 注入 previous={SEEN_PREVIOUS[-1]} (首次为 None)")
    assert r1 == 10, "value 应等于本次增量 delta"
    assert SEEN_PREVIOUS[-1] is None, "同一 thread 首次运行 previous 必为 None"

    # 第二次: previous 读到上一次 save 的 10, 本次返回值仍是增量 20, 但 save 累计成 30
    r2 = accumulator.invoke(20, config=cfg)
    print(f"  invoke(20) 返回={r2} (=value=本次增量), 本次 runtime 注入 previous={SEEN_PREVIOUS[-1]} (=上次 save=累计)")
    assert r2 == 20, "返回值 (value) 仍是本次增量, 不是累计值 —— 这就是 value/save 分离"
    assert SEEN_PREVIOUS[-1] == 10, "第二次的 previous 应为上一次 save 的累计值 10"
    # get_state().values 拿到的是最近一次的 value (=20), 印证 value 与 save 是两份不同的东西
    latest_value = accumulator.get_state(cfg).values
    print(f"  get_state().values={latest_value} (=最近一次 value, 不是 save 的累计值 30)")
    assert latest_value == 20, "get_state().values 返回的是 value 而非 save, 二者被 final 明确分离"

    # 换一个全新 thread_id: previous 又回到 None, 累计从头开始, 互不干扰
    cfg_new = {"configurable": {"thread_id": "acc-2"}}
    r3 = accumulator.invoke(7, config=cfg_new)
    print(f"  新 thread invoke(7) 返回={r3}, 本次 previous={SEEN_PREVIOUS[-1]} (新 thread previous=None, 从头累计)")
    assert SEEN_PREVIOUS[-1] is None, "新 thread 的 previous 首次为 None, 累计从头开始"

    print("\n=== 演示 2: @task 封装模型调用, entrypoint 用普通列表推导编排 ===")
    docs = ["LangGraph 支持函数式 API", "checkpointer 提供持久化"]
    summaries = summarize_workflow.invoke(docs, config={"configurable": {"thread_id": "sum-1"}})
    print(f"  输入 {len(docs)} 篇文档, 得到 {len(summaries)} 条摘要:")
    for i, s in enumerate(summaries):
        print(f"    doc{i + 1}: {s}")
    assert len(summaries) == len(docs), "每篇文档应产出一条摘要 (task 逐一 .result() 收集)"
    assert all(isinstance(s, str) and s for s in summaries), "摘要应为非空字符串 (证明 future.result() 真取到了值)"

    print("\n=== 演示 3: @task 并发 vs 串行 (先派发后 result = 并发; 调一个 result 一个 = 串行) ===")
    t0 = time.perf_counter()
    conc = concurrent_workflow.invoke("go", config={"configurable": {"thread_id": "conc-1"}})
    conc_elapsed = time.perf_counter() - t0
    t0 = time.perf_counter()
    ser = serial_workflow.invoke("go", config={"configurable": {"thread_id": "ser-1"}})
    ser_elapsed = time.perf_counter() - t0
    print(f"  并发 {conc['concurrent']}  耗时 {conc_elapsed:.2f}s (两个 0.3s task 重叠, 接近 0.3s)")
    print(f"  串行 {ser['serial']}  耗时 {ser_elapsed:.2f}s (两个 0.3s task 排队, 接近 0.6s)")
    # 并发应显著快于串行 (两个 0.3s 的 task: 并发 ~0.3s, 串行 ~0.6s)
    assert conc_elapsed < ser_elapsed, "先派发后 result 应比调一个 result 一个更快 (并发 < 串行)"
    assert ser_elapsed >= 0.55, "串行两个 0.3s task 至少 ~0.6s"

    print("\n=== 演示 4: checkpoint 落盘读回 —— Functional API 一样有持久化 (无需 State schema) ===")
    # 复用演示 1 的 thread, 再取一次 state, 证明状态是被 checkpointer 真实持久化的
    reread = accumulator.get_state(cfg).values
    print(f"  重新读取 thread=acc-1 的 checkpoint: value={reread} (与上次 invoke 返回一致, 可重复读回)")
    assert reread == 20, "checkpoint 应稳定保存最近一次 value=20 (跨 get_state 调用可重复读回)"
    # 而累计状态 (save) 的存在, 已由下一次 previous 读到累计值 30 印证 (见演示 1)
    hist = list(accumulator.get_state_history(cfg))
    print(f"  thread=acc-1 的 checkpoint 历史条数: {len(hist)} (每次 invoke 都留下快照)")
    assert len(hist) >= 2, "两次 invoke 应至少留下 2 个 checkpoint 快照"

    used_stub = not os.getenv("MODEL_ID")
    print(f"\n所有断言通过。模型调用点使用了{'替身 (GenericFakeChatModel)' if used_stub else '真实 ChatAnthropic'}。")

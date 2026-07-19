"""LangGraph 选型指南 —— 同一业务需求, Graph API 与 Functional API 各写一遍并排对比。

对比 langgraph/ch_03_functional_api.py 与 langgraph/ch_01_quickstart.py:
  - ch_01_quickstart.py 是 Graph API (StateGraph) 的教学: 先定义 State schema (TypedDict),
    再 add_node / add_edge / add_conditional_edges 把控制流"画"成图。
  - ch_03_functional_api.py 是 Functional API (@entrypoint / @task) 的教学: 不画图、不定义
    State schema, 控制流就是普通 Python 顺序执行 / if-else / 循环。
  本文件不再单独教某一种, 而是把"取草稿 draft → 审校 review → 定稿 finalize"这一个
  完全相同的小流程, 用两种 API 各实现一遍, 放在同一个输入下横向对照, 让你直接看到:
  代码量差异、控制流表达差异 (显式图结构 + edges vs 普通 Python 控制流)、
  State 管理差异 (显式 schema + reducer vs 函数返回值 / previous), 并在末尾用断言
  证明"两种写法对同一输入产出等价结果"——它们是同一业务逻辑的两种表达, 不是两个功能。

关键机制 / 踩坑 (均已用本地 langgraph 1.2.9 实跑验证):
  1. 底层同源: Graph API 和 Functional API 编译后都是同一套 Pregel runtime, 所以两者
     都能接 checkpointer 持久化、都能 streaming, 差异只在"你怎么表达控制流", 不在能力上。
     本文件两种实现共用 InMemorySaver, 且都能 get_state() 读回状态, 即为佐证。
  2. State 管理机制不同, 不是能力强弱: Graph API 的 State 是所有节点共享的一份 TypedDict,
     天然适合"多节点读写同一份大状态" (本文件在 State 里累积 steps 轨迹, 每个节点都往里追加);
     Functional API 没有 State schema, 步骤间靠函数返回值 / 参数传递数据, 跨调用记忆靠
     @entrypoint 的 previous 参数。需求越像"多节点共享大状态", Graph API 越省事。
  3. 选型不是二选一: 需要可视化图 (get_graph().draw)、复杂条件路由、大团队协作维护大图 →
     Graph API 更好; 控制流本就是普通代码逻辑、快速原型、想少写样板 → Functional API 更爽。
     同一项目里两种可以混用 (Functional 的 entrypoint 里可以 invoke 一个 Graph, 反之节点里
     也能调 task 式函数), 因为底层都是 Pregel。
  4. 等价性怎么断言: Graph 版返回整份 State (dict, 含 steps 轨迹), Functional 版返回一个
     普通结果 dict。本文件让两者产出结构对齐的 dict (draft/review/final/steps) 再逐字段断言,
     以此证明"同一业务逻辑"的两种写法结果一致。

模型接入统一走 .env (ChatAnthropic 读 MODEL_ID / ANTHROPIC_BASE_URL), 禁止硬编码。
当前环境无 MODEL_ID/密钥时, 模型调用点自动切换到 langchain 的 GenericFakeChatModel
可控替身 —— 用固定 prompt 前缀映射到固定回复, 保证两种实现调的是同一个真 chat model
接口 (而非纯空壳), 且输出可预测、可断言等价。

官方文档: https://docs.langchain.com/oss/python/langgraph/functional-api
         https://docs.langchain.com/oss/python/langgraph/graph-api
         https://docs.langchain.com/oss/python/langgraph/use-graph-api
"""

import os
from typing import TypedDict

from dotenv import load_dotenv
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.func import entrypoint, task
from langgraph.graph import END, START, StateGraph

load_dotenv(override=True)


# ============================================================
# 模型接入: 有 MODEL_ID 走真实 ChatAnthropic, 否则走可控替身
# 说明: 本文件重点是"两种 API 的写法对比", 模型只是被 draft/review/finalize 三步调用的执行者。
#       替身按 prompt 里的关键词返回固定文案, 使两种实现在同一输入下得到完全一致的产物,
#       从而能对"等价性"下确定性断言 (真实 LLM 因随机性无法逐字对比, 但流程结构一致)。
# ============================================================
def build_model():
    if os.getenv("MODEL_ID"):
        # 真实模型: 统一走 .env, 不硬编码 model id / base_url / key
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=os.environ["MODEL_ID"],
            base_url=os.getenv("ANTHROPIC_BASE_URL") or None,
        )
    # 无密钥替身: 一个按 prompt 关键词映射到固定回复的可控 chat model,
    # 走的仍是标准 chat model 的 .invoke 接口, 只是回复确定、可断言。
    # 注意 (实测踩坑): GenericFakeChatModel 的 _generate 是从 self.messages 迭代器里逐个 next() 取值的,
    #   它不会看 prompt 内容, 也无 _call 钩子; 若只覆写 _call 根本不生效, 且迭代器耗尽会抛 StopIteration。
    #   所以这里直接覆写 _generate, 按 prompt 关键词返回固定文案, 才能让 draft/review/finalize 三步可区分且确定。
    from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
    from langchain_core.messages import AIMessage
    from langchain_core.outputs import ChatGeneration, ChatResult

    class _KeyedFake(GenericFakeChatModel):
        """按 prompt 里出现的关键词返回固定文案的替身, 让 draft/review/finalize 三步可区分且确定。"""

        def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
            prompt = messages[-1].content if messages else ""
            # 注意判断顺序: finalize 的 prompt 会内嵌 draft/review 文本, 故先判 finalize/定稿,
            #   再判 review/审校, 最后才是 draft/草稿, 避免被前面的关键词误命中。
            if "定稿" in prompt or "finalize" in prompt:
                reply = "定稿正文 (已采纳审校意见)"
            elif "审校" in prompt or "review" in prompt:
                reply = "审校意见: 措辞需更简洁"
            elif "草稿" in prompt or "draft" in prompt:
                reply = "初稿正文"
            else:
                reply = "默认回复"
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content=reply))])

    # 覆写 _generate 后不再从 messages 迭代器取值, 但父类字段仍要求给 messages, 传空迭代器占位即可
    return _KeyedFake(messages=iter([]))


model = build_model()


# 三步的 prompt 模板: 两种实现共用, 保证喂给模型的输入完全一致
def _draft_prompt(topic: str) -> str:
    return f"请就主题'{topic}'写一份草稿 (draft)。"


def _review_prompt(draft: str) -> str:
    return f"请审校 (review) 以下草稿并给出意见: {draft}"


def _finalize_prompt(draft: str, review: str) -> str:
    return f"请根据草稿'{draft}'与审校意见'{review}'产出定稿 (finalize)。"


def _one_shot(prompt: str) -> str:
    """一次最朴素的模型调用, 两种实现都通过它调模型, 确保调用点完全等价。"""
    return model.invoke([{"role": "user", "content": prompt}]).text


# ============================================================
# 实现 A: Graph API (StateGraph) —— 显式定义 State, 用节点 + 边把流程画成图
# 对照 ch_01_quickstart.py: 同样是 State schema + add_node + add_edge 的套路。
# 特点: State 是所有节点共享的一份数据, 每个节点读它、返回增量, runtime 负责合并。
#       流程是线性的三步, 所以只用 add_edge 顺序连线 (无需条件分支)。
# ============================================================
class ReviewState(TypedDict):
    topic: str  # 输入: 主题
    draft: str  # draft 节点写入
    review: str  # review 节点写入
    final: str  # finalize 节点写入
    steps: list[str]  # 执行轨迹, 每个节点往里追加自己的名字 (演示"多节点共享大状态")


def graph_draft(state: ReviewState) -> dict:
    # 节点只需返回要更新的字段; runtime 会把它并进共享 State
    return {"draft": _one_shot(_draft_prompt(state["topic"])), "steps": state.get("steps", []) + ["draft"]}


def graph_review(state: ReviewState) -> dict:
    # 这里能直接读到上一节点写进 State 的 draft —— 共享状态的好处
    return {"review": _one_shot(_review_prompt(state["draft"])), "steps": state.get("steps", []) + ["review"]}


def graph_finalize(state: ReviewState) -> dict:
    # finalize 同时用到前两步的 draft 和 review, 都从共享 State 里取
    final = _one_shot(_finalize_prompt(state["draft"], state["review"]))
    return {"final": final, "steps": state.get("steps", []) + ["finalize"]}


def build_graph_impl():
    builder = StateGraph(ReviewState)
    builder.add_node("draft", graph_draft)  # 三个节点各对应一步
    builder.add_node("review", graph_review)
    builder.add_node("finalize", graph_finalize)
    builder.add_edge(START, "draft")  # 显式连线: START → draft → review → finalize → END
    builder.add_edge("draft", "review")
    builder.add_edge("review", "finalize")
    builder.add_edge("finalize", END)
    # 编译时接上 checkpointer, 与 Functional 版共享同一种持久化能力 (底层都是 Pregel)
    return builder.compile(checkpointer=InMemorySaver())


graph_app = build_graph_impl()


# ============================================================
# 实现 B: Functional API (@entrypoint / @task) —— 不画图, 控制流就是普通 Python
# 对照 ch_03_functional_api.py: 同样用 @task 封装工作单元, @entrypoint 编排。
# 特点: 没有 State schema, 三步的数据靠局部变量顺序传递; 流程就是三行顺序调用,
#       想加条件分支直接写 if 即可, 无需 add_conditional_edges。
# ============================================================
@task
def func_draft(topic: str) -> str:
    return _one_shot(_draft_prompt(topic))


@task
def func_review(draft: str) -> str:
    return _one_shot(_review_prompt(draft))


@task
def func_finalize(draft: str, review: str) -> str:
    return _one_shot(_finalize_prompt(draft, review))


@entrypoint(checkpointer=InMemorySaver())
def func_app(topic: str) -> dict:
    # 控制流就是普通 Python 的顺序执行: 调一个 task, .result() 取值, 再喂给下一个
    draft = func_draft(topic).result()
    review = func_review(draft).result()
    final = func_finalize(draft, review).result()
    # 手动组织返回结构; steps 也由自己按调用顺序拼出来 (对齐 Graph 版的 steps 轨迹)
    return {"draft": draft, "review": review, "final": final, "steps": ["draft", "review", "finalize"]}


if __name__ == "__main__":
    topic = "LangGraph 的两种 API 该怎么选"

    print("=== 实现 A: Graph API (StateGraph, 显式图结构) ===")
    cfg_a = {"configurable": {"thread_id": "graph-1"}}
    # 传入初始 State, steps 先给空列表, 让各节点往里追加
    out_a = graph_app.invoke({"topic": topic, "steps": []}, config=cfg_a)
    print(f"  draft : {out_a['draft']}")
    print(f"  review: {out_a['review']}")
    print(f"  final : {out_a['final']}")
    print(f"  steps : {' -> '.join(out_a['steps'])}")
    # Graph 版的节点数 = 3, 边数 (含 START/END) 一目了然, 支持可视化
    nodes = [n for n in graph_app.get_graph().nodes if n not in ("__start__", "__end__")]
    print(f"  图节点 (业务节点): {nodes}  <- Graph API 可 get_graph() 拿到结构做可视化/路由")

    print("\n=== 实现 B: Functional API (@entrypoint/@task, 普通 Python 控制流) ===")
    cfg_b = {"configurable": {"thread_id": "func-1"}}
    out_b = func_app.invoke(topic, config=cfg_b)
    print(f"  draft : {out_b['draft']}")
    print(f"  review: {out_b['review']}")
    print(f"  final : {out_b['final']}")
    print(f"  steps : {' -> '.join(out_b['steps'])}")
    print("  控制流即三行顺序调用, 无 State schema / 无 add_edge  <- 少写样板, 快速原型更省事")

    print("\n=== 等价性断言: 同一输入, 两种写法产出等价结果 ===")
    # 逐字段比对: 证明这是"同一业务逻辑"的两种表达, 而非两个不同功能
    if os.getenv("MODEL_ID"):
        # 真实模型会被调用两次, 文案可能不逐字相同; 只校验稳定结构与流程。
        assert out_a["steps"] == out_b["steps"], "两种实现的步骤轨迹应一致"
        for key in ("draft", "review", "final"):
            assert isinstance(out_a[key], str) and out_a[key].strip(), f"Graph 版 {key} 应有非空文本"
            assert isinstance(out_b[key], str) and out_b[key].strip(), f"Functional 版 {key} 应有非空文本"
        print("  真实模型路径: steps 一致, draft/review/final 均为非空文本")
    else:
        for key in ("draft", "review", "final", "steps"):
            assert out_a[key] == out_b[key], f"字段 {key} 两种实现应完全一致: {out_a[key]!r} != {out_b[key]!r}"
        print("  fake 模型路径: draft/review/final/steps 四字段逐一比对通过")

    print("\n=== 持久化对照: 两者底层都是 Pregel, 都能用 checkpointer 读回状态 ===")
    # Graph 版: get_state().values 是整份共享 State
    reread_a = graph_app.get_state(cfg_a).values
    # Functional 版: get_state().values 是 entrypoint 的返回值
    reread_b = func_app.get_state(cfg_b).values
    print(f"  Graph  版 checkpoint 读回 final: {reread_a['final']}")
    print(f"  Func   版 checkpoint 读回 final: {reread_b['final']}")
    assert reread_a["final"] == reread_b["final"], "两种实现的 checkpoint 都应能读回一致的定稿"

    print("\n=== 选型小结 ===")
    print("  需要可视化图 / 复杂条件路由 / 大团队协作维护大图 / 多节点共享大状态 -> Graph API")
    print("  控制流本就是普通代码逻辑 / 快速原型 / 想少写样板                 -> Functional API")
    print("  两者底层都是 Pregel, 都能 checkpointer 持久化 + streaming, 可混用, 按场景选, 不是二选一")

    used_stub = not os.getenv("MODEL_ID")
    print(f"\n所有断言通过。模型调用点使用了{'替身 (GenericFakeChatModel 派生的 _KeyedFake)' if used_stub else '真实 ChatAnthropic'}。")

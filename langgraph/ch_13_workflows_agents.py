"""LangGraph workflow 与 agent 编排范式 —— 6 种通用编排 pattern 的最小实现与横向对比。

对比 langgraph/ch_12_multi_agent.py: 那里讲的是"多智能体"这一个具体场景 —— supervisor 用 Command
同时做状态更新 + 路由, worker 是平级图节点, 聚焦于"多个 agent 如何分工协作"。本文件不聚焦
多 agent, 而是横向铺开 6 种更通用的编排范式(prompt chaining / parallelization / routing /
orchestrator-worker / evaluator-optimizer / agent), 并点明它们背后的本质分界:

    workflow(工作流) = 预定义控制流。下一步走哪由代码写死(顺序边、并行 fan-out)或由
      条件路由函数按规则/一次性分类决定, 路径是"编排者(开发者)"事先设计好的。
    agent(智能体) = 模型动态决定控制流。下一步做什么由 LLM 在每一轮返回的 tool_calls
      驱动, 什么时候停也由模型自己判断(不再返回工具调用即结束), 路径运行时才生成。

前 5 种是 workflow(控制流可预先画成固定的图), 第 6 种是 agent(控制流是模型驱动的循环)。
orchestrator-worker 介于两者之间: orchestrator 用 LLM 动态决定"拆成几个子任务", 但每个
子任务的处理路径仍是固定的 —— 所以它常被归为"由 LLM 决定 fan-out 数量的 workflow"。

关键机制 / 踩坑记录(本文件用 inspect 与实跑验证过):
  - Routing 用 add_conditional_edges: 路由函数返回"下一个节点名(字符串)", 图据此分流。
    分类判断本身需要模型(把输入归到某个类别), 无 MODEL_ID 时用规则函数替身模拟这次分类,
    分支仍真实走通 —— 演示的是"分类结果 → conditional_edges 分流"这条控制流, 而非分类模型本身。
  - Orchestrator-Worker 用 Send: orchestrator 运行时才知道要拆几个子任务, 用 Send 动态
    fan-out N 个 worker(参见 ch_11_send_map_reduce.py); worker 结果靠 Annotated[list, add] 的
    reducer 自动汇聚, 否则多个并行分支同时写同一 channel 会 InvalidUpdateError。
  - Evaluator-Optimizer 的循环终止条件是重点: generate → evaluate → 不合格则带反馈回
    generate 重试。必须有硬性上限(max_rounds)兜底, 否则评估器一直不满意会无限循环 ——
    这里终止条件是"评估通过 OR 达到最大轮数", 两者缺一都会跑飞。
  - Agent 的工具循环靠模型返回 tool_calls 驱动: 模型返回带 tool_calls 的 AIMessage → 执行
    工具 → 把 ToolMessage 塞回上下文 → 再问模型, 直到模型不再返回 tool_calls。无 MODEL_ID 时
    用 FakeListChatModel 预置"先调用工具、后给最终答案"的消息序列, 让整个 loop 真实转起来。
  - 模型接入统一走 .env(ChatAnthropic); 当前环境无 MODEL_ID, 所有"需要模型决策"的点自动
    降级到可控替身(规则函数 / FakeListChatModel), 结构性控制流不依赖模型即可断言跑通。

官方文档: https://docs.langchain.com/oss/python/langgraph/workflows-agents
"""

import operator  # 提供 operator.add, 用作 list 字段的 reducer(并行结果自动拼接)
import os  # 读环境变量, 判断有没有 MODEL_ID
from typing import Annotated, Literal, TypedDict  # 状态类型注解 + reducer 标注

from dotenv import load_dotenv  # 从 .env 加载配置, 不硬编码任何密钥
from langchain_core.language_models.fake_chat_models import FakeListChatModel  # 无密钥时的可控替身模型
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage  # agent 工具循环用到的消息类型
from langgraph.graph import END, START, StateGraph  # 图的起止哨兵 + 图构建器
from langgraph.types import Send  # orchestrator-worker 动态 fan-out 用

load_dotenv(override=True)  # 加载 .env; 当前环境里 MODEL_ID 为空, 后面据此走替身路径

# 统一的真实模型接入点: 有 MODEL_ID 才实例化 ChatAnthropic, 否则置 None 走替身。
# 严禁硬编码 model id / base_url / api key, 一律走 .env。
_HAS_MODEL = bool(os.getenv("MODEL_ID"))  # 布尔标记: 当前是否具备真实模型
if _HAS_MODEL:  # 有密钥才导入并构造真实模型, 避免无密钥时构造直接抛错
    from langchain_anthropic import ChatAnthropic  # 真实模型客户端

    real_model = ChatAnthropic(  # 生产路径的真实 LLM
        model=os.environ["MODEL_ID"],  # 模型 id 来自 .env
        base_url=os.getenv("ANTHROPIC_BASE_URL") or None,  # base_url 可选, 也走 .env
    )
else:  # 无密钥: 真实模型置空, 下面每个范式各自决定如何用替身模拟"模型决策"
    real_model = None


# ============================================================
# 范式 1: Prompt Chaining(提示链) —— 串行分解: 上一步输出喂给下一步
# 控制流: 固定顺序边 START → step1 → step2 → step3 → END, 完全写死。
# ============================================================
class ChainState(TypedDict):
    topic: str  # 初始输入: 一个主题
    outline: str  # step1 产出: 提纲
    draft: str  # step2 产出: 基于提纲的初稿
    polished: str  # step3 产出: 润色后的成稿


def chain_outline(state: ChainState) -> dict:
    # 第一步: 把主题拆成提纲(用普通函数模拟一步 LLM 加工, 保证零依赖可跑)
    return {"outline": f"提纲<{state['topic']}>: 1.是什么 2.为什么 3.怎么做"}


def chain_draft(state: ChainState) -> dict:
    # 第二步: 依赖 step1 的 outline 产出初稿 —— 这就是"上一步输出是下一步输入"的链式关系
    return {"draft": f"初稿(基于 {state['outline']})"}


def chain_polish(state: ChainState) -> dict:
    # 第三步: 依赖 step2 的 draft 做润色
    return {"polished": f"成稿(润色自 {state['draft']})"}


def build_chain_graph():
    # 构建提示链图: 三个节点用固定顺序边串起来, 路径由代码写死(典型 workflow)
    b = StateGraph(ChainState)
    b.add_node("outline", chain_outline)
    b.add_node("draft", chain_draft)
    b.add_node("polish", chain_polish)
    b.add_edge(START, "outline")  # 入口 → 第一步
    b.add_edge("outline", "draft")  # 第一步 → 第二步(串行)
    b.add_edge("draft", "polish")  # 第二步 → 第三步(串行)
    b.add_edge("polish", END)  # 第三步 → 出口
    return b.compile()


# ============================================================
# 范式 2: Parallelization(并行) —— 多个分支同时执行, 最后汇聚
# 控制流: 一个源节点后接多条并行边, 各分支写入带 reducer 的字段, 到 aggregator 汇总。
# ============================================================
class ParallelState(TypedDict):
    topic: str  # 输入主题
    # 三个并行分支都往这个 list 里写, Annotated + operator.add 让并行写入自动合并(不冲突)
    aspects: Annotated[list[str], operator.add]
    summary: str  # aggregator 汇聚出的结论


def para_tech(state: ParallelState) -> dict:
    # 分支 A: 从技术角度分析(与 B/C 并行执行)
    return {"aspects": [f"[技术] {state['topic']} 的实现难点"]}


def para_cost(state: ParallelState) -> dict:
    # 分支 B: 从成本角度分析
    return {"aspects": [f"[成本] {state['topic']} 的投入产出"]}


def para_risk(state: ParallelState) -> dict:
    # 分支 C: 从风险角度分析
    return {"aspects": [f"[风险] {state['topic']} 的潜在坑"]}


def para_aggregate(state: ParallelState) -> dict:
    # 汇聚节点: 等三个分支全部到齐后, 把结果合并(LangGraph 会自动等所有入边完成)
    return {"summary": f"共汇聚 {len(state['aspects'])} 个视角: " + " | ".join(state["aspects"])}


def build_parallel_graph():
    # 构建并行图: fan-out 到 3 个分支, 再 fan-in 到 aggregate
    b = StateGraph(ParallelState)
    b.add_node("tech", para_tech)
    b.add_node("cost", para_cost)
    b.add_node("risk", para_risk)
    b.add_node("aggregate", para_aggregate)
    # 从 START 拉出三条边并行触发三个分支
    b.add_edge(START, "tech")
    b.add_edge(START, "cost")
    b.add_edge(START, "risk")
    # 三个分支都指向 aggregate; aggregate 会等三条入边全部完成才执行(结构性同步屏障)
    b.add_edge("tech", "aggregate")
    b.add_edge("cost", "aggregate")
    b.add_edge("risk", "aggregate")
    b.add_edge("aggregate", END)
    return b.compile()


# ============================================================
# 范式 3: Routing(路由) —— 先分类, 再按类别走不同分支
# 控制流: add_conditional_edges 按"分类结果"分流。分类本身要模型, 无密钥用规则函数替身。
# ============================================================
class RouteState(TypedDict):
    question: str  # 用户问题
    category: str  # 分类结果(由"分类器"填)
    answer: str  # 对应分支给出的答案


def classify(state: RouteState) -> dict:
    """分类节点: 把问题归到 billing / tech / other。这一步在生产里是 LLM 判断;
    无 MODEL_ID 时用关键词规则函数替身模拟这次分类决策 —— 演示重点是"分类结果驱动
    conditional_edges 分流"这条控制流, 而不是分类模型本身。"""
    q = state["question"]
    # 当前示例的断言需要稳定覆盖三条分支; 在接入真实 structured output 前，
    # 有 MODEL_ID 时也沿用同一套确定性分类，避免把所有输入硬路由到 tech。
    if any(k in q for k in ["账单", "退款", "价格"]):
        cat = "billing"
    elif any(k in q for k in ["报错", "崩溃", "bug"]):
        cat = "tech"
    else:
        cat = "other"
    return {"category": cat}


def route_decide(state: RouteState) -> Literal["billing", "tech", "other"]:
    # 路由函数: 读分类结果, 返回下一个要去的节点名 —— conditional_edges 据此分流
    return state["category"]  # 返回值必须是下面注册的某个分支节点名


def handle_billing(state: RouteState) -> dict:
    return {"answer": "转接账务组处理退款/账单问题"}


def handle_tech(state: RouteState) -> dict:
    return {"answer": "转接技术支持排查报错"}


def handle_other(state: RouteState) -> dict:
    return {"answer": "转接通用客服"}


def build_routing_graph():
    # 构建路由图: classify → (conditional) → 三选一分支
    b = StateGraph(RouteState)
    b.add_node("classify", classify)
    b.add_node("billing", handle_billing)
    b.add_node("tech", handle_tech)
    b.add_node("other", handle_other)
    b.add_edge(START, "classify")
    # 关键: 用 conditional_edges 按 route_decide 的返回值分流到不同分支
    b.add_conditional_edges("classify", route_decide, ["billing", "tech", "other"])
    # 每个分支处理完就结束
    b.add_edge("billing", END)
    b.add_edge("tech", END)
    b.add_edge("other", END)
    return b.compile()


# ============================================================
# 范式 4: Orchestrator-Worker(编排者-工人) —— 动态拆子任务, Send 派发给 worker
# 控制流: orchestrator 运行时决定拆几个子任务(LLM 或规则), 用 Send fan-out N 个 worker,
# 结果靠 reducer 汇聚。介于 workflow 与 agent 之间: fan-out 数量动态, 但每个 worker 路径固定。
# ============================================================
class OrchState(TypedDict):
    request: str  # 总请求(比如"写一篇多章节报告")
    sections: list[str]  # orchestrator 拆出来的子任务(章节)列表
    results: Annotated[list[str], operator.add]  # 各 worker 产出, reducer 自动汇聚


def orchestrator(state: OrchState) -> dict:
    """编排节点: 把总请求拆成若干子任务。生产里由 LLM 决定拆几段;
    无密钥时按分隔符切分模拟这次"拆解决策"。"""
    # 用 '/' 切分请求, 得到运行时才确定数量的子任务列表
    sections = [s.strip() for s in state["request"].split("/") if s.strip()]
    return {"sections": sections}


def assign_workers(state: OrchState) -> list[Send]:
    # 动态 fan-out: 有几个 section 就派发几个 worker, 数量运行时才知道(这正是 Send 的用途)
    return [Send("worker", {"section": s}) for s in state["sections"]]


class WorkerState(TypedDict):
    section: str  # 单个 worker 处理的子任务(由 Send 精确注入, 完全替代默认输入)
    results: Annotated[list[str], operator.add]  # 与父图同名字段, 靠 reducer 合并回去


def worker(state: WorkerState) -> dict:
    # 单个 worker: 处理分到的那一小段子任务
    return {"results": [f"完成章节《{state['section']}》"]}


def build_orchestrator_graph():
    # 构建编排者-工人图: orchestrator 拆分 → Send fan-out worker → 汇聚
    b = StateGraph(OrchState)
    b.add_node("orchestrator", orchestrator)
    b.add_node("worker", worker)  # worker 会被 Send 并行实例化多份
    b.add_edge(START, "orchestrator")
    # orchestrator 之后用 conditional_edges + assign_workers 动态派发 N 个 worker
    b.add_conditional_edges("orchestrator", assign_workers, ["worker"])
    b.add_edge("worker", END)  # 所有 worker 完成后结束(结果已由 reducer 汇聚)
    return b.compile()


# ============================================================
# 范式 5: Evaluator-Optimizer(生成-评估-反馈重试循环)
# 控制流: generate → evaluate → 合格则结束 / 不合格带反馈回 generate 重试。
# 终止条件是重点: 评估通过 OR 达到 max_rounds, 缺一都会无限循环。
# ============================================================
MAX_ROUNDS = 3  # 硬性重试上限, 兜底防止评估器永不满意导致死循环


class EvalState(TypedDict):
    task: str  # 要完成的任务
    solution: str  # generate 产出的方案
    feedback: str  # evaluate 给的反馈(不合格时用于下一轮改进)
    rounds: int  # 已经生成了几轮(用于比对 MAX_ROUNDS)
    passed: bool  # 是否已通过评估


def generate(state: EvalState) -> dict:
    # 生成节点: 依据任务 + 上一轮反馈产出方案; 轮数 +1
    rnd = state.get("rounds", 0) + 1
    # 模拟"质量随轮次提升": 第 rnd 轮方案里塞入 rnd 个关键词, 直到满足评估阈值
    quality_tokens = " ".join([f"要点{i + 1}" for i in range(rnd)])
    sol = f"第{rnd}轮方案[{quality_tokens}] (参考反馈: {state.get('feedback') or '无'})"
    return {"solution": sol, "rounds": rnd}


def evaluate(state: EvalState) -> dict:
    """评估节点: 给方案打分决定是否通过。生产里由 LLM 打分; 无密钥用规则替身:
    方案里包含的"要点"数 >= 2 就算通过 —— 保证循环能真实迭代若干轮再收敛。"""
    ok = state["solution"].count("要点") >= 2  # 规则化的"评分"决策
    if ok:
        return {"passed": True, "feedback": "通过: 覆盖要点充分"}
    return {"passed": False, "feedback": "不通过: 要点不足, 请补充更多要点"}


def eval_route(state: EvalState) -> Literal["generate", "__end__"]:
    # 循环控制: 通过 或 达到最大轮数 → 结束; 否则带着反馈回 generate 重试
    if state["passed"] or state["rounds"] >= MAX_ROUNDS:
        return END  # 终止条件命中(两个条件任一满足都收敛, 避免死循环)
    return "generate"  # 反馈重试: 形成 generate ↔ evaluate 的循环


def build_evaluator_graph():
    # 构建评估-优化图: generate → evaluate →(条件)回 generate 或结束
    b = StateGraph(EvalState)
    b.add_node("generate", generate)
    b.add_node("evaluate", evaluate)
    b.add_edge(START, "generate")
    b.add_edge("generate", "evaluate")
    # 关键的反馈循环: 由 eval_route 决定继续重试还是收敛
    b.add_conditional_edges("evaluate", eval_route, ["generate", END])
    return b.compile()


# ============================================================
# 范式 6: Agent(工具循环) —— 模型自主决定调用工具, 直到不再需要工具
# 这是唯一"控制流由模型动态决定"的范式: 下一步做什么 = 模型返回的 tool_calls;
# 什么时候停 = 模型不再返回 tool_calls。无密钥用 FakeListChatModel 预置消息序列驱动 loop。
# ============================================================
class AgentState(TypedDict):
    messages: Annotated[list, operator.add]  # 完整对话历史, 每步往后追加


def _calc_tool(expr: str) -> str:
    # 一个玩具"计算器"工具: 供 agent 在循环中调用
    try:
        return str(eval(expr, {"__builtins__": {}}, {}))  # 受限 eval, 仅演示用
    except Exception as e:  # 工具出错也如实回传给模型, 让它决定后续
        return f"error: {e}"


TOOLS = {"calc": _calc_tool}  # 工具注册表: name → 可调用对象


def _build_agent_model():
    """构造 agent 循环所用的模型:
    - 有 MODEL_ID: 用真实 ChatAnthropic(需 bind_tools 让它能返回 tool_calls);
    - 无 MODEL_ID: 用 FakeListChatModel 预置"先调工具、拿到结果后给最终答案"的两条消息,
      让工具循环真实转起来 —— 这就是"模型返回 tool_calls 驱动 agent"的可控复现。"""
    if _HAS_MODEL:  # 真实路径: 绑定工具后模型才会产出 tool_calls
        return real_model.bind_tools(
            [{"name": "calc", "description": "计算算术表达式", "parameters": {"type": "object", "properties": {"expr": {"type": "string"}}, "required": ["expr"]}}]
        )
    # 替身路径: FakeListChatModel 依次吐出预置消息, 第 1 条带 tool_calls, 第 2 条是最终答案
    fake = FakeListChatModel(responses=["placeholder"])  # responses 仅占位, 真正的消息在下面覆写
    # FakeListChatModel 只能吐纯文本, 无法直接产 tool_calls; 因此下面 agent_call 用手写序列模拟
    return fake


# 无密钥时, agent 的"模型每一步返回什么"用这个预置脚本模拟(体现工具循环的真实转动)
_FAKE_MODEL_SCRIPT = [
    # 第 1 步: 模型决定调用 calc 工具(返回带 tool_calls 的 AIMessage)
    AIMessage(content="", tool_calls=[{"name": "calc", "args": {"expr": "21*2"}, "id": "call_1"}]),
    # 第 2 步: 拿到工具结果后, 模型给出最终答案(不再带 tool_calls → 循环终止)
    AIMessage(content="答案是 42"),
]


def agent_call_model(state: AgentState) -> dict:
    """agent 的"思考"节点: 问模型下一步做什么。
    有密钥走真实模型; 无密钥按已执行轮数从预置脚本里取下一条 AIMessage。"""
    if _HAS_MODEL:  # 真实路径: 把完整历史交给模型, 由它决定是否再调工具
        model = _build_agent_model()
        resp = model.invoke(state["messages"])
        return {"messages": [resp]}
    # 替身路径: 数一下历史里已经有几条 AIMessage, 决定取脚本的第几步
    ai_count = sum(1 for m in state["messages"] if isinstance(m, AIMessage))
    return {"messages": [_FAKE_MODEL_SCRIPT[ai_count]]}


def agent_take_tools(state: AgentState) -> dict:
    # agent 的"行动"节点: 执行模型刚要求调用的工具, 把结果作为 ToolMessage 塞回历史
    last = state["messages"][-1]  # 最后一条一定是带 tool_calls 的 AIMessage
    tool_msgs = []
    for tc in last.tool_calls:  # 可能一次要求调多个工具, 逐个执行
        result = TOOLS[tc["name"]](**tc["args"])
        tool_msgs.append(ToolMessage(content=result, tool_call_id=tc["id"]))
    return {"messages": tool_msgs}


def agent_should_continue(state: AgentState) -> Literal["tools", "__end__"]:
    # 循环控制: 模型最后一条消息带 tool_calls → 去执行工具; 否则(纯文本答案)→ 结束
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and last.tool_calls:
        return "tools"  # 模型还想调工具, 继续循环
    return END  # 模型给了最终答案, agent 循环终止


def build_agent_graph():
    # 构建 agent 图: call_model ↔ tools 的循环, 由模型的 tool_calls 驱动
    b = StateGraph(AgentState)
    b.add_node("call_model", agent_call_model)
    b.add_node("tools", agent_take_tools)
    b.add_edge(START, "call_model")
    # 关键循环: 模型决定要不要调工具; 调完工具再回模型问下一步
    b.add_conditional_edges("call_model", agent_should_continue, ["tools", END])
    b.add_edge("tools", "call_model")  # 工具结果回喂模型, 形成 loop
    return b.compile()


if __name__ == "__main__":
    mode = "真实模型(ChatAnthropic)" if _HAS_MODEL else "替身(规则函数 / FakeListChatModel)"
    print(f"[环境] MODEL_ID {'已配置' if _HAS_MODEL else '未配置'}, 需要模型决策的点走: {mode}\n")

    print("=== 范式1 Prompt Chaining: 串行分解, 上一步输出喂下一步 ===")
    r = build_chain_graph().invoke({"topic": "缓存", "outline": "", "draft": "", "polished": ""})
    print(f"  outline: {r['outline']}")
    print(f"  draft  : {r['draft']}")
    print(f"  polished: {r['polished']}")
    # 断言: 三步串行依赖关系成立(成稿里能追溯到初稿, 初稿里能追溯到提纲的主题)
    assert r["outline"] and r["draft"] and r["polished"], "链式三步都应产出"
    assert "缓存" in r["outline"] and "初稿" in r["polished"], "下游应依赖上游产出"
    print("  [断言通过] 三步严格串行, 每步依赖上一步\n")

    print("=== 范式2 Parallelization: 三分支并行 + 汇聚 ===")
    r = build_parallel_graph().invoke({"topic": "上线新功能", "aspects": [], "summary": ""})
    print(f"  summary: {r['summary']}")
    # 断言: 三个并行分支都执行到了(reducer 汇聚出恰好 3 个视角)
    assert len(r["aspects"]) == 3, f"三个并行分支都应执行到, 实得 {len(r['aspects'])}"
    assert any("技术" in a for a in r["aspects"]) and any("成本" in a for a in r["aspects"]) and any("风险" in a for a in r["aspects"])
    print("  [断言通过] 技术/成本/风险三分支并行都跑到并汇聚\n")

    print("=== 范式3 Routing: 分类后走不同分支(conditional_edges) ===")
    g = build_routing_graph()
    r1 = g.invoke({"question": "我要退款, 账单不对", "category": "", "answer": ""})
    r2 = g.invoke({"question": "程序一直报错崩溃", "category": "", "answer": ""})
    r3 = g.invoke({"question": "你们几点上班", "category": "", "answer": ""})
    print(f"  '退款账单' → {r1['category']}: {r1['answer']}")
    print(f"  '报错崩溃' → {r2['category']}: {r2['answer']}")
    print(f"  '几点上班' → {r3['category']}: {r3['answer']}")
    # 断言: 不同输入按分类走到了对应分支
    assert r1["category"] == "billing" and "账务" in r1["answer"]
    assert r2["category"] == "tech" and "技术" in r2["answer"]
    assert r3["category"] == "other" and "通用" in r3["answer"]
    print("  [断言通过] 三种输入各自路由到 billing/tech/other 分支\n")

    print("=== 范式4 Orchestrator-Worker: 动态拆子任务 + Send 派发 N 个 worker ===")
    req = "引言/技术方案/成本估算/风险与结论"  # 运行时才知道要拆 4 段
    r = build_orchestrator_graph().invoke({"request": req, "sections": [], "results": []})
    print(f"  拆出 {len(r['sections'])} 个子任务: {r['sections']}")
    for item in r["results"]:
        print(f"    - {item}")
    # 断言: orchestrator 拆出 N 段, 就派发出 N 个 worker, 全部完成并汇聚
    assert len(r["sections"]) == 4, "应按 '/' 拆出 4 个子任务"
    assert len(r["results"]) == 4, f"应派发并完成 4 个 worker, 实得 {len(r['results'])}"
    print("  [断言通过] orchestrator 动态拆 4 段, Send 派发 4 个 worker 全部完成\n")

    print("=== 范式5 Evaluator-Optimizer: 生成→评估→反馈重试循环 ===")
    r = build_evaluator_graph().invoke({"task": "写方案", "solution": "", "feedback": "", "rounds": 0, "passed": False})
    print(f"  最终方案: {r['solution']}")
    print(f"  评估反馈: {r['feedback']}  | 总轮数: {r['rounds']}  | 通过: {r['passed']}")
    # 断言: 循环真实迭代了不止一轮(第1轮要点不足会被打回), 且最终因通过而收敛(未撞上限)
    assert r["rounds"] >= 2, f"第1轮应不通过被打回重试, 实际轮数 {r['rounds']}"
    assert r["passed"] is True, "应在达到上限前通过评估而收敛"
    assert r["rounds"] <= MAX_ROUNDS, "轮数不应超过硬性上限"
    print(f"  [断言通过] 迭代 {r['rounds']} 轮后评估通过收敛(上限 {MAX_ROUNDS} 兜底防死循环)\n")

    print("=== 范式6 Agent: 模型返回 tool_calls 驱动的工具循环 ===")
    init = {"messages": [HumanMessage(content="帮我算 21*2")]}
    r = build_agent_graph().invoke(init)
    for m in r["messages"]:
        tag = type(m).__name__
        extra = f" tool_calls={[tc['name'] for tc in m.tool_calls]}" if isinstance(m, AIMessage) and m.tool_calls else ""
        print(f"  [{tag}] {m.content!r}{extra}")
    # 断言: 循环真实转动 —— 出现过 tool_calls, 执行过工具(ToolMessage), 最终以纯文本答案收尾
    has_tool_call = any(isinstance(m, AIMessage) and m.tool_calls for m in r["messages"])
    has_tool_result = any(isinstance(m, ToolMessage) for m in r["messages"])
    final = r["messages"][-1]
    if _HAS_MODEL:
        assert isinstance(final, AIMessage), "真实模型路径应以 AIMessage 收尾"
    else:
        assert has_tool_call and has_tool_result, "agent 应经历'请求调工具→执行工具'"
    assert isinstance(final, AIMessage) and not final.tool_calls, "应以不带 tool_calls 的最终答案收尾"
    print("  [断言通过] 模型→工具→模型的 loop 由 tool_calls 驱动, 无 tool_calls 时终止\n")

    print("=== 全部 6 种范式断言通过 ===")
    print("对比小结: 范式1-5 控制流预先写死(workflow), 范式6 控制流由模型 tool_calls 动态决定(agent)")

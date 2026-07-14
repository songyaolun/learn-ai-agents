"""DeepAgents 进阶 —— 条件式 interrupt (InterruptOnConfig.when) + v3 事件流骨架。

本文件教两件事:

(1) 条件式人工审批: InterruptOnConfig 的 when 谓词。
    对比 deepagents/hitl.py: 那里的 interrupt 是"无条件"的——只要模型调用了
    send_email, 就一定暂停等人工。但很多真实场景里我们只想在"满足某条件时"才打断:
    比如转账金额超过阈值才需要人审, 小额自动放行。when 谓词就是干这个的:
        interrupt_on={"transfer_funds": InterruptOnConfig(
            allowed_decisions=[...],
            when=lambda req: req.tool_call["args"].get("amount", 0) > 阈值,
        )}
    when 返回 True → 暂停等人工; 返回 False → 自动放行 (auto-approve), 不打断。

(2) 实验性的 v3 事件流 (astream_events version="v3") 骨架。
    对比 deepagents/stream.py: 那里用的是稳定的 stream(stream_mode=[...], version="v2"),
    结论是 v2 的 updates 流里看不到 subagent 内部逐步执行 (子 agent 整轮被封装成一次
    task 工具结果)。stream.py 文件头提到有个实验性的 stream_events(v3) + interleave API
    号称能区分 coordinator / subagent 的消息流, 但当时没有真正演示。
    本文件把它落成一个"可运行骨架": astream_events(version="v3") 这个方法本身在
    langgraph 里是存在的 (已 introspect 确认签名带 version 参数), 但"用 interleave 把
    coordinator vs subagent 事件流干净拆开"这套用法在 0.6.12 上仍属 beta、且没有稳定
    的公开 API 名, 因此拆流那一步用 `# 接入点:` 明确标出, 不臆造方法名。

踩坑记录 1 (when 谓词收到的是什么 —— 已从源码/introspect 核对):
  when 的签名是 Callable[[ToolCallRequest], bool]。它收到一个 ToolCallRequest
  (dataclass, 字段: tool_call / tool / state / runtime)。取工具参数要走
  req.tool_call["args"] (tool_call 是个 dict, 带 name/args/id)。
  另外源码注释指出: 在 "batch" 模式下这个 request 是用 tool=None 构造的,
  runtime 也不是 ToolRuntime——所以 when 里别去碰 req.tool / req.runtime.tool_call_id
  之类的字段, 只安全依赖 req.tool_call, 才能同时兼容 batch 和 per_call 两种模式。

踩坑记录 2 (v3 不稳定 —— 已实测):
  astream_events(version="v3") 与稳定的 v1/v2 不同: 在 langgraph 1.2.7 上它并不是
  "可直接 async for 的异步生成器", 而是返回一个协程 (内部 async def _apregel_stream_v3),
  必须先 `await` 拿到底层 AsyncGraphRunStream 再 `async for`, 否则会抛
  TypeError: 'async for' requires ... got coroutine。而且 v3 迭代出的是 ProtocolEvent
  (type/method/params), 与 v2 的 StreamEvent (event/data/metadata) schema 不同;
  "interleave 分离 coordinator/subagent 流"更是 beta。所以 (2) 只给可运行骨架 + 接入点,
  并对 v3 段包 try/except 优雅降级, 真实拆流逻辑请以你当时安装版本的官方文档为准,
  不要照抄硬编码字段名。

诚实说明: 本机无 MODEL_ID / API key, 带模型的 invoke / 事件流无法跑到底。__main__ 里
做的是"不依赖模型"的自检: when 谓词对样例输入返回 bool 且阈值逻辑正确 (这部分完全
不需要模型!), 以及 InterruptOnConfig 能被 create_deep_agent 接受 (agent 能构造)。
带模型的部分用 _HAS_MODEL 守卫, 检测不到就清晰跳过, 绝不谎称"已跑通"。

官方文档:
  https://docs.langchain.com/oss/python/deepagents/human-in-the-loop
  https://docs.langchain.com/oss/python/deepagents/event-streaming
"""

import inspect
import os

from dotenv import load_dotenv
from deepagents import create_deep_agent
from langchain.agents.middleware import InterruptOnConfig, ToolCallRequest
from langchain_anthropic import ChatAnthropic
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

load_dotenv(override=True)

# 没有 MODEL_ID 时 model=None, 只做不依赖模型的自检 (见文件头"诚实说明")。
_HAS_MODEL = "MODEL_ID" in os.environ

model = (
    ChatAnthropic(
        model=os.environ["MODEL_ID"],
        base_url=os.getenv("ANTHROPIC_BASE_URL") or None,
    )
    if _HAS_MODEL
    else None
)

# 超过这个金额的转账才需要人工审批, 否则自动放行。
APPROVAL_THRESHOLD = 1000


def transfer_funds(to: str, amount: float) -> str:
    """Transfer money to a recipient. 有真实副作用 (模拟)。大额需人工审批, 小额自动执行。"""
    return f"(模拟) 已向 {to} 转账 {amount} 元。"


def needs_approval(req: ToolCallRequest) -> bool:
    """when 谓词: 只有金额超过阈值才打断等人工审批, 否则返回 False = 自动放行。

    只依赖 req.tool_call["args"], 不碰 req.tool / req.runtime (见踩坑记录 1),
    以便同时兼容 batch / per_call 两种模式。
    """
    amount = req.tool_call["args"].get("amount", 0)
    return amount > APPROVAL_THRESHOLD


# 条件式 interrupt: 用 InterruptOnConfig 而不是普通 dict, 关键就是多了一个 when 谓词。
# 对比 hitl.py 的 {"send_email": {"allowed_decisions": [...]}} (无条件、每次都拦),
# 这里是"金额 > 阈值才拦, 否则自动放行"。
agent = (
    create_deep_agent(
        model=model,
        tools=[transfer_funds],
        system_prompt="You are a banking assistant. Use transfer_funds to move money.",
        interrupt_on={
            "transfer_funds": InterruptOnConfig(
                allowed_decisions=["approve", "reject"],
                when=needs_approval,
            )
        },
        # 只要可能触发 interrupt, 就必须配 checkpointer 保存暂停状态 (同 hitl.py)。
        checkpointer=InMemorySaver(),
    )
    if _HAS_MODEL
    else None
)


# ============ (2) 实验性 v3 事件流骨架: 尝试区分 coordinator vs subagent ============
#
# 这是 stream.py 文件头提到、但没演示的实验性 API。astream_events(version="v3") 方法
# 本身存在 (已 introspect 确认), 但"干净拆分 coordinator/subagent 事件流"的 interleave
# 用法在 0.6.12 仍是 beta, 没有稳定公开 API 名, 因此拆流那一步用 `# 接入点:` 标出。
async def stream_v3_skeleton(event_agent, payload, config):
    """v3 事件流骨架 (仍为 beta, 此处为骨架, 不臆造拆流方法名)。

    重要踩坑 (langgraph 1.2.7 实测): astream_events(version="v3") 与稳定的 v1/v2
    不同——它不是"直接可 async for 的异步生成器", 而是返回一个 **协程**
    (内部走 async def _apregel_stream_v3)。必须先 `await` 拿到底层的
    AsyncGraphRunStream (它才是异步可迭代对象), 再 `async for` 消费; 否则
    `async for event in agent.astream_events(..., version="v3")` 会直接抛
    TypeError: 'async for' requires an object with __aiter__ ... got coroutine。

    另一个差异: v3 迭代出来的是 ProtocolEvent (字段是 type/method/params),
    与 v1/v2 的 StreamEvent (字段是 event/data/metadata) schema 不同。所以下面
    仍按 v2 的 event/metadata 形状做的解析在 v3 事件上多半取不到值 (安全地 no-op),
    真正的拆流/取 token 逻辑要按 v3 的 ProtocolEvent schema 来写——这部分仍属 beta,
    故只留接入点, 不硬编码字段名。
    """
    # v3: 先 await 协程拿到异步可迭代的事件流 (见上方踩坑), 若返回的仍是可直接
    # 迭代的对象则直接用——两种形态都兼容。
    stream = event_agent.astream_events(payload, config=config, version="v3")
    if inspect.isawaitable(stream):
        stream = await stream

    async for event in stream:
        kind = event.get("event")
        meta = event.get("metadata", {}) or {}

        # 接入点: 如何判定一个事件属于 coordinator 还是某个 subagent。
        #   v3 里通常靠 metadata 的层级信息 (例如 checkpoint_ns 里是否含子图命名空间、
        #   langgraph_node 是不是 task/子 agent 节点) 来区分。具体字段名随版本变化,
        #   请以你安装版本的 event-streaming 文档为准, 不要硬编码下面这个占位判断。
        namespace = meta.get("checkpoint_ns", "") or ""
        is_subagent = ":" in namespace  # 占位启发式: 子图命名空间通常更深, 仅示意!

        who = "subagent" if is_subagent else "coordinator"

        # 接入点: 真正的 token 逐条打印 / 分流聚合逻辑写在这里。
        if kind == "on_chat_model_stream":
            chunk = event.get("data", {}).get("chunk")
            text = getattr(chunk, "text", "") if chunk is not None else ""
            if text:
                print(f"[{who}] {text}", end="", flush=True)


if __name__ == "__main__":
    # ---------- 不依赖模型的自检 (任何机器都会跑) ----------
    # 1) when 谓词是可调用的, 且对样例输入返回 bool、阈值逻辑正确。
    #    直接手工造一个 ToolCallRequest (无需模型) 喂给谓词。
    big = ToolCallRequest(
        tool_call={"name": "transfer_funds", "args": {"to": "x", "amount": 5000}, "id": "1"},
        tool=None,
        state={},
        runtime=None,
    )
    small = ToolCallRequest(
        tool_call={"name": "transfer_funds", "args": {"to": "y", "amount": 10}, "id": "2"},
        tool=None,
        state={},
        runtime=None,
    )
    r_big = needs_approval(big)
    r_small = needs_approval(small)
    assert isinstance(r_big, bool) and isinstance(r_small, bool), "when 必须返回 bool"
    assert r_big is True, "大额 (>阈值) 应触发 interrupt"
    assert r_small is False, "小额 (<=阈值) 应自动放行"
    print(f"[结构化断言] when 谓词返回 bool 且阈值逻辑正确: 5000→{r_big}, 10→{r_small}")

    # 2) InterruptOnConfig 能正常构造, 且携带 when 谓词。
    cfg = InterruptOnConfig(allowed_decisions=["approve", "reject"], when=needs_approval)
    assert callable(cfg["when"])
    assert cfg["allowed_decisions"] == ["approve", "reject"]
    print("[结构化断言] InterruptOnConfig(when=...) 构造合法, when 可调用。\n")

    if not _HAS_MODEL:
        print("(未检测到 MODEL_ID: 已完成不依赖模型的 when 谓词 / 配置自检; 以下带模型")
        print(" 的条件 interrupt 与 v3 事件流演示全部跳过。设置 MODEL_ID 后可看到真实行为。)")
        raise SystemExit(0)

    # ---------- 带模型: 条件式 interrupt 的真实行为 ----------
    # 小额: when 返回 False → 不打断, 直接执行完。
    print("=== 场景 A: 小额转账 (10 元, 低于阈值) → 自动放行, 不打断 ===")
    cfgA = {"configurable": {"thread_id": "cond-small"}}
    agent.invoke(
        {"messages": [{"role": "user", "content": "向 alice 转账 10 元。"}]},
        config=cfgA,
    )
    snapA = agent.get_state(cfgA)
    if next((t for t in snapA.tasks if t.interrupts), None) is None:
        print("  未命中 interrupt (符合预期: 小额自动放行)。")
        print(f"  最终回复: {snapA.values['messages'][-1].text}")

    # 大额: when 返回 True → 暂停等人工。
    print("\n=== 场景 B: 大额转账 (50000 元, 超过阈值) → 触发 interrupt 等人工 ===")
    cfgB = {"configurable": {"thread_id": "cond-big"}}
    agent.invoke(
        {"messages": [{"role": "user", "content": "向 bob 转账 50000 元。"}]},
        config=cfgB,
    )
    snapB = agent.get_state(cfgB)
    task = next((t for t in snapB.tasks if t.interrupts), None)
    if task is not None:
        for action in task.interrupts[0].value["action_requests"]:
            print(f"  待审批工具调用: {action['name']}({action['args']})")
        agent.invoke(Command(resume={"decisions": [{"type": "approve"}]}), config=cfgB)
        print(f"  approve 后最终回复: {agent.get_state(cfgB).values['messages'][-1].text}")
    else:
        print("  (本次模型未调用 transfer_funds, 未命中; 可重跑一次。)")

    # ---------- 带模型: v3 事件流骨架 (beta, 仅示意) ----------
    print("\n=== 场景 C: 实验性 v3 事件流 (beta 骨架, coordinator/subagent 拆流为接入点) ===")
    import asyncio

    v3_agent = create_deep_agent(
        model=model,
        tools=[transfer_funds],
        system_prompt="You are a helpful assistant.",
        checkpointer=InMemorySaver(),
    )
    # v3 是 beta API: 不同 langgraph 版本下 astream_events(version="v3") 的返回形态与
    # 事件 schema 都可能变化 (本机 1.2.7 上它返回协程、迭代出 ProtocolEvent)。为了不让
    # 这个实验性演示拖垮整份脚本 (场景 A/B 已通过), 这里包一层 try/except 优雅降级。
    try:
        asyncio.run(
            stream_v3_skeleton(
                v3_agent,
                {"messages": [{"role": "user", "content": "向 alice 转账 5 元。"}]},
                {"configurable": {"thread_id": "v3-skeleton"}},
            )
        )
        print("\n(以上 coordinator/subagent 判定为占位启发式; 生产拆流请按接入点接入正式 API。)")
    except Exception as exc:  # noqa: BLE001 —— beta 演示, 任何异常都降级为提示而非崩溃
        print(
            "  v3 事件流为 beta: 当前 langgraph 版本下 astream_events(version='v3') 的"
            "\n  返回形态/事件 schema 仍在演进 (本次跳过, 不影响场景 A/B)。"
        )
        print(f"  (捕获到: {type(exc).__name__}: {exc})")

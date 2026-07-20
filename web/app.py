"""Chainlit 网页交互 —— 四个 chat profile, 把本仓库的四种 agent 玩法都搬到浏览器。

对比 claude-code/、langchain/、langgraph/、deepagents/、rag/ 里的脚本: 那些都是终端里
一次性跑完打印结果 (或者 REPL), 这里用 Chainlit 起一个本地网页, 通过左上角的 Chat Profile
切换四种不同玩法, 在浏览器里实时看到过程:
  - Deep Research:  deepagents 的 research agent (原有内容, 未改动)
  - 人工审批 HITL:   langchain/middleware_hitl.py 的网页版, 用 cl.AskActionMessage 弹窗审批
  - 多 Agent 编排:   langgraph/multi_agent.py 的网页版, 用 cl.Step 展示 supervisor 的路由过程
  - RAG 检索问答:    rag/quickstart.py 的网页版, 检索步骤由 LangchainCallbackHandler 自动折叠展示

四个 profile 的 agent/graph 构建逻辑是在本文件内独立实现的, 不 import 同名的 langchain/、
langgraph/、rag/ 目录 —— 那些目录名和同名的 pip 包 (langchain、langgraph) 撞名, 混着 import
容易出隐蔽的 shadowing bug, 也符合仓库里"每个脚本保持独立可读、不抽取公共模块"的一贯风格。

运行: uv run chainlit run web/app.py  (自动开 http://localhost:8000)
官方文档: https://docs.chainlit.com/integrations/langchain
        https://docs.chainlit.com/concepts/chat-profiles
        https://docs.chainlit.com/concepts/human-feedback (AskActionMessage)
"""

import base64
import os
from pathlib import Path
from typing import Annotated, Literal, TypedDict

import chainlit as cl
from deepagents import create_deep_agent
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain.messages import AIMessageChunk
from langchain_anthropic import ChatAnthropic
from langchain_tavily import TavilySearch
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_voyageai import VoyageAIEmbeddings
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph, add_messages
from langgraph.types import Command
from pydantic import BaseModel, Field

load_dotenv(override=True)

RESEARCH = "Deep Research"
HITL = "人工审批 (HITL)"
SUPERVISOR = "多 Agent 编排 (Supervisor)"
RAG = "RAG 检索问答"


def _make_model() -> ChatAnthropic:
    # 沿用 ch01-ch03 + deepagents 的接入方式: ANTHROPIC_BASE_URL + MODEL_ID
    return ChatAnthropic(
        model=os.environ["MODEL_ID"],
        base_url=os.getenv("ANTHROPIC_BASE_URL") or None,
    )


# ============================================================
# Profile 1: Deep Research —— 对应 deepagents/research.py
# ============================================================
def build_research_agent() -> object:
    search = TavilySearch(max_results=5)  # Tavily 搜索, 需要 TAVILY_API_KEY
    return create_deep_agent(
        model=_make_model(),
        tools=[search],
        system_prompt=(
            "You are a research assistant. Use the search tool to find current, accurate "
            "information. Plan your research, search multiple queries if needed, then "
            "synthesize a clear answer with sources."
        ),
        checkpointer=InMemorySaver(),
        subagents=[
            {
                "name": "researcher",
                "description": "Delegate a focused research subtopic to this subagent.",
                "system_prompt": "You are a great researcher. Search and return a brief, accurate summary.",
            }
        ],
    )


async def handle_research(message: cl.Message) -> None:
    agent = cl.user_session.get("agent")
    config = {
        "configurable": {"thread_id": cl.context.session.id},
        "callbacks": [cl.LangchainCallbackHandler()],
    }
    final_answer = cl.Message(content="")
    user_content = _build_user_content(message)
    async for chunk, _metadata in agent.astream(
        {"messages": [{"role": "user", "content": user_content}]},
        stream_mode="messages",
        config=config,
    ):
        if isinstance(chunk, AIMessageChunk) and chunk.text:
            await final_answer.stream_token(chunk.text)
    await final_answer.send()


def _build_user_content(message: cl.Message) -> str | list[dict]:
    """把 chainlit 消息(可能带图片附件)转成 LangChain 消息 content 格式."""
    image_elements = [
        el for el in (message.elements or [])
        if getattr(el, "mime", "") and el.mime.startswith("image/") and el.path
    ]
    if not image_elements:
        return message.content

    blocks: list[dict] = [{"type": "text", "text": message.content or "请识别这些图片。"}]
    for el in image_elements:
        image_bytes = Path(el.path).read_bytes()
        blocks.append({
            "type": "image",
            "base64": base64.b64encode(image_bytes).decode("utf-8"),
            "mime_type": el.mime,
        })
    return blocks


# ============================================================
# Profile 2: 人工审批 HITL —— 对应 langchain/middleware_hitl.py
# ============================================================
def get_weather(city: str) -> str:
    """Get weather for a given city. 无副作用, 无需审批."""
    return f"It's always sunny in {city}!"


def send_email(to: str, subject: str, body: str) -> str:
    """Send an email. 有真实副作用 (模拟), 需要人工审批."""
    return f"(模拟) 已发送邮件给 {to}, 主题: {subject}"


def build_hitl_agent() -> object:
    return create_agent(
        model=_make_model(),
        tools=[get_weather, send_email],
        system_prompt="You are a helpful assistant.",
        middleware=[
            HumanInTheLoopMiddleware(
                interrupt_on={
                    "send_email": {"allowed_decisions": ["approve", "edit", "reject"]}
                },
            )
        ],
        checkpointer=InMemorySaver(),
    )


async def _resolve_hitl(agent, config: dict, result: dict) -> None:
    """跑完了就发最终回复; 命中 interrupt 就弹窗问人, 处理完决策后递归继续跑."""
    snapshot = await agent.aget_state(config)
    if not snapshot.next:
        await cl.Message(content=str(result["messages"][-1].text)).send()
        return

    request = snapshot.tasks[0].interrupts[0].value
    action = request["action_requests"][0]  # demo 里一次只有一个待审批工具调用
    res = await cl.AskActionMessage(
        content=f"agent 想要调用工具 **{action['name']}**, 参数: `{action['args']}`\n\n是否批准?",
        actions=[
            cl.Action(name="approve", payload={}, label="✅ 批准"),
            cl.Action(name="edit", payload={}, label="✏️ 修改收件人后执行"),
            cl.Action(name="reject", payload={}, label="❌ 拒绝"),
        ],
    ).send()

    if res is None or res.get("name") == "reject":
        decision = {"type": "reject", "message": "用户在网页上拒绝了该操作。"}
    elif res.get("name") == "edit":
        answer = await cl.AskUserMessage(content="请输入修改后的收件人邮箱:").send()
        new_to = answer["output"] if answer else action["args"].get("to")
        decision = {
            "type": "edit",
            "edited_action": {
                "name": action["name"],
                "args": {**action["args"], "to": new_to},
            },
        }
    else:
        decision = {"type": "approve"}

    new_result = await agent.ainvoke(Command(resume={"decisions": [decision]}), config=config)
    await _resolve_hitl(agent, config, new_result)


async def handle_hitl(message: cl.Message) -> None:
    agent = cl.user_session.get("agent")
    config = {"configurable": {"thread_id": cl.context.session.id}}
    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": message.content}]}, config=config
    )
    await _resolve_hitl(agent, config, result)


# ============================================================
# Profile 3: 多 Agent 编排 —— 对应 langgraph/multi_agent.py
# ============================================================
class _SupervisorState(TypedDict):
    messages: Annotated[list, add_messages]


class _RouteDecision(BaseModel):
    next: Literal["researcher", "writer", "FINISH"] = Field(
        description="下一步交给谁处理; 所有信息都已齐备、writer 已给出最终答案时输出 FINISH"
    )
    instruction: str = Field(
        default="", description="给下一个 worker 的具体指令 (FINISH 时可以不填)"
    )


_SUPERVISOR_PROMPT = (
    "你是任务调度者 (supervisor), 手下有两个 worker:\n"
    "- researcher: 只负责上网搜索、收集事实信息, 不写最终答案\n"
    "- writer: 只负责根据已收集的信息整理成最终答案, 不做搜索\n"
    "根据当前对话历史决定下一步交给谁、给出具体指令。"
    "在 writer 产出最终答案之前不能输出 FINISH —— 即使 researcher 收集的信息已经够用, "
    "也必须先交给 writer 整理成一份完整答案, 再输出 FINISH。"
)


def build_supervisor_graph() -> object:
    model = _make_model()
    router_model = model.with_structured_output(_RouteDecision)
    researcher_agent = create_agent(
        model=model,
        tools=[TavilySearch(max_results=5)],
        system_prompt="You are a researcher. Search and report concise, factual findings.",
    )
    writer_agent = create_agent(
        model=model,
        system_prompt="You are a writer. Synthesize a clear final answer from the conversation so far.",
    )

    def supervisor_node(state: _SupervisorState) -> Command[Literal["researcher", "writer", "__end__"]]:
        decision = router_model.invoke(
            [{"role": "system", "content": _SUPERVISOR_PROMPT}, *state["messages"]]
        )
        if decision.next == "FINISH":
            return Command(goto=END)
        return Command(
            goto=decision.next,
            update={"messages": [HumanMessage(content=decision.instruction, name="supervisor")]},
        )

    def researcher_node(state: _SupervisorState) -> Command[Literal["supervisor"]]:
        instruction = state["messages"][-1].content
        result = researcher_agent.invoke({"messages": [{"role": "user", "content": instruction}]})
        findings = str(result["messages"][-1].text)
        return Command(
            goto="supervisor",
            update={"messages": [HumanMessage(content=findings, name="researcher")]},
        )

    def writer_node(state: _SupervisorState) -> Command[Literal["supervisor"]]:
        result = writer_agent.invoke({"messages": state["messages"]})
        draft = str(result["messages"][-1].text)
        return Command(
            goto="supervisor",
            update={"messages": [HumanMessage(content=draft, name="writer")]},
        )

    builder = StateGraph(_SupervisorState)
    builder.add_node("supervisor", supervisor_node)
    builder.add_node("researcher", researcher_node)
    builder.add_node("writer", writer_node)
    builder.add_edge(START, "supervisor")
    return builder.compile()


async def handle_supervisor(message: cl.Message) -> None:
    graph = cl.user_session.get("graph")
    writer_text = ""
    last_text = ""  # 兜底: 万一 supervisor 没按提示走 writer 就 FINISH 了, 也不至于什么都不发
    async for update in graph.astream(
        {"messages": [{"role": "user", "content": message.content}]},
        config={"recursion_limit": 15},
        stream_mode="updates",
    ):
        for source, data in update.items():
            if not data or not data.get("messages"):
                continue  # supervisor 判定 FINISH 时没有 messages 更新
            content = str(data["messages"][-1].text)
            async with cl.Step(name=source, type="run") as step:
                step.output = content
            if source == "writer":
                writer_text = content
            elif source == "researcher":
                last_text = content
    await cl.Message(content=writer_text or last_text or "(没有生成最终答案)").send()


# ============================================================
# Profile 4: RAG 检索问答 —— 对应 rag/quickstart.py
# ============================================================
_RAG_DOCS = [
    "claude-code/ 目录用裸 Anthropic SDK 手写 agent loop: ch01 只有 bash 工具, "
    "ch02 加入 read/write/edit 文件工具和消息规范化, ch03 加入 todo 计划工具。",
    "langchain/ 目录用 create_agent 一行组装 model+tools+system_prompt, "
    "agent loop 由框架托管, 不需要手写工具分发和消息拼接逻辑。",
    "langchain/middleware_hitl.py 用 HumanInTheLoopMiddleware 给特定工具加人工审批, "
    "支持 approve/edit/reject/respond 四种人工决策。",
    "langchain/middleware_summarization.py 用 SummarizationMiddleware 在对话历史过长时"
    "自动压缩成摘要, 避免超出模型上下文窗口。",
    "langgraph/ 目录手动用 StateGraph 搭图, 展示 state、条件路由、interrupt 人工审批、"
    "SqliteSaver 持久化等 runtime 层概念。",
    "langgraph/multi_agent.py 用 supervisor 模式编排多个平级 agent, "
    "每个节点返回 Command(goto=...) 同时完成状态更新和路由, 不需要手写条件边函数。",
    "deepagents/ 目录用 create_deep_agent 在 create_agent 之上叠加 planning、"
    "虚拟文件系统、subagents 委派, 适合长时运行的复杂任务。",
    "web/app.py 用 Chainlit 把仓库里的四种 agent 玩法都搬到浏览器, "
    "支持流式输出、多轮记忆、人工审批弹窗和图片输入。",
]


def build_rag_agent() -> object:
    embeddings = VoyageAIEmbeddings(model="voyage-3.5")
    vector_store = InMemoryVectorStore.from_texts(_RAG_DOCS, embedding=embeddings)
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    @tool
    def search_repo_docs(query: str) -> str:
        """Search this repo's module descriptions for the given query and return the most relevant snippets."""
        results = retriever.invoke(query)
        if not results:
            return "(没有找到相关内容)"
        return "\n---\n".join(doc.page_content for doc in results)

    return create_agent(
        model=_make_model(),
        tools=[search_repo_docs],
        system_prompt=(
            "You answer questions about this repository. Always call search_repo_docs "
            "first to ground your answer in the retrieved snippets; do not make things up."
        ),
        checkpointer=InMemorySaver(),
    )


async def handle_rag(message: cl.Message) -> None:
    agent = cl.user_session.get("agent")
    config = {
        "configurable": {"thread_id": cl.context.session.id},
        "callbacks": [cl.LangchainCallbackHandler()],
    }
    final_answer = cl.Message(content="")
    async for chunk, _metadata in agent.astream(
        {"messages": [{"role": "user", "content": message.content}]},
        stream_mode="messages",
        config=config,
    ):
        if isinstance(chunk, AIMessageChunk) and chunk.text:
            await final_answer.stream_token(chunk.text)
    await final_answer.send()


# ============================================================
# Chainlit 生命周期钩子: chat profile 选择 → 按 profile 分发
# ============================================================
_WELCOME = {
    RESEARCH: "👋 我是 deep research agent, 接了 Tavily 真实搜索。问我任何需要查资料的问题吧。",
    HITL: (
        "👋 我可以查天气 (自动放行), 也可以帮你发邮件 (`send_email`, 需要你在弹窗里批准/修改/拒绝)。\n\n"
        "试试: '给 alice@example.com 发一封邮件, 说明天的会议改到周一'"
    ),
    SUPERVISOR: (
        "👋 我是 supervisor, 手下有 researcher (搜索) 和 writer (整理答案) 两个 worker。"
        "问我一个需要先查资料再总结的问题, 每一步路由都会展示成可折叠步骤。"
    ),
    RAG: "👋 问我关于这个仓库各个模块的问题, 我会先检索再回答 (需要配置 VOYAGE_API_KEY)。",
}


@cl.set_chat_profiles
async def chat_profiles(current_user) -> list[cl.ChatProfile]:
    return [
        cl.ChatProfile(
            name=RESEARCH,
            markdown_description="deepagents 的 deep research agent: 接 Tavily 搜索, 支持多轮记忆和图片输入。",
            default=True,
        ),
        cl.ChatProfile(
            name=HITL,
            markdown_description="给 `send_email` 工具加人工审批 (approve/edit/reject), 用弹窗模拟工具调用确认流程。",
        ),
        cl.ChatProfile(
            name=SUPERVISOR,
            markdown_description="supervisor 编排 researcher/writer 两个 worker agent, 每一步路由都会展示成可折叠步骤。",
        ),
        cl.ChatProfile(
            name=RAG,
            markdown_description="检索本仓库各模块的简介文本回答问题 (agentic RAG), 需要 VOYAGE_API_KEY。",
        ),
    ]


@cl.on_chat_start
async def on_chat_start() -> None:
    """会话开始时按当前 chat profile 构建对应的 agent/graph, 存入 user_session."""
    profile = cl.user_session.get("chat_profile")
    if profile == HITL:
        cl.user_session.set("agent", build_hitl_agent())
    elif profile == SUPERVISOR:
        cl.user_session.set("graph", build_supervisor_graph())
    elif profile == RAG:
        cl.user_session.set("agent", build_rag_agent())
    else:  # 默认 / RESEARCH
        cl.user_session.set("agent", build_research_agent())
    await cl.Message(content=_WELCOME.get(profile, _WELCOME[RESEARCH])).send()


@cl.on_message
async def on_message(message: cl.Message) -> None:
    profile = cl.user_session.get("chat_profile")
    if profile == HITL:
        await handle_hitl(message)
    elif profile == SUPERVISOR:
        await handle_supervisor(message)
    elif profile == RAG:
        await handle_rag(message)
    else:
        await handle_research(message)

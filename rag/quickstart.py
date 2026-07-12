"""RAG quickstart —— embedding + 向量库 + retriever, 包成 agent 工具做检索增强。

对比 deepagents/*: 那里的"虚拟文件系统"管理的是 agent 自己产出的中间内容 (笔记、草稿),
检索方式是精确路径读写; 这里的向量库管理的是外部知识库, 检索方式是语义相似度匹配 ——
两者都是"给 agent 扩展上下文"的手段, 但解决的是不同问题 (自身状态 vs 外部知识)。

三个新概念, 也是 LangChain 里最基础的 RAG 组件:
  - Embeddings:   把文本转成向量 (这里用 Voyage AI, Anthropic 官方推荐的 embedding 服务)
  - VectorStore:  按向量相似度存取文档 (这里用纯内存的 InMemoryVectorStore, 无需额外部署)
  - Retriever:    vector_store.as_retriever(), 把"向量检索"包装成统一的 invoke(query) 接口

检索本身不直接调用 LLM, 这里把 retriever 包成一个工具交给 create_agent, 让 agent 自己
决定"要不要查、查什么关键词", 比手写"先检索再拼 prompt"的固定流程更灵活 (agentic RAG)。

需要环境变量 VOYAGE_API_KEY (https://www.voyageai.com/ 注册获取, 有免费额度)。
官方文档: https://docs.langchain.com/oss/python/langchain/retrieval
"""

import os

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_voyageai import VoyageAIEmbeddings
from langchain.agents import create_agent

load_dotenv(override=True)

model = ChatAnthropic(
    model=os.environ["MODEL_ID"],
    base_url=os.getenv("ANTHROPIC_BASE_URL") or None,
)

# 知识库: 用本仓库自身各模块的介绍做语料, 方便直接验证检索结果是否准确
DOCS = [
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
    "web/app.py 用 Chainlit 把 deepagents 的 research agent 搬到浏览器, "
    "支持流式输出、多轮记忆和图片输入。",
]

embeddings = VoyageAIEmbeddings(model="voyage-3.5")
vector_store = InMemoryVectorStore.from_texts(DOCS, embedding=embeddings)
retriever = vector_store.as_retriever(search_kwargs={"k": 3})


@tool
def search_repo_docs(query: str) -> str:
    """Search this repo's module descriptions for the given query and return the most relevant snippets."""
    results = retriever.invoke(query)
    if not results:
        return "(没有找到相关内容)"
    return "\n---\n".join(doc.page_content for doc in results)


agent = create_agent(
    model=model,
    tools=[search_repo_docs],
    system_prompt=(
        "You answer questions about this repository. Always call search_repo_docs "
        "first to ground your answer in the retrieved snippets; do not make things up."
    ),
)


if __name__ == "__main__":
    questions = [
        "这个仓库里, 哪个模块负责人工审批工具调用?",
        "多 agent 编排的 supervisor 模式是在哪个文件里实现的?",
        "对话历史太长会被自动压缩吗? 用的是什么机制?",
    ]
    for q in questions:
        result = agent.invoke({"messages": [{"role": "user", "content": q}]})
        print(f"Q: {q}")
        print(f"A: {result['messages'][-1].content}\n")

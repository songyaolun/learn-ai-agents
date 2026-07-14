"""LangChain RAG 演示 —— 检索增强生成 (Retrieval-Augmented Generation)。

对比 langchain/quickstart.py: 那里的 agent 只能调用工具获取实时数据,
这里的 agent 会先从向量库中检索相关文档, 再结合检索结果生成回答,
适合需要结合私有知识或历史数据的场景。

RAG = Retrieval-Augmented Generation, 检索增强生成 —— 先检索资料再让模型据此作答,
核心链路: 文档加载 → 文本切分 → 嵌入 → 向量库存储 → 检索 → 生成回答

官方文档: https://docs.langchain.com/oss/python/langchain/retrievers

前置依赖与降级方案:
- 真实嵌入模型需要 API key (如: voyageai、openai)
- 无网络时可用 FakeEmbeddings 演示链路 (不生成真实向量, 但能跑通流程)
- 文本切分和向量库存储可无网络运行
"""

import os
from typing import List

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_anthropic import ChatAnthropic
from langchain_core.documents import Document
from langchain_core.embeddings import FakeEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore

# 加载环境变量
load_dotenv(override=True)

# 模型初始化沿用既有约定。注意: 这里用一个工厂函数延迟创建模型, 而不是在模块顶层直接
# 构造 —— 因为本文件的核心价值是演示"文档加载→切分→嵌入→向量库→检索"这条 **无网络**
# 链路, 它完全不依赖真实模型。若在顶层写 model=ChatAnthropic(model=os.environ["MODEL_ID"]),
# 没配 .env 时 import 阶段就会 KeyError, 连无网络链路都跑不起来。延迟到真正要调用
# agent 时 (需本地配置 .env) 再构造, 才能让前半段随时可跑可验证。
def build_model() -> ChatAnthropic:
    """需本地配置 .env (MODEL_ID / ANTHROPIC_API_KEY) 后才能成功构造。"""
    return ChatAnthropic(
        model=os.environ["MODEL_ID"],
        base_url=os.getenv("ANTHROPIC_BASE_URL") or None,
    )

# 1. 文档加载 (示例用硬编码文档, 实际可从文件/网页/数据库加载)
documents = [
    Document(
        page_content="LangChain 1.0 发布于 2024 年 10 月, 引入了 create_agent 等新原语",
        metadata={"source": "langchain-docs"}
    ),
    Document(
        page_content="LangChain 支持多种向量库, 包括 FAISS、Pinecone、Chroma 等",
        metadata={"source": "langchain-docs"}
    ),
    Document(
        page_content="MCP 协议是 LangChain 1.0 新增的外部工具接入标准",
        metadata={"source": "langchain-docs"}
    )
]

# 2. 文本切分
# 踩坑记录: 生产 RAG 里通常用 langchain_text_splitters.RecursiveCharacterTextSplitter
# (按段落/句子/字符逐级递归切分, 更"聪明")。但当前环境未安装 langchain_text_splitters 包,
# import 会失败, 所以这里用一个"按定长切片"的简化版切分器把链路跑通。真实项目中请:
#   pip install langchain-text-splitters
# 然后替换为:
#   from langchain_text_splitters import RecursiveCharacterTextSplitter
#   splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
#   chunks = splitter.split_documents(documents)
def simple_text_splitter(documents: List[Document], chunk_size: int = 100) -> List[Document]:
    """简化版定长文本切分 (替代未安装的 RecursiveCharacterTextSplitter)。"""
    chunks = []
    for doc in documents:
        content = doc.page_content
        for i in range(0, len(content), chunk_size):
            chunk = content[i:i+chunk_size]
            chunks.append(Document(page_content=chunk, metadata=doc.metadata.copy()))
    return chunks

# 3. 嵌入 (embedding, 把文本转成向量)
# 踩坑记录: 真实嵌入 (如 voyageai / openai) 需要 API key 且要联网。为了让本文件的
# 检索链路无网络也能跑通, 这里用 FakeEmbeddings —— 它生成固定维度的"假向量", 不代表
# 真实语义, 所以检索结果的"相关性"是随机的, 仅用于演示链路是否打通, 切勿据此评估召回质量。
embeddings = FakeEmbeddings(size=10)

# 4. 向量库存储 (InMemoryVectorStore, 纯内存, 无网络依赖)
vector_store = InMemoryVectorStore.from_documents(
    documents=simple_text_splitter(documents),
    embedding=embeddings
)

# 5. 检索器
def retrieve_relevant_docs(query: str, k: int = 2) -> str:
    """检索与查询相关的文档"""
    docs = vector_store.similarity_search(query, k=k)
    return "\n".join([f"[来源: {doc.metadata['source']}] {doc.page_content}" for doc in docs])


# 创建 agent 也延迟到需要时再做 (依赖真实模型, 需本地配置 .env)
def build_agent():
    """把检索器挂成工具, 组装成一个会"先检索再作答"的 RAG agent (需本地配置 .env)。"""
    return create_agent(
        model=build_model(),
        tools=[retrieve_relevant_docs],
        system_prompt=("You are a helpful assistant that uses retrieval to answer questions."
                       "First retrieve relevant documents, then use them to answer the user's question."
                       "Always cite your sources from the retrieved documents."),
    )


if __name__ == "__main__":
    print("=== RAG 演示 ===")
    print("使用 FakeEmbeddings 演示链路 (无网络依赖)")

    # 测试文本切分 (用较小的 chunk_size=20 才能把这几条短文档真正切成多片, 便于验证)
    chunks = simple_text_splitter(documents, chunk_size=20)
    assert len(chunks) > len(documents), "文本切分失败"
    print(f"文本切分测试通过: {len(documents)} 个文档 → {len(chunks)} 个 chunk (chunk_size=20)")

    # 测试向量库存储与检索
    test_query = "LangChain 1.0 有什么新特性?"
    retrieved_docs = retrieve_relevant_docs(test_query)
    assert retrieved_docs, "检索失败"
    print(f"\n检索测试通过: 找到与 '{test_query}' 相关的文档:")
    print(retrieved_docs)

    # 需要模型调用的部分 (需本地配置 .env)
    print("\n=== 需要模型调用的部分 ===")
    if os.getenv("MODEL_ID"):
        agent = build_agent()
        result = agent.invoke({
            "messages": [
                {"role": "user", "content": test_query}
            ]
        })
        print("\n最终回答:")
        print(result["messages"][-1].text)
    else:
        print("未检测到 MODEL_ID, 跳过 agent 调用。配置 .env 后可看到『先检索再作答』的完整效果。")
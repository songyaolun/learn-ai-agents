"""LangGraph Agentic RAG —— 智能体式检索增强生成, 让图自主决定检索/打分/改写循环。

RAG = Retrieval-Augmented Generation(检索增强生成): 先从知识库检索相关片段, 再把片段
拼进 prompt 让模型基于它作答, 缓解模型知识过时/胡编的问题。

对比 langgraph/quickstart.py: 那里是"分类→按 category 直接路由到某个 handler"的一次性
线性分支, 走到 handler 就结束; 这里是一个**带回路的图**: grade_documents 给检索结果打分,
分数决定是"够用→generate"还是"不相关→rewrite_query 改写后回到 retrieve 重新检索",
retrieve↔grade↔rewrite 之间形成一个真实的循环, 而不是走完一条边就到 END。

对比"朴素 RAG"(naive RAG, 一次检索→无脑拼接→回答): 朴素 RAG 无论检索质量好坏都直接把
命中文档塞给模型, 检索没命中也照答。Agentic RAG 把"是否检索/检索到的东西够不够用/要不要
换个说法再检索"这些决策交给图里的节点来做——本文件演示的核心正是 grade(打分过滤不相关
文档)与 rewrite(改写 query 再检索)这两个朴素 RAG 没有的自主环节。

关键机制/踩坑记录:
  1. grade 打分驱动条件边: grade_documents 节点不改检索结果, 只往 state 写一个布尔
     relevant; 真正决定走 generate 还是 rewrite 的是条件边函数 decide_next, 它读
     relevant + rewrites 计数。节点负责"判断", 条件边负责"路由", 两者分开(和 quickstart
     里 classify 写 category、route 读 category 是同一套分工)。
  2. rewrite 循环必须有最大次数兜底: retrieve→grade→rewrite→retrieve 是个回路, 如果
     query 始终检索不到相关文档(比如库里根本没有), 没有上限就会无限改写。这里在 state 里
     记 rewrites 计数, decide_next 超过 MAX_REWRITES 时强制走 generate(生成"没找到"的
     兜底答案)而不是继续 rewrite。LangGraph 本身还有 recursion_limit 作第二道保险。
  3. 假 embedding 只保证结构可跑, 不代表语义质量: 本文件用 DeterministicFakeEmbedding
     (确定性假向量, 零外部依赖、可复现)灌本地样例库, similarity_search 是真实执行的向量
     检索, 但假向量没有真实语义, "相似度"排序不反映真实语义相关性。所以打分 grade 不能
     只信向量距离——本文件的 grade 用"关键词是否命中"这类可控规则(无模型密钥时)或模型
     判断(有密钥时)来判定相关性, 让分支逻辑真实转动。真实场景请换成真实 embedding + 模型
     打分。

降级说明: 环境无 MODEL_ID/密钥时, grade/generate 的模型判断用规则替身(关键词命中判相关、
模板拼接生成答案)模拟决策; 有密钥时走 ChatAnthropic。无论哪种, 向量检索与图的循环都是
真实执行的, 主逻辑非空壳。embedding 恒用 DeterministicFakeEmbedding(纯本地)。

官方文档: https://docs.langchain.com/oss/python/langgraph/rag
         https://docs.langchain.com/oss/python/integrations/vectorstores/in_memory
"""

import os
from typing import TypedDict

from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.vectorstores import InMemoryVectorStore
from langgraph.graph import END, START, StateGraph

# 有密钥才导入 ChatAnthropic; 无密钥时下面走规则替身, 不 import 也不报错
HAS_MODEL = bool(os.getenv("MODEL_ID"))
if HAS_MODEL:
    from dotenv import load_dotenv
    from langchain_anthropic import ChatAnthropic

    load_dotenv(override=True)
    model = ChatAnthropic(
        model=os.environ["MODEL_ID"],
        base_url=os.getenv("ANTHROPIC_BASE_URL") or None,
    )
else:
    model = None  # 无密钥, grade/generate 走规则替身


# ============================================================
# 1. 本地样例知识库: 4~6 条自造小知识片段 + 假向量检索(零外部依赖)
# ============================================================
# 每条片段配一组关键词, 供无密钥时的规则版 grade 判断相关性(假向量没有真实语义,
# 不能只靠向量距离判相关, 见文件头踩坑记录 #3)
DOCS = [
    Document(
        page_content="LangGraph 是一个用于构建有状态、多步骤智能体应用的框架, 用图(节点+边)来编排流程。",
        metadata={"keywords": ["langgraph", "框架", "图", "智能体", "是什么"]},
    ),
    Document(
        page_content="checkpointer 是 LangGraph 的持久化机制, 在每一步之后保存状态快照, 配合 thread_id 可断点恢复。",
        metadata={"keywords": ["checkpointer", "持久化", "状态", "快照", "恢复"]},
    ),
    Document(
        page_content="Send API 允许一个节点在运行时动态产出任意数量的并行任务, 适合数量不固定的批量处理。",
        metadata={"keywords": ["send", "并行", "map", "批量", "动态"]},
    ),
    Document(
        page_content="conditional_edges(条件边)根据状态返回值决定下一步走哪个节点, 是图里实现 if-else 分支的方式。",
        metadata={"keywords": ["条件边", "conditional", "路由", "分支"]},
    ),
    Document(
        page_content="Store 按自定义 namespace 组织长期记忆, 可跨 thread_id 共享, 和按 thread 隔离的 checkpointer 互补。",
        metadata={"keywords": ["store", "长期记忆", "namespace", "共享"]},
    ),
]

# DeterministicFakeEmbedding: 确定性假向量, 同样文本每次向量一致, 可复现, 不连任何外部服务
embeddings = DeterministicFakeEmbedding(size=256)
vector_store = InMemoryVectorStore(embeddings)  # 纯内存向量库, 无需落盘
vector_store.add_documents(DOCS)  # 灌入样例库


# ============================================================
# 2. 图状态: 记录问题、检索结果、打分结论、改写计数
# ============================================================
MAX_REWRITES = 2  # rewrite 循环上限, 超过就兜底生成"没找到", 防死循环(踩坑记录 #2)


class RagState(TypedDict):
    question: str  # 当前(可能已被改写的)问题
    original_question: str  # 最初的问题, 生成兜底答案时用得上
    documents: list[Document]  # retrieve 节点写入: 本轮检索到的文档
    relevant: bool  # grade 节点写入: 检索结果是否相关
    rewrites: int  # 已改写次数, decide_next 用它判断是否到上限
    answer: str  # generate 节点写入: 最终答案


# ============================================================
# 3. 节点定义
# ============================================================
def retrieve(state: RagState) -> dict:
    """向量检索: 用当前 question 去样例库做 similarity_search(真实执行的向量检索)。"""
    docs = vector_store.similarity_search(state["question"], k=2)  # 取相似度 top-2
    return {"documents": docs}


def _rule_grade(question: str, docs: list[Document]) -> bool:
    """规则版打分(无密钥替身): 问题里若命中任一文档的关键词, 判为相关。

    假向量没有真实语义, 不能只信向量距离(踩坑 #3), 这里用关键词命中作可控判据。
    """
    q = question.lower()
    for doc in docs:
        for kw in doc.metadata.get("keywords", []):
            if kw.lower() in q:  # 问题里出现了该文档的关键词
                return True
    return False


def grade_documents(state: RagState) -> dict:
    """给检索到的文档打分, 判断是否与问题相关; 只写结论, 不做路由(路由交给条件边)。"""
    if HAS_MODEL:
        # 有密钥: 让模型判断检索到的文档能否回答问题, 输出 yes/no
        joined = "\n".join(f"- {d.page_content}" for d in state["documents"])
        resp = model.invoke(
            [
                {
                    "role": "user",
                    "content": (
                        f"下面这些资料能否回答问题「{state['question']}」? "
                        f"只回答 yes 或 no。\n\n资料:\n{joined}"
                    ),
                }
            ]
        )
        relevant = "yes" in resp.text.lower()
    else:
        # 无密钥: 走规则替身
        relevant = _rule_grade(state["question"], state["documents"])
    return {"relevant": relevant}


def rewrite_query(state: RagState) -> dict:
    """改写问题后重新检索: 把问题往库里的术语上靠, 并把改写计数 +1。"""
    if HAS_MODEL:
        resp = model.invoke(
            [
                {
                    "role": "user",
                    "content": (
                        f"把下面这个问题改写得更利于检索一个 LangGraph 知识库(保留原意, "
                        f"补上可能的技术术语), 只输出改写后的问题:\n{state['question']}"
                    ),
                }
            ]
        )
        new_q = resp.text.strip()
    else:
        # 规则替身: 换个说法重新表述, 演示 query 被改写后再走一遍 retrieve。
        # 注意不硬塞库里的术语——那样等于"作弊命中", 就演示不出"库里真没有→改到上限兜底"了。
        new_q = f"请换个角度再说一遍: {state['question']}"
    return {"question": new_q, "rewrites": state["rewrites"] + 1}


def generate(state: RagState) -> dict:
    """基于相关文档生成答案; 若是兜底进来(不相关且到上限)则生成"没找到"式答案。"""
    context = "\n".join(f"- {d.page_content}" for d in state["documents"])
    if state["relevant"]:
        if HAS_MODEL:
            resp = model.invoke(
                [
                    {
                        "role": "user",
                        "content": f"根据下面资料回答问题「{state['original_question']}」:\n{context}",
                    }
                ]
            )
            answer = resp.text
        else:
            # 规则替身: 把命中文档拼成答案, 体现"基于检索结果生成"
            answer = f"根据知识库: {context}"
    else:
        # 到达最大改写次数仍不相关, 兜底告知没找到, 而不是硬编造
        answer = f"知识库里没有找到与「{state['original_question']}」直接相关的内容。"
    return {"answer": answer}


# ============================================================
# 4. 条件边: grade 结论 + 改写计数 共同决定 generate 还是 rewrite
# ============================================================
def decide_next(state: RagState) -> str:
    """相关→generate; 不相关但还没到上限→rewrite; 不相关且到上限→generate(兜底)。"""
    if state["relevant"]:
        return "generate"  # 打分通过, 去生成答案
    if state["rewrites"] >= MAX_REWRITES:
        return "generate"  # 改写到上限还不相关, 兜底生成(防死循环, 踩坑 #2)
    return "rewrite_query"  # 不相关且还有改写机会, 改写后重新检索


# ============================================================
# 5. 组图: retrieve → grade →(条件边)→ generate / rewrite → retrieve(回路)
# ============================================================
builder = StateGraph(RagState)
builder.add_node("retrieve", retrieve)
builder.add_node("grade_documents", grade_documents)
builder.add_node("rewrite_query", rewrite_query)
builder.add_node("generate", generate)

builder.add_edge(START, "retrieve")
builder.add_edge("retrieve", "grade_documents")
# grade 后按 decide_next 分流: 去 generate 或去 rewrite_query
builder.add_conditional_edges("grade_documents", decide_next, ["generate", "rewrite_query"])
builder.add_edge("rewrite_query", "retrieve")  # 改写完回到 retrieve, 形成循环
builder.add_edge("generate", END)

graph = builder.compile()


if __name__ == "__main__":
    print(f"=== 环境: {'有 MODEL_ID, 走真实模型' if HAS_MODEL else '无 MODEL_ID, grade/generate 走规则替身'} ===")

    # -- 先单独验证向量检索本身真实跑通(不经过图) --
    print("\n=== (0) 向量检索自测: InMemoryVectorStore + DeterministicFakeEmbedding ===")
    hits = vector_store.similarity_search("checkpointer 是什么", k=2)
    assert len(hits) > 0, "向量检索必须返回文档"
    print(f"  similarity_search 返回 {len(hits)} 篇, top1: {hits[0].page_content[:30]}...")

    # -- 用例 1: 库里有答案的问题 → 检索命中 → grade 通过 → generate(不触发 rewrite) --
    print("\n=== (1) 库里有答案: 「checkpointer 的作用是什么?」 ===")
    r1 = graph.invoke(
        {
            "question": "checkpointer 的作用是什么?",
            "original_question": "checkpointer 的作用是什么?",
            "documents": [],
            "relevant": False,
            "rewrites": 0,
            "answer": "",
        }
    )
    print(f"  检索到文档数: {len(r1['documents'])}")
    print(f"  grade 判定 relevant: {r1['relevant']}")
    print(f"  改写次数 rewrites: {r1['rewrites']}")
    print(f"  答案: {r1['answer'][:60]}...")
    assert len(r1["documents"]) > 0, "应检索到文档"
    assert r1["relevant"] is True, "库里有答案, grade 应判相关"
    assert r1["rewrites"] == 0, "命中即通过, 不应触发改写"

    # -- 用例 2: 库里没有的问题 → grade 判不相关 → rewrite 循环 → 到上限兜底 --
    print("\n=== (2) 库里没有/问得模糊: 「今天午饭吃什么?」 ===")
    r2 = graph.invoke(
        {
            "question": "今天午饭吃什么?",
            "original_question": "今天午饭吃什么?",
            "documents": [],
            "relevant": False,
            "rewrites": 0,
            "answer": "",
        },
        config={"recursion_limit": 25},  # 第二道保险, 防回路跑飞
    )
    print(f"  最终 grade relevant: {r2['relevant']}")
    print(f"  实际改写次数 rewrites: {r2['rewrites']}")
    print(f"  答案: {r2['answer']}")
    assert r2["rewrites"] >= 1, "库里没有, 应至少触发一次 rewrite"
    assert r2["rewrites"] <= MAX_REWRITES, f"改写次数必须 <= 上限 {MAX_REWRITES}, 防死循环"
    if not HAS_MODEL:
        # 规则替身下, 库里确实没有午饭相关内容, 应改到上限仍不相关→兜底
        assert r2["rewrites"] == MAX_REWRITES, "库里没有, 应改写到上限"
        assert r2["relevant"] is False, "到上限仍不相关, 应走兜底而非硬答"

    print("\n=== 结论 ===")
    print("  用例1: 检索命中→grade 通过→直达 generate, 未触发 rewrite")
    print(f"  用例2: grade 不相关→rewrite 循环, 命中上限 {MAX_REWRITES} 次后兜底 generate")
    print("  → grade 分支被走到、rewrite 循环有上限, 均已断言")

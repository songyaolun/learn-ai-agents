"""LCEL —— 不用 agent, 用管道操作符 `|` 手动拼一条"提示词模板 → 模型 → 输出解析"的链。

对比这个仓库里其他所有例子: 前面全部是"agent"范式 (create_agent/create_deep_agent),
核心是"模型自己决定要不要调用工具、要调用几轮", 循环由框架托管。
但很多场景根本不需要 agent 那套循环——流程是固定的、一条道走到底
(拼提示词 → 调模型 → 解析输出), 这种"链"式场景用 LCEL (LangChain Expression
Language) 更直接: 用 `|` 把几个 Runnable 对象串起来, 每一步的输出自动变成下一步的输入,
跟 Unix 管道 `cmd1 | cmd2 | cmd3` 是同一个思路。

三个新概念:
  - ChatPromptTemplate: 提示词模板, 用 {变量名} 占位, invoke 时传字典填空
  - `|` (管道操作符): LangChain 给 Runnable 对象重载了 `|`, `a | b` 等价于
    "a 的输出喂给 b 做输入", 串起来就是一条 RunnableSequence
  - StrOutputParser: 从模型返回的消息对象里把纯文本提取出来 (变成普通字符串)

官方文档: https://docs.langchain.com/oss/python/langchain/lcel
"""

import os

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

load_dotenv(override=True)

model = ChatAnthropic(
    model=os.environ["MODEL_ID"],
    base_url=os.getenv("ANTHROPIC_BASE_URL") or None,
)

# ChatPromptTemplate.from_messages 接收一个 (角色, 模板字符串) 列表, {language}/{text}
# 是占位符, 之后 invoke 传对应的字典就会被替换进去; system 角色对应"人设", human 角色
# 对应"用户这句话"。
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a translator. Translate the user's text to {language}. "
                   "Only output the translation, nothing else."),
        ("human", "{text}"),
    ]
)

# prompt | model | StrOutputParser() 拼出一条"链": 调用 chain.invoke(输入) 时,
# 输入先经过 prompt 变成一组消息, 这组消息喂给 model 得到模型回复 (一个消息对象),
# 消息对象再经过 StrOutputParser 提取出纯文本字符串。整条链本身也是一个 Runnable,
# 可以像 model 一样直接 .invoke()/.stream()/.batch()。
chain = prompt | model | StrOutputParser()


if __name__ == "__main__":
    # 输入是一个字典, key 要跟模板里的占位符名字对上 (language / text)
    result = chain.invoke({"language": "French", "text": "Hello, how are you?"})
    # 注意: 用 str() 包一层——链里最后一步理论上应该是纯字符串, 但如果模型开了
    # extended thinking, StrOutputParser 在这个版本里拿到的实际是 AIMessage.text
    # 这个 "TextAccessor" 对象 (打印出来跟字符串一样, 但 isinstance(..., str) 是
    # False), 用 str() 转一下更安全, 也是这个仓库里统一遵循的写法。
    print(f"翻译结果: {str(result)}")

    # .batch() 是 LCEL 链自带的能力: 一次性并发处理多个输入, 不用自己写 for 循环
    # (对比 for 循环挨个 invoke, batch 内部会做并发调用, 更快)。
    print("\n=== batch: 一次翻译多句话 ===")
    inputs = [
        {"language": "Japanese", "text": "Good morning"},
        {"language": "Spanish", "text": "Thank you very much"},
    ]
    for original, translated in zip(inputs, chain.batch(inputs)):
        print(f"{original['text']} ({original['language']}) → {str(translated)}")

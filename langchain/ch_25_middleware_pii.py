"""LangChain middleware —— 用 PIIMiddleware 自动脱敏用户消息里的敏感信息。

对比前面几个 middleware: HumanInTheLoopMiddleware 管"要不要人工审批",
SummarizationMiddleware/trim_messages 管"历史太长怎么办", 这个 middleware 管的是
数据安全——用户在对话里贴了邮箱、信用卡号这类敏感信息, PIIMiddleware 会在消息真正
被模型看到之前自动检测并脱敏, 既保护了用户隐私, 也避免这些敏感信息被原样存进日志/
对话历史/后续可能发生的模型训练数据里。

每个 PIIMiddleware 实例只负责一种 pii_type, 需要脱敏几种类型就叠几个实例。
strategy 决定脱敏方式:
  - "redact": 整个替换成占位符, 如 [REDACTED_EMAIL]
  - "mask":   保留部分特征, 其余打码, 如信用卡号只留末 4 位 ************1111
  - "hash":   替换成不可逆的哈希值 (脱敏后仍能判断"是不是同一个值", 但看不出原文)
  - "block":  直接拒绝这条消息, 不让它进入对话 (最严格)

apply_to_input/apply_to_output/apply_to_tool_results 控制在哪个环节生效
(默认只处理用户输入, 这里保持默认)。

官方文档: https://docs.langchain.com/oss/python/langchain/middleware#pii-detection
"""

import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import PIIMiddleware
from langchain_anthropic import ChatAnthropic

load_dotenv(override=True)

model = ChatAnthropic(
    model=os.environ["MODEL_ID"],
    base_url=os.getenv("ANTHROPIC_BASE_URL") or None,
)

agent = create_agent(
    model=model,
    system_prompt=(
        "You are a customer support assistant. Acknowledge what the user told you "
        "and confirm you've recorded their request."
    ),
    middleware=[
        # 邮箱地址: 整体替换成占位符, 反正后续也用不上具体地址
        PIIMiddleware("email", strategy="redact"),
        # 信用卡号: 打码但保留末 4 位, 方便人工核对"是不是那张卡"而不暴露完整号码
        PIIMiddleware("credit_card", strategy="mask"),
    ],
)


if __name__ == "__main__":
    # 用户消息里同时包含邮箱和信用卡号, 都会在进入模型之前被自动处理掉——模型看到的
    # 是脱敏后的版本, 从一开始就"看不到"原始敏感信息, 而不是"看到了但被要求不要说"。
    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "我的邮箱是 zhangsan@example.com, "
                        "信用卡号是 4111111111111111, 请帮我确认收到这个请求。"
                    ),
                }
            ]
        }
    )

    print("=== 模型实际看到的用户消息 (已脱敏) ===")
    print(result["messages"][0].content)

    print("\n=== agent 的回复 ===")
    print(result["messages"][-1].text)

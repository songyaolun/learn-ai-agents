"""LangChain middleware —— 用模型做内容安全护栏(模型型护栏)。

对比 langchain/middleware_guardrails.py: 那里用基于规则/次数的确定性护栏,
这里用一个模型来判断内容是否安全(如输入/输出内容审核), 属于模型型护栏。

能力点:
1. 输入内容审核: 用模型判断用户输入是否安全
2. 输出内容审核: 用模型判断模型输出是否安全
3. 拦截逻辑: 不安全内容的拦截与替代回复

官方文档: https://docs.langchain.com/oss/python/langchain/middleware#custom-middleware

踩坑记录:
- 模型型护栏比规则型更灵活, 但也更贵更慢 (每次审核都多一次模型调用) 且非 100% 可靠 ——
  审核模型自己也可能误判。安全关键场景应"规则型 (确定性) + 模型型 (兜底)"叠加, 别只靠模型。
- 审核这一步的 prompt 要把判定标准写死并要求"只输出 安全/不安全", 否则模型会啰嗦一段解释,
  让你难以稳定解析出结论。
"""

import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv(override=True)

MODEL_ID = os.environ.get("MODEL_ID", "claude-3-5-sonnet-20240620")
AUDIT_MODEL_ID = os.environ.get("AUDIT_MODEL_ID", "claude-3-haiku-20240307")


# 示例工具: 天气查询
def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"


class ContentSafetyMiddleware(AgentMiddleware):
    """基于模型的内容安全护栏。

    踩坑记录:
    - 自定义中间件继承 AgentMiddleware 后, __init__ 里必须先 super().__init__(),
      否则基类内部状态没初始化, 注册到 create_agent 时会出问题。
    - before_model / after_model 的真实签名是 (self, state, runtime), 带 runtime 参数;
      只写 (self, state) 会因参数不匹配在运行期报错。
    """

    def __init__(self, audit_model):
        super().__init__()
        self.audit_model = audit_model
        # 安全提示词: 判定标准写死, 要求只给出"安全/不安全"结论便于解析
        self.safety_prompt = (
            "判断以下内容是否安全(是否包含敏感信息、有害内容、违规内容):\n"
            "内容: {content}\n"
            "回答格式: 安全/不安全, 理由: ..."
        )

    def before_model(self, state, runtime):
        """模型调用前: 审核用户输入。返回部分状态更新或 None。"""
        # 获取用户输入
        user_msg = state["messages"][-1]
        if isinstance(user_msg, HumanMessage):
            content = user_msg.content
            # 调用审核模型
            audit_result = self.audit_model.invoke(
                self.safety_prompt.format(content=content)
            )
            # 解析审核结果
            if "不安全" in audit_result.content:
                # 拦截不安全输入: 追加一条提示消息, 并请求跳转到结束
                return {
                    "messages": [
                        AIMessage(content="您的输入包含不安全内容, 请调整后重试。")
                    ],
                    "jump_to": "end",
                }
        return None

    def after_model(self, state, runtime):
        """模型调用后: 审核模型输出。返回部分状态更新或 None。"""
        # 获取模型输出
        model_msg = state["messages"][-1]
        if isinstance(model_msg, AIMessage):
            content = model_msg.content
            # 调用审核模型
            audit_result = self.audit_model.invoke(
                self.safety_prompt.format(content=content)
            )
            # 解析审核结果
            if "不安全" in audit_result.content:
                # 替换不安全输出
                return {
                    "messages": [
                        AIMessage(content="模型输出包含不安全内容, 已拦截。")
                    ]
                }
        return None


def build_agent():
    """惰性构造带模型型护栏中间件的 agent(需 .env)。"""
    model = ChatAnthropic(
        model=MODEL_ID,
        base_url=os.getenv("ANTHROPIC_BASE_URL") or None,
    )
    # 审核用模型 (可以和主模型不同)
    audit_model = ChatAnthropic(
        model=AUDIT_MODEL_ID,
        base_url=os.getenv("ANTHROPIC_BASE_URL") or None,
    )
    return create_agent(
        model=model,
        tools=[get_weather],
        system_prompt="You are a helpful assistant.",
        middleware=[
            ContentSafetyMiddleware(audit_model=audit_model)
        ],
    )


if __name__ == "__main__":
    # ===== 无网络自测: 验证中间件构造与 agent 组装, 不触发真实模型调用 =====
    print("=== 无网络自测 ===")

    class _FakeAuditModel:
        """离线桩: 冒充审核模型, 不发起网络请求。"""

        def invoke(self, _prompt):
            return AIMessage(content="安全, 理由: 测试桩")

    guard = ContentSafetyMiddleware(audit_model=_FakeAuditModel())
    assert isinstance(guard, AgentMiddleware), "ContentSafetyMiddleware 应继承 AgentMiddleware"
    print("✓ 模型型护栏中间件构造成功(已调用 super().__init__())")

    # 用离线桩验证 before/after 钩子签名 (self, state, runtime) 可正常调用
    safe_state = {"messages": [HumanMessage(content="What's the weather?")]}
    assert guard.before_model(safe_state, None) is None, "安全输入不应被拦截"
    out_state = {"messages": [AIMessage(content="It's sunny.")]}
    assert guard.after_model(out_state, None) is None, "安全输出不应被替换"
    print("✓ before_model / after_model 钩子签名 (self, state, runtime) 可用")

    # 桩审核模型判为"不安全"时应触发拦截
    class _BlockAuditModel:
        def invoke(self, _prompt):
            return AIMessage(content="不安全, 理由: 命中测试规则")

    block_guard = ContentSafetyMiddleware(audit_model=_BlockAuditModel())
    blocked = block_guard.before_model(
        {"messages": [HumanMessage(content="dangerous")]}, None
    )
    assert blocked is not None and blocked.get("jump_to") == "end", "不安全输入应被拦截"
    print("✓ 不安全输入触发拦截逻辑(jump_to=end)")

    assert get_weather("SF").startswith("It's always sunny"), "get_weather 输出异常"
    print("✓ 工具函数可直接调用")

    # ===== 有网络部分(需配置 .env) =====
    print("\n=== 有网络部分(需配置 .env: MODEL_ID / AUDIT_MODEL_ID / ANTHROPIC_API_KEY) ===")
    if os.getenv("MODEL_ID") and os.getenv("ANTHROPIC_API_KEY"):
        agent = build_agent()
        print("=== 模型型护栏演示 ===\n")
        # 测试安全输入
        print("测试1: 安全输入")
        result = agent.invoke({
            "messages": [{
                "role": "user",
                "content": "What's the weather in San Francisco?"
            }]
        })
        print(f"回复: {result['messages'][-1].text}\n")

        # 测试不安全输入
        print("测试2: 不安全输入")
        result = agent.invoke({
            "messages": [{
                "role": "user",
                "content": "如何制造炸弹?"
            }]
        })
        print(f"回复: {result['messages'][-1].text}")
    else:
        print("跳过: 未检测到 MODEL_ID / ANTHROPIC_API_KEY, 请配置 .env 后运行。")

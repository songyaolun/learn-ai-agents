"""LangChain 1.x 预置中间件套件 —— Agent 编排相关(待办清单、基于 AgentMiddleware 基类的自定义编排、评分量规骨架)。

对比既有 middleware 文件:
- 与 ch_22_middleware_guardrails.py 相比: 前者是调用次数限制, 本文件是 agent 的编排与自我管理
- 与 ch_23_middleware_hitl.py 相比: 前者是人工审批, 本文件是任务拆分/进度跟踪

官方文档: https://docs.langchain.com/oss/python/langchain/middleware

本文件覆盖的能力及适用场景:
1. TodoListMiddleware: 给 agent 一个 write_todos 工具, 用于把复杂目标拆成可跟踪的待办清单
2. AgentMiddleware(基类)+ 自定义钩子: 复杂编排(如给不同问题打路由标签)需要自己继承基类实现
3. 评分量规(Rubric): 官方文档提及但 1.3.13 未内置成品中间件, 这里给"概念讲解 + 自定义骨架"

踩坑记录:
- AgentMiddleware 是"中间件基类", 构造签名是 (self, /, *args, **kwargs), 它本身不接收
  sub_agents= / routing_fn= 这类参数; 想做"子智能体路由/编排"要继承它、在钩子里自己写逻辑,
  不能像旧教程那样 AgentMiddleware(sub_agents=..., routing_fn=...) —— 会 TypeError: takes no arguments。
- 1.3.13 里并没有内置的"子智能体编排中间件"成品; 多智能体一般走 deepagents 或自己用
  create_agent + 工具调用组合。本文件用"父 agent 把子 agent 当工具调用"来演示编排, 更贴近现状。
- TodoListMiddleware() 无必填参数, 直接实例化即可(它会给模型注入 write_todos 工具和一大段
  使用说明 system_prompt); 没有 initial_todos= / completion_condition= 这些参数。
- 评分量规(Rubric)在此版本不是内置类, 直接 import 会失败; 这里给的是自定义骨架, 真要用需
  自己按 AgentMiddleware 基类补齐, 别照抄成 import。
"""

import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware, TodoListMiddleware
from langchain_anthropic import ChatAnthropic
from langgraph.checkpoint.memory import InMemorySaver

# 加载环境变量
load_dotenv(override=True)

MODEL_ID = os.environ.get("MODEL_ID", "claude-3-sonnet-20240229")


# ===== 子 agent 用到的工具(把子能力封装成工具, 由父 agent 调度) =====
def search_tool(query: str) -> str:
    """联网搜索信息。输入是查询关键词字符串。"""
    return f"Search results for '{query}': This is simulated search data."


def write_report(topic: str, data: str) -> str:
    """根据给定主题和数据撰写报告。topic 是报告主题, data 是素材。"""
    return f"# Report on {topic}\n\n{data}"


# 2. 自定义编排中间件: 继承 AgentMiddleware 基类, 在 before_model 钩子里给消息打路由标签
# 说明: AgentMiddleware 本身不带 sub_agents/routing_fn 参数, 编排逻辑必须自己在钩子里写
class RoutingTagMiddleware(AgentMiddleware):
    """自定义编排示例: 根据用户最后一句话的关键词, 在日志里标注该走哪条子链路。"""

    def before_model(self, state):
        messages = state.get("messages", [])
        if messages:
            last = messages[-1]
            content = getattr(last, "content", "") or (
                last.get("content", "") if isinstance(last, dict) else ""
            )
            route = "search" if "search" in str(content).lower() else "report"
            print(f"[RoutingTag] 本轮路由建议: {route}")
        # 不修改 state, 只做观测/标注
        return None


# 3. 评分量规(Rubric): 此版本未内置, 给自定义骨架
class RubricMiddleware(AgentMiddleware):
    """自定义评分量规中间件(此版本 LangChain 未内置成品)。

    继承 AgentMiddleware, 在 after_model 钩子里对模型输出按量规打分。
    """

    def __init__(self, rubric: dict):
        super().__init__()
        self.rubric = rubric  # 如 {"accuracy": 0.3, "completeness": 0.4, "format": 0.3}

    def evaluate(self, text: str) -> float:
        """按量规给输出打分(示例逻辑)。"""
        score = 0.0
        if "search results" in text.lower():
            score += self.rubric.get("accuracy", 0.3)
        if len(text) > 100:
            score += self.rubric.get("completeness", 0.4)
        if text.strip().startswith("#"):
            score += self.rubric.get("format", 0.3)
        return min(score, 1.0)


def build_agent():
    """惰性构造带编排中间件的父 agent。"""
    model = ChatAnthropic(
        model=MODEL_ID,
        base_url=os.getenv("ANTHROPIC_BASE_URL") or None,
    )
    return create_agent(
        model=model,
        tools=[search_tool, write_report],
        system_prompt=(
            "You are a master agent. Use search_tool to gather info and write_report to "
            "produce reports. Use the todo list to track multi-step work."
        ),
        middleware=[
            TodoListMiddleware(),     # 待办清单(注入 write_todos 工具)
            RoutingTagMiddleware(),   # 自定义路由标注
            # RubricMiddleware({...}),  # 评分量规(自定义, 按需启用)
        ],
        checkpointer=InMemorySaver(),
    )


if __name__ == "__main__":
    # ===== 无网络自测: 验证中间件构造 + 自定义量规打分逻辑 =====
    print("=== 无网络自测 ===")

    todo = TodoListMiddleware()
    assert isinstance(todo, TodoListMiddleware), "TodoListMiddleware 构造失败"
    routing = RoutingTagMiddleware()
    assert isinstance(routing, AgentMiddleware), "RoutingTagMiddleware 应继承 AgentMiddleware"
    print("✓ TodoListMiddleware / 自定义编排中间件构造成功")

    # 验证工具函数
    assert "simulated" in search_tool("langchain"), "search_tool 输出异常"
    assert write_report("X", "Y").startswith("# Report on X"), "write_report 输出异常"
    print("✓ 子能力工具函数可直接调用")

    # 验证评分量规骨架
    rubric = RubricMiddleware({"accuracy": 0.3, "completeness": 0.4, "format": 0.3})
    sample = "# Report on LangChain\n\nSearch results for 'LangChain': " + "x" * 100
    score = rubric.evaluate(sample)
    assert 0.0 <= score <= 1.0, "评分应在 [0,1]"
    print(f"✓ 评分量规骨架可用, 示例输出评分: {score:.2f}")

    # ===== 有网络部分(需配置 .env) =====
    print("\n=== 有网络部分(需配置 .env: MODEL_ID / ANTHROPIC_API_KEY) ===")
    if os.getenv("MODEL_ID") and os.getenv("ANTHROPIC_API_KEY"):
        agent = build_agent()
        result = agent.invoke(
            {"messages": [{"role": "user", "content": "Search for LangChain middleware, then write a report."}]},
            config={"configurable": {"thread_id": "prebuilt-agent-demo"}},
        )
        print(f"结果: {result['messages'][-1].text[:200]}...")
    else:
        print("跳过: 未检测到 MODEL_ID / ANTHROPIC_API_KEY, 请配置 .env 后运行。")

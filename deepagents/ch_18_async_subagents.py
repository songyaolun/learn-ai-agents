"""DeepAgents async_subagents —— 用 AsyncSubAgent 把子任务委派到"远程 LangGraph 服务器"。

对比 deepagents/ch_01_quickstart.py: 那里 subagents 里的 researcher 是一个同步子 agent ——
主 agent 调用内置 task 工具时, researcher 在【本进程内】同步跑完, 主 agent 会【阻塞】
等它返回结果, 再继续。适合短、快、能立即拿到结果的子任务。

本文件演示的 AsyncSubAgent 完全不同: 子 agent 不在本进程跑, 而是【远程】的一个
Agent-Protocol / LangGraph 服务器上的图 (由 graph_id + url 指定)。主 agent 通过
AsyncSubAgentMiddleware 暴露出来的 5 个工具, 以"发起后立即交还控制权、绝不轮询"的
方式驱动它 —— 类似"提交一个后台任务拿到 task_id, 之后用户想看结果时再去查一次",
而不是傻等。适合长耗时、可后台跑、可并发多个的子任务。

AsyncSubAgentMiddleware 暴露的 5 个异步工具:
  - start_async_task : 启动一个后台任务, 立刻返回 task_id (然后就该停下来, 别自动查)
  - check_async_task : 按需查询某个任务的状态/结果 (用户主动问时才查, 不要循环轮询)
  - update_async_task: 给正在跑的任务发新指令 (会中断当前 run 在同一 thread 上重开)
  - cancel_async_task: 取消不再需要的任务
  - list_async_tasks : 一次性列出所有已跟踪任务的实时状态 (上下文被压缩后用来找回 id)
核心工作流: "launch then return control, never poll" ——
  发起后立即把控制权还给用户, 只有用户明确追问时才 check 一次; 历史里的状态永远是
  过期的, 报告状态前必须重新调工具查实时值。

【降级说明 (诚实标注)】AsyncSubAgent 指向一个【远程服务器】。本地环境没有这样的服务器,
而且"自己起一个监听端口的服务器"会违反本仓库的红线 (禁止起端口进程)。因此本文件属于
【降级-骨架】: 只离线构建并断言 AsyncSubAgent 规格 + AsyncSubAgentMiddleware 的接线,
不真正发起远程任务、不真正 invoke。url/headers/graph_id 处用 "# 接入点" 标出你需要
填入真实远程部署信息的位置。

踩坑记录:
  1. AsyncSubAgent 不进 create_deep_agent(subagents=...), 而是先包进
     AsyncSubAgentMiddleware, 再通过 middleware=[...] 挂上去 —— 这和同步 SubAgent
     直接放 subagents=[...] 的接线方式不一样, 容易搞混。
  2. url 可省略: 省略时走 LangGraph SDK 默认端点 (托管平台); 自托管服务器才需要显式
     url + headers 传鉴权。graph_id 是远程服务器上"图/assistant 的名字", 不是本地对象。
  3. 模型天生倾向"发起后马上 check 看看好了没" —— 这恰恰是这套工具最不希望的行为。
     default system_prompt 里反复强调"never poll / 发起后交还控制权", 但模型不一定
     严格照做, 必要时要在自己的 system_prompt 里再次强调。参数/提示存在 != 模型照办。

官方文档: https://docs.langchain.com/oss/python/deepagents/subagents
"""

import os

from dotenv import load_dotenv
from deepagents import create_deep_agent, AsyncSubAgent, AsyncSubAgentMiddleware
from langchain_anthropic import ChatAnthropic

load_dotenv(override=True)


# ---- 1. 声明一个远程异步子 agent 的规格 ----
# 接入点: graph_id / url / headers 都要换成你真实远程 LangGraph 部署的值。
remote_researcher = AsyncSubAgent(
    name="remote_researcher",
    description="把耗时的深度调研任务派给远程 LangGraph 服务器后台执行。一次给一个主题。",
    graph_id="research_graph",  # 接入点: 远程服务器上图/assistant 的名字
    url="https://your-langgraph-server.example.com",  # 接入点: 远程 LangGraph server URL (自托管才需要)
    headers={"Authorization": "Bearer <YOUR_TOKEN>"},  # 接入点: 远程服务器鉴权头
)


# ---- 2. 把异步子 agent 包进 AsyncSubAgentMiddleware ----
# 这一步会给主 agent 注入 start/check/update/cancel/list 这 5 个异步任务工具,
# 以及一段告诉模型"发起后别轮询"的默认 system_prompt。
def build_async_middleware() -> AsyncSubAgentMiddleware:
    """构建挂载了远程异步子 agent 的中间件。"""
    return AsyncSubAgentMiddleware(async_subagents=[remote_researcher])


def build_agent(model):
    """通过 middleware=[...] 把异步子 agent 能力接到主 agent 上。"""
    return create_deep_agent(
        model=model,
        tools=[],
        system_prompt=(
            "You are a coordinator. For long research tasks, launch a remote async "
            "subagent and IMMEDIATELY return control to the user. Never poll."
        ),
        # 关键: 异步子 agent 通过 middleware 挂载, 而非 subagents 参数
        middleware=[build_async_middleware()],
    )


if __name__ == "__main__":
    # --- 不依赖模型、也不连远程服务器的结构验证 ---
    # (a) AsyncSubAgent 规格字段就位
    assert remote_researcher["name"] == "remote_researcher"
    assert remote_researcher["graph_id"] == "research_graph"
    assert remote_researcher["url"].startswith("https://")
    assert "Authorization" in remote_researcher["headers"]

    # (b) 中间件能构建 (它内部会准备好 5 个异步工具)
    mw = build_async_middleware()
    assert mw is not None

    # (c) agent 能在 model=None 下把中间件接进去
    agent_struct = build_agent(model=None)
    assert agent_struct is not None
    print("结构验证通过: AsyncSubAgent 规格 + AsyncSubAgentMiddleware + agent 接线 OK")
    print("(降级-骨架: 无本地远程服务器, 且禁止起端口进程, 故不真正发起远程任务。)")

    # --- 需要模型 + 真实远程服务器的部分 ---
    if os.getenv("MODEL_ID"):
        model = ChatAnthropic(
            model=os.environ["MODEL_ID"],
            base_url=os.getenv("ANTHROPIC_BASE_URL") or None,
        )
        agent = build_agent(model=model)
        print("已构建带异步子 agent 的主 agent。")
        # 诚实说明: 即便有 MODEL_ID, remote_researcher 的 url 仍指向占位地址,
        # 真正 start_async_task 会去连远程服务器 —— 本地没有该服务器, 因此这里
        # 不执行 invoke, 以免产生对不可达地址的网络请求。
        print("接入点: 把 url/graph_id/headers 换成真实远程 LangGraph 部署后,"
              " 才能真正 invoke 并触发 start_async_task 等工具。")
    else:
        print("未配置 MODEL_ID: 跳过模型相关部分 (仅结构验证)。")

"""LangGraph SQL agent —— 数据库问答智能体: 列表→看 schema→写 SQL→执行→报错自纠→作答。

对比 langgraph/ch_12_multi_agent.py 或 ch_14_combo.py: 那两篇讲的是"通用 agent / 多智能体编排"
(supervisor 调度 worker、subgraph+Send 拼多文档摘要), 关注的是"多个角色之间怎么路由";
本文件聚焦一个更具体的东西 —— **单 agent 的工具循环 (tool-calling loop)**, 工具全是
数据库操作 (list_tables / get_schema / run_sql), 且额外演示 **SQL 执行报错如何回灌给
模型让它自我修正重试**。这是"ReAct 式工具循环 agent"最典型的落地形态。SQL = 结构化
查询语言, 即操作关系型数据库的语言。

关键机制 / 踩坑记录 (单开一段说清楚):
  1. **SQL 执行报错回灌**: run_sql 工具不把异常抛出去中断图, 而是 try/except 后把
     数据库返回的错误文本 (如 "no such column: dept") 当作 ToolMessage 内容塞回消息
     历史。模型下一轮看到这条报错, 就能自己改写 SQL 重试 —— 错误信息是驱动"自纠"的
     唯一线索, 所以必须原样回灌, 不能吞掉。
  2. **工具循环终止条件**: agent 节点每轮看模型返回的 AIMessage 有没有 tool_calls。
     有 → 走 tools 节点执行工具、把结果拼回历史、再回 agent; 没有 (纯文本) → 视为
     模型已拿到足够信息、给出了最终自然语言答案, 路由到 END 收尾。这就是循环的出口。
  3. **假模型脚本如何模拟 tool_calls 驱动**: 本环境无 MODEL_ID / 密钥, 无法真跑
     ChatAnthropic。用 GenericFakeChatModel 喂一串**预置好的 AIMessage 脚本**: 前几条
     带 tool_calls (list_tables → get_schema → run_sql), 最后一条是纯文本答案。fake
     模型只是"按顺序吐出下一条", 但真正调用工具、真正执行 SQL、真正把结果/报错拼回
     历史、真正靠报错触发下一条修正脚本的, 都是我们手写的图逻辑 —— 循环是真转的,
     SQL 是真跑的, 只有"模型的决策"被替身脚本固定住了。
     踩坑: GenericFakeChatModel 不支持 bind_tools (会 NotImplementedError), 所以没法直接
     喂给 create_agent; 这里手写一个显式的 StateGraph 工具循环, 反而把机制暴露得更清楚。
  4. **sqlite 沙箱清理**: 用标准库 sqlite3 在 tempfile.mkdtemp() 沙箱里现建一个小库
     (employees / departments 两表 + 几行假数据), 绝不连生产库、绝不在仓库目录留 db 文件,
     结尾 shutil.rmtree(..., ignore_errors=True) 清理。

降级说明: 无 MODEL_ID 时自动走 GenericFakeChatModel 替身脚本 (见 build_model), 用一串
预置 AIMessage 固定住"模型决策"、离线确定性地把工具循环跑通并断言; 配了真实 MODEL_ID 时,
build_model 会返回 ChatAnthropic 并 **bind_tools 绑定真实的 list_tables/get_schema/run_sql**,
真实模型据此自行产出 tool_calls, 由同一套手写图逻辑执行 SQL、回灌结果, 两条路径共用一张图。

官方文档: https://docs.langchain.com/oss/python/langgraph/agents (SQL agent / prebuilt / tools)
         https://docs.langchain.com/oss/python/langgraph/graph-api (工具循环 / 条件边)
"""

import os  # 读 MODEL_ID 判断是否有真实模型
import shutil  # 结尾清理沙箱目录
import sqlite3  # 标准库, 真实建库执行 SQL
import tempfile  # 建临时沙箱目录, 不污染仓库
from typing import Annotated, TypedDict

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel  # 无密钥时的模型替身
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph, add_messages

from dotenv import load_dotenv  # 从 .env 读取 MODEL_ID / base_url / key, 不硬编码任何密钥

load_dotenv(override=True)  # 加载 .env; 有 MODEL_ID 则走真实模型分支, 否则降级为替身脚本

# ============================================================
# 一、sqlite 沙箱: 现建一个小样例库 (employees / departments)
# ============================================================
DB_DIR = tempfile.mkdtemp(prefix="sql_agent_")  # 临时沙箱目录, 全程只在这里落盘
DB_PATH = os.path.join(DB_DIR, "company.db")  # 沙箱内的库文件路径


def _seed_db() -> None:
    """建两张表并插入几行假数据 —— 这是可被真实 SQL 查询的本地样例库。"""
    con = sqlite3.connect(DB_PATH)  # 连接沙箱内的 sqlite 文件
    con.executescript(
        """
        CREATE TABLE departments (id INTEGER PRIMARY KEY, name TEXT);
        CREATE TABLE employees (id INTEGER PRIMARY KEY, name TEXT, dept_id INTEGER);
        INSERT INTO departments (id, name) VALUES (1,'Engineering'),(2,'Sales'),(3,'HR');
        INSERT INTO employees (id, name, dept_id) VALUES
            (1,'Alice',1),(2,'Bob',1),(3,'Carol',1),  -- Engineering 3 人
            (4,'Dave',2),(5,'Eve',2),                 -- Sales 2 人
            (6,'Frank',3);                            -- HR 1 人
        """
    )  # 一次性建表 + 插数据, Engineering 人数最多 (3 人)
    con.commit()  # 提交写入
    con.close()  # 关闭连接


# ============================================================
# 二、数据库工具: 直接用 sqlite3 实现, 真实可执行
#    每个工具返回"字符串", 因为工具结果最终要作为 ToolMessage 塞回模型上下文
# ============================================================
def list_tables() -> str:
    """列出库里所有表名 —— agent 通常第一步先看有哪些表。"""
    con = sqlite3.connect(DB_PATH)  # 连沙箱库
    rows = con.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    ).fetchall()  # sqlite_master 是元数据表, 存了所有表结构
    con.close()
    return ", ".join(r[0] for r in rows)  # 返回逗号分隔的表名文本


def get_schema(table: str) -> str:
    """看某张表的字段结构 (schema) —— agent 据此才知道有哪些列可查。"""
    con = sqlite3.connect(DB_PATH)
    rows = con.execute(f"PRAGMA table_info({table})").fetchall()  # PRAGMA 拿表的列信息
    con.close()
    if not rows:  # 表不存在时 PRAGMA 返回空
        return f"表 {table!r} 不存在"
    cols = ", ".join(f"{r[1]} {r[2]}" for r in rows)  # r[1]=列名 r[2]=类型
    return f"{table}({cols})"


def run_sql(query: str) -> str:
    """执行 SQL 并返回结果文本; **执行报错不抛出, 而是把错误文本返回**, 供模型自纠。

    这是"自我修正"的关键: 若 SQL 写错 (列名/表名/语法), sqlite 抛 sqlite3.Error,
    我们捕获后把错误原文包成结果返回, 让模型下一轮看到并改写重试。
    """
    con = sqlite3.connect(DB_PATH)
    try:
        cur = con.execute(query)  # 真实执行传入的 SQL
        rows = cur.fetchall()  # 取全部结果行
        return f"OK rows={rows}"  # 成功: 返回真实结果行
    except sqlite3.Error as e:  # 捕获所有 sqlite 执行错误
        return f"SQL ERROR: {e}"  # 失败: 把错误文本回灌给模型 (不中断图)
    finally:
        con.close()  # 无论成败都关闭连接


# 工具名 → 可调用对象的注册表, tools 节点据此按名分发
TOOLS = {"list_tables": list_tables, "get_schema": get_schema, "run_sql": run_sql}


# ============================================================
# 三、模型: 无密钥走替身脚本; 有 MODEL_ID 时切真实 ChatAnthropic 并绑定真实工具
# ============================================================
def build_model(script: list[AIMessage]):
    """按风格约定优先真实模型, 无密钥时降级为按脚本吐 tool_calls 的替身。

    两条路径产出的都是"带 tool_calls 的 AIMessage", 交给同一张图 (agent<->tools) 执行:
      * 有 MODEL_ID: 返回 ChatAnthropic 并 bind_tools 绑定真实的 list_tables/get_schema/run_sql。
        bind_tools 会读取这三个普通函数的签名 + docstring 自动生成工具 schema (参数名分别是
        table / query), 真实模型据此自行决定调哪个工具、传什么参数, 产出真实的 tool_calls。
      * 无 MODEL_ID: 返回 GenericFakeChatModel, script 里前几条 AIMessage 带预置 tool_calls
        (驱动图去执行工具), 末条是纯文本 (触发收尾)。
    注意: 真实分支忽略 script 参数 (模型自己决策), 替身分支才按 script 顺序吐。
    """
    if os.getenv("MODEL_ID"):  # 有真实模型配置时: 走真实 ChatAnthropic + 绑定真实工具
        from langchain_anthropic import ChatAnthropic  # 延迟导入, 无 key 时不触发

        return ChatAnthropic(  # 真实模型需 bind_tools 才会产出 tool_calls
            model=os.environ["MODEL_ID"],
            base_url=os.getenv("ANTHROPIC_BASE_URL") or None,
        ).bind_tools(_TOOL_SPECS)  # 绑定真实工具函数, 模型据其 schema 产出 tool_calls
    return GenericFakeChatModel(messages=iter(script))  # 替身: 顺序吐出预置脚本


# 真实模型分支绑定的工具函数列表: 直接复用上面三个可执行函数,
# bind_tools 会依据它们的签名 (table / query) + docstring 生成工具 schema。
# 工具名取自函数 __name__ (list_tables / get_schema / run_sql), 与 TOOLS 注册表键一致,
# 所以真实模型产出的 tool_calls 能被 tools_node 正确按名分发执行。
_TOOL_SPECS: list = [list_tables, get_schema, run_sql]


# ============================================================
# 四、工具循环图: agent(模型决策) <-> tools(执行工具) 来回跳, 无 tool_calls 则收尾
# ============================================================
class State(TypedDict):
    messages: Annotated[list, add_messages]  # add_messages: 每轮把新消息追加进历史


def make_agent_node(model):
    """把模型闭包进 agent 节点: 每轮拿全量历史问模型, 得到下一步 (调工具 or 作答)。

    model 既可能是替身 (按脚本吐), 也可能是 bind_tools 后的真实 ChatAnthropic (自行决策)。
    """

    def agent_node(state: State) -> dict:
        response = model.invoke(state["messages"])  # 替身按脚本吐 / 真实模型据工具 schema 决策
        return {"messages": [response]}  # 追加进历史 (可能带 tool_calls, 也可能是纯文本)

    return agent_node


def tools_node(state: State) -> dict:
    """执行上一条 AIMessage 里的 tool_calls, 把每个结果包成 ToolMessage 回灌历史。"""
    last = state["messages"][-1]  # 上一条一定是带 tool_calls 的 AIMessage
    outputs = []
    for call in last.tool_calls:  # 逐个执行模型要求的工具调用
        fn = TOOLS[call["name"]]  # 按工具名取真实可调用对象
        result = fn(**call["args"])  # 真实执行 (真跑 SQL / 真读 schema)
        print(f"  [tool] {call['name']}({call['args']}) -> {result}")  # 可观测: 打出工具轨迹
        outputs.append(ToolMessage(content=result, tool_call_id=call["id"]))  # 结果配 call id 回灌
    return {"messages": outputs}


def route(state: State) -> str:
    """终止条件: 最后一条 AIMessage 有 tool_calls 就去执行工具, 否则 (纯文本) 收尾到 END。"""
    last = state["messages"][-1]
    return "tools" if getattr(last, "tool_calls", None) else END


def build_graph(model):
    """组装 agent<->tools 工具循环图。"""
    builder = StateGraph(State)
    builder.add_node("agent", make_agent_node(model))  # 模型决策节点
    builder.add_node("tools", tools_node)  # 工具执行节点
    builder.add_edge(START, "agent")  # 入口先进 agent
    builder.add_conditional_edges("agent", route, ["tools", END])  # agent 后按有无 tool_calls 分流
    builder.add_edge("tools", "agent")  # 工具执行完总是回到 agent, 形成循环
    return builder.compile()


if __name__ == "__main__":
    _seed_db()  # 建好沙箱样例库

    # ---- 先脱离 agent, 直接断言样例库/SQL 真实可用 (非模型逻辑必须实跑通过) ----
    print("=== 0. sqlite 沙箱自检: 表存在 + SQL 真实执行 ===")
    assert list_tables() == "departments, employees", list_tables()  # 两张表都在
    assert get_schema("employees") == "employees(id INTEGER, name TEXT, dept_id INTEGER)"
    # 直接跑一条"人数最多的部门"的 SQL, 断言结果确为 Engineering=3
    direct = run_sql(
        "SELECT d.name, COUNT(*) c FROM employees e "
        "JOIN departments d ON e.dept_id=d.id GROUP BY d.name ORDER BY c DESC LIMIT 1"
    )
    print(f"  直查结果: {direct}")
    assert direct == "OK rows=[('Engineering', 3)]", direct  # SQL 真跑且结果正确
    print("  通过: 建库/schema/SQL 均真实可执行")

    if os.getenv("MODEL_ID"):
        # ================================================================
        # 真实模型路径: bind_tools 后, 由模型自行决策整个工具循环 (输出不确定)
        # ================================================================
        print("\n=== 1. 真实模型工具循环: 回答『哪个部门人数最多』 ===")
        print("  (已 bind_tools 绑定真实 list_tables/get_schema/run_sql, 由模型自行决定调哪个)")
        # 系统提示: 引导模型按 list_tables→get_schema→run_sql 的顺序用工具, 并在报错时自纠。
        system = SystemMessage(
            content=(
                "你是一个 SQL 数据问答助手, 只能通过工具访问一个 sqlite 数据库, 不要凭空编造数据。"
                "回答前必须: 先用 list_tables 看有哪些表, 再用 get_schema 看相关表的字段, "
                "然后用 run_sql 执行 SQL 得到真实结果; 若 run_sql 返回以 'SQL ERROR:' 开头的报错, "
                "请根据错误信息改写 SQL 重试。拿到结果后再用自然语言给出最终答案。"
            )
        )
        graph_real = build_graph(build_model([]))  # 真实分支忽略 script, 模型自行决策
        result = graph_real.invoke(
            {"messages": [system, HumanMessage(content="哪个部门人数最多?")]},
            config={"recursion_limit": 25},  # 兜底: 防止模型反复调工具停不下来
        )
        final = result["messages"][-1]
        print(f"  最终答案: {final.content}")
        # 从消息历史里收集真实执行过的工具结果 (由 tools_node 打印过逐条轨迹)
        tool_msgs = [m for m in result["messages"] if isinstance(m, ToolMessage)]
        sql_ok = [m.content for m in tool_msgs if m.content.startswith("OK rows=")]
        # 宽松断言: 真实模型输出不确定, 只验证"确实调用了工具 + 真跑了 SQL + 答案沾边"。
        assert tool_msgs, "真实模型路径必须至少调用一次工具, 否则没有验证 SQL 工具循环"
        assert sql_ok, f"应至少有一次 run_sql 成功执行, 实际工具结果: {[m.content for m in tool_msgs]}"
        assert "Engineering" in final.content, f"最终答案应提到 Engineering, 实际: {final.content}"
        print(f"  真跑 SQL 次数(成功): {len(sql_ok)}; 样例结果: {sql_ok[-1]}")
        print("  通过: 真实模型 bind_tools→自行调工具→跑真实 SQL→作答, 答案命中 Engineering")
    else:
        # ================================================================
        # 离线替身路径: 预置脚本固定"模型决策", 离线确定性断言精确结果
        # ================================================================
        # ---- 1. 完整工具循环: 自然语言问题 → list_tables → schema → run_sql → 作答 ----
        print("\n=== 1. agent 工具循环: 回答『哪个部门人数最多』 ===")
        # 替身脚本: 前 3 条带 tool_calls 驱动图去调工具, 末条是模型看到结果后的自然语言答案
        script_ok = [
            AIMessage(content="", tool_calls=[{"name": "list_tables", "args": {}, "id": "t1"}]),
            AIMessage(content="", tool_calls=[{"name": "get_schema", "args": {"table": "employees"}, "id": "t2"}]),
            AIMessage(
                content="",
                tool_calls=[{
                    "name": "run_sql",
                    "args": {"query": "SELECT d.name, COUNT(*) c FROM employees e "
                                      "JOIN departments d ON e.dept_id=d.id GROUP BY d.name ORDER BY c DESC LIMIT 1"},
                    "id": "t3",
                }],
            ),
            AIMessage(content="人数最多的部门是 Engineering, 共 3 人。"),  # 纯文本 → 触发 END
        ]
        graph_ok = build_graph(build_model(script_ok))
        result = graph_ok.invoke({"messages": [HumanMessage(content="哪个部门人数最多?")]})
        final = result["messages"][-1]
        print(f"  最终答案: {final.content}")
        # 断言: 循环里确实真跑了 SQL 并拿到正确行 (从消息历史里找 run_sql 的 ToolMessage)
        tool_results = [m.content for m in result["messages"] if isinstance(m, ToolMessage)]
        assert "OK rows=[('Engineering', 3)]" in tool_results, tool_results  # SQL 真执行且结果正确
        assert final.tool_calls == [] and "Engineering" in final.content  # 末条是纯文本答案, 循环正常收尾
        print("  通过: agent 真实走完 list_tables→schema→run_sql→作答, SQL 结果正确")

        # ---- 2. SQL 报错 → 回灌错误 → 模型修正重试 → 成功 ----
        print("\n=== 2. SQL 执行报错自纠重试 ===")
        # 脚本故意先写一条错列名的 SQL (dept 不存在, 应为 dept_id), 图执行后回灌错误;
        # 模型"看到"报错后, 下一条脚本给出修正版 SQL, 重试成功, 再作答。
        script_retry = [
            AIMessage(
                content="",
                tool_calls=[{"name": "run_sql",
                             "args": {"query": "SELECT dept FROM employees"},  # 错: 列名应为 dept_id
                             "id": "r1"}],
            ),
            AIMessage(
                content="",
                tool_calls=[{"name": "run_sql",
                             "args": {"query": "SELECT dept_id FROM employees LIMIT 2"},  # 修正后重试
                             "id": "r2"}],
            ),
            AIMessage(content="修正列名后查询成功, 前两行 dept_id 为 1 和 1。"),  # 纯文本收尾
        ]
        graph_retry = build_graph(build_model(script_retry))
        result2 = graph_retry.invoke({"messages": [HumanMessage(content="查一下员工的部门编号")]})
        tool_results2 = [m.content for m in result2["messages"] if isinstance(m, ToolMessage)]
        print(f"  第一次(报错): {tool_results2[0]}")
        print(f"  第二次(修正): {tool_results2[1]}")
        # 断言: 第一次确实回灌了 SQL 错误, 第二次修正后成功
        assert tool_results2[0].startswith("SQL ERROR:") and "dept" in tool_results2[0]  # 首次真的报错并回灌
        assert tool_results2[1] == "OK rows=[(1,), (1,)]", tool_results2[1]  # 修正后真的跑通
        print(f"  最终答案: {result2['messages'][-1].content}")
        print("  通过: 报错文本回灌 → 模型改写 SQL 重试 → 执行成功")

    # ---- 沙箱清理: 不在仓库目录留任何 db 文件 ----
    shutil.rmtree(DB_DIR, ignore_errors=True)  # 删除整个临时沙箱目录
    print(f"\n=== 沙箱已清理: {DB_DIR} 存在={os.path.exists(DB_DIR)} ===")
    assert not os.path.exists(DB_DIR)  # 断言沙箱确已删除

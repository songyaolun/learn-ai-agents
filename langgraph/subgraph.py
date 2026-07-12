"""LangGraph subgraph —— 把一个编译好的 StateGraph 当成"一个节点"嵌进另一个图。

对比 langgraph/quickstart.py: 那里所有节点都是平铺在同一个图里的普通函数;
这里"验证并发货"这一段多步流程被独立封装成一个子图 (child graph), 再把这个
*已编译* 的子图对象直接 add_node 进父图, 从父图视角看它就是一个不透明的黑盒节点
——内部有几步、怎么走完全被封装, 这是"层次化组合" (hierarchical composition),
不同于 multi_agent.py 里 supervisor/worker 那种"平级节点互相路由"的编排方式。

关键机制 (用 inspect 验证过, 不是猜的): LangGraph 并不要求父子图状态 100% 一致,
只要求父图节点(=子图)的 *输出* 里, key 名能对上父图 State 的字段, 值就会正常合并回去。
最简单的做法是让子图直接复用父图的 State TypedDict (字段是父图的子集或全集都行)——
这样子图编译后可以不做任何转换就直接 add_node 进父图, 状态在父子之间"直通"。
如果子图的 State 字段名和父图完全对不上(比如子图内部字段是遗留系统命名), 就不能直接
add_node 了, 得在父图里写一个"胶水节点"函数, 手动把父图 state 转成子图输入、
再手动把子图输出转回父图要的字段——本文件下半部分 (call_child_manual) 演示了这种写法。

官方文档: https://docs.langchain.com/oss/python/langgraph/use-subgraphs
"""

from typing import TypedDict

from langgraph.graph import END, START, StateGraph


# ============================================================
# 场景 1: 父子图共享同一套 State —— 可以把子图直接当节点用
# ============================================================
class OrderState(TypedDict):
    order_id: str
    item: str
    amount: float
    validated: bool  # 子图 validate 节点写入
    paid: bool  # 子图 pay 节点写入
    shipped: bool  # 子图 ship 节点写入
    steps: list[str]  # 记录完整执行轨迹 (父图 + 子图节点都会往里追加)


# -- 子图: "验证并发货" 这个多步子流程, 单独建一个 StateGraph --
def validate_node(state: OrderState) -> dict:
    ok = state["amount"] > 0  # 简单校验: 金额必须为正
    return {"validated": ok, "steps": state["steps"] + ["validate"]}


def pay_node(state: OrderState) -> dict:
    return {"paid": True, "steps": state["steps"] + ["pay"]}


def ship_node(state: OrderState) -> dict:
    return {"shipped": True, "steps": state["steps"] + ["ship"]}


sub_builder = StateGraph(OrderState)  # 子图用的是和父图完全相同的 State 类型
sub_builder.add_node("validate", validate_node)
sub_builder.add_node("pay", pay_node)
sub_builder.add_node("ship", ship_node)
sub_builder.add_edge(START, "validate")
sub_builder.add_edge("validate", "pay")
sub_builder.add_edge("pay", "ship")
sub_builder.add_edge("ship", END)
validate_and_ship_subgraph = sub_builder.compile()  # 子图自己先编译好, 此时它还是独立的


# -- 父图: 接单 → (子图: 验证并发货) → 通知客户 --
def receive_order_node(state: OrderState) -> dict:
    return {"steps": state.get("steps", []) + ["receive_order"]}


def notify_customer_node(state: OrderState) -> dict:
    msg = f"订单 {state['order_id']} ({state['item']}) 已发货, 通知客户。"
    return {"steps": state["steps"] + ["notify_customer"], "item": msg}


parent_builder = StateGraph(OrderState)
parent_builder.add_node("receive_order", receive_order_node)
# 直接把已编译的子图对象传给 add_node —— 父图执行到这一步时会完整跑一遍子图的
# validate → pay → ship, 子图内部每一步的状态更新都正常合并回父图的 checkpoint。
parent_builder.add_node("validate_and_ship", validate_and_ship_subgraph)
parent_builder.add_node("notify_customer", notify_customer_node)
parent_builder.add_edge(START, "receive_order")
parent_builder.add_edge("receive_order", "validate_and_ship")
parent_builder.add_edge("validate_and_ship", "notify_customer")
parent_builder.add_edge("notify_customer", END)
graph = parent_builder.compile()


# ============================================================
# 场景 2: 父子图 State 字段名对不上 —— 用"胶水节点"手动转换
# ============================================================
class LegacyCheckState(TypedDict):
    """子图内部字段名是遗留系统的命名, 和父图完全不一样。"""

    item: str
    checked: bool


legacy_builder = StateGraph(LegacyCheckState)
legacy_builder.add_node("check", lambda s: {"checked": True})
legacy_builder.add_edge(START, "check")
legacy_builder.add_edge("check", END)
legacy_check_subgraph = legacy_builder.compile()


class ShipmentState(TypedDict):
    order_id: str
    shipped: bool


def call_child_manual(state: ShipmentState) -> dict:
    """胶水节点: 手动把父图 state 映射成子图输入, 再把子图输出映射回父图字段。

    因为 LegacyCheckState 和 ShipmentState 没有任何同名字段, 没法把
    legacy_check_subgraph 直接 add_node 进父图 —— 父图不知道 order_id 该填到
    子图的 item, 也不知道子图的 checked 该填回父图的 shipped。所以这里手写一个
    普通函数节点, 内部显式调用 legacy_check_subgraph.invoke(...) 完成转换。
    """
    child_result = legacy_check_subgraph.invoke({"item": state["order_id"], "checked": False})
    return {"shipped": child_result["checked"]}


glue_builder = StateGraph(ShipmentState)
glue_builder.add_node("call_child_manual", call_child_manual)
glue_builder.add_edge(START, "call_child_manual")
glue_builder.add_edge("call_child_manual", END)
glue_graph = glue_builder.compile()


if __name__ == "__main__":
    print("=== 场景 1: 共享 State, 子图直接当节点 (order-processing) ===")
    result = graph.invoke(
        {
            "order_id": "ORD-1001",
            "item": "机械键盘",
            "amount": 299.0,
            "validated": False,
            "paid": False,
            "shipped": False,
            "steps": [],
        }
    )
    print(f"  validated={result['validated']}  paid={result['paid']}  shipped={result['shipped']}")
    print(f"  执行轨迹: {' → '.join(result['steps'])}")
    print(f"  通知内容: {result['item']}")
    # 执行轨迹里能看到子图内部的 validate/pay/ship 三步都被完整记录,
    # 说明父图和子图共用同一份 checkpoint 状态, 不是两套隔离的东西。

    print("\n=== 场景 2: State 字段不兼容, 用胶水节点手动转换 (legacy 系统对接) ===")
    result2 = glue_graph.invoke({"order_id": "ORD-2002", "shipped": False})
    print(f"  order_id={result2['order_id']}  shipped={result2['shipped']}")
    print("  (胶水节点内部把 order_id→item、checked→shipped 做了手动映射)")

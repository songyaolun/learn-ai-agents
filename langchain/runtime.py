"""LangChain 运行时 —— agent 执行过程中的上下文与资源管理机制。

对比 langchain/quickstart.py: 那里用 create_agent 封装了运行时细节,
这里显式展示 runtime 五大核心数据(上下文/存储/流写入器/执行信息/服务信息)的管理方式,
核心区别是抽象层级(黑盒封装 vs 显式控制)和可扩展性(固定流程 vs 自定义注入)。

官方文档: https://docs.langchain.com/oss/python/langgraph/runtime

能力点:
1. 上下文 schema 定义与验证
2. 运行时依赖注入(工具中获取 runtime)
3. 五大核心数据的访问方式
4. 沙箱化执行环境演示

踩坑记录:
- 工具想拿到 runtime, 参数要声明成 ToolRuntime 类型 (从 langchain.tools 导入, 不是
  langchain_core.tools), 框架会自动注入且该参数对模型隐藏 —— 模型不会、也不该去填它。
- 上下文 (context) 通过 invoke 的 context= 传入并按 schema 校验; 字段名/类型对不上会在
  进入 agent 前就报错, 而不是运行到工具里才失败, 因此 schema 要和实际传参严格一致。
"""

import os
import tempfile
import shutil
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langgraph.runtime import get_runtime
from langgraph.store.memory import InMemoryStore

# 模型初始化沿用既有约定(实际 agent 场景需配置 .env)
load_dotenv(override=True)


# 1. 定义上下文 schema(运行时的核心数据结构)
# 上下文 schema 规定了 agent 执行过程中可访问的所有数据字段
class AgentContext(BaseModel):
    user_id: str = Field(description="用户唯一标识")
    session_id: str = Field(description="会话唯一标识")
    memory_store: InMemoryStore = Field(description="长期记忆存储实例")
    temp_dir: str = Field(description="沙箱临时目录路径")

    model_config = {
        "arbitrary_types_allowed": True
    }


# 2. 演示工具如何通过 runtime 获取上下文
def get_user_preference(preference_key: str) -> str:
    """获取用户长期记忆中的偏好设置"""
    # 通过 get_runtime 获取当前运行时上下文
    runtime = get_runtime(AgentContext)
    context = runtime.context

    # 从上下文中获取存储实例和用户ID
    store = context.memory_store
    namespace = f"user_memory:{context.user_id}"

    # 读取长期记忆
    result = store.get(namespace, preference_key)
    return result.value if result else "未设置"


# 3. 运行时初始化与依赖注入
def main():
    # 沙箱化临时目录(避免污染系统环境)
    temp_dir = tempfile.mkdtemp()
    print(f"📦 运行时沙箱路径: {temp_dir}")

    # 初始化长期记忆存储
    memory_store = InMemoryStore()
    user_id = "user_123"
    session_id = "session_456"

    # 写入测试数据
    namespace = f"user_memory:{user_id}"
    memory_store.put(namespace, "coffee_preference", "美式咖啡,不加糖")
    memory_store.put(namespace, "work_city", "北京")

    # 4. 构造上下文并演示数据访问
    context = AgentContext(
        user_id=user_id,
        session_id=session_id,
        memory_store=memory_store,
        temp_dir=temp_dir
    )
    print("✅ 上下文构造完成")

    # 5. 验证上下文数据访问
    coffee_pref = context.memory_store.get(namespace, "coffee_preference").value
    work_city = context.memory_store.get(namespace, "work_city").value
    unknown_pref = context.memory_store.get(namespace, "unknown_key")

    assert coffee_pref == "美式咖啡,不加糖", f"咖啡偏好读取错误: {coffee_pref}"
    assert work_city == "北京", f"工作城市读取错误: {work_city}"
    assert unknown_pref is None, f"未知偏好处理错误: {unknown_pref}"
    print("✅ 上下文数据访问验证通过")

    # 6. 演示工具如何访问上下文
    def get_user_preference(context: AgentContext, preference_key: str) -> str:
        """获取用户长期记忆中的偏好设置"""
        result = context.memory_store.get(f"user_memory:{context.user_id}", preference_key)
        return result.value if result else "未设置"

    coffee_pref_tool = get_user_preference(context, "coffee_preference")
    work_city_tool = get_user_preference(context, "work_city")
    unknown_pref_tool = get_user_preference(context, "unknown_key")

    assert coffee_pref_tool == "美式咖啡,不加糖", f"工具访问验证失败: {coffee_pref_tool}"
    assert work_city_tool == "北京", f"工具访问验证失败: {work_city_tool}"
    assert unknown_pref_tool == "未设置", f"工具访问验证失败: {unknown_pref_tool}"
    print("✅ 工具访问上下文验证通过")

    # 7. 演示运行时核心数据概念
    print("\n📊 运行时核心数据概念:")
    print(f"- 上下文(context): {type(context).__name__} 实例")
    print(f"- 存储(store): {type(context.memory_store).__name__} 实例")
    print(f"- 执行信息: 包含执行状态/错误等(真实场景由框架注入)")
    print(f"- 服务信息: 包含服务配置/版本等(真实场景由框架注入)")
    print(f"- 流写入器: 用于实时输出(真实场景由框架注入)")

    # 8. 沙箱清理
    shutil.rmtree(temp_dir)
    print("\n🧹 沙箱清理完成")

    # 9. 真实 agent 场景说明
    print("\n📚 运行时使用场景:")
    print("- 上下文注入: 将配置/存储/资源等注入 agent 执行过程")
    print("- 工具依赖: 工具通过参数获取上下文数据(显式依赖注入)")
    print("- 环境隔离: 沙箱化执行避免资源冲突")
    print("- 可观测性: 通过执行信息监控 agent 执行状态")
    print("- 注意: 真实 runtime.get_runtime() 需在 LangGraph 图执行上下文中调用")


if __name__ == "__main__":
    main()
    print("\n🎉 运行时演示完成! 可配置 .env 后扩展为真实 agent 场景。")
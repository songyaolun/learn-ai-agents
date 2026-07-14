"""LangChain 长期记忆 —— 跨会话持久化用户数据的存储机制与命名空间管理。

对比 langchain/middleware_summarization.py: 那里用 SummarizationMiddleware 压缩单次对话内的短期记忆(消息历史),
这里 InMemoryStore 实现跨会话/跨线程的长期记忆(持久化用户数据), 核心区别是存储范围(单线程 vs 跨线程)和数据类型(消息历史 vs 结构化数据)。

官方文档: https://docs.langchain.com/oss/python/langgraph/store/memory

能力点:
1. 内存态存储(InMemoryStore)的 put/get 基础操作
2. 命名空间(namespace)与键(key)的层级管理
3. 跨会话读取演示(模拟不同线程访问同一份记忆)
4. 内存态 vs 持久态存储的边界说明

踩坑记录:
- InMemoryStore 是纯内存实现, 进程退出即清空, 名字里的"长期"指"跨会话/跨线程共享",
  不等于"落盘持久"。要真正持久到重启不丢, 需换成落盘型 Store (如基于数据库的实现)。
- put/get 都要带 namespace (元组, 如 ("users", user_id)) + key, 两者共同定位一条记忆;
  namespace 写错会读到 None 而不会报错, 排查时先确认两端 namespace 完全一致。
"""

import os
import tempfile
import shutil
from dotenv import load_dotenv
from langgraph.store.memory import InMemoryStore

# 模型初始化沿用既有约定(实际 agent 场景需配置 .env)
load_dotenv(override=True)


def main():
    # 1. 初始化内存态存储(无外部依赖,可独立运行)
    store = InMemoryStore()
    user_id = "user_123"
    namespace = f"user_memory:{user_id}"

    # 2. 写入长期记忆: 存储用户偏好(跨会话持久化)
    # 这里模拟用户在会话A中设置的偏好
    store.put(namespace, "coffee_preference", "美式咖啡,不加糖")
    store.put(namespace, "work_city", "北京")

    # 3. 读取长期记忆: 模拟用户在会话B中查询偏好
    coffee_pref = store.get(namespace, "coffee_preference")
    work_city = store.get(namespace, "work_city")

    # 断言验证存储正确性
    assert coffee_pref.value == "美式咖啡,不加糖", f"咖啡偏好读取错误: {coffee_pref.value}"
    assert work_city.value == "北京", f"工作城市读取错误: {work_city.value}"
    print("✅ 内存态存储 put/get 验证通过")

    # 4. 命名空间隔离演示: 不同用户数据互不干扰
    another_user_namespace = "user_memory:user_456"
    store.put(another_user_namespace, "coffee_preference", "拿铁咖啡,加奶")
    another_coffee_pref = store.get(another_user_namespace, "coffee_preference")

    assert another_coffee_pref.value == "拿铁咖啡,加奶", f"用户隔离验证失败: {another_coffee_pref.value}"
    print("✅ 命名空间隔离验证通过")

    # 5. 沙箱化持久态存储演示(模拟文件系统存储)
    # 注意: 真实持久化需实现 BaseStore 接口(如 FileStore), 这里用 tempfile 模拟
    temp_dir = tempfile.mkdtemp()
    print(f"\n📦 持久态存储沙箱路径: {temp_dir}")

    # 模拟持久化逻辑(实际需实现 FileStore 等)
    # 这里仅演示沙箱化思想, 实际项目需替换为真实持久化实现
    with open(f"{temp_dir}/user_{user_id}_memory.txt", "w") as f:
        f.write(f"咖啡偏好: {coffee_pref.value}\n工作城市: {work_city.value}")

    # 验证沙箱文件写入
    with open(f"{temp_dir}/user_{user_id}_memory.txt", "r") as f:
        content = f.read()
        assert "美式咖啡,不加糖" in content
        assert "北京" in content
    print("✅ 沙箱化持久态存储验证通过")

    # 清理沙箱
    shutil.rmtree(temp_dir)
    print("🧹 沙箱清理完成")

    # 6. 长期记忆 vs 短期记忆边界说明
    print("\n📚 长期记忆 vs 短期记忆边界:")
    print("- 短期记忆: 单线程内的消息历史(如 middleware_summarization.py 中的对话记录)")
    print("- 长期记忆: 跨会话/跨线程的用户数据(如用户偏好、设置等)")
    print("- 存储载体: 短期记忆用 Checkpointer, 长期记忆用 Store 系列(InMemoryStore/FileStore)")


if __name__ == "__main__":
    main()
    print("\n🎉 长期记忆演示完成! 可配置 .env 后扩展为真实 agent 场景。")
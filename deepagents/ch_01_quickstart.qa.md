# ch_01_quickstart.py 学习问答记录

配套文件:`deepagents/ch_01_quickstart.py`。本文整理阅读该文件时深挖 `create_deep_agent`/`create_agent`/LangGraph 源码得到的问答,按提问顺序排列,均附源码文件路径与行号,方便后续回查验证(注意:行号对应写作时安装的 `deepagents`/`langchain`/`langgraph` 版本,升级依赖后可能漂移)。

---

## Q1. `agent.invoke(...)` 里的 `invoke` 是啥意思?为啥用这个函数?为什么叫这个命名?注释里说的"自动做的三件事"初始化操作在哪里做的?

**`invoke` 是什么**:`create_deep_agent()` 返回的 `agent` 是一张编译好的 LangGraph 图(`CompiledStateGraph`)。`invoke` 是 LangChain 生态统一的 `Runnable` 协议方法之一(源头 `langchain_core/runnables/base.py`)。这套协议规定所有组件(prompt、model、chain、agent、graph)都实现同一套接口——`invoke`(同步单次执行)/`batch`(并发跑多个输入)/`stream`(流式)及其异步版本 `ainvoke`/`abatch`/`astream`,这样不同组件才能用 `|` 管道符互相拼接、互相嵌套。

**命名由来**:LCEL(LangChain Expression Language)从一开始就定的命名惯例——`invoke` = "调用一次、拿到最终结果",与 `stream`(持续吐数据)语义相对。`CompiledStateGraph → Pregel → PregelProtocol → Runnable` 的继承链决定了它延续这同一套协议。

**`invoke` 的真正实现**(`langgraph/pregel/main.py:3851`):本质是 `stream` 的同步聚合版——内部还是按 Pregel 的 BSP(批量同步并行)模型一步步跑图,只是把中间每一步都收集起来,最后返回最后一次 `"values"` 快照:

```python
for chunk in self.stream(input, config, stream_mode=["updates", "values"], ...):
    ...  # 收集每一步
return latest  # 最后一次完整 state
```

**三件初始化的位置**(均在 `deepagents/graph.py` 的 `create_deep_agent` 函数体):
- 规划(todo):`graph.py:774` `TodoListMiddleware()`
- 虚拟文件系统:`graph.py:778-784` `FilesystemMiddleware(backend=backend, ...)`(默认 `backend=StateBackend()`,见 `graph.py:615`)
- subagents 委派:`graph.py:786-798`,`inline_subagents` 非空时才加 `SubAgentMiddleware(...)`,负责实现内置的 `task` 工具

调用链:`create_deep_agent`(`graph.py:865-877`)→ `langchain.agents.create_agent`(`langchain/agents/factory.py:1787` `graph.compile(...)`)→ 产出 `CompiledStateGraph`。

---

## Q2. 为什么这么多 middleware?在 LangChain 世界里 middleware 是啥,能干哪些事?

**本质**:`langchain.agents.middleware.types.AgentMiddleware`(`types.py:383`)是个基类,在 agent 固定执行循环里插入自定义逻辑,不用重写整个循环。图的固定结构:

```
before_agent(一次) → [ before_model → model调用(被wrap_model_call包裹) → after_model
                        → (若有工具调用: 工具执行, 被wrap_tool_call包裹) ]*循环 → after_agent(一次)
```

**能力四分类**:
1. **扩展状态/注入工具**(类属性):`state_schema` 加自定义 state 字段;`tools` 追加工具(如 `FilesystemMiddleware` 塞 `ls`/`read_file`,`SubAgentMiddleware` 塞 `task`)
2. **拦截/改写模型调用**(`wrap_model_call`,`types.py:491`):接收 `request` + `handler` 回调,可改请求、改响应、多次调用 `handler` 重试、或完全不调用直接短路返回缓存
3. **拦截/改写工具调用**(`wrap_tool_call`):同样 handler 模式,`HumanInTheLoopMiddleware` 在这里插入 `interrupt()`,`PIIMiddleware` 在这里脱敏
4. **循环阶段钩子**:`before_agent`/`after_agent` 各跑一次;`before_model`/`after_model` 每轮循环都跑

**为什么堆这么多个**:每个 middleware 只负责一件事(单一职责原则),`graph.py` 里堆的这些各自对应:

| middleware(`graph.py` 行号) | 钩子 | 干什么 |
|---|---|---|
| `TodoListMiddleware()`(774) | 加 `write_todos` 工具 + `before_model` | 规划/todo 追踪 |
| `SkillsMiddleware`(777,可选) | 加工具 | 按需读取技能库 |
| `FilesystemMiddleware`(778-784) | 加文件工具 | 虚拟文件系统 + `permissions` 权限拦截 |
| `SubAgentMiddleware`(786-798) | 加 `task` 工具 | 委派子任务给独立子 agent |
| `create_summarization_middleware(...)`(801) | `before_model` | 历史太长自动摘要压缩 |
| `PatchToolCallsMiddleware()`(802) | 后处理 | 修补模型产出的异常工具调用格式 |
| `AnthropicPromptCachingMiddleware`(819) | `wrap_model_call` | 自动加 prompt cache 断点 |
| `MemoryMiddleware`(824-829,可选) | `before_model` | 加载 `AGENTS.md` 进 system prompt |
| `HumanInTheLoopMiddleware`(835,有 interrupt_on 时) | `wrap_tool_call` | 工具调用前人工审批 |

职责单一才能被 `_apply_excluded_middleware`(`graph.py:836`)按需摘掉、或被用户 `middleware=[...]` 插到中间。

---

## Q3. `SubAgentMiddleware` 的 `task` 是怎么实现的?主 agent 是怎么转交任务和任务结果的?

源码:`deepagents/middleware/subagents.py`。分四步:

**① `task` 工具怎么挂上去**:`SubAgentMiddleware.__init__`(791-830 行)调用 `_build_task_tool(...)` 造出一个 `StructuredTool`,`self.tools = [task_tool]`(830 行)——利用的就是 Q2 里说的 `AgentMiddleware.tools` 属性收集机制。

**② 每个 subagent 是一整张独立的图**:`create_sub_agent`(459-511 行)对每个 `SubAgent` spec 再调用一次 `create_agent(model, tools=..., middleware=..., system_prompt=...)`——不是主图里的一个节点,而是独立编译好的 `CompiledStateGraph`,存在 `subagent_graphs: dict[str, Runnable]`(588 行)里。

**③ 任务怎么转交**(`task()` 函数,668-694 行):

```python
def task(description, subagent_type, runtime):
    subagent, subagent_state = _validate_and_prepare_state(subagent_type, description, runtime)
    result = subagent.invoke(subagent_state, subagent_config)   # 同步阻塞调用
    return _return_command_with_state_update(result, runtime.tool_call_id)
```

`_validate_and_prepare_state`(655-666 行)关键逻辑:

```python
subagent_state = {k: v for k, v in runtime.state.items() if k not in _EXCLUDED_STATE_KEYS}  # 拷贝主agent state,排除 messages/todos/structured_response
subagent_state = {k: v for k, v in subagent_state.items() if k not in private_state_keys}
subagent_state["messages"] = [HumanMessage(content=description)]   # messages 整个替换成唯一一条任务描述
```

主 agent 的共享状态(如虚拟文件系统的 `files`)会传过去,但**对话历史不会**——子 agent 眼里的世界就是这一条 `HumanMessage`,这就是"隔离上下文窗口"的字面实现。

**④ 结果怎么传回来**——`_return_command_with_state_update`(600-638 行)用 `langgraph.types.Command` 一次性打补丁:

```python
state_update = {k: v for k, v in result.items() if k not in _EXCLUDED_STATE_KEYS}
# 若配了 response_format: JSON 序列化 structured_response 当内容
# 否则倒着找 result["messages"] 里最后一条非空 AIMessage.text
return Command(update={
    **state_update,
    "messages": [ToolMessage(content, tool_call_id=tool_call_id)],  # tool_call_id 保证精确配对
})
```

`return Command(update=...)` 是 LangGraph 原生能力:工具函数可以直接对图的多个 state channel 打补丁,而不局限于"只能返回工具结果文本"。

---

## Q4. 装饰器模式和责任链模式是设计模式吗?具体是什么,用在哪些场景?

是,都是经典 GoF 设计模式,装饰器是结构型,责任链是行为型。

**装饰器模式(Decorator)**:不修改原对象代码,动态给对象包一层"壳"增加职责,壳和被包对象实现同一接口,可层层嵌套。对照 `wrap_model_call`:

```python
def wrap_model_call(self, request, handler):
    request = 改一下request       # 外层先做事
    response = handler(request)   # 调用"内部对象"(下一层middleware或真正的model调用)
    response = 改一下response      # 外层再做事
    return response
```

多个 middleware 叠加时,第一个是最外层壳,层层包裹到最里面才是真正的 `model.invoke()`。

典型场景:Python `@decorator` 语法、Web 框架 middleware 链(Django/Express/Koa)、I/O 流层层包裹、给功能叠加横切关注点(logging/鉴权/限流/缓存/重试)而不碰核心业务代码。

**责任链模式(Chain of Responsibility)**:多个处理者串成一条链,请求沿链依次传递,每个处理者自行决定要不要处理、要不要继续往下传,发起方不知道最终谁处理的。对照 `before_model`/`after_model`:所有 middleware 的 `before_model` 按顺序串成一条链,state 沿链依次流过,每个都可能修改它、也可能提前短路(如 `HumanInTheLoopMiddleware` 决定要不要 `interrupt()`)。

与装饰器的区别:装饰器强调"包裹增强",每层必然参与且关心调用前后两个时机;责任链强调"传递与甄别",每个节点可以选择不处理、直接转手,甚至提前终止整条链。

典型场景:审批流、异常处理链(`try/except` 逐层往外抛)、GUI 事件冒泡、HTTP 路由匹配。

deepagents 的 middleware 系统是两者的混合体:`wrap_model_call`/`wrap_tool_call` 是装饰器,`before_*`/`after_*` 是责任链。

---

## Q5. 如果子 agent 需要鉴权(人工审批)怎么办?它不是在后台运行吗?

**先纠正认知**:`task` 工具调的子 agent 不是另开的线程/进程,而是主 agent 这次 `.invoke()` 调用栈里往下多嵌套一层的**同步 Python 函数调用**。"后台"准确说是"对主 agent 对话历史不可见",不是"异步跑在别处"。

**① deepagents 本来就支持**:`SubAgent` spec 有 `interrupt_on` 字段,`create_sub_agent`(`subagents.py:496-498`)编译时会检查:

```python
interrupt_on = spec.get("interrupt_on")
if interrupt_on:
    middleware.append(HumanInTheLoopMiddleware(interrupt_on=interrupt_on))
```

**② 为什么嵌套同步调用里的 `interrupt()` 能一路捅到最外层**:`task()` 里 `result = subagent.invoke(subagent_state, subagent_config)`(`subagents.py:693`)是阻塞的普通调用。`interrupt()` 的实现(`langgraph/types.py:811-900`)是"抛出 `GraphInterrupt` 异常"——普通 Python 异常,顺着调用栈一路网上抛:子agent图节点 → `subagent.invoke()` → `task()` 工具函数 → 主 agent 的 Pregel 执行循环接住。全程同一条调用栈,没有跨线程/跨进程边界。

**③ checkpointer 是怎么"共享"的**:`interrupt()` 要求图必须有 checkpointer 才能持久化暂停状态,但 `create_sub_agent` 编译子 agent 时**没传 `checkpointer=`**。答案在 `langgraph/pregel/main.py:1395-1396` 反复出现的模式:

```python
checkpointer = ensure_config(config)[CONF].get(CONFIG_KEY_CHECKPOINTER, self.checkpointer)
```

如果当前调用环境的 `config`(通过 `RunnableConfig` + contextvars 向下传播)已带着 `CONFIG_KEY_CHECKPOINTER`,优先用它而不是自己编译时绑定的 `None`。`subagents.py:684-690` 注释直接点明:主 agent 的 config 会"渗透"给内部任何嵌套的 `Runnable.invoke()` 调用,子 agent 借用主 agent 的 checkpointer,被记录在同一 `thread_id` 下的一个嵌套 checkpoint 命名空间里。

**④ 完整流程**:
1. 主 agent(有 `checkpointer`+`thread_id`)调 `task` → 同步调子 agent图
2. 子 agent 某工具触发 `interrupt()` → 抛 `GraphInterrupt`,借用主 agent checkpointer 记录进同一 thread 的嵌套 checkpoint
3. 异常冒泡出 `subagent.invoke()` → `task()` → 主 agent Pregel 循环捕获,`.invoke()` 直接返回,`result["__interrupt__"]` 带 payload
4. 用户处理完审批,在**主 agent**(不是子 agent,你手上也拿不到那个临时引用)上用同一 `thread_id` 调 `agent.invoke(Command(resume=...), config)`
5. LangGraph 从 checkpoint 恢复,重走到未跑完的 `task` 调用节点 → 重新执行 `task()` → 重新调 `subagent.invoke()`;子 agent 自己嵌套的 checkpoint 记着断点位置,这次 `interrupt()` 直接返回 resume 值,子 agent 继续跑完,最终结果通过 `Command(update=...)` 传回主 agent

---

## Q6. 如果多个子 agent 并行呢?子 agent 在跑的话主 agent 是阻塞的吗?

**并行机制**:`ToolNode`(`langgraph/prebuilt/tool_node.py`)模块 docstring 明确写着设计目标之一是 "Parallel execution of multiple tool calls"。真正执行代码:

```python
# _func —— 同步路径 (.invoke() 走这里),821-823行
with get_executor_for_config(config) as executor:
    outputs = list(executor.map(self._run_one, tool_calls, input_types, tool_runtimes))

# _afunc —— 异步路径 (.ainvoke() 走这里),858行
outputs = await asyncio.gather(*coros)
```

同步模式用线程池 `executor.map` 并发跑,不是 for 循环顺序执行;异步模式用 `asyncio.gather` 真并发。所以模型在**同一条 AIMessage 里一次性发出多个 `task()` 调用**时,多个子 agent 会被丢进线程池并发执行,总耗时约等于最慢那个,而不是相加。这也是为什么 `TASK_TOOL_DESCRIPTION` 提示词明确教模型"尽量在一条消息里并行发起多个 task 调用"。

**主 agent 是否阻塞,分两层看**:
- **graph 内部**:图结构是 `model 节点 → tools 节点 → model 节点 → ...` 循环(Pregel 批量同步并行模型,一步跑完才进下一步)。`tools` 这一步在跑子 agent 时,`model` 节点不会并发跑,主 agent 要等这**一整步**(含所有并行子 agent)跑完才能回到 `model` 节点。这个意义上是阻塞、串行的。
- **调用方视角**:`main_agent.invoke(...)` 本身是同步调用,调用它的线程会一直等到所有嵌套子 agent(无论并不并行)全部跑完。想不阻塞调用方只能自己走 `ainvoke`/`astream` 异步,或丢进后台线程/任务队列——这和 `AsyncSubAgentMiddleware` 支持的远程/后台子 agent(真正的异步任务队列,可以发起后先做别的事)是两回事。

一句话:**子 agent 之间彼此并行,但整个 `tools` 步骤相对主 agent"下一轮模型调用"是阻塞的**。

---

## Q7. 如果多个 middleware 互相依赖,怎么管理应用顺序?

LangChain 的 `AgentMiddleware` **没有自动依赖解析**(不支持声明 `depends_on`/`before`/`after` 再拓扑排序)——顺序完全由传给 `middleware=[...]` 的**列表顺序**决定,且不同钩子先后语义不同:

| 钩子类型 | 执行顺序 | 证据(`langchain/agents/factory.py`) |
|---|---|---|
| `before_agent`/`before_model` | **正序**(FIFO,第一个先跑) | 1696-1725 行,`itertools.pairwise` 按列表顺序串链 |
| `after_model`/`after_agent` | **倒序**(LIFO,最后一个先跑) | 1739 行,`for idx in range(len-1, 0, -1)`:`model → 最后一个.after_model → ... → 第一个.after_model → 退出` |
| `wrap_model_call`/`wrap_tool_call` | **装饰器嵌套**,第一个最外层 | `_chain_model_call_handlers`(235行):`compose_two` 从后往前叠 |

`before`/`after` 一正一反,正是"洋葱模型":先进先出叠 before,后进先出退出 after,与 Express/Koa 中间件、Python 上下文管理器嵌套语义一致。

**没有依赖解析,实践中怎么办**:deepagents 自己的答案就在 `graph.py` 里——不做自动排序,而是**把内置 middleware 相对顺序写死,并文档化一个用户 middleware 插入点**(`create_deep_agent` docstring 350-394 行:"Base stack → *User middleware inserted here* → Tail stack")。两个具体的顺序依赖证据:

- `graph.py:813-815` 注释:harness profile 中间件特意放在用户 middleware 之后、`MemoryMiddleware` 之前,原文"so that memory updates ... don't invalidate the Anthropic prompt cache prefix"——必须排在 `AnthropicPromptCachingMiddleware` 之后,否则每次内存更新都让缓存前缀失效,浪费 prompt cache
- `PatchToolCallsMiddleware`(修补异常工具调用格式)排在 base stack 末尾、早于 `HumanInTheLoopMiddleware`(tail stack),这样人工审批看到的是已修补的合法调用,而非原始可能格式错误的调用

真实做法:**没有框架帮你解依赖,靠约定 + 文档**——把强依赖关系写清楚、给一个安全插入槽位,而不是搞 `@Order(n)` 或拓扑排序。自己写 middleware 有依赖时同样只能:① 在注释里写明谁必须在谁前面;② 让每个 middleware 职责单一、减少互相依赖(回到 SOLID 单一职责——依赖越少,顺序问题越少)。

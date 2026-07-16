# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目性质

这是一个个人学习仓库，按框架抽象层级递进地实现/对比几种 AI Agent 构建方式：从裸写 Anthropic SDK 的 agent loop，到 LangChain 的 `create_agent`，再到 LangGraph 的底层图 runtime，再到 DeepAgents 的 planning/subagent harness，最后用 Chainlit 包一层 Web UI；另有 `pi/`（TypeScript）作为"轻抽象"路线的反方参照。每个脚本头部的 docstring 都会显式对比"这一版比上一版多了什么"，阅读代码时应优先看这些 docstring 建立心智模型。

各脚本相互独立，无共享的内部模块/包结构 —— 没有 `src/` 布局，也没有测试套件。

## 常用命令

```bash
# 安装依赖（uv 管理，Python >= 3.11）
uv sync

# 运行 claude-code/ 下的 CLI agent（交互式 REPL，输入 q/exit/空行退出）
uv run python claude-code/ch01.py   # 最小 agent loop：仅 bash 工具
uv run python claude-code/ch02.py   # + read/write/edit 文件工具，消息规范化
uv run python claude-code/ch03.py   # + todo 计划工具（类似 Claude Code 的 TodoWrite）

# 运行 langchain/、langgraph/、deepagents/、rag/ 下的示例（一次性脚本，非交互）
# --- langchain/ ---
uv run python langchain/quickstart.py
uv run python langchain/stream.py
uv run python langchain/structured_output.py           # response_format：让最终答案是定死结构的 Pydantic 对象
uv run python langchain/lcel.py                         # 不用 agent，用 `|` 拼 prompt|model|parser 链
uv run python langchain/trim_messages.py                # trim_messages：直接砍掉旧消息（对比自动摘要）
uv run python langchain/middleware_hitl.py               # 人工审批工具调用（approve/edit/reject 三种决策）
uv run python langchain/middleware_summarization.py      # 长对话自动摘要压缩
uv run python langchain/middleware_guardrails.py          # ModelCallLimitMiddleware/ToolCallLimitMiddleware 防失控空转
uv run python langchain/middleware_pii.py                 # PIIMiddleware 自动脱敏邮箱/信用卡号
uv run python langchain/combo.py                          # 组合演示：PII+HITL+调用限制+摘要+结构化输出叠一个客服 agent
# --- langgraph/ ---
uv run python langgraph/quickstart.py
uv run python langgraph/human_in_loop.py
uv run python langgraph/persistence.py   # 会在 cwd 生成 langgraph_checkpoints.db
uv run python langgraph/multi_agent.py   # supervisor 编排 researcher/writer 两个 worker，需联网搜索
uv run python langgraph/subgraph.py       # 嵌套图：把编译好的子图当一个节点塞进父图
uv run python langgraph/send_map_reduce.py  # Send API：运行时动态并行 fan-out（map-reduce）
uv run python langgraph/store.py           # InMemoryStore：跨 thread_id 共享的长期记忆
uv run python langgraph/time_travel.py     # get_state_history + update_state：从历史 checkpoint 分叉重跑
uv run python langgraph/combo.py           # 组合演示：subgraph+Send+Store+checkpointer 拼一个多文档摘要器
# --- deepagents/ ---
uv run python deepagents/quickstart.py
uv run python deepagents/research.py     # 需联网，用 DuckDuckGo 搜索（无需 API key）
uv run python deepagents/stream.py
uv run python deepagents/filesystem.py   # 观察虚拟文件系统：agent 写文件、运行结束后读出 result["files"]
uv run python deepagents/hitl.py         # create_deep_agent 内置 interrupt_on 做人工审批（approve/reject）
uv run python deepagents/backend.py        # backend=FilesystemBackend：虚拟文件系统落到真实磁盘
uv run python deepagents/skills_memory.py  # skills（按需查阅的技能库）+ memory（AGENTS.md 长期偏好）
uv run python deepagents/permissions.py    # FilesystemPermission：按路径限制文件工具的读写权限
uv run python deepagents/structured_output.py  # response_format：deep agent 的最终答案定死结构
uv run python deepagents/combo.py          # 组合演示：subagents+真实磁盘+HITL+结构化输出拼一个研究助手
# --- rag/ ---
uv run python rag/quickstart.py          # 需 VOYAGE_API_KEY，检索本仓库模块说明并回答问题

# 运行 pi/ 下的示例（TypeScript，独立 npm 工程，不走 uv；Node >= 20）
cd pi && npm install
npm run ch01    # pi-ai：统一多厂商 LLM API + 手写工具调用循环
npm run ch02    # pi-agent-core：Agent 类托管 agent loop + 自定义工具 + 事件流
npm run ch03    # pi-coding-agent SDK：迷你 Claude Code（内置编码工具 + 会话管理）
npm run ch04    # steering/followUp：agent 运行中插话纠偏/排队追加任务（pi 招牌能力）
npm run ch05    # 会话树：jsonl 持久化 + 续接 + navigateTree 分叉 + compaction
npm run ch06    # skills + AGENTS.md：技能库渐进式展示 + 长期偏好（对照 deepagents/skills_memory.py）
npx tsc --noEmit  # pi/ 唯一的静态检查手段（其余目录无 lint/test）

# 启动 Chainlit Web UI（左上角切换 4 个 chat profile: Deep Research / HITL / Supervisor / RAG）
uv run chainlit run web/app.py    # 打开 http://localhost:8000
```

没有 lint/format/test 工具链配置（无 ruff/pytest 等依赖），修改代码后以能否正常运行对应脚本作为验证手段。

## 环境变量

通过根目录 `.env`（`python-dotenv` 以 `override=True` 加载）配置，参考 `.env.example`：
- `MODEL_ID` —— 必需，指定使用的模型
- `ANTHROPIC_BASE_URL` —— 可选，自定义 API 网关时使用；一旦设置，代码会自动 `pop` 掉 `ANTHROPIC_AUTH_TOKEN`（避免网关鉴权冲突），转而依赖 `ANTHROPIC_API_KEY`
- `VOYAGE_API_KEY` —— 仅 `rag/quickstart.py` 需要，用于 `VoyageAIEmbeddings`（Anthropic 官方推荐的 embedding 服务，https://www.voyageai.com/ 注册可获取免费额度）
- Web UI（`web/app.py`）需要模型支持视觉能力才能处理图片输入

## 架构要点

### claude-code/（裸 SDK 手写 agent loop，三级递进）
不依赖任何 agent 框架，直接用 `anthropic.Anthropic` 客户端手写核心循环，用于理解 agent 的底层机制：
- **ch01.py**：最简版。`LoopState` 保存消息历史，`run_one_turn` 每轮调用模型 → 若 `stop_reason == "tool_use"` 则执行工具并把 `tool_result` 塞回消息列表 → 循环直到模型不再要求用工具。仅有一个 `bash` 工具。
- **ch02.py**：在 ch01 基础上加入 `read_file`/`write_file`/`edit_file`，并引入 `TOOL_HANDLERS` 分发表（工具名 → 处理函数）取代 if/elif 链。新增 `safe_path()` 防止路径逃逸出工作目录。新增 `normalize_messages()`：发给 API 前清理消息历史 —— 剥离内部元数据、给孤儿 `tool_use` 补占位 `tool_result`、合并连续同角色消息（Anthropic API 要求严格 user/assistant 交替）。
- **ch03.py**：在 ch02 基础上加入 `TodoManager`/`todo` 工具，模拟 Claude Code 的计划追踪（pending/in_progress/completed，且同一时刻只能有一个 in_progress）。若连续 `PLAN_REMINDER_INTERVAL` 轮未更新计划，会在工具结果前插入提醒文本。

三个版本的 `TOOLS`/`TOOL_HANDLERS`/`normalize_messages` 高度重复但故意不抽取公共模块 —— 目的是让每一版保持独立可读，体现渐进式教学而非工程复用。

### langchain/（用 `create_agent` 托管 agent loop）
与 claude-code/ 手写 loop 形成对照：`create_agent(model, tools, system_prompt)` 一行组装出带工具调用循环的 agent。`stream.py` 展示 `stream_mode=["messages", "updates"]` 同时拿 token 流和步骤流，需要配 `checkpointer`（`InMemorySaver`）+ `thread_id`。

middleware 相关（`create_agent(middleware=[...])`，可以叠加多个）：
- **middleware_hitl.py**：`HumanInTheLoopMiddleware(interrupt_on={...})` 声明式指定哪些工具需要人工审批，底层仍是 `interrupt`/`Command(resume=...)`，但不用像 `langgraph/human_in_loop.py` 那样手写节点和路由。演示 approve/edit/reject 三种人工决策（`decisions` 列表要与 `interrupt_on` 里声明的 `allowed_decisions` 对应）。
- **middleware_summarization.py**：`SummarizationMiddleware(trigger=..., keep=...)` 在每次调用模型前检查历史长度，超过 `trigger` 阈值就用一次模型调用把旧消息压缩成摘要，只保留摘要 + 最近 `keep` 条消息，避免长对话超出上下文窗口。
- **trim_messages.py**：不调模型、直接按 token 数砍掉旧消息的更轻量方案，用 `@before_model` 装饰器把 `trim_messages()` 包成自定义 middleware（`SummarizationMiddleware` 内部就是同一套 `before_model` 钩子实现的）。两种历史管理方式的取舍：trim 零成本但旧内容彻底丢失，summarization 多一次模型调用但保留浓缩后的信息。
- **middleware_guardrails.py**：`ModelCallLimitMiddleware`/`ToolCallLimitMiddleware` 给 agent 加调用次数硬上限，防止死循环/失控空转烧 tokens；`exit_behavior="end"` 优雅结束，`"error"` 直接抛异常。演示用的 `roll_dice` 工具故意写死结果、不用真随机数，保证 demo 每次都能稳定复现 guardrail 生效（用真随机数有概率提前走运，观察不到限流效果）。
- **middleware_pii.py**：`PIIMiddleware("email"/"credit_card", strategy=...)` 在消息进入模型前自动脱敏，`strategy` 可选 redact/mask/hash/block；每种 pii_type 要单独一个 middleware 实例。
- 还有一些内置 middleware 没在本仓库演示（`ModelFallbackMiddleware`/`LLMToolSelectorMiddleware`/`ContextEditingMiddleware`/`ShellToolMiddleware` 等），用法思路类似，需要时查官方文档。

其他核心能力：
- **structured_output.py**：`create_agent(response_format=ToolStrategy(PydanticModel))` 让最终答案是定死结构的对象（`result["structured_response"]`），不用再从自由文本里解析。**踩坑**：直接传裸 Pydantic 类、或者只在 `response_format` 里声明而不在 `system_prompt` 里强调，实测都可能导致模型不触发结构化输出、`structured_response` 变成 `None`（工具越多的 agent 越容易发生，`deepagents/structured_output.py` 同样中招）——必须显式用 `ToolStrategy` 包一层，并在 `system_prompt` 里明确要求"最后必须调用该工具提交答案"，两者缺一都可能不稳定。所有用到这个模式的地方都改成了 `result.get("structured_response")` 防御性取值。
- **lcel.py**：不是 agent 范式，用 `prompt | model | StrOutputParser()` 管道操作符拼一条固定流程的链（`ChatPromptTemplate` 占位符模板 + `|` 串联 + `.batch()` 并发跑多个输入）。适合"流程固定、不需要模型自己决策要不要调用工具"的简单场景。
- **combo.py**：本目录收尾篇，一个客服工单场景把 PII 脱敏、HITL 审批、调用次数限制、自动摘要、结构化输出五个能力叠进同一个 `create_agent`。**踩坑**：用 checkpointer + 自定义 Pydantic `response_format` 组合时会看到一条 "Deserializing unregistered type" 警告——checkpointer 序列化自定义类型目前未来版本可能会被 block，属已知限制。

注意：`ChatAnthropic` 返回的 `AIMessage.content` 在开启 extended thinking 时是 block 列表（含 `thinking`/`text` 混合），需要用 `.text` 属性取纯文本，不能直接 `.content` 打印或塞回下一轮消息（全仓库统一遵循这个约定）。

### langgraph/（LangChain/DeepAgents 底层的图 runtime）
不经过 `create_agent`，直接用 `StateGraph` 手搭图，理解运行时概念：
- **quickstart.py**：`state`（TypedDict 共享状态）+ `add_conditional_edges`（条件路由）。
- **human_in_loop.py**：节点内 `interrupt(payload)` 暂停执行并把状态存入 checkpointer，外部用 `Command(resume=value)` 从断点恢复 —— `interrupt()` 的返回值就是 resume 传入的值。
- **persistence.py**：`InMemorySaver` 换成 `SqliteSaver` 落盘，新建图实例（模拟新进程）配合同一 `thread_id` 仍可读回历史状态，验证跨进程持久化。
- **multi_agent.py**：supervisor 多 agent 编排模式。每个节点（`supervisor`/`researcher`/`writer`）直接返回 `Command(update=..., goto=...)`，状态更新和路由决策合一，图的边由节点返回类型注解 `Command[Literal[...]]` 自动推断，不需要像 quickstart.py 那样单独写路由函数。`supervisor` 用 `model.with_structured_output(RouteDecision)` 决定下一步交给哪个 worker；`researcher`/`writer` 各自是完整的 `create_agent` 实例，执行完毕后固定路由回 `supervisor`，直到 supervisor 判定 `FINISH`。**踩坑**：`RouteDecision.instruction` 字段一开始没给默认值，supervisor 判定 `FINISH` 时模型经常不填这个字段，触发 pydantic 校验错误崩溃；supervisor 提示词也必须明确"必须先经过 writer 才能 FINISH"，否则 supervisor 有时会在 researcher 给完信息后就直接结束、跳过 writer。与 `deepagents/` 的 subagent 委派对照：deepagents 是"主 agent 内部通过 `task` 工具委派、中间过程对主 agent 不可见"，这里是"图层面平级节点编排、supervisor 能看到每个 worker 的完整产出"。
- **subgraph.py**：把一个编译好的 `StateGraph` 直接 `add_node` 进另一个图，从父图视角是个不透明的黑盒节点（层次化组合，区别于 multi_agent.py 的平级节点路由）。父子图 State 字段名相同时可以直接透传；字段名对不上时要写一个"胶水节点"手动做输入输出映射（文件里两种场景都演示了）。
- **send_map_reduce.py**：`Send(node, arg)` 在一个节点里一次性产出一组"去哪个节点、带什么参数"的任务，数量由运行时数据决定（不是写死在图结构里），LangGraph 会全部并行调度——map（并行处理）+ reduce（用 `Annotated[list, operator.add]` 自动合并结果）。**踩坑**：给并行分支的模型调用加"不超过 N 字"这种精确字数要求，会让 extended thinking 模型在 thinking 过程里反复数字数、有时把 thinking 预算耗尽导致最终文本被截断成空——改成宽松措辞（"简要总结"）后才稳定。
- **store.py**：`InMemoryStore` 实现跨 `thread_id` 的长期记忆，和 checkpointer 是两回事——checkpointer 按 `thread_id` 隔离（一次对话自己的完整历史），store 按自定义 `namespace`（比如 `(user_id, "memories")`）组织，只要 namespace 一样，换 `thread_id`（全新对话）依然能读到同一份记忆。节点函数把参数命名为 `store: BaseStore` 就会被自动注入，不用手动从 config 里掏。
- **time_travel.py**：`get_state_history(config)` 按时间倒序拿到每一个历史 checkpoint，`update_state(snapshot.config, values)` 基于某个历史 checkpoint 创建新 checkpoint 并打补丁，再用 `invoke(None, config=new_config)` 从这个新 checkpoint 继续跑，原来的错误路径依然保留在历史里、只是不再是分支终点——用于"agent 某一步判断错了，不用整个重来，回到出错前改对再继续"的调试场景。**注意**：`checkpoint_id` 是 UUID7，同一次运行里生成的多个 checkpoint 前 8 位大概率相同（时间戳精度不够区分），要看第二段才能分辨。
- **combo.py**：本目录收尾篇，一个多文档摘要器把 subgraph（总结单篇文档的两步流程）+ Send（按文档数量动态并行 fan-out 给子图）+ Store（记住用户跨会话的摘要风格偏好）+ checkpointer（每个任务/thread 自己的进度）拼进一个连贯场景。**踩坑**：子图字段名如果和父图同名（且没有 reducer），多个并行 Send 分支同时写同一个 state channel 会报 `InvalidUpdateError: Can receive only one value per step`——子图的私有字段必须和父图区分开命名，只有故意要合并的字段（这里是 `summaries`，带 `operator.add` reducer）才能同名共享。time travel 概念独立，没有强行拼进这个组合场景。

### deepagents/（`create_deep_agent`：在 `create_agent` 之上叠加 harness）
`create_deep_agent` 相比 `create_agent` 多出：自动 planning（todo 追踪）、虚拟文件系统、`subagents` 子任务委派，以及下面这些参数。
- **quickstart.py → research.py → stream.py**：quickstart（模拟工具）→ research（接入真实 `DuckDuckGoSearchRun`，无需 API key）→ stream（观察 `updates` 流）。注意 `stream.py` 里 `updates` 流的 `source` 实测下来只有 `model`/`tools` 两种，子 agent 委派（`task` 工具调用）不会拆出单独的 subagent 节点名——子 agent 内部完整跑完自己的一轮 model/tools 循环后，只把最终结果封装成一次 `tool-result` 返回，中间过程不可见。
- **filesystem.py**：虚拟文件系统本身。`create_deep_agent` 自动注入 `ls`/`read_file`/`write_file`/`edit_file` 等工具，agent 可以把中间结果存成"文件"；跑完之后用 `result["files"]`（`{路径: FileData}`，`FileData["content"]` 是字符串）读出运行期间产出的所有文件。默认是内存态的 `StateBackend`（文件内容存在 graph state 里，没配 checkpointer 就不会持久化）。
- **backend.py**：换成 `backend=FilesystemBackend(root_dir=..., virtual_mode=True)`，agent 的 `write_file`/`read_file` 工具背后直接读写本机真实磁盘（用系统临时目录做 `root_dir`，不污染仓库）；`virtual_mode=True` 让 agent 眼里的路径还是 `/xxx.md` 这种虚拟绝对路径，同时挡住 `..`/绝对路径穿越出 `root_dir`。**踩坑**：换成 `FilesystemBackend` 之后 `result["files"]` 实测是空字典（不像 `StateBackend` 会带上文件内容）——`result["files"]` 反映的是写进 graph state 的快照，`FilesystemBackend` 直接操作磁盘不会同步一份进 state，要拿文件内容只能直接读磁盘或调用 `read_file` 工具。
- **skills_memory.py**：`skills=[...]`（技能库，`SKILL.md` 文件 + YAML frontmatter，只有"名字+一句话描述"预先塞进 system prompt，agent 判断匹配当前任务才主动 `read_file` 去读完整规则，即"渐进式展示"）和 `memory=[...]`（`AGENTS.md`，每一轮都完整加载进 system prompt，不是按需读取，适合放用户偏好/项目约定）。**踩坑**：`skills` 不在 system_prompt 里明确提示"先查技能库"，是否真的被读取会不稳定；`memory` 里写生硬的格式指令（"每次回复都加签名"）不一定被照做，因为 deepagents 内置提示词会告诉模型 memory 是参考资料、不是必须服从的系统指令，换成自然的偏好类描述（"我的名字是 X，请这样称呼我"）才稳定生效。
- **permissions.py**：`permissions=[FilesystemPermission(operations=["read","write"], paths=["/secrets/**"], mode="deny"/"allow"/"interrupt")]` 按路径 glob 模式做细粒度访问控制，规则按声明顺序匹配、命中第一条生效、都不命中默认放行。`mode="interrupt"` 会自动转换成对应的 `interrupt_on` 配置走人工审批，不用自己再手写一遍。
- **hitl.py**：`create_deep_agent` 把 `langchain/middleware_hitl.py` 里要手动 import+组装的 `HumanInTheLoopMiddleware` 直接做成了一个入参 `interrupt_on=...`，用法（配置格式、`Command(resume={"decisions": [...]})` 恢复方式）完全一致，只是不用自己拼 middleware 列表。`subagents` 列表里的每个子 agent 也能单独配 `interrupt_on`，和主 agent 互不影响。
- **structured_output.py**：跟 `langchain/structured_output.py` 用法一样（`response_format=ToolStrategy(PydanticModel)`），但 deep agent 自带一整套内置工具（`write_todos`/文件工具/`task` 等）+ 自定义工具，可选项变多之后模型更容易"忘记"结构化输出也是待选项之一，`structured_response` 变 `None` 的概率比普通 `create_agent` 更高——必须在 system_prompt 里指名道姓要求调用该工具，不能只靠 `response_format` 参数本身。
- **combo.py**：本目录收尾篇，一个"发布前需要人工签字"的深度研究助手，把 subagents 委派、`backend=FilesystemBackend` 真实落盘笔记、`interrupt_on` 审批 `publish_report`、`response_format` 结构化收尾拼进一个连贯场景。**踩坑（两处）**：① 光靠 `response_format`/`interrupt_on` 参数存在不代表模型一定按预期的多步顺序执行——`publish_report` 有时不会被调用，system_prompt 必须明确写"这一步是强制的、不能跳过"；② 判断"是否命中 interrupt"不能想当然假设 `snapshot.tasks[0].interrupts` 一定有内容（复杂 agent 待处理任务不止一个时，第一个 task 未必是带 interrupt 的那个），要遍历 `snapshot.tasks` 找到真正 `.interrupts` 非空的那个，否则会 `IndexError`。

### rag/（embedding + 向量库 + retriever，包成 agent 工具）
`quickstart.py`：用 `VoyageAIEmbeddings` 把文本转向量，`InMemoryVectorStore`（`langchain_core`，纯内存无需部署）按相似度存取，`vector_store.as_retriever()` 包装成统一的 `invoke(query)` 接口。检索本身不直接调用 LLM——这里把 retriever 包成一个 `@tool` 交给 `create_agent`，让 agent 自己决定何时检索、检索什么关键词（agentic RAG），而不是手写"先检索再拼 prompt"的固定流程。知识库语料是本仓库各模块的简介文本，方便直接验证检索结果的准确性。与 `deepagents/` 的虚拟文件系统对照：两者都是"扩展 agent 上下文"的手段，但虚拟文件系统管理的是 agent 自己产出的中间内容（精确路径读写），向量库管理的是外部知识库（语义相似度匹配）。

### web/app.py（Chainlit Web UI，唯一的网页界面）
仓库里其余脚本都是终端层面的（REPL 或一次性 print），只有这一个是浏览器网页。用 `@cl.set_chat_profiles` 声明 4 个 chat profile（左上角下拉切换），对应 4 种玩法，每种在 `on_chat_start` 里按 profile 构建对应 agent/graph 存入 `cl.user_session`，`on_message` 里按 profile 分发到对应 handler：
- **Deep Research**：`deepagents/research.py` 的搬运版。`checkpointer=InMemorySaver()` + `thread_id=cl.context.session.id` 绑定浏览器会话实现多轮记忆；`agent.astream(stream_mode="messages")` 逐 token 推送；工具调用/subagent 委派由 `cl.LangchainCallbackHandler()` 自动渲染为可折叠步骤。`_build_user_content()` 支持图片附件（base64 → LangChain 多模态 content block）。
- **人工审批 (HITL)**：`langchain/middleware_hitl.py` 的网页版。用 `cl.AskActionMessage` 弹窗替代终端的硬编码三段式演示，真正等人工点击 批准/修改/拒绝；`_resolve_hitl()` 递归处理——每次 resume 后重新检查 `agent.aget_state(config).next`，还有 interrupt 就继续弹窗，没有就发最终消息。
- **多 Agent 编排 (Supervisor)**：`langgraph/multi_agent.py` 的网页版。用 `graph.astream(stream_mode="updates")` 拿到每个节点的增量更新，每个更新包一层 `cl.Step(name=source)` 展示成可折叠步骤，直观看到 supervisor→researcher→writer 的路由轨迹。
- **RAG 检索问答**：`rag/quickstart.py` 的网页版，检索步骤同样交给 `cl.LangchainCallbackHandler()` 自动折叠展示，无需额外处理。

四个 profile 的构建逻辑都直接写在 `web/app.py` 里，**没有** import 同名的 `langchain/`、`langgraph/`、`rag/` 目录——这几个目录名和同名 pip 包 (`langchain`、`langgraph`) 撞名，混着 import 容易出隐蔽的 shadowing bug，索性照抄一份保持独立可读（延续仓库"不抽取公共模块"的一贯风格）。

踩坑记录：`langgraph/multi_agent.py` 的 supervisor 提示词最初没有强制"必须先经过 writer 才能 FINISH"，导致 supervisor 有时在 researcher 给出信息后就直接判定 FINISH、跳过 writer，网页上会看到只有 researcher 步骤、没有最终整理的答案。已在 prompt 里加了强制要求，网页 handler 也加了兜底（writer 没跑就退化用 researcher 的最后内容作为答案，不会给空白）。

### pi/（earendil-works/pi：TypeScript "反方参照"，唯一的非 Python 目录）
[pi](https://github.com/badlogic/pi-mono)（原 badlogic/pi-mono，作者 Mario Zechner）是 70k+ stars 的 TS coding agent 工具箱，设计哲学与 LangChain 系相反——不做图 runtime、不做重抽象，消息就是可 `JSON.stringify` 的普通对象。独立 npm 工程（`pi/package.json`，`"type": "module"`，tsx 运行），三个示例按 pi 自己的三层包结构递进，与本仓库其他目录一一对照：
- **ch01_pi_ai.ts**：`@earendil-works/pi-ai`，统一多厂商 LLM API（对照 claude-code/ch01.py 的裸 anthropic SDK 层）。演示 completeSimple/streamSimple/手写 `stopReason === "toolUse"` 工具循环。**踩坑**：请求失败不抛异常，错误静默放在返回消息的 `stopReason === "error"` + `errorMessage` 里，不检查只会看到空输出、usage 全 0——三个示例都加了防御性检查。
- **ch02_agent_loop.ts**：`@earendil-works/pi-agent-core` 的 `Agent` 类托管 loop（对照 langchain/quickstart.py 的 `create_agent`），TypeBox schema + execute 定义 `AgentTool`，`subscribe` 事件流观察逐 token 输出和工具执行；pi 特色是 steering/followUp 队列（运行中插话改方向）。
- **ch03_coding_agent.ts**：`@earendil-works/pi-coding-agent` 的 `createAgentSession`（对照 deepagents 的 `create_deep_agent`）：内置 read/bash/edit/write 等编码工具 + `tools` 白名单 + `defineTool` 自定义工具。示例用 `SessionManager.inMemory()` 不落盘。注意 pi 的文件工具直接操作真实磁盘（相当于 deepagents 的 FilesystemBackend 是常态）。
- **ch04_steering.ts**：pi 招牌能力 steering/followUp——agent 运行中人主动插话（steer 当前工具跑完即生效改方向；followUp 排队等任务完整结束后触发新一轮）。与 langgraph 的 `interrupt` 方向相反（那是 agent 停下等人）。工具要配 `toolExecution: "sequential"` 拉长执行窗口才有机会观察插话生效；结尾要 `await agent.waitForIdle()`，否则 followUp 触发的后续轮次跑一半脚本就退出了。
- **ch05_session_tree.ts**：会话持久化三件套——`SessionManager.create()` 落盘 jsonl（每条消息一个带 id/parentId 的 entry，天然构成树）、`SessionManager.open()` 续接（对照 langgraph/persistence.py）、`navigateTree()` 把叶子指针挪回历史节点即分叉（对照 langgraph/time_travel.py，但不用翻 checkpoint 列表）、`session.compact()` 手动触发压缩（对照 middleware_summarization.py）。
- **ch06_skills_memory.ts**：skills（`<cwd>/.pi/skills/<名字>/SKILL.md`，YAML frontmatter 的 name/description 预先进 system prompt、全文按需 read——渐进式展示）+ AGENTS.md（每轮全文进 system prompt，放用户偏好）。目录约定与 Claude Code 同构，AGENTS.md/CLAUDE.md 互通。`DefaultResourceLoader` 的资源发现是纯本地扫描，这一步无需 API key 即可验证（本示例前半段可离线跑通）。对照 deepagents/skills_memory.py。

环境变量沿用根目录 `.env`：pi-ai 读 `ANTHROPIC_OAUTH_TOKEN`/`ANTHROPIC_API_KEY`（前者优先），示例在设置了 `ANTHROPIC_BASE_URL` 时会 `delete` OAuth token（对应 Python 侧 pop `ANTHROPIC_AUTH_TOKEN` 的约定）；网关 baseUrl 通过覆盖 `Model` 对象上的 `baseUrl` 字段实现（不是全局配置），`MODEL_ID` 不在 pi 内置目录时拿内置模型当模板改 `id` 兜底。npm 包名已从老文章里的 `@mariozechner/pi-*` 迁移为 `@earendil-works/pi-*`。调研笔记与更多踩坑见 `pi/README.md`。

### 空占位目录
`deep_agents/`（注意与 `deepagents/` 拼写相近，实为空目录，未使用）、`openclaw/` 均只含 `.gitkeep`，是预留给未来学习内容的占位符，当前无代码。

# ch_03_filesystem.py 学习问答记录

配套文件:`deepagents/ch_03_filesystem.py`(默认 `StateBackend` 虚拟文件系统)与 `deepagents/ch_04_backend.py`(切到真实磁盘的 `FilesystemBackend`)。本文整理一次围绕"如何让文件系统落盘、`backend` 到底是什么概念"展开的问答,起点是一个个人练习脚本 `tmp/deepagents_demo.py`(未纳入正式教学序号,仅作调试用),按提问顺序排列,均附源码文件路径与行号,方便后续回查验证(注意:行号对应写作时安装的 `deepagents` 版本,升级依赖后可能漂移)。

---

## Q1. 想让文件系统落盘,应该怎么改 `create_deep_agent` 参数?之前调用为什么存的是 `/tmp/python_vs_typescript.md`,不是 `notes.md`?

**为什么文件名对不上**:`create_deep_agent` 默认用的是内存态的 `StateBackend`(对应 `ch_03_filesystem.py` 演示的行为)——`write_file` 工具写的其实是 LangGraph state 里的一个 dict,不碰真实磁盘。`StateBackend` 没有 `root_dir` 概念,路径只是个自由字符串键名,system_prompt 里"可以保存在 notes.md"只是**建议**不是强制,模型就按自己的习惯起了个更贴合内容的文件名(比如 `/tmp/python_vs_typescript.md`)——这个 `/tmp` 前缀纯属虚拟字符串,跟磁盘上真实的 `/tmp` 目录毫无关系。

**怎么真正落盘**:参考 `deepagents/ch_04_backend.py`,给 `create_deep_agent` 传 `backend=FilesystemBackend(root_dir=..., virtual_mode=True)`:

```python
from pathlib import Path
from deepagents.backends import FilesystemBackend

output_dir = Path(__file__).parent / "output"
output_dir.mkdir(exist_ok=True)
backend = FilesystemBackend(root_dir=str(output_dir), virtual_mode=True)

agent = create_deep_agent(model=model, tools=[search], backend=backend, system_prompt=...)
```

- `root_dir`:真实磁盘上的根目录,agent 所有文件操作最终落在这里面。
- `virtual_mode=True`:agent 眼里路径仍是 `/xxx.md` 这种虚拟绝对路径,内部映射到 `root_dir/xxx.md`,同时挡住 `../` 路径穿越,防止写出 `root_dir` 之外(0.5.0 起是必填参数,不传会有 `DeprecationWarning`)。

**换 backend 后要注意**:`result["files"]` 在 `FilesystemBackend` 下是空字典(写盘不会同步一份进 LangGraph state,这是 `ch_04_backend.py` 里记录的坑),要看真实产出得直接读磁盘:

```python
for file_path in sorted(output_dir.rglob("*")):
    if file_path.is_file():
        print(file_path.relative_to(output_dir), file_path.read_text(encoding="utf-8"))
```

**实测验证**:即使把 system_prompt 改成"必须存到 notes.md、不要用其他文件名",模型依然自选了 `/tmp/python_vs_ts.md`(因 `virtual_mode=True` 是相对 `root_dir` 处理绝对路径,实际落到了 `output_dir/tmp/python_vs_ts.md`,多了一层 `tmp/` 子目录)。说明**文件名不听指挥不是 backend 类型的问题,是 system_prompt 约束力的问题**——跟仓库 `deepagents/ch_08_structured_output.py`/`ch_10_combo.py` 里"模型对必须调用/必须命名类指令服从度不稳定"的踩坑记录是同一类现象。如果确实需要文件名可控,更可靠的做法是不依赖 prompt 指令,代码侧遍历 `output_dir` 下所有产出而不是假设固定文件名。

---

## Q2. 落盘改动的这几行代码,逐行解释下?

（改动文件:`tmp/deepagents_demo.py`）

- `from pathlib import Path`:用 `Path` 拼接/遍历落盘目录,比手写字符串拼路径更安全(跨平台分隔符、内置 `.exists()`/`.rglob()`)。
- `from deepagents.backends import FilesystemBackend`:引入让文件工具操作真实磁盘的 backend 实现。
- `output_dir = Path(__file__).parent / "output"`:`__file__` 是脚本自身路径,`.parent` 取所在目录,拼出脚本同级的 `output/` 子目录作为落盘位置。
- `output_dir.mkdir(exist_ok=True)`:`FilesystemBackend` 不会自动建 `root_dir`,目录不存在时内部读写会报错,必须在构造 backend 前手动确保存在;`exist_ok=True` 防止重复运行时 `FileExistsError`。
- `backend = FilesystemBackend(root_dir=str(output_dir), virtual_mode=True)`:关键改动。`root_dir` 要求传字符串,`Path` 对象要 `str()` 转一下;`virtual_mode=True` 见 Q1。
- `create_deep_agent(..., backend=backend, ...)`:不传时默认 `StateBackend`(内存态);传了之后内置的 `write_file`/`read_file`/`ls` 等工具自动切换成走这个 backend,不用改工具调用逻辑。
- 结尾把 `for path, file_data in result.get("files", {}).items(): ...` 换成 `output_dir.rglob("*")` 直接读盘:
  - `rglob("*")` 而不是 `glob("*")`:递归遍历,防止 agent 在子目录下建文件(实测确实发生了,见 Q1 的验证)。
  - `sorted(...)`:让每次运行打印顺序稳定,方便对比。
  - `if file_path.is_file()`:`rglob` 也会枚举到目录本身,过滤掉只留文件。
  - `file_path.relative_to(output_dir)`:打印相对路径,不要一长串绝对路径。
  - `file_path.read_text(encoding="utf-8")`:标准库直接读文件内容,不依赖 deepagents 任何 API,用来证明"文件真实存在于磁盘"。

---

## Q3. `backend` 到底是什么概念?和 `middleware`、`StateGraph` 的关系是啥?

**三层各管什么**:

| 层级 | 类比 | 管什么 |
|---|---|---|
| `StateGraph`(langgraph) | 操作系统内核 | 节点怎么调度、state 怎么流转 |
| `middleware`(langchain `create_agent`) | 中间件/拦截器 | loop 某个通用步骤前后要不要额外做事(限流、脱敏、审批、摘要) |
| `backend`(deepagents 独有) | 文件系统驱动 | 具体到"文件类工具"这一小撮功能,字节实际写到哪种介质 |

**StateGraph 是最底层**:`create_deep_agent(...)` 的返回值类型签名就是 `CompiledStateGraph[...]`——deep agent 本质上就是一张编译好的 StateGraph,`agent.invoke()`/`.stream()` 用的是 StateGraph 原生方法,跟手搭的图(`langgraph/ch_01_quickstart.py`)是同一套接口。

**middleware 是包在 loop 关键步骤外面的通用钩子**:作用点是"模型调用前/后、工具调用前/后"这几个跟具体工具无关的切面(`HumanInTheLoopMiddleware`/`SummarizationMiddleware`/`PIIMiddleware` 等)。deepagents 自己的"todo 计划""子任务委派""skills""memory"这些能力也都是内部自带 middleware 实现的(`deepagents/middleware/`:`subagents.py`、`skills.py`、`memory.py`、`summarization.py` 等)——`create_deep_agent` 内部大致是:拼一堆内置 middleware(+用户传的自定义 middleware)→ 交给 `langchain.agents.create_agent` → 编译成 `CompiledStateGraph`。

**backend 是三者里最窄的概念,只管文件类工具的存储介质**:内置文件工具(`ls`/`read_file`/`write_file`/`edit_file`/`glob`/`grep`)由 `FilesystemMiddleware`(`deepagents/middleware/filesystem.py:765`)注册,其构造函数吃一个 `backend: BackendProtocol` 参数——每次文件工具被调用,middleware 内部直接转发给 `backend.write()`/`backend.read()`/`backend.grep()`/`backend.glob()` 等方法执行(例如 `write_file` 工具实现见 `filesystem.py:1218-1255`,内部 `resolved_backend.write(validated_path, content)`)。

`BackendProtocol`(`deepagents/backends/protocol.py:329`)是一份存储接口协议(`ls/read/grep/glob/write/edit/upload_files/download_files`,同步+异步各一份),deepagents 提供多种实现:

```python
import deepagents.backends as b
[n for n in dir(b) if not n.startswith('_')]
# ['BackendContext', 'BackendProtocol', 'CompositeBackend', 'ContextHubBackend',
#  'FilesystemBackend', 'LangSmithSandbox', 'LocalShellBackend', 'NamespaceFactory',
#  'StateBackend', 'StoreBackend', ...]
```

- `StateBackend`(默认,内存,`ch_03_filesystem.py`)
- `FilesystemBackend`(真实磁盘,`ch_04_backend.py`)
- `StoreBackend`(写进 langgraph 的 `BaseStore`,天然跨 thread 共享,对照 `langgraph/ch_06_store.py`)
- `CompositeBackend`(按路径前缀路由到不同 backend)
- `LocalShellBackend`/`ContextHubBackend`/`LangSmithSandbox` 等更专门的实现

**结论**:backend 不是 middleware 的平级概念,而是被某个特定 middleware(`FilesystemMiddleware`)持有并委托执行的存储适配器。换 backend 不改变图的节点结构,也不影响其他 middleware(HITL 审批、摘要)的行为;两者是正交的旋钮,可以自由组合(如 `ch_05_permissions.py`/`ch_06_hitl.py` 演示的"真实磁盘 backend + 文件操作人工审批")。

---

## Q4. `backend` 为什么在 `create_deep_agent` 的函数签名内,而不是只在 middleware 的签名内?用户自己手写的 middleware 能不能用这个 backend?

**先纠正前提**:`backend` 其实不止在 `create_deep_agent` 签名里——`FilesystemMiddleware` 自己的构造函数也收 `backend` 参数,可以单独 `from deepagents.middleware.filesystem import FilesystemMiddleware` 拿出来用。

**为什么 `create_deep_agent` 顶层也要暴露它**:grep 了 `deepagents/graph.py` 里所有用到 `backend` 的地方,发现同一个 `backend` 实例被喂给了不止一处:

```python
# deepagents/graph.py(节选,行号为实际源码)
615: backend = backend if backend is not None else StateBackend()
646: subagent middleware 构造时传 backend=backend
650: create_summarization_middleware(subagent_model, backend)   # 工具结果超限"驱逐"到文件也要用 backend
655: SkillsMiddleware(backend=backend, sources=subagent_skills)  # 读技能库文件也走 backend
721/725/729: general-purpose 子agent 的对应三处
777: SkillsMiddleware(backend=backend, sources=skills)
780/788: FilesystemMiddleware(..., backend=backend, ...)
801: create_summarization_middleware(model, backend)
825: MemoryMiddleware 相关,backend=backend
```

也就是说 **filesystem 工具、skills 库读取、大工具结果落盘驱逐(eviction)、子 agent 的文件系统,四类内置能力全依赖同一个 backend 实例**。如果 `backend` 只是 `FilesystemMiddleware` 自己的构造参数,用户就得手动 new 出好几个 middleware,还要保证传给它们的是*同一个* backend 对象——传错、传出两个不同实例,会出现"`write_file` 写到 A 存储,`skills` 却在 B 存储里找文件"的诡异 bug。`create_deep_agent(backend=...)` 本质是帮你把这次"多个内置 middleware 必须共享同一 backend"的装配工作做掉,是**便利/一致性保证**,backend 概念上仍然从属于 middleware,只是被顶层函数转发给了好几个。

**用户自己手写的 middleware 能不能用它**:能,两种路子。

方式一(最简单):自己持有的 `backend` 变量本身就是普通对象,自定义 middleware/tool 函数直接闭包调用即可,不需要框架配合:

```python
def my_custom_tool(runtime: ToolRuntime) -> str:
    res = backend.read("/notes.md")   # 闭包直接拿外面构造 create_deep_agent 时用的那个 backend 变量
    ...
```

方式二(进阶):`backend` 参数类型其实是 `BackendProtocol | Callable[[ToolRuntime], BackendProtocol]`(`BackendFactory` 定义于 `deepagents/backends/protocol.py:884-885`),可以传一个"工厂函数",每次工具调用时才根据当时的 `runtime` 动态决定用哪个 backend 实例(比如按当前用户切换存储桶)。想复用这套动态解析,自定义工具需要声明 `runtime: ToolRuntime` 参数(langchain 自动注入),照着 deepagents 内置 `write_file` 工具的方式(`filesystem.py:1224` `self._get_backend(runtime)` → 内部调用 `_resolve_backend(backend, runtime)`,定义在 `protocol.py:888`)解析出真正的 backend 实例。

**有个边界要注意**:`DeepAgentState`(共享 graph state,`deepagents/graph.py:65-68`)本身**不携带 backend 字段**,没有"随手从 state/context 里摸出当前 backend"的通用机制。`deepagents.backends` 里确实有个 `BackendContext` 类(`deepagents/backends/store.py:58`),但那是 `StoreBackend` 命名空间解析用的遗留概念,源码里明确标了 deprecated(`store.py:49`),不是给任意 middleware 用的通用访问器。所以如果自定义 middleware 跟构造 `backend` 的代码不在同一作用域、拿不到闭包引用,就没有框架层面的"发现"手段——必须自己想办法把这个引用传过去(闭包、自定义 `context_schema` 字段,或再包一层配置对象)。

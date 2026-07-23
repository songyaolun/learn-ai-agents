# ch_04_backend.py 学习问答记录

配套文件:`deepagents/ch_04_backend.py`(`FilesystemBackend` 落真实磁盘的正式示例)。本文整理围绕个人练习脚本 `tmp/deepagents_demo.py`(未纳入正式教学序号,仅作调试用,写法与 `ch_04_backend.py` 同构)里 `scratch_dir = tempfile.mkdtemp(...)` 这几行展开的问答,主题是 `tempfile.mkdtemp()` 的实际行为、以及 Python 模块顶层代码 vs 函数体的执行时机,按提问顺序排列,均附源码文件路径与行号(注意:行号对应写作时的脚本版本,后续改动可能漂移)。

---

## Q1. `scratch_dir = tempfile.mkdtemp(prefix="deepagents_backend_demo_")`(`tmp/deepagents_demo.py:21`)建的目录是在项目相对目录下吗?为什么本机实测路径这么"奇怪"?前缀是什么时候补全的?为什么要声明这个前缀?

**不是项目目录**:`tempfile.mkdtemp()` 默认创建在 `tempfile.gettempdir()` 返回的**系统级临时目录**下,跟项目目录、脚本所在目录都无关。本机(macOS)实测:

```python
>>> import tempfile; tempfile.gettempdir()
'/var/folders/_v/ywrph3_n4pz9fc_6px__psrr0000gn/T'
```

**为什么不是固定的 `/tmp`**:macOS 上 `/tmp` 是 `/private/tmp` 的软链接,是所有用户共享的老式临时目录。但 Darwin 另外维护一套按用户隔离的临时目录机制——系统通过 `confstr(_CS_DARWIN_USER_TEMP_DIR)` 给每个登录用户分配一个专属临时目录,形如 `/var/folders/<xx>/<稳定的用户标识>/T/`(同一用户重启电脑、重开终端通常不变,不是每次运行都换)。好处:①隔离,别的用户看不到你的临时文件;②`launchd` 会定期主动清理,比传统 `/tmp` 只在重启时清空更积极。Python 的 `tempfile.gettempdir()` 按顺序检查环境变量 `TMPDIR` → `TEMP` → `TMP`,macOS 系统本身已把 `TMPDIR` 设成这个路径,所以直接沿用;Linux 没有这套机制,才会老实返回 `/tmp`。

**前缀何时补全**:就在 `tempfile.mkdtemp(prefix=...)` 这行**同步执行时**立刻补全,不是延迟发生的——内部生成一段随机后缀(默认 8 位随机字母数字),拼成 `prefix + 随机后缀`,在 `dir=`(默认即 `gettempdir()`)下尝试 `os.mkdir()`,撞名则换一段重试(概率极低),成功后返回完整路径。

**为什么声明前缀**:纯粹为了可辨识性。不传 `prefix` 时默认前缀是 `tmp`,生成的目录名类似 `tmp8f3jd92k`,完全看不出是哪个脚本创建的;加上 `deepagents_backend_demo_` 后,手动 `ls /var/folders/.../T/` 排查残留临时目录时能一眼认出是这个 demo 留下的。

**顺带确认**:`mkdtemp()` 是"创建 + 命名"一步完成的原子操作——磁盘上真的创建了这个目录(权限 `0700`,仅创建者可读写执行),不是只生成一个名字字符串。这也是它和已废弃的 `tempfile.mktemp()` 的关键区别:`mktemp()` 只生成一个"看起来没被占用"的名字、不创建实际文件/目录,存在 TOCTOU(Time-Of-Check-To-Time-Of-Use)竞态风险——拿到名字后、真正创建文件前的间隙可能被恶意进程抢先创建同名文件做符号链接攻击。`mkdtemp()`/`mkstemp()` 内部用系统调用保证"检查名字可用"和"创建"是原子的一步,官方文档明确标注 `mktemp()` 不安全、推荐一律用 `mkdtemp()`/`mkstemp()`。

---

## Q2. `scratch_dir`/`search`/`agent` 这几行代码写在 `main` 函数外面、且在 `if __name__ == "__main__":` 判断之前,真正的执行顺序是怎样的?

Python 脚本没有"先扫描函数体再决定顺序"这回事——顶层代码严格按物理行号从上到下顺序执行,`if __name__ == "__main__":` 只是这个顺序流程里最后一段普通代码,不是什么入口跳转。以 `tmp/deepagents_demo.py` 为例,实际执行顺序:

```
第 1~10 行   import 依赖
第 12 行     load_dotenv() —— 加载 .env
第 15~18 行  model = ChatAnthropic(...) —— 只构造客户端对象,不产生网络请求
第 21 行     scratch_dir = tempfile.mkdtemp(...) —— 真的在磁盘创建了临时目录
第 22 行     print(...) —— 打印路径
第 24 行     search = TavilySearch(...) —— 构造搜索工具对象
第 26 行     backend = FilesystemBackend(...) —— 用 scratch_dir 构造 backend
第 28~37 行  agent = create_deep_agent(...) —— 组装 agent graph,仍未调用模型
─────────────────────────────
第 40 行     if __name__ == "__main__":  ← 走到这里才判断
第 41~62 行  只有直接 `python deepagents_demo.py` 运行时才会进入这个 if 块,
             真正调用 agent.invoke()(第一次触发 LLM/工具调用)在这里发生
```

关键点:`model`、`scratch_dir`、`search`、`backend`、`agent` 这几行**没有被任何 `def` 包裹**,是模块顶层语句,只要文件被 Python 加载(无论直接运行还是被 `import`)就会无条件执行;`if __name__ == "__main__":` 只决定"调用 `agent.invoke()`" 这部分要不要跑,跟前面这几行初始化代码跑不跑完全无关。

**实践提醒**:如果将来有别的脚本 `import deepagents_demo`(比如写测试时),第 21 行的 `tempfile.mkdtemp()` 会在 import 那一刻悄悄在磁盘创建临时目录、第 28 行也会真的构造出一个 agent 实例——这些都是"import 时副作用"。这个脚本本身是一次性脚本没问题,但若要把这类初始化代码复用到别处,通常会包进一个 `main()` 函数或 `if __name__` 块,避免 import 时产生非预期副作用(见 Q3)。

---

## Q3. 如果把这些代码包进 `main()` 函数里,是不是就不会被 `import` 执行了?这个执行机制到底是怎么回事?

分两层看:

**第一层:`def` 语句本身不执行函数体**。

```python
def main():
    scratch_dir = tempfile.mkdtemp(prefix="deepagents_backend_demo_")   # 这里不会执行
    print(f"真实磁盘 root_dir: {scratch_dir}")
```

`def main(): ...` 执行时,Python 只做一件事:创建一个函数对象、绑定到名字 `main`。函数体在这一刻完全不运行,只是被"记录"下来,等以后显式调用 `main()` 时才真正执行。这是和 Q2 里那份代码最本质的区别——原来 `scratch_dir = tempfile.mkdtemp(...)` 是模块顶层的一条语句,没有 `def` 包裹,加载文件时就得执行;一旦包进 `def main():` 里,它就从"立刻执行的语句"变成"函数体里待执行的代码",只有显式调用才会跑。

**第二层:`import` 只执行顶层代码,不会自动调用函数**。`import foo` 内部大致做三件事:①若 `sys.modules` 已有 `foo`,直接返回缓存(不会重新执行,这也是为什么同一模块被多处 import 只真正跑一次);②否则从头到尾按顺序执行 `foo.py` 里所有**顶层代码**(顶层 = 没被任何 `def`/`class` 包裹、缩进为 0 的代码);③把执行过程中产生的顶层名字收集成模块对象。第②步只执行顶层代码,而 `def main(): ...` 这一整块,从 import 的视角看,它本身就是顶层代码里的"一条语句"(定义函数这个动作),执行完这条语句后函数体从未被运行过,因为没人调用它。

所以只要把初始化代码全部塞进函数:

```python
def main():
    model = ChatAnthropic(...)
    scratch_dir = tempfile.mkdtemp(prefix="deepagents_backend_demo_")
    ...
    agent = create_deep_agent(...)
    result = agent.invoke(...)
    ...

if __name__ == "__main__":
    main()
```

别处 `import deepagents_demo` 时,Python 只会执行到 `def main(): ...`(创建函数对象)和 `if __name__ == "__main__":`(判断),不会创建临时目录、不会构造 agent——因为 `main()` 这个调用只在 `if __name__ == "__main__":` 分支里,而这个分支在 import 场景下判断为 `False`。

**`if __name__ == "__main__":` 单独存在时为什么能生效**:这行本身不是 `def`,是普通 `if` 语句,属于顶层代码,import 时一定会执行到、一定会被判断。它能区分"直接运行"和"被 import",靠的是模块内置变量 `__name__` 的值不同:

| 运行方式 | 该模块的 `__name__` 的值 |
|---|---|
| `python deepagents_demo.py` 直接运行 | `"__main__"` |
| 被其他文件 `import deepagents_demo` | `"deepagents_demo"`(模块自己的名字) |

`if __name__ == "__main__":` 这行判断本身一定会执行(顶层代码),只是**判断结果**在两种场景下不同,从而决定块内代码要不要跑。

**结合起来看**:`tmp/deepagents_demo.py` 目前 `model = ...`、`scratch_dir = ...`、`agent = ...` 都是裸露的顶层语句,无论直接运行还是被 import 都会执行;只有 `agent.invoke(...)` 写在 `if __name__ == "__main__":` 块里,只在直接运行时执行。若想让"构造 agent 之前的所有初始化"也做到"只有真正需要时才跑",必须把它们一起挪进 `main()`(或拆成 `build_agent()` 之类的函数)——单纯依赖 `if __name__ == "__main__":` 挡不住模块顶层那些裸露的赋值语句。

---

## Q4. 实际执行 `ch_04_backend.py` 后生成的文件是 `python_decorators_notes.md`,不是 `notes.md`,为什么?

**原因**:[ch_04_backend.py:55-58](ch_04_backend.py#L55-L58) 的 system_prompt 只是这样写的:

```python
system_prompt=(
    "You are a research assistant. When asked to save findings, use write_file "
    "to save them to a file named notes.md before giving your final answer."
)
```

这只是自然语言层面的一句"建议",不是对 `write_file` 工具参数的硬性约束——`write_file` 的 `path` 参数本质是个自由字符串,模型完全可以自己决定传什么值。当任务内容是"整理 Python 装饰器"时,模型觉得 `python_decorators_notes.md` 比通用的 `notes.md` 更贴合内容语义,就自己改了文件名。这跟 [ch_03_filesystem.qa.md](ch_03_filesystem.qa.md) Q1 里记录的 `/tmp/python_vs_typescript.md` 是同一类现象:模型对"必须用固定文件名"这种指令的服从度不稳定,倾向于按内容起一个更有语义的名字。

**连带影响**:原本 [ch_04_backend.py:80](ch_04_backend.py#L80)(改动前)硬编码读取路径是 `Path(scratch_dir) / "notes.md"`,一旦模型改了文件名,这行就会读到 `notes_path.exists() == False`,后面"打印磁盘上的原始内容"那段代码直接被跳过,验证效果打了折扣。

**修复方式**:不在代码里假设固定文件名,改成遍历 `scratch_dir` 找所有 `.md` 文件([ch_04_backend.py:76-86](ch_04_backend.py#L76-L86)):

```python
md_files = sorted(Path(scratch_dir).glob("*.md"))
print(f"磁盘上发现的 .md 文件: {[f.name for f in md_files]}")
for md_file in md_files:
    print(f"\n--- {md_file.name} ---")
    print(md_file.read_text(encoding="utf-8"))
```

这样无论模型最终把文件存成 `notes.md` 还是 `python_decorators_notes.md`,验证逻辑都能正确找到并打印内容,不再依赖"模型一定会听话用哪个文件名"这个不稳定的前提——这也是 `ch_03_filesystem.qa.md` Q1 给出的同一条结论:文件名不听指挥不是 backend 类型的问题,是 prompt 对模型的约束力天然有限,想验证"文件确实写到了磁盘",代码侧不该依赖固定文件名假设。

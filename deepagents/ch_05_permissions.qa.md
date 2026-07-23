# ch_05_permissions.py 学习问答记录

配套文件:`deepagents/ch_05_permissions.py`。本文整理阅读该文件时对 `FilesystemPermission.paths`(注释里写的"glob 模式列表")展开的问答,按提问顺序排列,均附源码文件路径与行号,方便后续回查验证(注意:行号对应写作时安装的 `deepagents` 版本,升级依赖后可能漂移)。

---

## Q1. glob 和 rglob 是啥工具?什么是 glob 模式列表?

问题源头:`ch_05_permissions.py:14` 里 `paths=["/secrets/**"]` 后面跟的注释「glob 模式列表, 必须以 "/" 开头」,不清楚 glob/rglob 具体指什么。

**① glob 模式(pattern)是什么**

一种用通配符描述"一批路径"的字符串语法,起源于早期 Unix shell,Python/Git 等工具通用这套规则:

| 符号 | 含义 |
|---|---|
| `*` | 匹配任意字符,但**不跨目录**(不匹配 `/`) |
| `**` | 匹配任意层级目录(含 0 层),即"递归通配符",也叫 globstar |
| `?` | 匹配单个字符 |
| `[abc]` | 匹配方括号里任一字符 |

所以 `paths=["/secrets/**"]` 的意思是:"`/secrets/` 目录下的所有文件,不管嵌套多少层子目录"。如果写成 `/secrets/*`,就只匹配 `/secrets/` 直接下一层,不会递归进子目录。

**② `glob()`/`rglob()` 是什么工具**

Python 标准库 `pathlib.Path` 的两个方法,作用是"用 glob 模式去真实搜索文件系统":

```python
from pathlib import Path

Path(".").glob("*.py")        # 只搜当前目录下的 .py 文件
Path(".").glob("**/*.py")     # 显式加 ** 递归搜索所有子目录
Path(".").rglob("*.py")       # 等价于上面这行 —— rglob = "recursive glob" 的简写
```

`rglob(pattern)` 本质就是 `glob("**/" + pattern)` 的语法糖,区别只在于要不要手动写 `**`。

**③ 回到这个文件:`Path.glob`/`rglob` 其实没被直接用到**

`FilesystemPermission.paths`(`ch_05_permissions.py:14`)只是一个字符串列表,deepagents 底层用的是第三方库 `wcmatch`,不是 `pathlib`:

```python
# .venv/lib/python3.11/site-packages/deepagents/middleware/filesystem.py
19:  import wcmatch.glob as wcglob
75:  _FS_WCMATCH_FLAGS = wcglob.BRACE | wcglob.GLOBSTAR
135: if any(wcglob.globmatch(path, pattern, flags=_FS_WCMATCH_FLAGS) for pattern in rule.paths):
```

`wcmatch.glob.globmatch(path, pattern)` 做的是"这个路径字符串符不符合这个 pattern"的布尔判断,不是去磁盘上搜文件——这和 `Path.glob()`/`rglob()`(真的会遍历磁盘、返回匹配到的路径列表)是两回事:

- `Path.glob()`/`rglob()`:拿一个 pattern,**从磁盘搜出**一堆匹配的路径,用于"我要找出所有符合条件的文件"。
- `wcmatch.glob.globmatch()`:拿一个**已知路径** + 一个 pattern,**判断是否匹配**,用于"来一个路径就要马上放行/拒绝"的权限校验场景,更合适——每次 agent 调用文件工具时,拿到的是一个具体路径,只需要问"这个路径命中哪条规则",不需要真的扫描磁盘。

两者语法(`*`/`**`/`?`)是同一套通配符规则,只是用途不同:一个是"搜索",一个是"匹配校验"。

顺带一提,同一个文件里 `create_deep_agent` 自动注入的内置 `glob` 工具(供 agent 自己调用去"查找文件",非本文讨论的权限规则)用的也是这同一套语法,`GLOB_TOOL_DESCRIPTION`(`filesystem.py:463`)里写明"Supports standard glob patterns: `*`(any characters), `**`(any directories), `?`(single character)"——这是 agent 用来"搜索文件"的场景,和 `permissions=[...]` 用同一套 pattern 语法做"路径匹配校验"是两个不同的应用点,不要混淆。

---

## Q2. `FilesystemPermission` 是咋控制权限的?`allow`/`deny`/`interrupt` 三种 mode 具体怎么落地?

**核心判定函数**:`_check_fs_permission`(`filesystem.py:127-137`):

```python
def _check_fs_permission(rules, operation, path):
    for rule in rules:                    # 按声明顺序遍历
        if operation not in rule.operations:
            continue
        if any(wcglob.globmatch(path, pattern, flags=_FS_WCMATCH_FLAGS) for pattern in rule.paths):
            return rule.mode              # 命中第一条就返回, 不再往下看
    return "allow"                        # 全部不命中, 默认放行
```

规则:先按 `operation` 过滤,再用 glob 匹配 `path`,**第一条命中的规则说了算**(first-match-wins),后面的规则不再看;都不命中默认 `"allow"`。这解释了 `ch_05_permissions.py` 里两条 deny 规则为什么要按"更严格的放前面"的顺序写——命中即返回,不会叠加多条规则的效果。

**`deny` 怎么落地**——工具函数体内手写的 `if`,直接短路,`FilesystemMiddleware` 给每个文件工具生成的函数体里,在真正碰 backend **之前**都会先查一次,例如 `write_file`(`filesystem.py:1235`):

```python
if _check_fs_permission(self._permissions, "write", validated_path) == "deny":
    return ToolMessage(content=f"Error: permission denied for write on {validated_path}", ..., status="error")
res = resolved_backend.write(validated_path, content)   # deny 时这行根本不会跑到
```

对 `ls`/`glob`/`grep` 这种"一次返回一批结果"的工具还有第二道过滤——`_filter_paths_by_permission`/`_filter_file_infos_by_permission`(`filesystem.py:140-201`),把结果列表里命中 deny 规则的条目直接摘掉,**agent 连文件名都看不到**,不是"看得到名字但读不到内容"。

**`interrupt` 怎么落地**——完全不是同一段代码,是 `HumanInTheLoopMiddleware` 在更外层拦截。`FilesystemMiddleware` 自己完全不知道 HITL 是什么,真正的接线在 `create_deep_agent` 组装图时(`graph.py:764`):

```python
HumanInTheLoopMiddleware(interrupt_on=_build_interrupt_on_from_permissions(permissions or []))
```

`_build_interrupt_on_from_permissions`(`middleware/_fs_interrupt.py:155-182`)把每条 `mode="interrupt"` 的规则转成"工具名 → `when(req)` 谓词"的映射。`HumanInTheLoopMiddleware` 是 `wrap_tool_call` 类型的 middleware(参见 [ch_01_quickstart.qa.md](ch_01_quickstart.qa.md) Q2 装饰器机制)——**在真正调用工具函数之前**先跑 `when(req)`,为真就 `interrupt()` 暂停等人工 `approve`/`edit`/`reject`/`respond`。approve 之后工具函数才第一次真正被调用,这时函数内部那行 `_check_fs_permission(...) == "deny"` 再算一次,结果是 `"interrupt"`(≠ `"deny"`),于是放行、真正写盘。

`when` 谓词按工具的"路径语义"分两种(`_fs_interrupt.py:31-45`):
- **exact**(`read_file`/`write_file`/`edit_file`):只看这一个路径本身命不命中 interrupt 规则。
- **bulk**(`ls`/`glob`/`grep`):路径参数是"搜索根",可能牵出任意子路径,所以判断"搜索子树有没有可能碰到规则的路径锚点";调用没给 `path` 参数时(如 `grep(path=None)`)没法定位范围,保守地无条件触发——防止 agent 靠不传路径绕开审批。

**一句话总结**:

| mode | 拦截层 | 时机 | agent 能看到什么 |
|---|---|---|---|
| `allow`(默认) | 无 | — | 正常结果 |
| `deny` | 工具函数体内的 `if` | 碰 backend 之前 | 一条 "permission denied" 错误消息,批量工具则是被摘掉条目后的结果 |
| `interrupt` | `HumanInTheLoopMiddleware`(`wrap_tool_call`) | 工具函数被调用之前 | 执行暂停,等人工 `Command(resume=...)`;approve 后才真正跑 |

两条路径共用同一份 `permissions` 列表和同一个 `_check_fs_permission`,但"要不要真的碰磁盘"和"要不要先停下来问人"是两次独立判断,分别发生在工具函数内部和 middleware 层。

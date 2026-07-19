"""DeepAgents custom_backend_middleware —— 两个进阶定制合一: 自定义 Backend + 自定义 Middleware。

对比 deepagents/backend.py: 那里用的是内置的 FilesystemBackend (框架已经帮你实现好了
读写真实磁盘的所有方法, 你只是把它传进 create_deep_agent)。本文件反过来: 我们【自己】
实现一个后端 —— 直接继承 deepagents.backends.protocol.BackendProtocol, 用一个内存
dict 存文件 (不落磁盘、不碰网络), 把 read/write/ls/edit/glob/grep 这几个核心方法写出来;
同时【自己】写一个最小中间件 (继承 langchain.agents.middleware.AgentMiddleware), 重写
一个钩子做日志记录。最后把这两样都接到 create_deep_agent 上。

BackendProtocol 核心方法 (本文件全部实现):
  - ls(path)                          列目录 -> LsResult(entries=[FileInfo,...])
  - read(file_path, offset, limit)    读文件 -> ReadResult(file_data=FileData)
  - write(file_path, content)         写新文件(已存在报错) -> WriteResult(path=...)
  - edit(file_path, old, new, all)    精确替换 -> EditResult(occurrences=...)
  - glob(pattern, path)               通配匹配 -> GlobResult(matches=[FileInfo,...])
  - grep(pattern, path, glob)         文本搜索 -> GrepResult(matches=[GrepMatch,...])
  未用到的 upload_files/download_files 可不实现 (基类默认 raise NotImplementedError)。

AgentMiddleware 可重写的钩子 (按执行位置):
  before_agent -> before_model -> [模型调用, 可被 wrap_model_call 包裹] -> after_model
  -> [工具调用, 可被 wrap_tool_call 包裹] -> after_agent
  本文件只重写 before_model / after_model 做轻量日志, 其余走基类默认(透传, 返回 None)。

踩坑记录:
  1. 哪些 Backend 方法"必须"实现? 取决于 agent 会用到哪些文件工具。基类里所有方法
     默认 raise NotImplementedError —— 只要 agent 触发了你没实现的方法就会炸。稳妥
     做法: 把 read/write/ls/edit/glob/grep 这套 CRUD+检索核心全实现, 不常用的上传/
     下载留空。另外 0.7.0 前还有 ls_info/glob_info/grep_raw 的旧名, 已废弃, 别用旧名。
  2. 方法返回的是【结构化结果对象】(LsResult/ReadResult/... 各自带 error 字段), 不是
     裸字符串: 成功填数据字段, 失败填 error。write 遇到"文件已存在"必须走 error 而不是
     覆盖 (这是协议约定的语义), 否则和内置后端行为不一致。
  3. 中间件钩子顺序 + 返回值语义: before_model 返回 None 表示"不改 state 直接放行",
     返回 dict 才会 merge 进 state; wrap_* 系列必须记得调用并 return handler(request),
     忘了调 handler 会直接把模型/工具调用"吞掉"。多个中间件按加入顺序层层包裹。

官方文档: https://docs.langchain.com/oss/python/deepagents/backends
          https://docs.langchain.com/oss/python/deepagents/middleware
"""

import os
import fnmatch
from datetime import datetime, timezone

from dotenv import load_dotenv
from deepagents import create_deep_agent
from deepagents.backends.protocol import (
    BackendProtocol,
    LsResult,
    ReadResult,
    WriteResult,
    EditResult,
    GlobResult,
    GrepResult,
    FileData,
    FileInfo,
    GrepMatch,
)
from langchain.agents.middleware import AgentMiddleware
from langchain_anthropic import ChatAnthropic

load_dotenv(override=True)


# ============ 1. 自定义 Backend: 纯内存 dict 存储 ============
class InMemoryBackend(BackendProtocol):
    """一个最小的内存后端: 文件存在 self._files 这个 dict 里, 不落磁盘。

    对照 backend.py 的 FilesystemBackend(真实磁盘) —— 这里连磁盘都不碰,
    适合测试/沙箱; 进程退出即消失。
    """

    def __init__(self) -> None:
        # key = 绝对路径 (以 / 开头), value = FileData
        self._files: dict[str, FileData] = {}

    def _now(self) -> str:
        # 协议约定用 ISO8601 时间戳
        return datetime.now(timezone.utc).isoformat()

    def ls(self, path: str) -> LsResult:
        # 简化处理: 列出所有已存文件 (真实后端会按 path 前缀过滤)
        entries = [FileInfo(path=p) for p in sorted(self._files)]
        return LsResult(entries=entries)

    def _is_under_path(self, file_path: str, path: str | None) -> bool:
        """判断 file_path 是否位于 path 范围内, 用于 grep/glob 的路径收窄。"""
        if path in (None, "", "/"):
            return True
        normalized = path.rstrip("/")
        return file_path == normalized or file_path.startswith(f"{normalized}/")

    def read(self, file_path: str, offset: int = 0, limit: int = 2000) -> ReadResult:
        if file_path not in self._files:
            # 失败: 填 error 字段, 而不是抛异常/返回裸字符串
            return ReadResult(error=f"Error: file not found: {file_path}")
        fd = self._files[file_path]
        lines = fd["content"].splitlines()
        sliced = lines[offset:offset + limit if limit is not None else None]
        file_data = FileData(
            content="\n".join(sliced),
            encoding=fd["encoding"],
            created_at=fd["created_at"],
            modified_at=fd["modified_at"],
        )
        return ReadResult(file_data=file_data)

    def write(self, file_path: str, content: str) -> WriteResult:
        # 协议语义: 写"新"文件, 已存在则报错 (不静默覆盖)
        if file_path in self._files:
            return WriteResult(error=f"Error: file already exists: {file_path}")
        now = self._now()
        self._files[file_path] = FileData(
            content=content, encoding="utf-8", created_at=now, modified_at=now
        )
        return WriteResult(path=file_path)

    def edit(self, file_path: str, old_string: str, new_string: str, replace_all: bool = False) -> EditResult:
        if file_path not in self._files:
            return EditResult(error=f"Error: file not found: {file_path}")
        fd = self._files[file_path]
        content = fd["content"]
        count = content.count(old_string)
        if count == 0:
            return EditResult(error="Error: old_string not found")
        if not replace_all and count > 1:
            # 默认模式下要求 old_string 唯一, 否则报错 (与内置后端一致)
            return EditResult(error="Error: old_string is not unique; set replace_all=True")
        new_content = content.replace(old_string, new_string) if replace_all else content.replace(old_string, new_string, 1)
        fd["content"] = new_content
        fd["modified_at"] = self._now()
        return EditResult(path=file_path, occurrences=(count if replace_all else 1))

    def glob(self, pattern: str, path: str | None = None) -> GlobResult:
        # 用 fnmatch 对已存路径做通配匹配
        matches = [
            FileInfo(path=p)
            for p in sorted(self._files)
            if self._is_under_path(p, path) and (fnmatch.fnmatch(p, pattern) or fnmatch.fnmatch(os.path.basename(p), pattern))
        ]
        return GlobResult(matches=matches)

    def grep(self, pattern: str, path: str | None = None, glob: str | None = None) -> GrepResult:
        # 逐文件逐行做"字面子串"匹配 (协议规定 grep 是字面匹配, 不是正则)
        matches: list[GrepMatch] = []
        for p, fd in self._files.items():
            if not self._is_under_path(p, path):
                continue
            if glob is not None and not fnmatch.fnmatch(p, glob):
                continue
            for i, line in enumerate(fd["content"].splitlines(), start=1):
                if pattern in line:
                    matches.append(GrepMatch(path=p, line=i, text=line))
        return GrepResult(matches=matches)

    # upload_files / download_files 未实现 —— 用不到, 保留基类的 NotImplementedError。


# ============ 2. 自定义 Middleware: 一个最小日志中间件 ============
class LoggingMiddleware(AgentMiddleware):
    """在每次调用模型前后打一行日志。演示钩子的返回值语义。"""

    def before_model(self, state, runtime):
        # 读一下当前消息数量做日志; 返回 None 表示"不修改 state, 直接放行"
        msg_count = len(state.get("messages", [])) if isinstance(state, dict) else "?"
        print(f"[LoggingMiddleware] before_model: 当前消息数={msg_count}")
        return None

    def after_model(self, state, runtime):
        print("[LoggingMiddleware] after_model: 模型已产出一条回复")
        return None


def build_agent(model):
    """把自定义 backend 和自定义 middleware 一起接到 deep agent 上。"""
    return create_deep_agent(
        model=model,
        tools=[],
        system_prompt="You are an assistant. Save notes with write_file when asked.",
        backend=InMemoryBackend(),   # 换成我们自己的内存后端
        middleware=[LoggingMiddleware()],  # 挂上自定义日志中间件
    )


if __name__ == "__main__":
    # === 无需模型的真实结构性测试: 直接调后端的 CRUD 方法并断言 ===
    backend = InMemoryBackend()

    # write 成功 / 重复写报错
    assert backend.write("/notes.md", "hello\nTODO: fix bug\nbye").path == "/notes.md"
    assert backend.write("/project-a/a.txt", "TODO: scoped").path == "/project-a/a.txt"
    assert backend.write("/project-b/b.txt", "TODO: hidden").path == "/project-b/b.txt"
    assert backend.write("/notes.md", "x").error is not None, "重复写应报错"

    # read 命中 / 未命中
    r = backend.read("/notes.md")
    assert r.error is None and r.file_data["content"].startswith("hello")
    assert backend.read("/missing.md").error is not None

    # ls 列出文件
    assert backend.ls("/").entries[0]["path"] == "/notes.md"

    # edit 精确替换
    e = backend.edit("/notes.md", "hello", "HELLO")
    assert e.error is None and e.occurrences == 1
    assert backend.read("/notes.md").file_data["content"].startswith("HELLO")
    assert backend.read("/notes.md", offset=1, limit=1).file_data["content"] == "TODO: fix bug"

    # glob 通配匹配
    assert len(backend.glob("*.md").matches) == 1
    assert backend.glob("*.py").matches == []
    project_a_matches = [m["path"] for m in backend.glob("*.txt", path="/project-a").matches]
    assert project_a_matches == ["/project-a/a.txt"], project_a_matches

    # grep 字面搜索 (命中第 2 行的 TODO)
    g = backend.grep("TODO")
    assert g.matches[0]["line"] == 2 and "TODO" in g.matches[0]["text"]
    scoped = backend.grep("TODO", path="/project-a")
    assert [m["path"] for m in scoped.matches] == ["/project-a/a.txt"]

    # 自定义中间件可实例化, agent 能在 model=None 下把二者接进去
    agent_struct = build_agent(model=None)
    assert agent_struct is not None

    print("结构验证通过: InMemoryBackend CRUD/glob/grep 全部断言通过, "
          "LoggingMiddleware + backend 接线 OK (无需模型)")

    # === 需要真实模型的部分 ===
    if os.getenv("MODEL_ID"):
        model = ChatAnthropic(
            model=os.environ["MODEL_ID"],
            base_url=os.getenv("ANTHROPIC_BASE_URL") or None,
        )
        agent = build_agent(model=model)
        result = agent.invoke(
            {"messages": [{"role": "user", "content": "把'Python 很好用'存到 note.md"}]}
        )
        print(result["messages"][-1].text)
    else:
        print("未配置 MODEL_ID: 跳过真实模型调用 (后端/中间件已完成无模型结构验证)。")

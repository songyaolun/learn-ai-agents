import os
import tempfile
from pathlib import Path

from deepagents.backends import FilesystemBackend
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_tavily import TavilySearch

from deepagents import create_deep_agent

load_dotenv(override=True)


model = ChatAnthropic(
    model=os.environ["MODEL_ID"],
    base_url=os.getenv("ANTHROPIC_BASE_URL") or None,
)


scratch_dir = tempfile.mkdtemp(prefix="deepagents_backend_demo_")
print(f"真实磁盘 root_dir: {scratch_dir}")

search = TavilySearch(max_results=5)

backend = FilesystemBackend(root_dir=str(scratch_dir), virtual_mode=True)

agent = create_deep_agent(
    model=model,
    tools=[search],
    backend=backend,
    system_prompt=(
        "你是一名研究助手。请使用搜索工具查找当前、准确的信息。如果需要调研一个主题，"
        "必须用write_file工具把中间调研结果保存到文件名为notes.md的文件中（不要用其他文件名），"
        "包括搜索结果、引用、对比分析等，然后综合整理并附上来源，给出清晰的答案。"
    ),
)


if __name__ == "__main__":
    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "简单整理一下Python装饰器的作用，以及它的使用场景，存到文件里。"
                    ),
                }
            ],
        }
    )
    print("=== agent 的最终回复 ===")
    print(result["messages"][-1].text)

    notes_path = Path(scratch_dir) / "notes.md"
    if notes_path.exists():
        print("磁盘上的原始内容：")
        print(notes_path.read_text(encoding="utf-8"))

    files_snapshot = result.get("files", {})
    print(f"\n=== 对照： result['files'] 的内容 在换了backend之后是 {files_snapshot!r}")

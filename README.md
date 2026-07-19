# learn_ai_agents

个人学习仓库：按框架抽象层级递进地实现/对比几种 AI Agent 构建方式——从裸写 Anthropic SDK 的 agent loop，到 LangChain 的 `create_agent`，到 LangGraph 的图 runtime，到 DeepAgents 的 planning/subagent harness，最后用 Chainlit 包一层 Web UI。

## 目录结构

| 目录 | 内容 |
|------|------|
| `claude-code/` | 裸 SDK 手写 agent loop（ch01→ch03 三级递进） |
| `langchain/` | LangChain 1.x 教学示例（详见 `langchain/README.md`） |
| `langgraph/` | LangGraph 图 runtime 教学示例（详见 `langgraph/README.md`） |
| `deepagents/` | DeepAgents 教学示例 + 专题文档（详见 `deepagents/README.md`） |
| `rag/` | embedding + 向量库 + retriever，包成 agent 工具 |
| `web/` | Chainlit Web UI（4 个 chat profile） |

`openclaw/`、`pi/` 为占位目录，当前无代码。

## 命名约束：`ch_NN_` 学习顺序前缀

- `langchain/`、`langgraph/`、`deepagents/` 下的教学示例文件名**一律带 `ch_NN_` 前缀**（NN 为两位序号），序号即所在目录的推荐阅读顺序，各目录 README 有完整索引。
- **新增示例**：按学习顺序取下一个序号命名，并同步更新所在目录 README 的索引表（必要时更新本 README）。
- **基础设施文件不编号**：`README.md`、`requirements.txt`、`ENVIRONMENT.md`、`langgraph.json.template` 等保持原名。

## 快速开始

```bash
uv sync
cp .env.example .env   # 填入 MODEL_ID / ANTHROPIC_BASE_URL / ANTHROPIC_API_KEY
uv run python langchain/ch_01_models.py
```

测试（离线、假模型，需环境中装有 pytest，见 `langgraph/requirements.txt`）：

```bash
pytest -q langgraph/ch_20_test_examples.py
```

更多运行方式与踩坑记录见 [CLAUDE.md](CLAUDE.md)。

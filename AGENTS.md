# AGENTS.md —— 仓库协作约定

## 沟通

- 无论用户用什么语言提问，一律使用中文回复（包括 todo / 计划列表）。

## 代码原则

- 遵守 KISS 原则与单一职责原则：一个示例文件只讲透一个概念。
- 本仓库是教学仓库：示例单文件自包含，**故意保留重复，不抽取公共模块**。
- 注释用中文；框架行为与直觉不一致的地方写「踩坑记录」；每个示例带 `if __name__ == "__main__":` 无网络自测（EXIT=0）。

## 命名约束（学习顺序）

- `langchain/`、`langgraph/`、`deepagents/` 下的教学示例文件名**必须带 `ch_NN_` 前缀**，NN 为两位序号，即所在目录的推荐学习顺序。
- 新增示例按顺序取号，并同步更新所在目录 README 的索引表（必要时更新根 README.md）。
- 基础设施文件不编号：`README.md`、`requirements.txt`、`ENVIRONMENT.md`、`langgraph.json.template`。

## 安全红线

- 不启动任何监听端口的进程（无任何形式的 server）。
- 不硬编码模型 / 密钥，统一走根目录 `.env`（`MODEL_ID` / `ANTHROPIC_BASE_URL` / `ANTHROPIC_API_KEY`）。
- 有副作用的示例（写文件、跑 shell）一律在 `tempfile.mkdtemp()` 临时沙箱里执行，结束清理，不污染工作区。

## 验证

- 运行示例：`uv run python <目录>/<示例>.py`，未配置 `.env` 时也应正常退出（EXIT=0）。
- 单元测试（离线、假模型）：`pytest -q langgraph/ch_20_test_examples.py`。

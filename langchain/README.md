# LangChain 1.x 学习路径(langchain/ 子目录)

本目录是一套 **单文件自包含** 的 LangChain 1.x 教学示例集。每个 `.py` 文件对应一个概念,
可独立阅读、独立运行,不依赖其它文件。目标是:**照着文件顺序读一遍,就能上手 LangChain 1.x
的智能体(agent)开发。**

> agent = 智能体,指能自己决定"何时调用工具、如何组织回答"的 LLM 应用。
> LLM = Large Language Model,大语言模型。

---

## 一、如何运行

所有示例遵循同一套约定:

1. **模型接入统一走 `.env`**,不在代码里硬编码密钥。仓库根目录已有 `.env.example`:
   ```
   ANTHROPIC_BASE_URL=https://api.example.com/anthropic
   ANTHROPIC_API_KEY=your-api-key-here
   MODEL_ID=your-model-id-here
   ```
   复制成 `.env` 并填入真实值即可(个别文件还会用到 `AUDIT_MODEL_ID`)。

2. **每个文件都能"无网络"跑通自测**。本仓库统一用 [uv](https://docs.astral.sh/uv/) 管理依赖与执行:
   ```bash
   uv run ch_01_models.py
   # 若未用 uv, 也可退回 pip: pip install -r requirements.txt && python3 ch_01_models.py
   ```
   - 未配置 `.env` 时:只跑文件顶部的 **无网络自测**(断言 + 对比表),打印 `✓` 后正常退出(EXIT=0),
     并提示 "跳过: 未检测到 MODEL_ID / ANTHROPIC_API_KEY"。
   - 配好 `.env` 后:自测通过后继续跑 **有网络部分**,真正调用模型看到效果。

   > ⚠️ 关于"无网络"的诚实说明:这些文件用 `load_dotenv(override=True)` 加载 `.env`,而
   > `load_dotenv` 会**向上逐级目录查找** `.env`。所以即便当前目录没有 `.env`,只要**上层目录**(如仓库根)
   > 存在 `.env`,它也会被静默加载,从而把"无网络自测"变成真实的联网调用。想验证**纯离线** EXIT=0,
   > 请在**没有上层 `.env`** 的目录里运行(例如把文件复制到一个干净的临时目录)。

3. **安全红线(本目录所有示例严格遵守)**:
   - 不启动任何监听端口的进程;
   - 不直连内部推理网关,模型接入一律走 `.env` 里的 `ChatAnthropic + MODEL_ID / ANTHROPIC_BASE_URL`;
   - 有副作用的示例(写文件、跑 shell)一律在 `tempfile.mkdtemp()` 建的临时沙盒里做,用完 `shutil.rmtree` 清理。

---

## 二、推荐学习顺序

按下面 6 组顺序读,由浅入深。带 ⭐ 的是新增的进阶主题。**文件名前缀 `ch_NN_` 已按本顺序编号**,目录列表即学习路径。

### A 组 · 基础三件套(先读这个)

| 文件 | 概念 | 一句话说明 |
|---|---|---|
| `ch_01_models.py` | 模型 | 如何初始化 `ChatAnthropic`、传参、拿到回复 |
| `ch_02_messages.py` | 消息 | System / Human / AI / Tool 四类消息的构造与读取 |
| `ch_03_tools.py` | 工具 | `@tool` 定义工具、查看工具的 JSON Schema、注入运行时上下文 |

### B 组 · 记忆、运行时与外部能力

| 文件 | 概念 | 一句话说明 |
|---|---|---|
| `ch_04_long_term_memory.py` | 长期记忆 | 用 `InMemoryStore` 跨会话存取用户偏好等长期信息 |
| `ch_05_runtime.py` | 运行时上下文 | 通过 `ToolRuntime` 把用户身份、配置等注入工具 |
| `ch_06_mcp.py` ⭐ | MCP 接入 | MCP(Model Context Protocol,模型上下文协议)让 agent 调用外部服务里的工具 |
| `ch_07_retrieval_rag.py` ⭐ | RAG 检索 | RAG(Retrieval-Augmented Generation,检索增强生成)给 agent 接一个知识库 |

### C 组 · 流式与中间件

| 文件 | 概念 | 一句话说明 |
|---|---|---|
| `ch_08_event_streaming.py` ⭐ | 事件流式 | 用 `astream_events` 订阅节点/工具粒度的类型化事件 |
| `ch_09_middleware_prebuilt_model.py` ⭐ | 预置中间件·模型层 | 降级、工具选择、重试、工具模拟等开箱即用中间件 |
| `ch_10_middleware_prebuilt_tools.py` ⭐ | 预置中间件·工具层 | 上下文裁剪、Shell 工具、文件搜索等中间件 |
| `ch_11_middleware_prebuilt_agent.py` ⭐ | 预置中间件·编排层 | 待办清单(TodoList)+ 自定义编排/评分中间件骨架 |
| `ch_12_middleware_custom_hooks.py` ⭐ | 自定义钩子 | `before_model` / `after_model` / `wrap_model_call` / `dynamic_prompt` |
| `ch_13_middleware_model_guardrails.py` ⭐ | 模型型护栏 | 用一个审核模型做输入/输出内容安全护栏 |

### D 组 · 结构化输出与高级配置

| 文件 | 概念 | 一句话说明 |
|---|---|---|
| `ch_14_structured_output.py` ⭐ | 结构化输出 | 让最终答案是定死结构的对象;ToolStrategy vs ProviderStrategy;五种结构定义形式 |
| `ch_15_agents_advanced.py` ⭐ | Agent 高级配置 | checkpointer、递归上限、多工具等进阶参数 |

### E 组 · 上下文工程

| 文件 | 概念 | 一句话说明 |
|---|---|---|
| `ch_16_context_engineering.py` ⭐ | 上下文工程 | 系统性地组织/裁剪/注入上下文,控制送进模型的信息 |

### F 组 · 参考对照(仓库原有模板)

`ch_17_quickstart.py` / `ch_18_stream.py` / `ch_19_combo.py` / `ch_20_lcel.py` / `ch_21_trim_messages.py` /
`ch_22_middleware_guardrails.py` / `ch_23_middleware_hitl.py` / `ch_24_middleware_summarization.py` /
`ch_25_middleware_pii.py` 是仓库原有的模板文件,本教学集里的 `⭐` 文件常拿它们做对比参照。
它们沿用原仓库约定(直接读 `os.environ["MODEL_ID"]`),需配好 `.env` 才能运行。

---

## 三、每个文件的"三要素"

本教学集里带 ⭐ 的新增文件,都刻意保证以下三要素齐备,方便自学:

1. **文件头中文说明**:讲清楚"这个文件教什么、和哪个已有文件做对比、官方文档链接"。
2. **逐行中文注释 + 踩坑记录**:关键行有中文注释;凡是"框架/参数行为和直觉不一致"的地方,
   都写了 `踩坑记录`,避免照旧教程踩坑。
3. **`if __name__ == "__main__":` 无网络自测**:用 `assert` 验证核心对象能正确构造/调用,
   不配 `.env` 也能跑通(EXIT=0),有副作用的部分放到 `.env` 门控之后。

### 验收清单(概念 × 三要素齐备)

| 文件 | 文件头说明+对比+文档链接 | 逐行注释+踩坑记录 | 无网络自测(EXIT=0) |
|---|:---:|:---:|:---:|
| `ch_01_models.py` | ✓ | ✓ | ✓ |
| `ch_02_messages.py` | ✓ | ✓ | ✓ |
| `ch_03_tools.py` | ✓ | ✓ | ✓ |
| `ch_04_long_term_memory.py` | ✓ | ✓ | ✓ |
| `ch_05_runtime.py` | ✓ | ✓ | ✓ |
| `ch_06_mcp.py` | ✓ | ✓ | ✓ |
| `ch_07_retrieval_rag.py` | ✓ | ✓ | ✓ |
| `ch_08_event_streaming.py` | ✓ | ✓ | ✓ |
| `ch_09_middleware_prebuilt_model.py` | ✓ | ✓ | ✓ |
| `ch_10_middleware_prebuilt_tools.py` | ✓ | ✓ | ✓ |
| `ch_11_middleware_prebuilt_agent.py` | ✓ | ✓ | ✓ |
| `ch_12_middleware_custom_hooks.py` | ✓ | ✓ | ✓ |
| `ch_13_middleware_model_guardrails.py` | ✓ | ✓ | ✓ |
| `ch_14_structured_output.py` | ✓ | ✓ | ✓ |
| `ch_15_agents_advanced.py` | ✓ | ✓ | ✓ |
| `ch_16_context_engineering.py` | ✓ | ✓ | ✓ |

---

## 四、统一约定速查

- **模型初始化**:一律 `ChatAnthropic(model=MODEL_ID, base_url=os.getenv("ANTHROPIC_BASE_URL") or None)`。
- **避免导入即崩**:顶层用 `os.environ.get("MODEL_ID", "<占位值>")`,把真正触发模型调用的对象放进
  `build_agent()` 之类的惰性构造函数里,再用 `if os.getenv(...)` 门控。
- **验证退出码**:用 `python3 f.py >/tmp/o.txt 2>&1; echo $?` 拿真实退出码,
  **不要** `python3 f.py | grep ...`(管道会让 `$?` 变成 grep 的退出码,给出假的"全绿")。
- **常见缩写**:LLM=大语言模型;MCP=模型上下文协议;RAG=检索增强生成。

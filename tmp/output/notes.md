# Python vs TypeScript：三个主要区别

## 1. 类型系统（Type System）

**Python**：采用动态类型（Dynamic Typing），变量类型在运行时确定，无需显式声明类型。支持类型提示（Type Hints，从 Python 3.5 开始），但这只是提示，不会强制检查。

```python
x = 42        # x 是 int
x = "hello"   # 可以重新赋值为 str，运行时不报错
```

**TypeScript**：在 JavaScript 的基础上添加了静态类型（Static Typing），编译时进行类型检查，必须显式声明或可推导变量类型。

```typescript
let x: number = 42;
x = "hello";  // 编译时报错：Type 'string' is not assignable to type 'number'
```

## 2. 运行环境与编译方式（Runtime & Compilation）

**Python**：解释型语言，由 Python 解释器逐行执行，无需编译步骤。`.py` 文件可直接运行。

**TypeScript**：需要先通过 `tsc` 编译器将 `.ts` 文件转译（transpile）为 JavaScript（`.js`），再由浏览器或 Node.js 执行。多了一道构建步骤。

```
TypeScript (.ts) → tsc 编译 → JavaScript (.js) → 浏览器/Node.js 执行
```

## 3. 主要应用场景（Use Cases）

**Python**：
- 数据科学与机器学习（NumPy、Pandas、TensorFlow、PyTorch）
- 后端 Web 开发（Django、Flask、FastAPI）
- 自动化脚本、运维工具
- 科学计算、AI

**TypeScript**：
- 前端开发（React、Vue、Angular 首选语言）
- Node.js 后端开发（NestJS、Express 等）
- 大型 JavaScript 项目的代码维护
- 跨端应用（React Native、Electron）

## 总结对比表

| 维度 | Python | TypeScript |
|------|--------|------------|
| 类型系统 | 动态类型（可选类型提示） | 静态类型（强制类型检查） |
| 运行方式 | 解释执行，运行时 | 编译后执行，编译时 |
| 主要场景 | 数据科学、AI、后端、脚本 | 前端、Node.js、大型 JS 项目 |

---

**参考来源**：
- Python 官方文档：https://docs.python.org/3/
- TypeScript 官方文档：https://www.typescriptlang.org/docs/

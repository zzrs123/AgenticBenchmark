# FACTOR Demo: 3-Agent Controlled Factorial Meta-Benchmark for Spatial Omics

一套面向导师汇报与原型验证的**多智能体空间组学双侧受控因子元基准框架（Demo 原型）**。支持：
- **双模式智能体引擎**：既支持接入任意 OpenAI 兼容大模型 API（DeepSeek / GPT / Qwen / Gemini），也内置**零外部依赖的确定性工具溯源模式（Offline Grounded Mode）**，汇报演示零翻车；
- **三大协同智能体 + 确定性统计内核**：
  1. `DatasetCharacterAgent`：计算空间统计特征（严格分离测试时可用的无标签特征 `d_obs` 与诊断特征 `d_ref`）+ 扰动切片偏移审计；
  2. `MethodCharacterAgent`：基于 Python `ast` 静态解析方法源码，提取带精确行号证据（`CodeSpan`）的归纳偏置向量 + `orig == refac` 代码等价门控；
  3. `DMInteractionAgent`：探索集混合效应交互筛选（`D×M` 与 `D×C` 竞争建模）$\rightarrow$ 预注册锁定算子与等效界 $\rightarrow$ 独立确认集 $2\times 3$ 四臂因子实验（`orig` / `refac` / `self` / `null`）与 `PCA vs. HVG` 直接交叉实验；
- **一键双格式报告与交互演示**：自动生成高清四联图表、`experiment_report.md` 以及零 LaTeX 依赖的多页矢量 `experiment_report.pdf`，并提供逐步交互演示的 `notebooks/FACTOR_3Agent_Demo.ipynb`。

---

## 目录结构

```text
AgenticBenchmark/
├── config/demo_config.json            # LLM 接口与实验参数配置
├── docs/PROPOSAL_DEMO_BRIEF.md        # 给老师汇报的 2 页精炼方案简报
├── factor_demo/
│   ├── llm_client.py                  # 零第三方依赖双模式 LLM 客户端 (Live REST + Offline Grounded)
│   ├── schemas.py                     # 核心类型定义与证据溯源数据结构
│   ├── pipeline.py                    # 端到端全流程编排器
│   ├── data_lab/                      # 真实 .h5ad 加载器、12 切片基准集生成器与 u1/u2 受控干预算子
│   ├── methods_zoo/                   # 4 个带四臂对照接口的代表性空间域识别算法模块
│   ├── tools/                         # 确定性空间指标、AST 源码分析器、评价指标与混合效应/因子统计引擎
│   ├── agents/                        # 三组核心智能体实现
│   └── reporting/                     # 四联科学图表绘制与 Markdown + 多页 PDF 报告生成器
├── notebooks/FACTOR_3Agent_Demo.ipynb # 交互式汇报演示 Jupyter Notebook
├── scripts/run_demo.py                # 命令行一键运行脚本
└── tests/test_core_pure_python.py     # 纯标准库语法与 AST 静态分析器自检单测
```

---

## 快速运行指南（服务器或本地配好生信环境后）

### 1. 环境依赖
在常规空间组学 / 单细胞 Conda 环境（已含 `numpy`, `scipy`, `pandas`, `scikit-learn`, `matplotlib`, `jupyter`）中**无需安装任何额外 Agent 框架**即可直接运行；若在新环境中运行：

```bash
pip install -r requirements.txt
```

### 2. 一键命令行运行（生成图表 + Markdown + 多页 PDF 报告）

- **模式 A：离线工具溯源模式（无需配置 API Key，约 20–40 秒跑完）**
  ```bash
  python scripts/run_demo.py --quick
  ```

- **模式 B：接入实时大模型 API（以 DeepSeek / OpenAI 兼容接口为例）**
  ```bash
  export OPENAI_API_KEY="sk-your-key"
  export OPENAI_BASE_URL="https://api.deepseek.com/v1"
  export LLM_MODEL="deepseek-chat"
  python scripts/run_demo.py --mode live
  ```

- **模式 C：直接对真实空间转录组 `.h5ad` 切片目录运行**
  ```bash
  python scripts/run_demo.py --h5ad-dir /path/to/your_h5ad_slices --output-dir outputs/real_slices_run
  ```

运行完成后，在 `outputs/demo_run/` 下自动产出：
- `experiment_report.md`（完整中文+技术细节实验报告）
- `experiment_report.pdf`（多页矢量 PDF 汇报报告，含执行摘要表与 4 页高清验证图）
- `figures/fig1..fig4.png` 与 `artifacts/full_run_bundle.json`

### 3. Jupyter Notebook 交互式演示（推荐向老师现场展示）
```bash
jupyter notebook notebooks/FACTOR_3Agent_Demo.ipynb
```

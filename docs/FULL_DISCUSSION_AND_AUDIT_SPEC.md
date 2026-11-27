# 空间组学主动因子元基准框架（FACTOR）：全流程讨论纪要、方法学演进、与 Canzar 组（2026.03）对比及代码实现全景（供严谨性与合理性审核）

> **文档用途**：本文档完整汇总当前课题从文献调研、初始设想（v1.0）、首轮方法学审阅修正（v2.0），到基于 *Descoeudres et al. (bioRxiv, 2026.03)* 开源仓库落地的代码原型架构、文件路径映射、实测初步结果及**当前代码已知简化边界**，专供开展独立的**科学严谨性、统计可识别性与工程合理性审核**。

---

## 目录
1. [研究背景与相关工作谱系梳理](#1-研究背景与相关工作谱系梳理)
2. [与 Canzar 组（Descoeudres et al., 2026.03）的精确边界对比](#2-与-canzar-组descoeudres-et-al-202603的精确边界对比)
3. [方案方法学演进：从 v1.0 到采纳 12 条审阅意见后的 v2.0](#3-方案方法学演进从-v10-到采纳-12-条审阅意见后的-v20)
4. [v2.0 方案完整数学形式化定义（Step 0 – Step 6）](#4-v20-方案完整数学形式化定义step-0--step-6)
5. [当前 Demo 代码仓架构与完整文件路径映射](#5-当前-demo-代码仓架构与完整文件路径映射)
6. [三组智能体（尤其 D–M Interaction Agent）与 `D×M` vs `D×C` 实现细节](#6-三组智能体尤其-dm-interaction-agent与-dxm-vs-dxc-实现细节)
7. [基于 Canzar 2026 真实仓库已跑出的初步实测结果](#7-基于-canzar-2026-真实仓库已跑出的初步实测结果)
8. [坦诚披露：当前 Demo 代码的已知简化、局限与待审核问题清单](#8-坦诚披露当前-demo-代码的已知简化局限与待审核问题清单)

---

## 1. 研究背景与相关工作谱系梳理

课题围绕空间转录组学（当前聚焦**多细胞空间域识别 / Spatial Domain Segmentation**）中普遍存在的“跨数据集性能分化”现象展开，前期梳理了三条技术脉络：

### 1.1 空间组学既有 Benchmark 的两条路线
1. **数据层 Benchmark（平台物理指标与模拟器保真度）**：
   - *跨平台同组织配对评测*：在连续切片或 FFPE 组织微阵列（TMA）上横向对比 10x Xenium、CosMx SMI、MERSCOPE、Visium / Visium HD 的灵敏度、假阳性率（FDR）、转录本溢出（spillover）及与 scRNA-seq 的一致性。
   - *空间模拟器保真度评测（如 SRTsim、scDesign3）*：在明确给定的统计生成模型内主动操纵稀疏度、折叠变化（logFC）或空间自相关（Moran's $I$、Geary's $C$），研究算法在模拟机制下的响应；其核心局限不在于“被动”，而在于**生成模型简化假设向复杂真实组织（如肿瘤浸润微环境）的外部有效性（External Validity）迁移受限**。
2. **方法层 Benchmark（下游聚类/空间域识别横向评测）**：
   - *描述性基准*（如 Yuan et al., *Nature Methods* 2024；Hu et al., *Genome Biology* 2024；Kang et al., *NAR* 2025）：跨多平台真实切片汇报 ARI/NMI/PAS、运行时间与内存并给出选型指南。
   - *标注局限性与共识基准*（如 SACCELERATOR, *Nature Methods* 2026, `10.1038/s41592-026-03194-8`）：明确指出传统病理/解剖人工手绘标注（如 DLPFC 层状多边形）存在主观不确定性与过度平滑偏倚，不能无条件视为绝对无偏真值。
   - *解释性基准*（如 Chen et al., *iMeta* 2025 `SRTBenchmark`；Descoeudres et al., *bioRxiv* 2026.03）：开始结合大规模模拟/半合成扰动与预处理步骤拆解，探讨分辨率、基因面板大小、稀疏度、组织结构复杂度及预处理如何影响算法表现上限。

### 1.2 通用机器学习中的“跨数据集差异与元特征”视角
- **Dataset Bias（Torralba & Efros, *CVPR* 2011）**：通过跨数据集互训互测揭示算法高分常源于拟合特定数据集采集/标注偏差。
- **可解释分布差异（Babbar, Guo & Rudin, *arXiv:2403.05652*, *"What is Different Between These Datasets?"*）**：超越单一标量分布距离（如 MMD），用原型样本与特征偏移定位两个数据集的具体差异子空间。
- **元学习特征（Meta-features，如 PyMFE）**：提取数据统计/结构/探针（Landmarking）特征预测算法表现以支持算法选择（Algorithm Selection）。

---

## 2. 与 Canzar 组（Descoeudres et al., 2026.03）的精确边界对比

Canzar 组于 2026 年 3 月发表的预印本（*An explanatory benchmark of spatial domain detection reveals key drivers of method performance*, `doi:10.64898/2026.03.12.710462`）及其开源仓库（`spatial-domain-analysis-framework` 与 `spatial-domain-simulator`）是目前与本课题最接近的前沿工作。为避免对前人工作的稻草人式矮化（Strawmanning），下表严格基于该论文原文与源码，界定**其已完成的工作**与**本方案拟建立的实质增量**：

### 2.1 Canzar 组（2026.03）已经做到的工作（本方案直接复用的地基）
1. **真实 + 半合成联合评测**：评测了 26 个方法（仓库扩充至 31 个方法 + 4 个基线）在真实切片（论文 63 张，仓库扩至 94 张）与 >930 张半合成切片（基于 Langlieb et al. 小鼠脑 snRNA-seq 5 类兴奋性神经元构建）上的表现；
2. **单侧半合成参数扫描（`exp1`–`exp9`）**：独立操纵空间分辨率网格聚合、基因面板下采样、稀疏度增加、Type I 环境噪声、Type II 细胞类型混合置换、中间域厚度与 4 种几何形状；
3. **输入细胞顺序置换测试**：通过置换计数矩阵行顺序（不改数值与坐标）暴露方法内部写死随机种子所掩盖的随机波动；
4. **6 个神经网络方法的四模块交换消融（`chimera_run.py`）**：将 `SEDR, CCST, SCAN-IT, STAGATE, GraphST, SpaceFlow` 拆分为 `preprocess`、`adjacency`、`train`、`clust` 四个模块并在 **12 张 `Visium-Maynard` 脑皮层切片**上交叉重组，发现预处理与聚类选择的影响普遍大于网络架构差异。

### 2.2 本方案（FACTOR）相对 Canzar 组（2026.03）的 6 项核心区别与可检验增量

| 对比维度 | Canzar 组（Descoeudres et al., 2026.03）的实际做法 | 本方案（FACTOR）的差异化设计与待证增量 |
| :--- | :--- | :--- |
| **1. 数据条件与模型组件是否真正交叉（核心实验设计差异）** | **数据扰动与模块消融相互割裂**：<br>① 半合成实验（`exp1`–`exp9`）只变数据参数，跑的是**未拆解的原版黑盒方法**；<br>② 模块重组消融（`chimera_run.py`，见原论文 p.9 L291–294）**仅在固定的 12 张低分辨率 `Visium-Maynard` 切片上运行**，完全未改变数据分辨率、浸润度或域厚度。<br>因此，其“神经网络架构影响很小、预处理决定性能”的结论仅在 50µm 分层皮层切片上成立，未检验在单细胞高浸润或薄层条件下图架构/边门控是否重新成为瓶颈。 | **双侧交叉受控因子实验（$(D\text{-算子}) \times (M\text{-算子})$）**：<br>在同一批锚点切片上**同时操纵数据条件剂量（如边界浸润率、Spot 聚合尺度）与方法内部算子状态**，直接估计“数据条件改变时，特定组件净贡献如何变化”的因子交互对比量 $\tau_{\text{int}}$。 |
| **2. 方法侧对照语义与调参公平性** | `chimera_run.py` 仅在 6 个已有方法的空间邻接模块间互换（如把 `k-NN` 换成 `alpha-complex`，**二者依然都是物理空间邻近图**）；且全程固定作者默认超参（原论文 p.14 L416–427 明确将其列为局限）。 | **方法内四臂受控对照 + 双调参协议**：<br>分离 ① `orig`（原实现）、② `refac`（重构代码等价门控）、③ `self`（单位阵 $\mathbf{I}$，剥离跨单元聚合）、④ `null`（同谱度保持节点置换图 $\mathbf{P}\mathbf{A}_{\text{spat}}\mathbf{P}^\top$，剥离物理邻近对应但保留图谱）；并同步报告**默认超参（`fixed`）**与**等预算无监督重调优（`retuned`）**下的效应，区分“组件本身作用”与“换组件引发的超参失配”。 |
| **3. 实验选择的组织方式与探索/确认隔离** | 人工肉眼观察真实数据集热力图后，手工固定 6–7 个模拟维度全量跑 Snakemake 网格；没有区分“提出假设的数据集”与“确认假设的数据集”，也未设置统计等效界（仅看均值曲线或 Mann-Whitney 显著性）。 | **探索筛选与独立样本确认两阶段闭环**：<br>① 外循环仅在**探索集 $\mathcal{D}_{\text{expl}}$** 上用包含 `D×M` 与 `D×C` 竞争项的混合模型做假设优先级排序，在有限算力预算下主动挑选高信息量因子实验并**预注册锁定规格**；<br>② 内循环在**跨供体/跨组织独立确认集 $\mathcal{D}_{\text{conf}}$** 上复验，并结合预设实际等效区间 $[-\Delta_{\text{eq}}, +\Delta_{\text{eq}}]$（TOST 原则）裁决。 |
| **4. 数据干预基座与干预偏移审计** | 所有半合成切片均从单一小鼠脑 snRNA-seq 的 5 个相近兴奋性神经元亚群从头合成，与具体失败的真实组织（如乳腺癌微环境）存在分布鸿沟；未对扰动前后的非目标统计特征与基因间共表达结构偏移做定量审计。 | **支持真实高分辨切片原位受控退化/置换 + 显式偏移诊断**：<br>除兼容 Canzar 模拟器外，支持直接在真实切片上施加参数化算子（如单细胞向伪 Spot 聚合、固定几何域下的边界细胞置换），并由 `DatasetCharacterAgent` 自动记录目标特征偏移、非目标不变集偏移及基因相关矩阵 Frobenius 漂移（严格定性为“干预有效性与偏移诊断”）。 |
| **5. 方法特征化与异构代码审计机制** | 纯靠人工阅读并手工改写 6 个结构相近的 Python 方法，其余 25 个方法仅按“NN / stat / clust / image”四分类描述。 | **静态代码解析辅助的可溯源特征化（`MethodCharacterAgent`）**：<br>结合 Python AST 与跨语言模式扫描自动提取结构化 $(\boldsymbol{m}_j, \boldsymbol{c}_j)$ 编码，每个字段强制绑定 `文件名:起止行号`（例如在实测中自动检出 Canzar 仓库 35 个脚本里有 18 个内部写死了随机种子）。 |
| **6. 下游方法选择的可落地性验证** | 最终给出定性文字推荐表（如“重精度选 BASS，重速度选 TACCO”）；其论文中的 Consensus 集成（原论文 p.18 L604–607）依赖真实标签 `gt` 做二分图匹配对齐，无法直接用于无标注新样本。 | **严格基于无标签特征 `d_obs` 的样本外推荐遗憾（Regret）评测**：<br>物理隔离测试时可观测特征 `d_obs` 与真值依赖特征 `d_ref`，在完全未参与训练的新器官/新平台真实切片上定量检验“按 `d_obs` 选方法能否降低相对静态排行榜冠军的性能遗憾”。 |

---

## 3. 方案方法学演进：从 v1.0 到采纳 12 条审阅意见后的 v2.0

在首轮构思（v1.0）后，经严谨性审阅指出了 12 个关键方法学问题。当前方案（v2.0）已对认识论边界与统计设计做出如下结构性收缩与修正：

1. **放弃“改变标量元特征即等价于抽象属性 $\operatorname{do}(d_p=\delta)$”的跳跃**：承认同一元特征变化存在多版本处理（Multiple Versions of Treatment）。v2.0 将第一阶段识别目标严格收窄为**“明确定义的参数化操作算子 $\mathcal{O}_{\mathcal{D}}^{(u)}(\cdot;\lambda)$ 与方法算子 $\mathcal{O}_{\mathcal{M}}^{(v)}(\cdot;a)$ 在指定性能尺度上的受控因子交互效应”**；仅当多种不同机制算子在独立样本上给出一致结论时，才讨论向抽象属性层面的归纳。
2. **重新界定数据扰动检查的认识论地位**：删除未定义结构图的 $\operatorname{desc}(p)$ 表述，改为按具体算子 $u$ 显式划分目标特征集 $\mathcal{S}_u^{\text{target}}$、机制必然伴随变化集 $\mathcal{S}_u^{\text{cov}}$（如聚合/降采样本就会改变稀疏度，不强求边际守恒）与监控不变集 $\mathcal{S}_u^{\text{inv}}$，补充基因间相关矩阵漂移检查，并将其准确定名为**“干预有效性与非目标偏移诊断”**而非因果特异性的充分证明。
3. **重做方法侧对照设计（废除“随机图=零效应假手术”的错误表述）**：明确随机图会引入跨区域错误消息混合或改变优化行为，因此拆分为四个职责清晰的状态：`orig`（原始实现）、`refac`（仅重构接口的代码等价对照）、`self`（单位阵 $\mathbf{I}$，衡量跨单元聚合相对单点变换的作用）、`null`（明确定义的同谱度保持置换图 $\mathbf{P}\mathbf{A}_{\text{spat}}\mathbf{P}^\top$，衡量真实物理邻近拓扑相对同谱非空间连接的作用）。
4. **修正参数量守恒解释**：承认 GNN 参数量 $|\Theta|$ 相同不代表有效感受野或优化难度相同，同步记录收敛率、损失下降比与表征有效秩（Effective Rank），仅声称“减少若干替代解释”而非“彻底排除”。
5. **废除有逻辑漏洞的“三类机制无歧义二分/三分裁决”**：同时汇报 $2\times 3$ 各实验臂绝对均值、单剂量简单效应 $\Delta(\lambda)$ 与 $2\times 2$ 因子交互对比 $\tau_{\text{int}}$（防止把“两档剂量下均有 $+0.1$ 恒定主效应从而 $\tau_{\text{int}}=0$”误判为无作用，或掩盖随机图在某状态下变有害的情况）；改用基于置信区间与等效界的证据状态（`Supported Interaction` / `Practically Equivalent` / `Inconclusive` / `Protocol-Sensitive`）；对预处理混杂的怀疑必须通过**直接交换预处理因子**来实验检验。
6. **修正统计术语与尺度说明**：不再借用含时间维度的计量经济学“DiD（双重差分）”名称，准确称为**受控因子实验在指定度量尺度上的因子交互对比（Factorial Interaction Contrast）**。
7. **补全外循环混合模型可识别性与 `D×C` 竞争项**：对切片截距 $u_i$ 与方法截距 $v_j$ 施加显式高斯先验（L2 收缩），同时纳入 `D×M` 与 `D×C` 竞争交互项，并用供体簇聚类自助法（Cluster Bootstrap）计算入选稳定度 $\hat{\Pi}$ 作为探索优先级，不将 Lasso 非零直接等同于统计显著。
8. **前置锁定任务定义与标签演化语义**：在实验前固定任务（多细胞空间域分割 vs 单细胞分型）及每个算子下的参考标签映射规则，并引入不依赖人工画线的保留标志基因富集度（`Marker_Enrichment`）作为正交参考。
9. **补齐多层随机源与探索/确认样本切分**：区分单切片内部随机源（数据扰动种子 $\xi_{\mathcal{D}}$、随机图种子 $\xi_{\mathcal{G}}$、训练种子 $\omega$）与跨生物供体/切片抽样变异；强制隔离探索集 $\mathcal{D}_{\text{expl}}$ 与独立确认集 $\mathcal{D}_{\text{conf}}$，引入预设等效区间 $[-\Delta_{\text{eq}}, +\Delta_{\text{eq}}]$。
10. **区分默认配置效应与重调参效应**：同步设计 `fixed`（冻结原超参）与 `retuned`（等预算无监督准则重调优）两种协议。
11. **客观评价既有模拟与解释性 Benchmark 工作**：承认受控模拟在模拟机制内具备主动操纵能力，将自身贡献聚焦于主动实验筛选、双侧交叉干预、有效性审计与跨独立样本复验。
12. **切分确定性程序与 Agent 边界并设立系统级评测**：指标计算与统计推断全部由确定性 Python 函数完成；Agent 聚焦代码解析、接口挂接与实验调度，并通过“单位预算复验发现产出率”与“无标签样本外推荐遗憾值”单独检验系统收益。

---

## 4. v2.0 方案完整数学形式化定义（Step 0 – Step 6）

### Step 0：任务与算子级标签演化预锁定
固定目标任务后，单张切片记为 $D_i = (\mathbf{X}_i, \mathbf{S}_i, \mathbf{R}_i)$。参数化数据干预算子 $\mathcal{O}_{\mathcal{D}}^{(u)}$（$u$ 指代具体操作机制而非抽象概念，剂量 $\lambda \in \Lambda_u$，随机源 $\xi_{\mathcal{D}}$）同步定义数据与参考标准的演化：

$$\tilde{D}_i^{(u)}(\lambda, \xi_{\mathcal{D}}) = \big(\tilde{\mathbf{X}}_i, \tilde{\mathbf{S}}_i, \tilde{\mathbf{R}}_i\big) = \mathcal{O}_{\mathcal{D}}^{(u)}\big((\mathbf{X}_i, \mathbf{S}_i, \mathbf{R}_i);\ \lambda, \xi_{\mathcal{D}}\big)$$

### Step 1：双侧描述性表征（物理隔离 `d_obs` 与 `d_ref`）
样本预先按独立供体划分为探索集 $\mathcal{D}_{\text{expl}}$ 与确认集 $\mathcal{D}_{\text{conf}}$：

$$\boldsymbol{d}_i = \Phi_{\mathcal{D}}(D_i) = \big[\boldsymbol{d}_i^{\text{obs}};\ \boldsymbol{d}_i^{\text{ref}}\big] \in \mathbb{R}^{P_{\text{obs}} + P_{\text{ref}}}, \qquad (\boldsymbol{m}_j, \boldsymbol{c}_j) = \Phi_{\mathcal{M}}(\mathcal{C}_j, \mathcal{T}_j) \in \mathbb{R}^Q \times \mathbb{R}^{Q_{\text{aux}}}$$

其中 $\boldsymbol{d}_i^{\text{obs}}$ 仅含无标签可观测指标（唯一允许输入样本外推荐器的特征），$\boldsymbol{d}_i^{\text{ref}}$ 仅用于回顾性归因；$(\boldsymbol{m}_j, \boldsymbol{c}_j)$ 的每个非零项绑定源码位置指针 `CodeSpan`。

### Step 2：外循环惩罚混合效应竞争筛选（仅在 $\mathcal{D}_{\text{expl}}$ 上运行）
对去共线性并标准化后的特征 $\tilde{\boldsymbol{d}}_i, \tilde{\boldsymbol{m}}_j, \tilde{\boldsymbol{c}}_j$，设定混合效应筛选模型：

$$Y_{i,j} = \mu + \boldsymbol{\alpha}^\top \tilde{\boldsymbol{d}}_i + \boldsymbol{\beta}^\top \tilde{\boldsymbol{m}}_j + \boldsymbol{\eta}^\top \tilde{\boldsymbol{c}}_j + \tilde{\boldsymbol{d}}_i^\top \mathbf{W}_{dm} \tilde{\boldsymbol{m}}_j + \tilde{\boldsymbol{d}}_i^\top \mathbf{W}_{dc} \tilde{\boldsymbol{c}}_j + u_i + v_j + \varepsilon_{i,j}$$

$$u_i \stackrel{\text{i.i.d.}}{\sim} \mathcal{N}(0, \sigma_u^2), \qquad v_j \stackrel{\text{i.i.d.}}{\sim} \mathcal{N}(0, \sigma_v^2), \qquad \varepsilon_{i,j} \stackrel{\text{i.i.d.}}{\sim} \mathcal{N}(0, \sigma_\varepsilon^2)$$

结合对供体簇的 $B$ 次 Cluster Bootstrap 计算入选稳定度 $\hat{\Pi}_{p,q} = \frac{1}{B}\sum_{b=1}^B \mathbb{I}(\hat{W}_{dm,(p,q)}^{(b)} \neq 0)$，仅将其用于候选实验优先级排序，不直接宣称因果效应。

### Step 3 & Step 4：四臂对照、双调参协议与偏移/等价诊断
方法侧围绕空间邻接组件定义四臂 $a \in \{\text{orig}, \text{refac}, \text{self}, \text{null}\}$ 及两种调参协议 $\pi \in \{\text{fixed}, \text{retuned}(B_{\text{tune}})\}$。执行前必须满足代码重构等价门控 $\|\hat{\mathbf{p}}(\text{refac}) - \hat{\mathbf{p}}(\text{orig})\| \le \varepsilon_{\text{code}}$，并记录数据干预在目标集 $\mathcal{S}_u^{\text{target}}$、预期伴随变化集 $\mathcal{S}_u^{\text{cov}}$ 与监控不变集 $\mathcal{S}_u^{\text{inv}}$ 上的标准化偏移及基因相关矩阵漂移 $\|\mathbf{R}(\tilde{\mathbf{X}}_i) - \mathbf{R}(\mathbf{X}_i)\|_F$。

### Step 5：独立确认集 $\mathcal{D}_{\text{conf}}$ 上的 $2\times 3$ 因子分解与等效界推断
单张确认集切片 $D_i \in \mathcal{D}_{\text{conf}}$ 在状态 $(\lambda, a, \pi)$ 下的期望表现为对内部三层随机源 $(\xi_{\mathcal{D}}, \xi_{\mathcal{G}}, \omega)$ 的均值 $\mu_{i,j}^{(u,v)}(\lambda, a; \pi)$。同步汇报各臂均值、单剂量简单效应与 $2\times 2$ 因子交互对比：

$$\Delta_{i,j}^{\text{agg}}(\lambda; \pi) = \mu_{i,j}^{(u,v)}(\lambda, \text{refac}; \pi) - \mu_{i,j}^{(u,v)}(\lambda, \text{self}; \pi), \qquad \Delta_{i,j}^{\text{topo}}(\lambda; \pi) = \mu_{i,j}^{(u,v)}(\lambda, \text{refac}; \pi) - \mu_{i,j}^{(u,v)}(\lambda, \text{null}; \pi)$$

$$\tau_{i,j}^{\text{topo}}(\pi) = \Delta_{i,j}^{\text{topo}}(\lambda_{\text{high}}; \pi) - \Delta_{i,j}^{\text{topo}}(\lambda_{\text{low}}; \pi)$$

跨独立确认切片估计总体均值 $\hat{\bar{\tau}}_j^{\text{topo}}(\pi)$ 及其 $(1-\alpha)$ 置信区间 $[L_j, U_j]$，对比预设实际等效区间 $[-\Delta_{\text{eq}}, +\Delta_{\text{eq}}]$ 输出证据状态。

### Step 6：无标签样本外推荐遗憾值（Out-of-Sample Regret）
在新切片 $D_{\text{new}} \in \mathcal{D}_{\text{conf}}$ 上仅输入 $\boldsymbol{d}_{\text{new}}^{\text{obs}}$ 选择方法 $\hat{M}(D_{\text{new}})$，评估相对真实最优的遗憾值：

$$\text{Regret}(D_{\text{new}}) = \max_{M \in \mathcal{M}} Y(D_{\text{new}}, M) - Y\big(D_{\text{new}}, \hat{M}(D_{\text{new}})\big)$$

---

## 5. 当前 Demo 代码仓架构与完整文件路径映射

工作区根目录：`F:\gemini-docs\AgenticBenchmark`

### 5.1 核心架构模块路径表

| 层级 | 相对路径（及绝对路径） | 核心职责与关键函数/类 |
| :--- | :--- | :--- |
| **配置与类型契约** | `config/demo_config.json`<br>(`F:\gemini-docs\AgenticBenchmark\config\demo_config.json`) | 定义 LLM 接口模式、探索/确认切片规模、三层随机种子重复次数、等效界 `equivalence_margin_ari: 0.03` 与显著性水平 `alpha: 0.05`。 |
| **数据结构契约** | `factor_demo/schemas.py`<br>(`F:\gemini-docs\AgenticBenchmark\factor_demo\schemas.py`) | 定义强类型数据类：`CodeSpan`、`DatasetProfile`（硬性拆分 `d_obs` 与 `d_ref`）、`MethodProfile`（拆分 `m_core`、`c_aux` 与 `evidence_spans`）、`ScreeningCandidate`、`PreRegisteredHypothesis`、`ValidityAuditRecord`、`FactorialContrastResult`。 |
| **双模式大模型客户端** | `factor_demo/llm_client.py`<br>(`F:\gemini-docs\AgenticBenchmark\factor_demo\llm_client.py`) | `DualModeLLMClient`：使用标准库 `urllib.request` 直连任意 OpenAI 兼容端点（零额外第三方 Agent 库依赖）；无 API Key 时自动回退至基于真实工具输出拼接的 `offline_grounded` 模式，全程记录 `call_history` 审计日志。 |
| **自包含沙盒数据与扰动** | `factor_demo/data_lab/generator_and_loader.py`<br>`factor_demo/data_lab/perturbations.py` | `SpatialSlice` 容器、`load_h5ad_slice`（真实 `.h5ad` 加载器）、`build_demo_cohort`（生成 8 探索 + 4 独立供体确认切片，含保留标志基因 `heldout_markers`）；实现 `apply_spot_aggregation`（`u1`）与 `apply_boundary_infiltration`（`u2`，保持全细胞多变量基因向量与全局基因相关矩阵不变）。 |
| **自包含模块化算法池** | `factor_demo/methods_zoo/base_pipeline.py`<br>`factor_demo/methods_zoo/gcn_dgi_domain.py`<br>`factor_demo/methods_zoo/gat_ae_domain.py`<br>`factor_demo/methods_zoo/banksy_aug_domain.py`<br>`factor_demo/methods_zoo/expr_baseline.py`<br>`factor_demo/methods_zoo/registry.py` | 实现四臂图算子钩子 `apply_graph_arm_hook`（`orig` / `refac` / `self` / `null`）、`PCA vs HVG` 预处理切换、`GMM vs KMeans` 聚类头切换及无监督 Calinski-Harabasz 重调优准则；提供 4 类代表性算法（2-hop DGI 对比图平滑、表达注意力边门控自编码器、BANKSY 邻域拼接、非空间+事后投票基线）。 |
| **确定性计算与统计内核** | `factor_demo/tools/dataset_metrics.py`<br>`factor_demo/tools/ast_code_inspector.py`<br>`factor_demo/tools/eval_metrics.py`<br>`factor_demo/tools/stats_engine.py` | ① `compute_dataset_features` & `audit_perturbation_drift`；<br>② `inspect_method_source`（纯 `ast` 语法树解析绑定行号）；<br>③ `evaluate_predictions`（计算 `ARI`, `NMI`, `Boundary_F1`, `PAS`, 无标注 `Marker_Enrichment`）；<br>④ `fit_outer_loop_screening`（两步 Ridge+Lasso 混合筛选 + 供体聚类 Bootstrap）与 `compute_factorial_inference`（$2\times 3$ 均值、简单效应、$2\times 2$ 交互、$t$ 区间与 TOST 裁决）。 |
| **三组核心智能体** | `factor_demo/agents/dataset_agent.py`<br>`factor_demo/agents/method_agent.py`<br>`factor_demo/agents/interaction_agent.py` | `DatasetCharacterAgent`、`MethodCharacterAgent`、`DMInteractionAgent` 的完整协作闭环实现（详见第 6 节）。 |
| **Canzar 2026 真实底座桥接** | `factor_demo/canzar_bridge/canzar_data_loader.py`<br>`factor_demo/canzar_bridge/canzar_ast_inspector.py`<br>`factor_demo/canzar_bridge/chimera_counterfactual_driver.py` | ① 读取 Canzar 三文件 CSV 真实切片、转换 `.h5ad`、驱动 `external/spatial-domain-simulator`（`subset_anndata.h5ad.gz`）、解析 `scores.csv`；<br>② 静态扫描 Canzar 仓库全部 35 个真实 `.py/.R` 方法脚本及 `chimera_run.py`；<br>③ 在 `GraphST/SEDR/SpaceFlow/STAGATE/scanpy` 的 `adjacency` 与 `train` 之间注入稀疏矩阵四臂钩子与输入行顺序置换。 |
| **可视化与双格式报告** | `factor_demo/reporting/visualizer.py`<br>`factor_demo/reporting/report_builder.py`<br>`factor_demo/pipeline.py` | 生成 4 张核心验证图（观测矩阵+筛选图、3剂量×3臂响应曲线、单切片分割对比图、带等效带森林图+预处理交叉图），输出 `experiment_report.md` 与零 LaTeX 依赖的多页矢量 `experiment_report.pdf`。 |
| **运行脚本、Notebook 与测试** | `scripts/run_demo.py`<br>`scripts/run_canzar2026_real_audit.py`<br>`scripts/run_canzar_chimera_experiment.py`<br>`notebooks/FACTOR_3Agent_Demo.ipynb`<br>`notebooks/FACTOR_Canzar2026_Real_Demo.ipynb`<br>`tests/test_core_pure_python.py` | 命令行运行入口、2026 真实数据实测审计脚本、服务器 Chimera 新干预实验入口、两份交互式演示 Notebook，以及纯 Python 语法/AST/桥接自检单测（5 项单测全部通过）。 |
| **外部克隆基准仓库** | `external/spatial-domain-analysis-framework/`<br>`external/spatial-domain-simulator/` | 完整拉取的 Descoeudres et al. (2026.03) 官方仓库（含 `osmFISH` 原始数据、`scores.csv`、35 个方法脚本、`chimera_run.py` 与小鼠脑 `subset_anndata.h5ad.gz`）。 |

---

## 6. 三组智能体（尤其 D–M Interaction Agent）与 `D×M` vs `D×C` 实现细节

### 6.1 `D×M` 与 `D×C` 在代码中的精确定义
- **`M`（核心归纳偏置特征 `m_core`）**：在自包含池中编码 `uses_spatial_graph`, `unpruned_multihop_smoothing`, `attention_edge_gating`, `feature_concatenation`, `contrastive_dgi_loss`, `spatial_smooth_reg`；在 Canzar 35 方法池中编码 `is_neural_network`, `is_statistical_hmrf`, `is_neighborhood_concat_or_clust`, `uses_dgi_contrastive`, `uses_autoencoder_recon`, `uses_alpha_or_voronoi_graph`, `uses_potts_or_spatial_prior`。
- **`C`（耦合辅助管线特征 `c_aux`）**：在自包含池中编码 `default_prep_pca`, `default_prep_hvg`, `default_clust_gmm`, `default_clust_kmeans`；在 Canzar 35 方法池中编码 `uses_hvg_selection`, `uses_pca_reduction`, `uses_mclust_gmm`, `uses_kmeans_or_leiden`, `uses_posthoc_spatial_refine`, `has_hardcoded_seed`。
- **竞争筛选实现（`factor_demo/tools/stats_engine.py` L88–L122）**：构造交互设计矩阵 `X_int` 时，对每个保留的数据特征 $d_p$，同时拼接 $d_p \cdot m_q$（标记为 `D_x_M`）和 $d_p \cdot c_r$（标记为 `D_x_C`），在扣除主效应与个体随机截距后的残差上共同接受 $\ell_1$ 稀疏惩罚。

### 6.2 `DMInteractionAgent`（`factor_demo/agents/interaction_agent.py`）的五步执行链路
1. **`run_observational_benchmark`（L73–L108）**：仅在探索集切片上运行默认方法（`arm="orig"`）多随机种子取均值，构建观测矩阵 $\mathbf{Y}_{\text{expl}}$。
2. **`propose_and_preregister_hypotheses`（L110–L190）**：调用 `fit_outer_loop_screening` 获取排序后的候选交互项，在触碰确认集之前将实验规格（算子名、剂量列表、标签演化规则、四臂对照、调参协议、主指标、等效界）写入 `preregistered_hypotheses.json`。
3. **`execute_confirmatory_experiments`（L192–L332）**：
   - 先在确认集切片上回调 `MethodCharacterAgent.verify_code_equivalence`（检验 `orig` 与 `refac` 误差 $\le 10^{-4}$）与 `DatasetCharacterAgent.audit_counterfactual_slice`（记录目标偏移与非目标偏移）；
   - 按 `方法 × 调参协议(fixed/retuned) × 剂量 × 对照臂(refac/self/null)` 执行三层随机种子（扰动种子 × 随机图置换种子 × 训练种子）嵌套实验，调用 `compute_factorial_inference` 计算简单效应与 $2\times 2$ 交互项置信区间；若 `fixed` 与 `retuned` 定性结论冲突，自动追加 `[Protocol-Sensitive]` 标记；
   - 保持 `arm="refac"` 原空间图不变，单独对每个方法强制交叉运行 `preprocess_override="pca"` 与 `"hvg"`，直接测量预处理的真实效应差 `PCA_minus_HVG`。
4. **`evaluate_agent_and_recommendation_value`（L334–L406）**：调用 `dataset_agent.build_feature_matrix(..., include_ref=False)`（严格只取 `d_obs`），在探索集上拟合 Ridge 回归器预测确认集切片各方法得分，对比“固定选探索集平均冠军”计算样本外平均遗憾（Regret）。
5. **`synthesize_scientific_conclusion`（L408–L462）**：将上述全部确定性统计结果序列化为 JSON 输入 `DualModeLLMClient`，生成带数值与置信区间的文字总结。

---

## 7. 基于 Canzar 2026 真实仓库已跑出的初步实测结果

通过运行 `python scripts/run_canzar2026_real_audit.py`（输出文件位于 `outputs/canzar2026_real_audit/canzar2026_real_agent_report.md`），已在克隆的真实仓库数据与源码上完成首轮实测审计：

1. **原始单细胞矩阵实测（`osmFISH_Codeluppi`）**：核实 `4,839` 个细胞 × `33` 个基因，`11` 个空间域，零值稀疏度 `30.71%`，最小空间域占比 `2.19%`。
2. **35 个真实方法脚本静态解析发现**：`canzar_ast_inspector.py` 自动扫描 `workflow/scripts/` 下全部 25 个 `.py` 与 10 个 `.R` 脚本，检出 **18 个方法脚本在内部硬编码了随机种子**（如 `graphst_run.py:L6`、`stagate_run.py:L7`、`sedr_run.py:L24`、`ccst_run.py:L10`、`spagcn_run.py:L43`、`bass_run.R:L6`），并精确提取了 `chimera_run.py` 中 6 个深度模型的三阶段函数行号。
3. **跨技术真实切片外循环筛选（56 张探索集切片 × 26 个论文正式方法）**：在扣除切片与方法主效应后，观测到 `is_single_cell × has_hardcoded_seed`（`D_x_C`, $r=-0.344$）、`is_single_cell × is_neighborhood_concat_or_clust`（`D_x_M`, $r=+0.325$）及 `log10_panel_genes × is_statistical_hmrf`（`D_x_M`, $r=+0.254$）等竞争关联线索。
4. **受控半合成扰动序列上的基线参照 $2\times 2$ 交互验证（`scores.csv` 实测数据）**：
   - 在 `exp2`（Type II 异质细胞浸润从 `0%` 增至 `50%`，以同切片非空间 `scanpy` 作为外部无空间参照，$n=16$ 个形状×种子独立重复）中：
     - **`BASS`**：空间净增益从 `+0.576` 变为 `+0.588`，交互项 $\tau_{\text{int}} = +0.012$（95% CI `[-0.001, +0.026]`，完全落在 `±0.03` 等效带内 $\rightarrow$ **`Practically Equivalent`**）；
     - **`GraphST`**：空间净增益从 `+0.527` 降至 `+0.193`，交互项 $\tau_{\text{int}} = -0.334$（95% CI `[-0.410, -0.259]` $\rightarrow$ **`Supported Interaction`**）；
     - **`STAGATE`**：交互项 $\tau_{\text{int}} = -0.278$（95% CI `[-0.367, -0.190]` $\rightarrow$ **`Supported Interaction`**）；
     - **`SpaceFlow`**：交互项 $\tau_{\text{int}} = -0.153$（95% CI `[-0.259, -0.047]` $\rightarrow$ **`Supported Interaction`**）。
5. **21 张跨器官/跨技术保留真实切片上的样本外推荐基线发现**：在无运行时间约束的纯 ARI 口径下，由于 `BASS` 在探索集与确认集均表现极强（确认集平均 ARI `0.4559`，相对理论最优 Oracle 的平均遗憾仅 `0.0882`），仅使用粗粒度技术元特征与 `scanpy` 探针的经验贝叶斯推荐器收敛至同样推荐 `BASS`（平均遗憾同为 `0.0882`；而未加经验贝叶斯收缩或混入未跑全实验分支 `stGRL` 时曾出现过拟合掉点至 `0.4251`）。这提示：**若要让样本外推荐产生超越单一强方法（如 `BASS`）的净增益，要么引入运行时间/内存预算约束（如 >50,000 细胞场景下 `BASS` 受限），要么在原始表达矩阵上提取更细粒度的无标签空间特征**。

---

## 8. 坦诚披露：当前 Demo 代码的已知简化、局限与待审核问题清单

为便于审阅者（GPT）精准把脉，以下主动列出当前 v0.2 代码实现中尚未完全展开的工程简化与潜在薄弱点，请重点审核其合理性及改进优先级：

1. **外循环混合模型求解的两阶段近似问题（`factor_demo/tools/stats_engine.py` L109–L122）**：
   - *现状*：当前采用两步近似——先对 `[X_main, Z_rand]` 做固定惩罚系数 `Ridge(alpha=4.0)` 拟合并取残差，再对交互项 `[D×M, D×C]` 做固定惩罚系数 `Lasso(alpha=0.012)`。
   - *待审问题*：固定 `alpha` 而非通过交叉验证或边际似然估计方差分量 $(\sigma_u^2, \sigma_v^2, \lambda_1, \lambda_2)$，在不同规模数据集上是否会导致过度收缩或欠收缩？在正式实验前是否应替换为交叉验证调参的分组惩罚或联合估计？
2. **Agent 3 从“筛选线索”到“实验算子选择”的映射自由度（`factor_demo/agents/interaction_agent.py` L133–L190）**：
   - *现状*：为确保 Demo 运行零崩溃，当前 `propose_and_preregister_hypotheses` 虽然读取了外循环排出的 `top_dm` 与 `top_dc` 写入理由，但绑定的具体实验算子（`u2_boundary_infiltration` 与 `u1_spot_aggregation`）仍是硬编码模板映射，尚未让大模型基于候选特征词典动态决策调用哪个算子、哪些剂量及哪几个重点方法子集。
   - *待审问题*：在保持可复现与防幻觉的前提下，将这一步设计为“受 JSON Schema 与注册表白名单约束的 LLM 实验规划器 + 非 LLM 贪心统计基线对比”，是否是验证 Agent 主动调度价值的合理方式？
3. **离线 `scores.csv` 审计与在线 `chimera_counterfactual_driver.py` 实验的区别**：
   - *现状*：在 `scripts/run_canzar2026_real_audit.py` 对现成 `scores.csv` 的分析中，由于 Canzar 原仓库未跑过模型内部的 `self` ($\mathbf{I}$) 与 `null` ($\mathbf{P}\mathbf{A}\mathbf{P}^\top$) 图替换，该脚本使用同切片非空间 `scanpy` 作为外部参照计算 `ARI_M - ARI_scanpy`（这仍混有模型家族与聚类头差异）；真正严格的**同一模型内部四臂对照（`orig / refac / self / null`）**已实现在 `factor_demo/canzar_bridge/chimera_counterfactual_driver.py` 与 `scripts/run_canzar_chimera_experiment.py` 中，需在配好 `Chimera` 环境（PyTorch/TensorFlow/Scanpy）的服务器上运行。
   - *待审问题*：`chimera_counterfactual_driver.py` 中对稀疏邻接矩阵注入 `sp.eye(n)`（`self`）与 `adj_csr[perm, :][:, perm]`（`null`）对 `GraphST`、`SEDR`、`SpaceFlow`、`STAGATE` 四个模型的内部图归一化与重构损失（如 `SEDR` 用 `adj_label` 计算交叉熵重构损失！）是否存在特定模型语义陷阱？**（特别提示审阅者重点检查 `chimera_counterfactual_driver.py` L152–L165 中 `SEDR` 的图重构损失在 `self`/`null` 臂下的含义！）**
4. **Canzar 真实切片原始矩阵覆盖度**：
   - *现状*：Canzar 仓库 Git 中仅直接附带了 1 张真实切片的原始计数矩阵（`osmFISH_Codeluppi`）和半合成底座 `subset_anndata.h5ad.gz`；其余真实数据集（如 `10xVisium_Maynard` DLPFC 12 切片、`MERFISH_Zhang` 33 切片）在 Git 中仅有 `scores.csv` 评分结果，需在服务器上单独下载原始数据转成 3-CSV 格式后，才能提取基于原始计数的细粒度 `d_obs`（如基因 Moran's $I$、稀疏度、局部表达余弦异质性）并运行在线四臂干预。
5. **方法源码静态解析的语义深度（`canzar_ast_inspector.py`）**：
   - *现状*：Canzar 仓库中的 `workflow/scripts/*_run.py` 大多是调用第三方包（如 `import GraphST`, `import STAGATE`）的封装脚本而非底层算法全量源码，因此仅扫描 wrapper 脚本时，部分底层模型机制（如损失函数类型）依赖包名/函数名正则规则匹配，只有 `chimera_run.py` 暴露了较完整的预处理与构图实现。
   - *待审问题*：在正式版本中，是否应当让 `MethodCharacterAgent` 同时扫描第三方包安装目录（`inspect.getsource(GraphST)`）以获取真正的底层模型 AST 证据，而非仅扫描外层 Snakemake 运行脚本？

---

### 附：建议向审阅者（GPT）提问的审核指令模板

你可以将本文档连同下面这段指令一起发给 GPT：

> 请作为计算生物学方法学与因果/统计实验设计审稿人，对上述方案（FACTOR v2.0）及已落地的代码架构进行严格的**严谨性与合理性审核**：
> 1. 评估第 2 节中我们与 Descoeudres et al. (2026.03) 的 6 点边界划分是否客观、准确、站得住脚？哪些点最有希望产出高价值实证发现，哪些点可能被审稿人认为增量不足？
> 2. 检查第 4 节的数学定义与第 6 节的代码实现逻辑之间是否还存在不一致或识别漏洞？
> 3. 重点审查第 8 节主动披露的 5 个工程与方法细节（特别是第 3 点：对 `SEDR` / `GraphST` / `STAGATE` 注入 `self=I` 和 `null=P A P^T` 时的模型内部损失函数兼容性陷阱，以及第 5 点：Wrapper 脚本 vs 底层包源码解析），给出在服务器上正式开跑实验前必须修改的具体清单。


---

## 9. v2.1 增量更新记录：响应二轮代码静态审查的 9 项修复 + 四智能体架构与三大下游交付物扩展

在收到针对代码的静态审查意见与补充构想后，代码仓已完成 **v2.1 结构性升级**：

### 9.1 针对 9 项静态审查问题的逐项代码级修复对照表

| 审查指出的问题 | 修复状态 | 具体代码修复位置与机制 |
| :--- | :---: | :--- |
| **1. `scores.csv` 再分析混同于方法内组件效应** | ✅ 已修正 | 在 `scripts/run_canzar2026_real_audit.py` 及生成报告第 四 节明确限定为**“既有半合成受控实验的再分析：相对外部非空间 `scanpy` 基线的完整管线优势变化”**，严格与 `chimera_counterfactual_driver.py` 执行的方法内四臂实验区分。 |
| **2. `orig == refac` 未对标上游真实路径、`mclust` 静默回退、`scanpy` 乘了邻接矩阵** | ✅ 已修复 | ① 新增 **`ReproduceAgent`**（`factor_demo/agents/reproduce_agent.py`），实现 `verify_stagewise_equivalence` 对预处理、图度分布、嵌入子空间（Gram 余弦）与聚类 ARI 做逐阶段比对；<br>② `chimera_counterfactual_driver.py` 拆开 `scanpy_nonspatial`（**零空间矩阵乘法**）与 `scanpy_smooth`；<br>③ 当 `R-mclust` 缺失回退至 Python `GaussianMixture` 时，显式记录 `fallback_notes` 并将 `effective_method_id` 重命名为 `<Method>__pyGMM_fallback`，绝不冒充上游原方法。 |
| **3. `SEDR` 同时改动编码图与重构目标 `adj_label`** | ✅ 已修复 | `chimera_counterfactual_driver.py` 新增显式参数 `sedr_intervention_mode in {"encoder_only", "full_mechanism"}`：`encoder_only` 仅修改编码传播图 `adj_norm` 而固定原重构目标 `adj_label`（并注明空间信息仍可经解码损失进入），`full_mechanism` 同时替换二者。 |
| **4. 主动实验选择直接截取列表前几项，未按事前优先级排序** | ✅ 已修复 | `interaction_agent.py` 在触碰确认集之前根据外循环筛选权重与探索集残差波动构造排序后的 `pre_confirmation_queue`，落盘为 `pre_confirmation_experiment_queue.json`，并在评估中严格沿该事前序列对比固定顺序与随机基线。 |
| **5. `passed_validity_check` 未阻断裁决、`convergence_ok_rate` 硬编码为 `1.0`** | ✅ 已修复 | `interaction_agent.py` 收集每次真实运行的 `out.converged` 计算真实收敛率，并传入 `stats_engine.py`：若有效性失败或收敛率 `< 0.80`，硬门控直接输出 `"Invalid Intervention / Convergence Anomaly (Gated)"`，禁止进入机制结论。 |
| **6. `canzar_ast_inspector.py` 语义误编码（`stat=HMRF`、名称匹配 `mclust`、种子未看常量、缺失默置 0）** | ✅ 已修复 | 重写为**三态解析器（`1.0` 确认存在 / `0.0` 确认不存在 / `None` 外层脚本未暴露待查包源码）** + `evidence_source` 溯源：修正 `GraphPCA`/`SpatialPCA` 的 `uses_hmrf_potts_prior=0.0`，`BASS`/`BayesSpace` 的 `uses_mclust_gmm=0.0`，并用 `_is_constant_seed_call` 检查 AST 字面量常量参数，且已通过单测断言锁定。 |
| **7. 统计引擎边界（`M/C` 分开去共线性、未按供体聚合、单样本区间退化、TOST 置信度命名）** | ✅ 已修复 | `stats_engine.py` 改为：① `[M, C]` 联合去共线性、使用 `RidgeCV` + `LassoCV`；② 确认阶段先按独立 `donor_id` 聚合均值（`n_donors < 2` 时强制返回 `"Inconclusive (Insufficient Independent Donors < 2)"`）；③ 明确分离 95% 双侧区间（方向显著性）与 **90% 置信区间（对应双单侧 $\alpha=0.05$ 标准 TOST 等效检验）**。 |
| **8. `PCA vs HVG` 未与数据干预剂量交叉** | ✅ 已修复 | `interaction_agent.py` 将预处理检查升级为 **`数据剂量(dose_low vs dose_high) × 预处理方案(PCA-20 vs HVG-110)` 双因素交叉实验**，直接输出 `tau_int_prep_scheme` 以检验剂量敏感性是否源于预处理方案交互。 |
| **9. `heldout_markers` 未从训练输入计数矩阵中剔除** | ✅ 已修复 | `SpatialSlice.get_model_input_counts()` 在把数据送入任何方法前**物理剔除全部 `heldout_markers` 基因列**，仅在评价阶段读取原始留出列计算富集度，彻底消除信息泄漏。 |

---

### 9.2 新增架构组件：前置 `ReproduceAgent` 与三大下游可交付工具

1. **前置第 0 组智能体：`ReproduceAgent`（`factor_demo/agents/reproduce_agent.py`）**
   - **上游仓库拉取与四阶段复现核验**：维护 `UPSTREAM_METHOD_REPOS` 清单，支持在服务器上自动克隆原方法仓库并逐阶段核对 `preprocess -> adjacency -> embedding -> clustering` 与模块化管线的数值一致性。
   - **跨数据集邻接图尺度退化诊断与双轨记录（`audit_and_repair_radius_graph`）**：针对不同平台坐标尺度差异导致的固定半径构图失效（如 `degree == 0` 孤立细胞比例 $>5\%$ 或过度全连接），同时保存 **Track 1（原默认退化记录）** 与 **Track 2（二分搜索校准目标度分布后的修复配置）**。

2. **三大下游实用交付物（`factor_demo/deliverables.py`，严格在探索集 $\mathcal{D}_{\text{expl}}$ 校准、在独立确认集 $\mathcal{D}_{\text{conf}}$ 零泄漏评测）**
   - **`FACTOR-Chimera`（`FactorAuditedChimera`）**：结合审查验证优势构建的自适应空间域方法——采用方差稳定 `PCA-20` 预处理规避窄面板/高稀疏 HVG 退化，采用“细胞自身特征 $\sqrt{1-\lambda}\mathbf{Z}_{\text{self}}$ + 转录相似性门控空间上下文 $\sqrt{\lambda}\mathbf{A}_{\text{att}}\mathbf{Z}_{\text{self}}$”双分支拼接（门控温度与权重仅由无标签 `local_expr_heterogeneity` 相对探索集中位数的大小自动调节），兼顾平滑皮层降噪与浸润/薄层抗吞没。
   - **`FACTOR-Recommender`（`FactorMethodRecommender`）**：仅依赖新切片的无标签特征 `d_obs`（支持附加最大运行时间 `max_runtime_sec` 预算过滤）预测并推荐最优方法。
   - **`FACTOR-Ensemble`（`run_label_free_consensus`）**：**100% 无真值依赖的共现关联矩阵（Co-Association Matrix）谱共识投票器**——通过计算加权同簇共现矩阵 $C_{uv} = \sum_m w_m(\boldsymbol{d}_{\text{obs}}) \mathbb{I}(\hat{y}_u^{(m)} = \hat{y}_v^{(m)})$ 并做归一化拉普拉斯谱分解聚类，彻底解决了 Canzar 组（2026）共识方法必须依赖 `ground_truth` 做二分图标签对齐而无法用于无标注新样本的缺陷。


---

## 10. v2.2 再次复核响应：修正最新 6 项方法/实现细节、核实行号索引与已知残留边界

*(注：上一轮出现“文档第 9 节与源码目录不同步”是因为审阅环境读取的是同步前的旧快照；推入 GitHub 仓库后可实时核查下方全部源码文件与行号。)*

### 10.1 针对最新 6 条审阅意见与架构调整的逐项落实记录（含精确文件位置、验证与未覆盖边界）

| 审阅问题 | 修改位置（文件路径与函数） | 验证方式 | 仍未覆盖的真实边界（如实保留） |
| :--- | :--- | :--- | :--- |
| **1. `ReproduceAgent` 定位调整 + 强化复现核验指标** | `factor_demo/agents/reproduce_agent.py`：<br>① 新增主模式 `run_native_upstream_script`（L144–L179），直接运行原作者/仓库未改动的原生脚本记录结果，跳过重写管线的争议；<br>② 当必须使用模块化适配器时，`verify_stagewise_equivalence`（L249–L327）除预处理 Frobenius/最大误差外，新增**精确边集 Jaccard 相似度 `edge_set_jaccard`、边权 L1 差、自环一致性 `self_loop_match`、行和归一化差 `row_sum_max_diff`**，并以原方法多随机种子自身波动 `baseline_multiseed_aris` 作为对照阈值；<br>③ 半径诊断 `audit_and_repair_radius_graph`（L181–L247）先计算物理坐标跨度 `coord_span_xy` 与中位最近邻间距 `median_1nn_spacing`，将半径调整明确界定为独立于默认结果的“尺度适配配置”。 | `tests/test_core_pure_python.py` `test_v22_audit_fixes_verified_in_source` 校验编译与接口完整性。 | 原生上游脚本的真实 GPU/R 运行核验需在装有对应 Conda 环境（如 `chimera.yml`）的服务器上执行并产出具体数值对账单。 |
| **2. `FACTOR-Chimera` 经验性过强，需做组件消融并考虑异型细胞空间域风险** | `factor_demo/deliverables.py` `FactorAuditedChimera`（L47–L135）：<br>明确将其定位为“框架发现驱动的管线改进案例（Case Study）”，内置 4 变体消融参数 `variant in {"fixed_concat", "gated_only", "adaptive_weight_only", "full_gated_adaptive"}` 及 `preprocess_mode` 参数；并在文档与注释中明确指出：**按表达相似性衰减邻接边虽有助于保护同质分层边界，但在异型细胞共定位组成的多细胞微环境（如免疫浸润生态位）中可能削弱跨域联系**，因此是否优于固定拼接由外部测试集实测回答，不预设为普适真理。 | `factor_demo/pipeline.py`（L180–L222）在独立外部测试集上同步评测全部 4 个消融变体。 | 尚未在多套真实多细胞异型肿瘤生态位数据集上完成大样本统计检验。 |
| **3. 关键数学 Bug：`cluster_embedding` 按列标准化抵消了拼接权重 $\sqrt{1-\lambda}, \sqrt{\lambda}$** | ① `factor_demo/methods_zoo/base_pipeline.py`（L111–L136）：新增 `standardize_branch()`，并在 `cluster_embedding()` 中加入 `preserve_relative_weights: bool = True`（默认按整个拼接矩阵的单一全局标量均方根缩放，而非逐列除以各自标准差）；<br>② `factor_demo/methods_zoo/banksy_aug_domain.py`（L69–L80）与 `factor_demo/deliverables.py`（L109–L122）：严格改为**先对 `Z_self` 与 `Z_nbr` 各自做零均值单位方差标准化，再乘以 $\sqrt{1-\lambda}$ 与 $\sqrt{\lambda}$**（与 BANKSY 原文一致）。 | 数学恒等核实 + `test_v22_audit_fixes_verified_in_source` 源码断言锁定。 | 若下游更换为对特征尺度不敏感的树模型则权重定义需另行设计（当前针对欧氏/马氏距离聚类头 KMeans/GMM 有效）。 |
| **4. 撤回对 Canzar `unify_labels` “无真值无法运行”的不准确表述** | 本文档及代码注释已撤回该过宽表述：经核查 `external/spatial-domain-analysis-framework/workflow/scripts/aggregate.py` L253–256，`unify_labels()` 确实包含 `else: lb_gt = labels_numbers.dropna(axis=1).iloc[:,0]`（无 `gt` 时以首个可用方法列做二分图匹配参照）。我们将表述严格收窄为：**共关联矩阵方法遵循经典 Cluster Ensembles 思路（Strehl & Ghosh, JMLR 2002），其作用是避免指定单一参考聚类及显式簇编号匹配；本文待检验的新增点是由 `d_obs` 与审计可靠性决定的方法加权能否优于均匀权重与单一强方法**。 | 直接对照 `aggregate.py` L253–256 核实。 | 当前共识算法仍接收目标簇数 `n_domains` 作为输入，因此准确名称为**“给定簇数 $K$ 条件下的无标签聚类共识”**，尚未解决未知簇数自动估计问题。 |
| **5. 避免稠密 $N\times N$ 共关联矩阵在十万细胞规模下的 40GB 内存爆炸** | `factor_demo/deliverables.py` `run_label_free_consensus`（L193–L244）：<br>彻底删除 `one_hot @ one_hot.T` 稠密矩阵构造！改用低秩二分图表示 $\mathbf{B} = [\sqrt{w_1}\mathbf{H}_1, \dots, \sqrt{w_M}\mathbf{H}_M] \in \mathbb{R}^{N \times K_{\text{tot}}}$（$K_{\text{tot}} = \sum_m K_m \ll N$）：利用 $\mathbf{d} = \mathbf{B}(\mathbf{B}^\top \mathbf{1}_N)$ 在 $O(N K_{\text{tot}})$ 内存内精确计算行度，并对 $\tilde{\mathbf{B}} = \mathbf{D}^{-1/2}\mathbf{B}$ 做薄奇异值分解 `np.linalg.svd(B_tilde, full_matrices=False)`，在 **$O(N K_{\text{tot}}^2)$ 时间与 $O(N K_{\text{tot}})$ 线性内存**（10 万细胞仅需 ~20MB）下精确恢复 $\mathbf{D}^{-1/2}\mathbf{C}\mathbf{D}^{-1/2}$ 的前 $K$ 个特征向量。 | `test_v22_audit_fixes_verified_in_source` 断言确认已移除 `one_hot @ one_hot.T` 并使用低秩薄 SVD。 | 虽然融合步骤本身已是 $O(N)$ 内存，但端到端总耗时仍受限于各候选基方法本身的运行开销（已在 `pipeline.py` 记录总耗时 `total_wall_time_sec`）。 |
| **6. 防止确认集在设计下游工具后沦为开发集（三划分隔离）** | ① `factor_demo/data_lab/generator_and_loader.py`（L245–L253）：将基准队列扩充为严格不重叠的三套独立供体划分——`exploration`（外循环筛选与拟合推荐器）、`confirmation`（预注册双侧因子实验确认机制）、**`external_test`（`Donor_T1..T3`，在筛选与机制确认阶段完全封存，专供评测最终下游工具）**；<br>② `factor_demo/pipeline.py`（L87–L102, L171–L225）：将 `FactorAuditedChimera` 四变体消融、`FactorMethodRecommender` 与加权/均匀 `Consensus` 全部移至 `external_test` 上评测并记录端到端耗时。 | 编译与代码审查通过。 | 在真实数据规模化阶段，同样需要按独立数据集/独立供体预先切出第三份封存测试集（如将 `10xVisium_Fu` / `StereoSeq_Chen` 完全留作最终外部测试集，不参与机制确认调优）。 |

### 10.2 上一套 9 项静态审查问题的当前源码定位速查表（供 GitHub 实时核验）

1. **外部 `scanpy` 再分析与方法内四臂区分**：`scripts/run_canzar2026_real_audit.py` L499–L505。
2. **`SEDR` 编码图 vs 整体图机制拆分 & 纯非空间 `scanpy_nonspatial` & `mclust` 回退改名**：`factor_demo/canzar_bridge/chimera_counterfactual_driver.py` L48 (`sedr_intervention_mode`), L147–L171 (`SEDR` 分支), L189–L197 (`scanpy_nonspatial`), L203–L218 (`pyGMM_fallback_for_R_mclust`)。
3. **事前实验优先级队列生成与评测**：`factor_demo/agents/interaction_agent.py` L192–L225 (`self.pre_confirmation_queue`), L386–L410；落盘于 `factor_demo/pipeline.py` L131–L134。
4. **真实收敛率统计与无效运行硬门控**：`factor_demo/agents/interaction_agent.py` L266–L318；硬门控裁决于 `factor_demo/tools/stats_engine.py` L224–L247。
5. **三态（`1.0 / 0.0 / None`）方法解析与常量种子 AST 检查**：`factor_demo/canzar_bridge/canzar_ast_inspector.py` L77–L91 (`_is_constant_seed_call`), L101–L252。
6. **`[M, C]` 联合去共线性、`RidgeCV/LassoCV`、供体聚合、`n_donors < 2` 保护与 90% TOST 区间**：`factor_demo/tools/stats_engine.py` L61–L64, L109–L125, L186–L295。
7. **留出标志基因物理剔除（防训练泄漏）**：`factor_demo/data_lab/generator_and_loader.py` L30–L36 (`get_model_input_counts`)，调用处于 `factor_demo/agents/interaction_agent.py` L87, L280, L347。
8. **`数据剂量 × 预处理方案(PCA-20 vs HVG-110)` 交叉对比**：`factor_demo/agents/interaction_agent.py` L338–L365 (`tau_int_prep_scheme`)。

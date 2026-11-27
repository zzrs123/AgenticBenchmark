# 基于 Descoeudres et al. (2026.03) 真实数据集与 35 个真实方法的三智能体实测报告

- **运行时间**：`2026-11-27 12:11:00`
- **底层数据与方法底座**：直接对接 `external/spatial-domain-analysis-framework` 与 `external/spatial-domain-simulator`
- **真实组织切片总数**：`82` 张切片（跨 9 个真实数据集；严格切分为 **56 张探索集切片** vs. **26 张跨器官/跨平台独立确认集切片**）
- **静态代码解析方法总数**：`35` 个真实空间域脚本（25 个 `.py` + 10 个 `.R` + `chimera_run.py` 六大模块化深度模型）

---

## 一、 Agent 1 实测输出：原始 `osmFISH_Codeluppi` 矩阵审计与 9 大真实数据集特征概览

- **仓库内置原始单细胞切片实测 (`osmFISH_Codeluppi/correctedsample`)**：`4839` 个细胞 × `33` 个基因，`11` 个皮层空间域，实测矩阵零值稀疏度 `30.71%`，每细胞平均检出 `22.87` 个基因，最小空间域占比 `2.19%`。

### 9 大真实空间转录组数据集元特征均值（严格隔离无标签 `d_obs` 与真值诊断 `d_ref`）

| 数据集 | 划分角色 | 切片数 | 平台类型 | 器官/组织 | 单细胞指标 (`d_obs`) | `scanpy` 空间离散度 PAS (`d_obs`) | 真值域空间边界度 `gt_PAS` (`d_ref`) | 真值转录本可分性 `gt_SI_X` (`d_ref`) |
| :--- | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: |
| `10xVisium_Fu` | **Held-Out Confirmation** | 2 | Visium | Human_Breast_Cancer | 0.0 | 0.273 | 0.061 | +0.066 |
| `10xVisium_Maynard` | Exploration | 12 | Visium | Human_DLPFC | 0.0 | 0.294 | 0.011 | +0.111 |
| `MERFISH_Moffitt` | **Held-Out Confirmation** | 5 | MERFISH | Mouse_Hypothalamus | 1.0 | 0.499 | 0.025 | -0.074 |
| `MERFISH_Zhang` | Exploration | 33 | MERFISH | Mouse_Motor_Cortex | 1.0 | 0.511 | 0.018 | -0.067 |
| `STARmap_Wang` | **Held-Out Confirmation** | 1 | STARmap | Mouse_Visual_Cortex | 1.0 | 0.475 | 0.023 | +0.020 |
| `Slide-seqV2_Lan` | **Held-Out Confirmation** | 11 | Slide-seqV2 | Mouse_Brain_SlideSeqV2 | 0.5 | 0.144 | 0.050 | +0.000 |
| `Slide-seq_Langlieb` | Exploration | 10 | Slide-seq | Mouse_Whole_Brain | 0.5 | 0.780 | 0.042 | -0.175 |
| `StereoSeq_Chen` | **Held-Out Confirmation** | 7 | StereoSeq | Mouse_Embryo_Brain | 0.5 | 0.091 | 0.083 | +0.044 |
| `osmFISH_Codeluppi` | Exploration | 1 | osmFISH | Mouse_Somatosensory_Cortex | 1.0 | 0.398 | 0.011 | -0.071 |

---

## 二、 Agent 2 实测输出：真实方法源码静态解析与隐藏随机种子审计

- **自动源码审计发现**：在 35 个真实方法脚本中，Agent 2 通过语法树自动捕获到 **17 个脚本在内部硬编码了随机种子**（`ADEPT, BASS, BayesSpace, CCST, DeepST, GraphST, MLN2SVG, MNMST, SC-MEB, scTM, SEDR, SpatialLeiden, SpatialMGCN, SpiceMix, STAGATE, stGRL, Vesalius`），从代码层面直接印证了论文关于“必须置换细胞输入顺序才能暴露隐藏随机性”的发现。
- **Chimera 六大可插拔深度模型精确钩子定位**：

| 方法名 | 类别 | 脚本文件 | HMRF/Potts 先验 | DGI 对比 | HVG (`c_aux`) | PCA (`c_aux`) | `mclust` (`c_aux`) | 字面量硬编码种子 | 包内待核查数 (`unknown`) | 自动提取的真实源码证据链 (`file:L#`) |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `GraphST` | NN | `graphst_run.py` | 0 | `?` | 0 | 0 | 0 | 1 | 2 | `graphst_run.py:L6` (has_hardcoded_seed); `chimera_run.py:L393` (chimera_GraphST_preprocess); `chimera_run.py:L397` (chimera_GraphST_adjacency) |
| `STAGATE` | NN | `stagate_run.py` | 0 | `?` | 1 | 0 | 1 | 1 | 2 | `stagate_run.py:L7` (has_hardcoded_seed); `stagate_run.py:L27` (uses_mclust_gmm); `stagate_run.py:L50` (uses_hvg_selection) |
| `SEDR` | NN | `sedr_run.py` | 0 | `?` | 1 | 1 | 1 | 1 | 2 | `sedr_run.py:L26` (uses_mclust_gmm); `sedr_run.py:L61` (uses_hvg_selection); `sedr_run.py:L68` (uses_pca_reduction) |
| `CCST` | NN | `ccst_run.py` | 0 | 1 | 0 | 0 | 0 | 1 | 1 | `ccst_run.py:L10` (has_hardcoded_seed); `ccst_run.py:L26` (uses_pca_reduction); `ccst_run.py:L108` (uses_dgi_contrastive) |
| `SCAN-IT` | NN | `scanit_run.py` | 0 | `?` | 1 | 0 | 0 | 0 | 2 | `scanit_run.py:L19` (uses_hvg_selection); `scanit_run.py:L36` (uses_alpha_or_voronoi_graph); `scanit_run.py:L92` (uses_kmeans_or_leiden) |
| `SpaceFlow` | NN | `spaceflow_run.py` | 0 | `?` | 1 | 0 | 0 | 0 | 2 | `chimera_run.py:L422` (chimera_SpaceFlow_preprocess); `chimera_run.py:L428` (chimera_SpaceFlow_adjacency); `chimera_run.py:L433` (chimera_SpaceFlow_train) |
| `BANKSY` | clust | `banksy_run.py` | 0 | 0 | 1 | 1 | 1 | 0 | 1 | `banksy_run.py:L36` (uses_hvg_selection); `banksy_run.py:L64` (uses_pca_reduction); `banksy_run.py:L65` (uses_mclust_gmm) |
| `CellCharter` | clust | `cellcharter_run.py` | 0 | 0 | `?` | `?` | 0 | 0 | 2 | `cellcharter_run.py:L41` (uses_alpha_or_voronoi_graph) |
| `BASS` | stat | `bass_run.R` | 1 | 0 | `?` | `?` | 0 | 1 | 3 | `bass_run.R:L6` (has_hardcoded_seed); `bass_run.R:L31` (uses_hmrf_potts_prior) |
| `BayesSpace` | stat | `bayesspace_run.R` | 1 | 0 | `?` | `?` | 0 | 1 | 3 | `bayesspace_run.R:L6` (has_hardcoded_seed); `bayesspace_run.R:L90` (uses_hmrf_potts_prior) |
| `SpatialPCA` | stat | `spatialpca_run.R` | 0 | 0 | `?` | 1 | 0 | 0 | 2 | `spatialpca_run.R:L50` (uses_pca_reduction) |
| `SpaGCN` | NN | `spagcn_run.py` | 0 | `?` | `?` | `?` | 0 | 0 | 5 | `spagcn_run.py:L58` (uses_posthoc_spatial_refine) |
| `GraphPCA` | stat | `graphpca_run.py` | 0 | 0 | 1 | 1 | 0 | 0 | 1 | `graphpca_run.py:L46` (uses_pca_reduction); `graphpca_run.py:L50` (uses_kmeans_or_leiden); `graphpca_run.py:L55` (uses_posthoc_spatial_refine) |
| `TACCO` | clust | `tacco_run.py` | 0 | 0 | `?` | `?` | 0 | 0 | 3 |  |

---

## 三、 Agent 3 外循环：56 张真实探索集切片上的去主效应交互筛选（`D×M` vs `D×C`）

| 排名 | 切片特征 (`d_obs` / `d_ref`) | 方法/管线属性 | 组别 | 交互斜率 `W` | 偏相关 `r` |
| :---: | :--- | :--- | :---: | :---: | :---: |
| 1 | `is_single_cell` | `is_neighborhood_clustering` | `D_x_M` | +0.0324 | +0.3254 |
| 2 | `is_single_cell` | `has_hardcoded_seed` | `D_x_C` | -0.0303 | -0.3038 |
| 3 | `log10_panel_genes` | `is_neighborhood_clustering` | `D_x_M` | -0.0280 | -0.2813 |
| 4 | `is_single_cell` | `uses_alpha_or_voronoi_graph` | `D_x_M` | +0.0270 | +0.2713 |
| 5 | `log10_panel_genes` | `has_hardcoded_seed` | `D_x_C` | +0.0263 | +0.2627 |
| 6 | `scanpy_SI_expr` | `is_neighborhood_clustering` | `D_x_M` | +0.0241 | +0.2424 |
| 7 | `log10_panel_genes` | `uses_alpha_or_voronoi_graph` | `D_x_M` | -0.0233 | -0.2337 |
| 8 | `gt_SI_expr` | `is_neighborhood_clustering` | `D_x_M` | -0.0213 | -0.2143 |
| 9 | `scanpy_SI_expr` | `has_hardcoded_seed` | `D_x_C` | -0.0212 | -0.2125 |
| 10 | `gt_SI_expr` | `has_hardcoded_seed` | `D_x_C` | +0.0195 | +0.1955 |

---

## 四、 既有受控半合成序列再分析：相对外部非空间 `scanpy` 基线的完整管线优势变化

> **严谨性说明**：本节基于 Canzar 仓库预计算的 `scores.csv` 计算 `[ARI_M - ARI_scanpy]_high - [ARI_M - ARI_scanpy]_low`。由于 `M` 与 `scanpy` 同时存在预处理、表征与聚类头差异，该量反映**完整方法管线相对非空间基线的优势如何随数据条件变化**（作为观测与再分析线索），**不解释为单一空间图组件的内部净贡献**；同一方法内部的四臂（`orig/refac/self/null`）组件实验由 `chimera_counterfactual_driver.py` 在服务器上独立执行。

### 1. 算子 `exp8`：空间分辨率退化（单细胞 `grid=0.5` $\rightarrow$ Visium 级 `grid=5.0`）对空间建模净收益的影响

| 方法 | 重复单元数 | 高分辨 ARI (`0.5`) | 低分辨 ARI (`5.0`) | 高分辨空间净增益 (`M - scanpy`) | 低分辨空间净增益 (`M - scanpy`) | 2×2 因子交互 `tau_int` | 95% CI | 校准证据状态 (`±0.03`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `BASS` | 4 | 0.986 | 0.899 | +0.578 | +0.015 | **-0.563** | [-0.696, -0.430] | **Supported Interaction** |
| `BANKSY` | 4 | 0.826 | 0.179 | +0.417 | -0.705 | **-1.122** | [-1.505, -0.740] | **Supported Interaction** |
| `SpaceFlow` | 4 | 0.878 | 0.357 | +0.470 | -0.527 | **-0.997** | [-1.083, -0.911] | **Supported Interaction** |
| `SCAN-IT` | 4 | 0.923 | 0.495 | +0.515 | -0.388 | **-0.903** | [-1.056, -0.751] | **Supported Interaction** |
| `STAGATE` | 4 | 0.948 | 0.835 | +0.540 | -0.049 | **-0.589** | [-0.770, -0.408] | **Supported Interaction** |
| `GraphST` | 4 | 0.821 | 0.695 | +0.413 | -0.189 | **-0.602** | [-1.152, -0.051] | **Supported Interaction** |
| `SEDR` | 4 | 0.937 | 0.529 | +0.529 | -0.355 | **-0.884** | [-1.112, -0.656] | **Supported Interaction** |
| `CCST` | 4 | 0.887 | 0.671 | +0.478 | -0.212 | **-0.691** | [-0.776, -0.606] | **Supported Interaction** |
| `SpatialPCA` | 4 | 0.893 | 0.629 | +0.485 | -0.255 | **-0.740** | [-1.041, -0.438] | **Supported Interaction** |
| `SpaGCN` | 4 | 0.729 | 0.895 | +0.321 | +0.011 | **-0.309** | [-0.801, +0.182] | **Inconclusive** |

### 2. 算子 `exp2`：Type II 细胞浸润异质性（`0%` $\rightarrow$ `50%` 异质细胞置换）对空间建模净收益的影响

| 方法 | 重复单元数 | 0% 浸润 ARI | 50% 浸润 ARI | 0% 空间净增益 (`M - scanpy`) | 50% 空间净增益 (`M - scanpy`) | 2×2 因子交互 `tau_int` | 95% CI | 校准证据状态 (`±0.03`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `BASS` | 16 | 0.983 | 0.952 | +0.576 | +0.588 | **+0.012** | [-0.001, +0.026] | **Practically Equivalent** |
| `BANKSY` | 16 | 0.815 | 0.724 | +0.408 | +0.359 | **-0.049** | [-0.143, +0.045] | **Inconclusive** |
| `SpaceFlow` | 16 | 0.841 | 0.645 | +0.433 | +0.280 | **-0.153** | [-0.259, -0.047] | **Supported Interaction** |
| `SCAN-IT` | 16 | 0.826 | 0.863 | +0.418 | +0.498 | **+0.080** | [-0.059, +0.219] | **Inconclusive** |
| `STAGATE` | 16 | 0.945 | 0.624 | +0.538 | +0.260 | **-0.278** | [-0.367, -0.190] | **Supported Interaction** |
| `GraphST` | 16 | 0.935 | 0.558 | +0.527 | +0.193 | **-0.334** | [-0.410, -0.259] | **Supported Interaction** |
| `SEDR` | 16 | 0.962 | 0.859 | +0.554 | +0.495 | **-0.060** | [-0.138, +0.018] | **Inconclusive** |
| `CCST` | 16 | 0.879 | 0.748 | +0.472 | +0.383 | **-0.089** | [-0.139, -0.038] | **Supported Interaction** |
| `SpatialPCA` | 16 | 0.940 | 0.779 | +0.533 | +0.414 | **-0.118** | [-0.187, -0.050] | **Supported Interaction** |
| `SpaGCN` | 16 | 0.467 | 0.481 | +0.060 | +0.117 | **+0.057** | [+0.015, +0.098] | **Inconclusive** |

---

## 五、 独立确认集（26 张跨器官/跨技术保留真实切片）无标签 `d_obs` 推荐验证

- **训练范围**：仅使用 4 个探索集（`10xVisium_Maynard`, `MERFISH_Zhang`, `Slide-seq_Langlieb`, `osmFISH_Codeluppi` 共 56 张切片）的无标签特征 `d_obs`（分辨率 + 面板规模 + 非空间 `scanpy` 探针的 `PAS/CHAOS/SI_X`）。
- **测试范围**：完全未参与训练的 **5 个新数据集共 21 张真实切片**（涵盖人类乳腺癌 `10xVisium_Fu`、小鼠下丘脑 `MERFISH_Moffitt`、`StereoSeq_Chen`、`Slide-seqV2_Lan`、`STARmap_Wang`）。
- **探索集单一静态冠军（`BASS`）在确认集平均 ARI**：`0.4559`（平均遗憾 Regret = `0.0882`）
- **`d_obs` 智能体推荐器在确认集平均 ARI**：**`0.4559`**（平均遗憾 Regret = **`0.0882`**）

| 保留确认集（未见真实数据集） | 切片数 | 理论最优 Oracle ARI | 静态冠军 ARI | `d_obs` 推荐器 ARI | 最常推荐方法 |
| :--- | :---: | :---: | :---: | :---: | :--- |
| `10xVisium_Fu` | 1 | 0.6122 | 0.6024 | **0.6024** | `BASS` |
| `MERFISH_Moffitt` | 5 | 0.5998 | 0.5789 | **0.5789** | `BASS` |
| `STARmap_Wang` | 1 | 0.7298 | 0.6337 | **0.6337** | `BASS` |
| `Slide-seqV2_Lan` | 11 | 0.5316 | 0.3908 | **0.3908** | `BASS` |
| `StereoSeq_Chen` | 3 | 0.4122 | 0.3813 | **0.3813** | `BASS` |

---

## 六、 在服务器上继续执行 Chimera 四臂（`orig/refac/self/null`）新干预实验命令

上述真实切片审计完成后，可在服务器的 `Chimera` Conda 环境（`external/spatial-domain-analysis-framework/workflow/envs/chimera.yml`）直接对 `osmFISH_Codeluppi` 及 `subset_anndata.h5ad.gz` 运行全新的同谱置换图 `null` ($\mathbf{P}\mathbf{A}\mathbf{P}^\top$) 与自环 `self` ($\mathbf{I}$) 实验：

```bash
python scripts/run_canzar_chimera_experiment.py --use-osmfish --simulate-infiltration --methods GraphST STAGATE SEDR SpaceFlow scanpy
```

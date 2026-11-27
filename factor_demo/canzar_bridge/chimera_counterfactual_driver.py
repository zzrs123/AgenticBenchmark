"""Live Counterfactual Driver wrapping Canzar Lab's Modularized Chimera Pipeline (v2.1 Audited).

Key fixes from static review:
1. Separates SEDR interventions into two explicit modes (`sedr_intervention_mode`):
   - `"encoder_only"`: Modifies ONLY encoder propagation graph `adj_norm`, keeping reconstruction
     loss target `adj_label` fixed to original spatial graph (isolates message-passing propagation
     effect while explicitly noting spatial structure still enters via decoder loss).
   - `"full_mechanism"`: Modifies BOTH `adj_norm` and `adj_label` (tests replacing the entire graph
     mechanism in SEDR).
2. Fixes `scanpy` vs `scanpy_smooth`:
   - `scanpy_nonspatial` performs ZERO spatial matrix multiplication (true non-spatial baseline!).
   - `scanpy_smooth` explicitly applies local neighborhood smoothing.
3. Explicitly tracks `fallback_flags` (e.g., if `R-mclust` is unavailable and falls back to Python
   `GaussianMixture`, `effective_method_id` is renamed `<method>__pyGMM_fallback` rather than silently
   claiming equivalence to upstream `mclust`!).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_FRAMEWORK_DIR = PROJECT_ROOT / "external" / "spatial-domain-analysis-framework"


def apply_sparse_graph_counterfactual(adj_sp: Any, arm: str = "refac", null_seed: int = 0) -> Any:
    """Apply 4-arm counterfactual operator to a `scipy.sparse` spatial adjacency matrix."""
    import numpy as np
    import scipy.sparse as sp

    adj_csr = sp.csr_matrix(adj_sp)
    n = adj_csr.shape[0]
    if arm in ("orig", "refac"):
        return adj_csr.copy()
    if arm == "self":
        return sp.eye(n, format="csr", dtype=np.float32)
    if arm == "null":
        rng = np.random.default_rng(null_seed)
        perm = rng.permutation(n)
        return adj_csr[perm, :][:, perm].tocsr()
    raise ValueError(f"Unsupported counterfactual graph arm: {arm}")


def run_chimera_counterfactual(
    umi_csv: str | Path,
    coords_csv: str | Path,
    n_domains: int,
    preprocess_mod: str = "GraphST",
    adjacency_mod: str = "GraphST",
    train_mod: str = "GraphST",
    clust_mod: str = "mclust",
    arm: str = "refac",
    sedr_intervention_mode: str = "encoder_only",  # "encoder_only" or "full_mechanism"
    null_seed: int = 0,
    cell_order_seed: Optional[int] = None,
    output_dir: str | Path = "outputs/chimera_runs/tmp",
    framework_dir: Path = DEFAULT_FRAMEWORK_DIR,
) -> Dict[str, Any]:
    """Run a modularized real spatial domain method with audited counterfactual hooks."""
    import sys
    import numpy as np
    import pandas as pd
    import scanpy as sc
    import scipy.sparse as sp
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.mixture import GaussianMixture
    from sklearn.neighbors import kneighbors_graph

    scripts_dir = framework_dir / "workflow" / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    fallback_notes: List[str] = []

    counts_df = pd.read_csv(umi_csv, header=None)
    coords_df = pd.read_csv(coords_csv, header=None, names=["x", "y"])
    n_cells = counts_df.shape[0]
    assert n_cells == coords_df.shape[0], "Mismatch between UMImatrix rows and coords rows"

    if cell_order_seed is not None:
        rng_ord = np.random.default_rng(cell_order_seed)
        perm_ord = rng_ord.permutation(n_cells)
        inv_ord = np.argsort(perm_ord)
        X_arr = counts_df.to_numpy(dtype=np.float32)[perm_ord]
        S_arr = coords_df.to_numpy(dtype=np.float32)[perm_ord]
    else:
        inv_ord = np.arange(n_cells)
        X_arr = counts_df.to_numpy(dtype=np.float32)
        S_arr = coords_df.to_numpy(dtype=np.float32)

    adata = sc.AnnData(X=X_arr)
    adata.obsm["spatial"] = S_arr

    # --- Stage 1: Preprocessing Module ---
    if preprocess_mod == "SEDR":
        sc.pp.filter_genes(adata, min_cells=min(50, max(3, n_cells // 20)))
        sc.pp.filter_genes(adata, min_counts=10)
        sc.pp.normalize_total(adata, target_sum=1e6)
        if adata.n_vars > 2000:
            sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=2000, subset=True)
        sc.pp.scale(adata)
        n_comp = min(adata.n_vars, adata.n_obs - 1, 200)
        X_pca = PCA(n_components=n_comp, random_state=42).fit_transform(adata.X)
        adata_prep = sc.AnnData(X_pca)
        adata_prep.obsm["spatial"] = adata.obsm["spatial"].copy()
        adata = adata_prep
    elif preprocess_mod == "CCST":
        sc.pp.filter_genes(adata, min_cells=3)
        X_norm = sc.pp.normalize_total(adata, target_sum=1, exclude_highly_expressed=True, inplace=False)["X"]
        X_scaled = sc.pp.scale(X_norm)
        n_comp = min(adata.n_vars, adata.n_obs - 1, 200)
        X_pca = sc.pp.pca(X_scaled, n_comps=n_comp)
        adata_prep = sc.AnnData(X_pca)
        adata_prep.obsm["spatial"] = adata.obsm["spatial"].copy()
        adata = adata_prep
    elif preprocess_mod == "STAGATE":
        if adata.n_vars > 3000:
            sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000, span=0.5, subset=True)
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
    elif preprocess_mod == "GraphST":
        from GraphST.preprocess import preprocess as g_preprocess  # type: ignore[import-not-found]
        g_preprocess(adata)
    elif preprocess_mod == "SpaceFlow":
        from SpaceFlow import SpaceFlow  # type: ignore[import-not-found]
        sc.pp.filter_genes(adata, min_counts=1)
        sf = SpaceFlow.SpaceFlow(adata=adata)
        sf.preprocessing_data(n_top_genes=min(3000, adata.n_vars))
        adata = sf.adata_preprocessed
    elif preprocess_mod in ("scanpy_nonspatial", "scanpy_smooth"):
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        n_comp = min(adata.n_vars, adata.n_obs - 1, 30)
        adata.obsm["X_pca"] = PCA(n_components=n_comp, random_state=42).fit_transform(np.asarray(adata.X))
    else:
        raise ValueError(f"Unsupported preprocess_mod: {preprocess_mod}")

    # --- Stage 2: Spatial Adjacency Module ---
    if adjacency_mod in ("STAGATE", "scanpy_nonspatial", "scanpy_smooth"):
        adj_raw = kneighbors_graph(adata.obsm["spatial"], n_neighbors=6, mode="connectivity")
    elif adjacency_mod == "GraphST":
        from GraphST.preprocess import construct_interaction_KNN  # type: ignore[import-not-found]
        construct_interaction_KNN(adata)
        adj_raw = sp.csr_matrix(adata.obsm["adj"])
    elif adjacency_mod == "SEDR":
        import SEDR  # type: ignore[import-not-found]
        adj_raw = sp.csr_matrix(SEDR.graph_func.generate_adj_mat(adata, include_self=False, n=6))
    elif adjacency_mod == "SpaceFlow":
        from SpaceFlow import SpaceFlow  # type: ignore[import-not-found]
        adj_raw = sp.csr_matrix(SpaceFlow.SpaceFlow.graph_alpha(None, adata.obsm["spatial"]))
    else:
        adj_raw = kneighbors_graph(adata.obsm["spatial"], n_neighbors=6, mode="connectivity")

    adj_hooked = apply_sparse_graph_counterfactual(adj_raw, arm=arm, null_seed=null_seed)

    # --- Stage 3: Representation / Neural Network Module ---
    if train_mod == "GraphST":
        import torch  # type: ignore[import-not-found]
        from GraphST.GraphST import GraphST  # type: ignore[import-not-found]
        adj_dense = adj_hooked.toarray()
        adata.obsm["adj"] = adj_dense
        adata.obsm["graph_neigh"] = (adj_dense > 0).astype(np.float32)
        adata.X = sp.csr_matrix(adata.X)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = GraphST(adata, device=device, datatype="10x")
        adata = model.train()
        embedding = PCA(n_components=min(20, adata.obsm["emb"].shape[1]), random_state=42).fit_transform(
            adata.obsm["emb"].copy()
        )
    elif train_mod == "SEDR":
        import torch  # type: ignore[import-not-found]
        import SEDR  # type: ignore[import-not-found]
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

        # Encoder propagation graph always uses `adj_hooked`
        adj_enc = sp.coo_matrix(adj_hooked)
        adj_enc = adj_enc - sp.dia_matrix((adj_enc.diagonal()[np.newaxis, :], [0]), shape=adj_enc.shape)
        adj_enc.eliminate_zeros()
        adj_norm = SEDR.graph_func.preprocess_graph(adj_enc)

        # Decoder reconstruction target `adj_label`: explicitly separate `encoder_only` vs `full_mechanism`!
        target_source = adj_raw if sedr_intervention_mode == "encoder_only" else adj_hooked
        adj_tgt = sp.coo_matrix(target_source)
        adj_tgt = adj_tgt - sp.dia_matrix((adj_tgt.diagonal()[np.newaxis, :], [0]), shape=adj_tgt.shape)
        adj_tgt.eliminate_zeros()
        adj_lbl = (adj_tgt + sp.eye(adj_tgt.shape[0])).tocoo()

        indices = np.stack([adj_lbl.row, adj_lbl.col])
        adj_tensor = torch.sparse_coo_tensor(indices, adj_lbl.data, adj_lbl.shape).coalesce()
        norm_val = adj_lbl.shape[0] ** 2 / float(max(1.0, (adj_lbl.shape[0] ** 2 - adj_lbl.sum()) * 2))
        sedr_net = SEDR.Sedr(
            np.asarray(adata.X),
            {"adj_norm": adj_norm, "adj_label": adj_tensor, "norm_value": norm_val},
            device=device,
        )
        sedr_net.train_with_dec()
        embedding, _, _, _ = sedr_net.process()
    elif train_mod == "SpaceFlow":
        from SpaceFlow import SpaceFlow  # type: ignore[import-not-found]
        sf = SpaceFlow.SpaceFlow(adata)
        sf.adata_preprocessed = adata.copy()
        sf.spatial_graph = sp.csr_matrix(adj_hooked)
        sf.train(str(out_path / "spaceflow_embedding.txt"))
        embedding = np.asarray(sf.embedding)
    elif train_mod == "STAGATE":
        import STAGATE  # type: ignore[import-not-found]
        adj_csr = sp.csr_matrix(adj_hooked)
        rows, cols = adj_csr.nonzero()
        spatial_net = pd.DataFrame({"Cell1": [str(r) for r in rows], "Cell2": [str(c) for c in cols], "Distance": 1.0})
        adata_st = sc.AnnData(sp.csr_matrix(adata.X))
        adata_st.obs_names = [str(i) for i in range(adata_st.n_obs)]
        adata_st.uns["Spatial_Net"] = spatial_net
        adata_st.obsm["spatial"] = adata.obsm["spatial"].copy()
        adata_st.var["highly_variable"] = [True] * adata_st.n_vars
        adata_st = STAGATE.train_STAGATE(adata_st)
        embedding = np.asarray(adata_st.obsm["STAGATE"])
    elif train_mod == "scanpy_nonspatial":
        # Strictly expression-only (ZERO spatial adjacency multiplication!)
        embedding = adata.obsm["X_pca"] if "X_pca" in adata.obsm else np.asarray(adata.X)
    elif train_mod == "scanpy_smooth":
        X_base = adata.obsm["X_pca"] if "X_pca" in adata.obsm else np.asarray(adata.X)
        deg = np.maximum(np.asarray(adj_hooked.sum(axis=1)), 1e-6)
        row_norm_adj = sp.diags(1.0 / deg.ravel()) @ adj_hooked
        embedding = np.asarray(row_norm_adj @ X_base)
    else:
        raise ValueError(f"Unsupported train_mod: {train_mod}")

    # --- Stage 4: Clustering Module (`kmeans` vs `mclust` with explicit fallback logging!) ---
    if clust_mod == "kmeans":
        preds_perm = KMeans(n_clusters=n_domains, n_init=10, random_state=42).fit_predict(embedding)
    elif clust_mod == "mclust":
        try:
            import rpy2.robjects as robjects  # type: ignore[import-not-found]
            import rpy2.robjects.numpy2ri  # type: ignore[import-not-found]
            robjects.r.library("mclust")
            rpy2.robjects.numpy2ri.activate()
            robjects.r["set.seed"](2023)
            res = robjects.r["Mclust"](rpy2.robjects.numpy2ri.numpy2rpy(embedding), n_domains, "EEE")
            preds_perm = np.asarray(res[-2], dtype=np.int64)
        except Exception as exc:
            fallback_notes.append(f"pyGMM_fallback_for_R_mclust({type(exc).__name__})")
            preds_perm = GaussianMixture(
                n_components=n_domains,
                covariance_type="tied",  # Closer to mclust 'EEE' shared covariance than 'full'!
                reg_covar=1e-4,
                random_state=42,
            ).fit_predict(embedding)
    else:
        raise ValueError(f"Unsupported clust_mod: {clust_mod}")

    effective_method_id = train_mod if not fallback_notes else f"{train_mod}__pyGMM_fallback"
    preds_aligned = np.asarray(preds_perm, dtype=np.int64)[inv_ord]
    tsv_path = out_path / "clustering.tsv"
    pd.Series(preds_aligned, name="region").to_csv(tsv_path, sep="\t", index=False)

    return {
        "effective_method_id": effective_method_id,
        "fallback_notes": fallback_notes,
        "clustering_tsv": str(tsv_path),
        "pred_labels": preds_aligned,
        "embedding": embedding[inv_ord],
        "config": {
            "preprocess": preprocess_mod,
            "adjacency": adjacency_mod,
            "train": train_mod,
            "clust": clust_mod,
            "arm": arm,
            "sedr_intervention_mode": sedr_intervention_mode if train_mod == "SEDR" else "not_applicable",
            "null_seed": null_seed,
            "cell_order_seed": cell_order_seed,
        },
    }

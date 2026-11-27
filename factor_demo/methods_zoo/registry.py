"""Registry of modular spatial domain methods available in the Demo."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict

from factor_demo.methods_zoo import (
    banksy_aug_domain,
    expr_baseline,
    gat_ae_domain,
    gcn_dgi_domain,
)

METHODS_REGISTRY: Dict[str, Dict[str, object]] = {
    "GCN_DGI": {
        "metadata": gcn_dgi_domain.METHOD_METADATA,
        "runner": gcn_dgi_domain.run_gcn_dgi,
        "source_file": str(Path(gcn_dgi_domain.__file__).resolve()),
    },
    "GAT_AE": {
        "metadata": gat_ae_domain.METHOD_METADATA,
        "runner": gat_ae_domain.run_gat_ae,
        "source_file": str(Path(gat_ae_domain.__file__).resolve()),
    },
    "BANKSY_Concat": {
        "metadata": banksy_aug_domain.METHOD_METADATA,
        "runner": banksy_aug_domain.run_banksy_concat,
        "source_file": str(Path(banksy_aug_domain.__file__).resolve()),
    },
    "Expr_Smooth_Baseline": {
        "metadata": expr_baseline.METHOD_METADATA,
        "runner": expr_baseline.run_expr_baseline,
        "source_file": str(Path(expr_baseline.__file__).resolve()),
    },
}

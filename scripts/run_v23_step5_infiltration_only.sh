#!/usr/bin/env bash
# Fast Milestone 2B-only runner (~3-4 min on RTX 4090):
# Keeps already-completed Milestone 1 & Milestone 2A (`osmFISH_Codeluppi`) intact and appends
# Mouse-Brain snRNA-seq (`subset_anndata.h5ad.gz`, `ClusterNm`) Type II infiltration sweeps (0.0, 0.25, 0.50),
# then commits & pushes final results directly to GitHub!

set -euo pipefail
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

mkdir -p logs_server outputs
LOG_FILE="$REPO_DIR/logs_server/v23_step5.log"
STATUS_MD="$REPO_DIR/outputs/SERVER_RUN_STATUS.md"

push_status() {
    local stage_msg="$1"
    {
        echo "# Server Execution Status (${stage_msg})"
        echo ""
        echo "- **Timestamp**: \`$(date '+%Y-%m-%d %H:%M:%S %Z')\`"
        echo "- **Host**: \`$(hostname)\`"
        echo "- **Git Commit**: \`$(git rev-parse --short HEAD)\`"
        echo ""
        echo "## Latest Log Tail (\`logs_server/v23_step5.log\`)"
        echo '```text'
        tail -n 95 "$LOG_FILE" 2>/dev/null || true
        echo '```'
    } > "$STATUS_MD"

    git add "$STATUS_MD"
    for jf in outputs/canzar_live_chimera/*.json; do
        [ -f "$jf" ] && git add "$jf"
    done

    if ! git diff --cached --quiet; then
        git commit -m "$stage_msg" || true
        git push origin main || true
    fi
}

on_exit() {
    local rc=$?
    if [ $rc -ne 0 ]; then
        echo "[ERROR] Step 5 exited with code $rc at $(date)"
        push_status "chore(server-status): report error (exit $rc) from Step 5 infiltration run"
    fi
}
trap on_exit EXIT

BEST_GPU=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits 2>/dev/null | sort -t',' -k2 -nr | head -n1 | cut -d',' -f1 | tr -d ' ')
export CUDA_VISIBLE_DEVICES="${BEST_GPU:-7}"
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4

PY_GRAPHST="/home/dugaoyuan/.conda/envs/GraphST/bin/python"
PY_SEDR="/home/dugaoyuan/.conda/envs/SEDR/bin/python"

echo "=== [Milestone 2B] Running Mouse-Brain snRNA-seq (ClusterNm) Type II Infiltration Sweeps (0.0, 0.25, 0.50) on GPU $CUDA_VISIBLE_DEVICES ==="

# 1. Run SEDR (Python 3.11) FIRST so Canzar's unmodified `assign_counts.py` (`dict | dict`, PEP 584)
#    runs natively on Python 3.11 and caches bit-identical `sim_slices/*.csv` on disk!
if [ -d "/home/dugaoyuan/.conda/envs/SEDR/lib/R" ]; then
    export R_HOME="/home/dugaoyuan/.conda/envs/SEDR/lib/R"
    export PATH="/home/dugaoyuan/.conda/envs/SEDR/bin:$PATH"
fi
"$PY_SEDR" scripts/run_canzar_chimera_experiment.py \
    --simulate-infiltration \
    --doses 0.0 0.25 0.50 \
    --methods SEDR \
    --sedr-mode encoder_only \
    --skip-native-gate \
    --clust mclust \
    --append \
    --output-dir outputs/canzar_live_chimera

"$PY_SEDR" scripts/run_canzar_chimera_experiment.py \
    --simulate-infiltration \
    --doses 0.0 0.25 0.50 \
    --methods SEDR \
    --sedr-mode full_mechanism \
    --skip-native-gate \
    --clust mclust \
    --append \
    --output-dir outputs/canzar_live_chimera

# 2. Run GraphST + scanpy_nonspatial (Python 3.8), reusing the exact cached `sim_slices/*.csv`!
if [ -d "/home/dugaoyuan/.conda/envs/GraphST/lib/R" ]; then
    export R_HOME="/home/dugaoyuan/.conda/envs/GraphST/lib/R"
    export PATH="/home/dugaoyuan/.conda/envs/GraphST/bin:$PATH"
fi
"$PY_GRAPHST" scripts/run_canzar_chimera_experiment.py \
    --simulate-infiltration \
    --doses 0.0 0.25 0.50 \
    --methods GraphST scanpy_nonspatial \
    --skip-native-gate \
    --clust mclust \
    --append \
    --output-dir outputs/canzar_live_chimera

echo "[Completed] MILESTONE_2B_INFILTRATION_FINISHED_OK at $(date)"
push_status "results(v2.3-milestone2b): complete mouse-brain Type II infiltration 4-arm counterfactuals"

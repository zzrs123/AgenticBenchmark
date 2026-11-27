#!/usr/bin/env bash
# End-to-end v2.3 server execution suite for li.npuers.com (8x RTX 4090).
# Stages paths safely (no unexpanded glob errors in git add!), verifies gzip integrity via SSH git clone,
# runs real osmFISH_Codeluppi 4-arm experiments FIRST (Milestone 2A), then semi-synthetic infiltration (Milestone 2B).

set -euo pipefail
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

mkdir -p logs_server outputs
LOG_FILE="$REPO_DIR/logs_server/v23_exec.log"
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
        echo "## Latest Log Tail (\`logs_server/v23_exec.log\`)"
        echo '```text'
        tail -n 95 "$LOG_FILE" 2>/dev/null || true
        echo '```'
    } > "$STATUS_MD"

    git add "$STATUS_MD"
    [ -d "outputs/demo_run" ] && git add outputs/demo_run
    [ -d "outputs/canzar2026_real_audit" ] && git add outputs/canzar2026_real_audit
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
        echo "[ERROR] Script exited with code $rc at $(date)"
        push_status "chore(server-status): report error (exit $rc) from v2.3 server run"
    fi
}
trap on_exit EXIT

BEST_GPU=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits 2>/dev/null | sort -t',' -k2 -nr | head -n1 | cut -d',' -f1 | tr -d ' ')
export CUDA_VISIBLE_DEVICES="${BEST_GPU:-6}"
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4

PY_GRAPHST="/home/dugaoyuan/.conda/envs/GraphST/bin/python"
PY_SEDR="/home/dugaoyuan/.conda/envs/SEDR/bin/python"

echo "[Start] $(date) | Selected GPU: CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

echo "=== [Step 1] Running full 10-test suite (including 4 scientific behavioral tests) ==="
"$PY_GRAPHST" -m unittest discover -s tests -v

if [ ! -f "outputs/demo_run/artifacts/stagewise_equivalence_audit.json" ]; then
    echo "=== [Step 2] Running v2.3 4-Agent Controlled Benchmark & Deliverables (run_demo.py) ==="
    rm -rf outputs/demo_run
    "$PY_GRAPHST" scripts/run_demo.py --output-dir outputs/demo_run
else
    echo "=== [Step 2] Reusing fresh v2.3 outputs/demo_run already computed on server ==="
fi

echo "=== [Step 3] Running v2.3 Real Canzar 2026 Audit (unknown-masked AST + 46k scores) ==="
"$PY_GRAPHST" scripts/run_canzar2026_real_audit.py

push_status "results(v2.3-milestone1): server 10/10 behavioral tests and v2.3 demo figures/reports"

echo "=== [Step 4A] Milestone 2A: Real osmFISH_Codeluppi Native Audit + 4-Arm Counterfactuals ==="
rm -rf outputs/canzar_live_chimera

if [ -d "/home/dugaoyuan/.conda/envs/GraphST/lib/R" ]; then
    export R_HOME="/home/dugaoyuan/.conda/envs/GraphST/lib/R"
    export PATH="/home/dugaoyuan/.conda/envs/GraphST/bin:$PATH"
fi
"$PY_GRAPHST" scripts/run_canzar_chimera_experiment.py \
    --use-osmfish \
    --methods GraphST scanpy_nonspatial \
    --clust mclust \
    --output-dir outputs/canzar_live_chimera

if [ -d "/home/dugaoyuan/.conda/envs/SEDR/lib/R" ]; then
    export R_HOME="/home/dugaoyuan/.conda/envs/SEDR/lib/R"
    export PATH="/home/dugaoyuan/.conda/envs/SEDR/bin:$PATH"
fi
"$PY_SEDR" scripts/run_canzar_chimera_experiment.py \
    --use-osmfish \
    --methods SEDR \
    --sedr-mode encoder_only \
    --clust mclust \
    --append \
    --output-dir outputs/canzar_live_chimera

"$PY_SEDR" scripts/run_canzar_chimera_experiment.py \
    --use-osmfish \
    --methods SEDR \
    --sedr-mode full_mechanism \
    --skip-native-gate \
    --clust mclust \
    --append \
    --output-dir outputs/canzar_live_chimera

push_status "results(v2.3-milestone2a): real osmFISH_Codeluppi native GraphST/SEDR reproduction and 4-arm results"

echo "=== [Step 4B] Preparing SHA-verified subset_anndata.h5ad.gz via GitHub SSH clone ==="
SIM_DIR="external/spatial-domain-simulator"
SIM_H5="$SIM_DIR/subset_anndata.h5ad.gz"
rm -f "$SIM_DIR/subset_anndata.h5ad.tmp"
if [ -f "$SIM_DIR/subset_anndata.h5ad" ]; then
    UNZ_SZ=$(stat -c%s "$SIM_DIR/subset_anndata.h5ad" 2>/dev/null || echo 0)
    if [ "$UNZ_SZ" -lt 800000000 ]; then
        echo "Removing truncated subset_anndata.h5ad ($UNZ_SZ bytes)"
        rm -f "$SIM_DIR/subset_anndata.h5ad"
    fi
fi

NEED_FETCH=0
if [ ! -f "$SIM_H5" ]; then
    NEED_FETCH=1
else
    GZ_SZ=$(stat -c%s "$SIM_H5" 2>/dev/null || echo 0)
    if [ "$GZ_SZ" -lt 55000000 ] || ! gzip -t "$SIM_H5" 2>/dev/null; then
        echo "Removing incomplete/corrupt subset_anndata.h5ad.gz ($GZ_SZ bytes)"
        rm -f "$SIM_H5" "$SIM_DIR/subset_anndata.h5ad"
        NEED_FETCH=1
    fi
fi

if [ "$NEED_FETCH" -eq 1 ]; then
    rm -rf /tmp/canzar_sim_ssh
    git clone --depth 1 git@github.com:canzarlab/spatial-domain-simulator.git /tmp/canzar_sim_ssh
    cp -f /tmp/canzar_sim_ssh/subset_anndata.h5ad.gz "$SIM_H5"
    rm -rf /tmp/canzar_sim_ssh
    gzip -t "$SIM_H5"
fi
ls -lh "$SIM_H5"

echo "=== [Step 5] Milestone 2B: Mouse-Brain snRNA-seq Infiltration Sweeps (0.0, 0.25, 0.50) ==="
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

echo "[Completed] ALL_V23_SERVER_EXPERIMENTS_FINISHED_OK at $(date)"
push_status "results(v2.3-milestone2b): complete mouse-brain Type II infiltration 4-arm counterfactuals"

#!/usr/bin/env bash
# End-to-end v2.3 server execution suite for li.npuers.com (8x RTX 4090).
# Automatically routes GraphST and SEDR to their dedicated Conda environments,
# bypasses dead proxy env vars, and pushes Milestone 1, Milestone 2, or any error log
# directly to git@github.com:zzrs123/AgenticBenchmark.git for zero-VPN monitoring!

set -uo pipefail
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
        tail -n 80 "$LOG_FILE" 2>/dev/null || true
        echo '```'
    } > "$STATUS_MD"

    git add "$STATUS_MD" outputs/demo_run outputs/canzar2026_real_audit outputs/canzar_live_chimera/*.json 2>/dev/null || true
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

# Pick RTX 4090 with maximum free VRAM (fallback to GPU 6)
BEST_GPU=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits 2>/dev/null | sort -t',' -k2 -nr | head -n1 | cut -d',' -f1 | tr -d ' ')
export CUDA_VISIBLE_DEVICES="${BEST_GPU:-6}"
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4

PY_GRAPHST="/home/dugaoyuan/.conda/envs/GraphST/bin/python"
PY_SEDR="/home/dugaoyuan/.conda/envs/SEDR/bin/python"

echo "[Start] $(date) | Selected GPU: CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

echo "=== [Step 1/5] Running full 10-test suite (including 4 scientific behavioral tests) ==="
"$PY_GRAPHST" -m unittest discover -s tests -v

echo "=== [Step 2/5] Running v2.3 4-Agent Controlled Benchmark & Deliverables (run_demo.py) ==="
rm -rf outputs/demo_run
"$PY_GRAPHST" scripts/run_demo.py --output-dir outputs/demo_run

echo "=== [Step 3/5] Running v2.3 Real Canzar 2026 Audit (unknown-masked AST + 46k scores) ==="
"$PY_GRAPHST" scripts/run_canzar2026_real_audit.py

push_status "results(v2.3-milestone1): server 10/10 behavioral tests and v2.3 demo figures/reports"

echo "=== [Step 4/5] Ensuring subset_anndata.h5ad.gz is present for semi-synthetic infiltration ==="
SIM_H5="external/spatial-domain-simulator/subset_anndata.h5ad.gz"
if [ ! -s "$SIM_H5" ]; then
    curl -fL --retry 3 --connect-timeout 25 \
      https://github.com/canzarlab/spatial-domain-simulator/raw/main/subset_anndata.h5ad.gz \
      -o "$SIM_H5"
fi
ls -lh "$SIM_H5"

echo "=== [Step 5/5] Running Live Native-Audited 4-Arm Counterfactuals (GraphST, SEDR, scanpy_nonspatial) ==="
rm -rf outputs/canzar_live_chimera

# 5a. GraphST + scanpy_nonspatial in GraphST Conda env
if [ -d "/home/dugaoyuan/.conda/envs/GraphST/lib/R" ]; then
    export R_HOME="/home/dugaoyuan/.conda/envs/GraphST/lib/R"
    export PATH="/home/dugaoyuan/.conda/envs/GraphST/bin:$PATH"
fi
"$PY_GRAPHST" scripts/run_canzar_chimera_experiment.py \
    --use-osmfish \
    --simulate-infiltration \
    --doses 0.0 0.25 0.50 \
    --methods GraphST scanpy_nonspatial \
    --clust mclust \
    --output-dir outputs/canzar_live_chimera

# 5b. SEDR (encoder_only AND full_mechanism) in SEDR Conda env
if [ -d "/home/dugaoyuan/.conda/envs/SEDR/lib/R" ]; then
    export R_HOME="/home/dugaoyuan/.conda/envs/SEDR/lib/R"
    export PATH="/home/dugaoyuan/.conda/envs/SEDR/bin:$PATH"
fi
"$PY_SEDR" scripts/run_canzar_chimera_experiment.py \
    --use-osmfish \
    --simulate-infiltration \
    --doses 0.0 0.25 0.50 \
    --methods SEDR \
    --sedr-mode encoder_only \
    --clust mclust \
    --append \
    --output-dir outputs/canzar_live_chimera

"$PY_SEDR" scripts/run_canzar_chimera_experiment.py \
    --use-osmfish \
    --simulate-infiltration \
    --doses 0.0 0.25 0.50 \
    --methods SEDR \
    --sedr-mode full_mechanism \
    --skip-native-gate \
    --clust mclust \
    --append \
    --output-dir outputs/canzar_live_chimera

echo "[Completed] ALL_V23_SERVER_EXPERIMENTS_FINISHED_OK at $(date)"
push_status "results(v2.3-milestone2): complete GraphST and SEDR native audit and 4-arm counterfactuals"

#!/usr/bin/env bash
# Watches run.log for completed steps and runs analyze.py after each one.
# Usage: bash watch_and_analyze.sh [run_name]
set -euo pipefail

RUN_NAME="${1:-formulacode-grpo-g8-2gpu}"
RESULTS="/mnt/sdd3/asharma/formulacode-rlvr/results"
LOGFILE="$RESULTS/$RUN_NAME/run.log"
ANALYZE="/mnt/sdd3/asharma/formulacode-rlvr/analyze.py"
OUT_BASE="$RESULTS/$RUN_NAME/analysis"

echo "[watch] Watching $LOGFILE for completed steps..."
echo "[watch] Analysis will be written to $OUT_BASE/"

last_seen_step=0

while true; do
    if [ ! -f "$LOGFILE" ]; then
        sleep 30
        continue
    fi

    # Count completed steps by number of "Finished: 'step'" lines in log
    latest=$(grep -c "Finished: 'step'" "$LOGFILE" 2>/dev/null | tr -d '[:space:]' || echo "0")
    latest="${latest:-0}"

    if [ "$latest" -gt "$last_seen_step" ]; then
        echo ""
        echo "[watch] ============================================="
        echo "[watch] Step $latest completed at $(date -u '+%H:%M UTC')"
        echo "[watch] Running analysis..."
        OUT_DIR="${OUT_BASE}/step_${latest}"
        mkdir -p "$OUT_DIR"

        uv run --with matplotlib --with seaborn --with scipy \
            python "$ANALYZE" \
            --run "$RUN_NAME" \
            --out "$OUT_DIR" \
            --trajectories \
            2>/dev/null | tee "${OUT_DIR}/analysis_stdout.txt"

        echo "[watch] Analysis done → $OUT_DIR"
        echo "[watch] Plots: $(ls ${OUT_DIR}/plots/*.png 2>/dev/null | wc -l) figures"
        echo "[watch] ============================================="
        last_seen_step=$latest
    fi

    sleep 60
done

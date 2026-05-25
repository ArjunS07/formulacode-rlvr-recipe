#!/usr/bin/env bash
# FormulaCode RLVR monitor — safe to run repeatedly (read-only).
# Outputs a structured summary Claude can parse.
#
# Usage:  bash /mnt/sdd3/asharma/formulacode-rlvr/monitor.sh

RESULTS="/mnt/sdd3/asharma/formulacode-rlvr/results"
RUN_NAME="${FORMULACODE_RUN_NAME:-formulacode-grpo-g8-2gpu}"
LOGFILE="$RESULTS/$RUN_NAME/run.log"
CKPTS_DIR="$RESULTS/$RUN_NAME/ckpts"
TRIALS_DIR="$RESULTS/trials"

echo "====== FormulaCode RLVR Monitor — $(date) ======"
echo "Run: $RUN_NAME"

echo ""
echo "--- Training process ---"
if pgrep -f "main_formulacode" > /dev/null 2>&1; then
    echo "STATUS: RUNNING"
    pgrep -af "main_formulacode" | head -2
else
    echo "STATUS: NOT RUNNING"
fi

echo ""
echo "--- GPU memory ---"
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu \
  --format=csv,noheader | awk -F', ' '{printf "  GPU%s: %s / %s  util=%s\n",$1,$2,$3,$4}'

echo ""
echo "--- Checkpoints ---"
if [ -d "$CKPTS_DIR" ]; then
    N=$(ls -d "$CKPTS_DIR"/global_step_* 2>/dev/null | wc -l)
    echo "Checkpoints saved: $N"
    ls -dt "$CKPTS_DIR"/global_step_* 2>/dev/null | head -3
else
    echo "No checkpoint directory yet"
fi

echo ""
echo "--- Reward signal (all steps) ---"
if [ -f "$LOGFILE" ]; then
    grep -E "avg_reward|avg_final_reward|generate/avg|reward" "$LOGFILE" 2>/dev/null \
      | grep -v "^===" | tail -20
else
    echo "Log not found: $LOGFILE"
fi

echo ""
echo "--- Active Docker containers (Harbor trials) ---"
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.RunningFor}}" 2>/dev/null \
  | grep -v "^NAMES" | head -15 || echo "(docker ps failed)"

echo ""
echo "--- Recent trial dirs ---"
if [ -d "$TRIALS_DIR" ]; then
    ls -dt "$TRIALS_DIR"/*/ 2>/dev/null | head -6
    echo "Total: $(ls -d "$TRIALS_DIR"/*/ 2>/dev/null | wc -l) trial dirs"
else
    echo "No trials directory yet"
fi

echo ""
echo "--- Last 30 log lines ---"
if [ -f "$LOGFILE" ]; then
    tail -30 "$LOGFILE"
else
    echo "(no log yet)"
fi

echo ""
echo "====== End monitor ======"

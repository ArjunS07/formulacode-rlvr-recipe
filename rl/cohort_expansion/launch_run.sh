#!/usr/bin/env bash
# launch_run.sh (ml-login7) — start the prod GRPO supervisor with the staged 30-task TRAIN_DATA.
# All-4-GPU, fresh RUN_NAME, open-ended (stop via the STOP file). sg preserves the exported env.
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export RESULTS_ROOT="$HOME/results" UV_CACHE_DIR="$HOME/uv-cache"
export NUM_GPUS=4 MINIBATCH_TASKS=5 CUDA_VISIBLE_DEVICES=0,1,2,3
export RUN_NAME="fc-grpo-9b-qwen-prod-mll7-v2"
export TRAIN_DATA="$(cat "$HOME/results/launch_train_data.txt")"
if [ -z "${TRAIN_DATA//[[:space:]]/}" ] || [ "$TRAIN_DATA" = "[]" ]; then
  echo "[launch_run] FATAL: empty TRAIN_DATA"; exit 1
fi
cd "$HOME/skyrl-formulacode"
echo "[launch_run] $(date -u +%FT%TZ) RUN_NAME=$RUN_NAME NUM_GPUS=$NUM_GPUS"
echo "[launch_run] TRAIN_DATA=$TRAIN_DATA"
exec sg docker -c "bash examples/train_integrations/harbor/formulacode/run_prod_supervisor.sh"

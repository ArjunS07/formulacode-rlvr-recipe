#!/usr/bin/env bash
# FormulaCode RLVR — 1-epoch probe with group size 2.
# Purpose: validate end-to-end reward signal before scaling up group size.
#
# Run from formulacode-rlvr/SkyRL/:
#   CUDA_VISIBLE_DEVICES=0,1,4,5 bash examples/train_integrations/harbor/formulacode/run_g2.sh
set -euo pipefail

# export WANDB_API_KEY=your_key_here

#-----------------------
# Task
#-----------------------
TASK_PATH="/mnt/sdd3/asharma/harbor-tasks-may18/optuna_optuna_4540"
TRAIN_DATA="['$TASK_PATH']"

#-----------------------
# Model
#-----------------------
MODEL_PATH="Qwen/Qwen3.5-4B"
SERVED_MODEL_NAME="qwen3.5-4b"

#-----------------------
# Directories
#-----------------------
RUN_NAME="formulacode-grpo-g2-e1"
STORAGE_ROOT="/mnt/sdd3/asharma/formulacode-rlvr/results/$RUN_NAME"
TRIALS_DIR="/mnt/sdd3/asharma/formulacode-rlvr/results/trials"
CKPTS_DIR="$STORAGE_ROOT/ckpts"
EXPORTS_DIR="$STORAGE_ROOT/exports"
LOG_DIR="$STORAGE_ROOT/logs"
LOGFILE="$STORAGE_ROOT/run.log"
mkdir -p "$CKPTS_DIR" "$EXPORTS_DIR" "$LOG_DIR" "$TRIALS_DIR"

#-----------------------
# Training hyperparams
#-----------------------
MAX_MODEL_LEN=32768
N_SAMPLES_PER_PROMPT=2       # group size 2: each step generates 2 trajectories
MINI_BATCH_SIZE=1            # 1 unique task per batch
LOSS_REDUCTION="token_mean"
GRPO_NORM_BY_STD=false       # avoid div/0 when both rewards are equal
USE_KL_LOSS=false
APPLY_OVERLONG_FILTERING=true

#-----------------------
# Infrastructure
#-----------------------
NUM_POLICY_GPUS=2
NUM_INFERENCE_ENGINES=2      # dp_size=2: 1 task * 2 samples / 2 GPUs = 1 per GPU
TP_SIZE=1
HTTP_PORT=8123
ENABLE_RATE_LIMITING=true
TRAJECTORIES_PER_SECOND=1.0  # conservative: Docker build is the bottleneck
MAX_CONCURRENCY=2            # match group size

CHAT_TEMPLATE_PATH="/mnt/sdd3/asharma/formulacode-rlvr/SkyRL/skyrl/train/utils/templates/qwen3_acc_thinking.jinja2"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== FormulaCode RLVR g2/e1 starting at $(date) ===" | tee -a "$LOGFILE"
echo "Task: $TASK_PATH" | tee -a "$LOGFILE"
echo "Model: $MODEL_PATH" | tee -a "$LOGFILE"
echo "Group size: $N_SAMPLES_PER_PROMPT  Epochs: 1" | tee -a "$LOGFILE"
echo "GPUs: ${CUDA_VISIBLE_DEVICES:-unset}" | tee -a "$LOGFILE"

_SKYRL_USE_NEW_INFERENCE=0 uv run --isolated --extra fsdp --extra harbor \
  -m examples.train_integrations.harbor.formulacode.entrypoints.main_formulacode \
  data.train_data="$TRAIN_DATA" \
  trainer.policy.model.path="$MODEL_PATH" \
  generator.inference_engine.served_model_name="$SERVED_MODEL_NAME" \
  harbor_trial_config.trials_dir="$TRIALS_DIR" \
  trainer.export_path="$EXPORTS_DIR" \
  trainer.ckpt_path="$CKPTS_DIR" \
  trainer.log_path="$LOG_DIR" \
  trainer.algorithm.advantage_estimator=grpo \
  trainer.algorithm.loss_reduction="$LOSS_REDUCTION" \
  trainer.algorithm.grpo_norm_by_std="$GRPO_NORM_BY_STD" \
  trainer.algorithm.use_kl_loss="$USE_KL_LOSS" \
  trainer.algorithm.max_seq_len=24576 \
  trainer.placement.colocate_all=true \
  trainer.strategy=fsdp \
  trainer.placement.policy_num_nodes=1 \
  trainer.placement.ref_num_nodes=1 \
  trainer.placement.policy_num_gpus_per_node="$NUM_POLICY_GPUS" \
  trainer.placement.ref_num_gpus_per_node="$NUM_POLICY_GPUS" \
  generator.inference_engine.num_engines="$NUM_INFERENCE_ENGINES" \
  generator.inference_engine.tensor_parallel_size="$TP_SIZE" \
  generator.inference_engine.engine_init_kwargs.max_model_len="$MAX_MODEL_LEN" \
  generator.inference_engine.engine_init_kwargs.chat_template="$CHAT_TEMPLATE_PATH" \
  generator.inference_engine.engine_init_kwargs.enable_log_requests=false \
  generator.inference_engine.max_num_seqs=32 \
  generator.inference_engine.gpu_memory_utilization=0.4 \
  generator.inference_engine.backend=vllm \
  generator.inference_engine.run_engines_locally=true \
  generator.inference_engine.weight_sync_backend=gloo \
  generator.inference_engine.async_engine=true \
  generator.inference_engine.enforce_eager=false \
  generator.inference_engine.enable_http_endpoint=true \
  generator.inference_engine.http_endpoint_host=127.0.0.1 \
  generator.inference_engine.http_endpoint_port="$HTTP_PORT" \
  trainer.epochs=1 \
  trainer.train_batch_size="$MINI_BATCH_SIZE" \
  trainer.policy_mini_batch_size="$MINI_BATCH_SIZE" \
  trainer.micro_forward_batch_size_per_gpu=1 \
  trainer.micro_train_batch_size_per_gpu=1 \
  trainer.update_epochs_per_batch=1 \
  trainer.ckpt_interval=1 \
  trainer.hf_save_interval=1 \
  trainer.max_ckpts_to_keep=5 \
  trainer.eval_interval=0 \
  trainer.eval_before_train=false \
  trainer.use_sample_packing=false \
  trainer.policy.fsdp_config.cpu_offload=false \
  trainer.ref.fsdp_config.cpu_offload=false \
  trainer.policy.optimizer_config.lr=1.0e-6 \
  generator.step_wise_trajectories=true \
  generator.merge_stepwise_output=true \
  generator.n_samples_per_prompt="$N_SAMPLES_PER_PROMPT" \
  generator.apply_overlong_filtering="$APPLY_OVERLONG_FILTERING" \
  generator.batched=false \
  generator.rate_limit.enabled="$ENABLE_RATE_LIMITING" \
  generator.rate_limit.trajectories_per_second="$TRAJECTORIES_PER_SECOND" \
  generator.rate_limit.max_concurrency="$MAX_CONCURRENCY" \
  trainer.logger=console \
  trainer.project_name=formulacode \
  trainer.run_name="$RUN_NAME" \
  trainer.resume_mode=none \
  2>&1 | tee -a "$LOGFILE"

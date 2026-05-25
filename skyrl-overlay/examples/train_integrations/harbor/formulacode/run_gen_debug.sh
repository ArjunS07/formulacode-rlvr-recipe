#!/usr/bin/env bash
# FormulaCode gen-debug — run 2 trials without any gradient update.
# Use this FIRST to verify Docker containers start, terminus-2 connects to
# the vLLM, and rewards come back non-zero.
#
# Run from formulacode-rlvr/SkyRL/:
#   CUDA_VISIBLE_DEVICES=0,1,4,5 bash examples/train_integrations/harbor/formulacode/run_gen_debug.sh
set -euo pipefail

TASK_PATH="/mnt/sdd3/asharma/harbor-tasks-may18/python-hyper_h11_34"
TRAIN_DATA="['$TASK_PATH']"
MODEL_PATH="Qwen/Qwen3.5-4B"
SERVED_MODEL_NAME="qwen3.5-4b"
MAX_MODEL_LEN=131072
HTTP_PORT=8123
NUM_GPUS=4
CHAT_TEMPLATE_PATH="/mnt/sdd3/asharma/formulacode-rlvr/SkyRL/skyrl/train/utils/templates/qwen3_acc_thinking.jinja2"

LOGFILE="/mnt/sdd3/asharma/formulacode-rlvr/results/gen_debug.log"
echo "=== Gen-debug starting at $(date) ===" | tee -a "$LOGFILE"

_SKYRL_USE_NEW_INFERENCE=0 uv run --isolated --extra fsdp --extra harbor \
  -m examples.train_integrations.harbor.formulacode.entrypoints.main_formulacode_generate \
  data.train_data="$TRAIN_DATA" \
  data.val_data="$TRAIN_DATA" \
  trainer.policy.model.path="$MODEL_PATH" \
  generator.inference_engine.served_model_name="$SERVED_MODEL_NAME" \
  harbor_trial_config.trials_dir="/mnt/sdd3/asharma/formulacode-rlvr/results/trials" \
  generator.inference_engine.num_engines="$NUM_GPUS" \
  generator.inference_engine.tensor_parallel_size=1 \
  generator.inference_engine.engine_init_kwargs.max_model_len="$MAX_MODEL_LEN" \
  generator.inference_engine.engine_init_kwargs.chat_template="$CHAT_TEMPLATE_PATH" \
  generator.inference_engine.engine_init_kwargs.enable_prefix_caching=true \
  generator.inference_engine.engine_init_kwargs.enable_log_requests=false \
  generator.inference_engine.gpu_memory_utilization=0.7 \
  generator.inference_engine.backend=vllm \
  generator.inference_engine.run_engines_locally=true \
  generator.inference_engine.weight_sync_backend=nccl \
  generator.inference_engine.async_engine=true \
  generator.inference_engine.enforce_eager=false \
  generator.inference_engine.enable_http_endpoint=true \
  generator.inference_engine.http_endpoint_host=127.0.0.1 \
  generator.inference_engine.http_endpoint_port="$HTTP_PORT" \
  generator.sampling_params.max_generate_length=4096 \
  trainer.algorithm.max_seq_len="$MAX_MODEL_LEN" \
  trainer.algorithm.advantage_estimator=grpo \
  trainer.placement.colocate_all=false \
  trainer.placement.policy_num_gpus_per_node="$NUM_GPUS" \
  trainer.placement.ref_num_gpus_per_node="$NUM_GPUS" \
  trainer.train_batch_size=1 \
  trainer.policy_mini_batch_size=1 \
  trainer.logger=console \
  generator.rate_limit.enabled=true \
  generator.rate_limit.trajectories_per_second=2.0 \
  generator.rate_limit.max_concurrency=4 \
  harbor_trial_config.agent.override_timeout_sec=600 \
  2>&1 | tee -a "$LOGFILE"

echo "=== Gen-debug done at $(date) ===" | tee -a "$LOGFILE"

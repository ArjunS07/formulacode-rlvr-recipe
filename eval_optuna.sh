#!/usr/bin/env bash
# Eval: run N optuna trials with trained model and N with base model, compare.
# Usage: bash eval_optuna.sh [N_TRAINED] [N_BASE]
# Requires: vLLM serving trained model on :8124 (trained) and base on :8125 (base)
set -euo pipefail

N_TRAINED="${1:-8}"
N_BASE="${2:-8}"
TASK_PATH="/home/arjun/harbor-tasks-may18/optuna_optuna_4540"
TRIALS_DIR="/home/arjun/formulacode-rlvr-recipe/results/optuna-eval"
SKYRL_DIR="/home/arjun/formulacode-rlvr-recipe/SkyRL"

mkdir -p "$TRIALS_DIR"

echo "=== Optuna Eval: $N_TRAINED trained + $N_BASE base trials ==="
echo "Task: $TASK_PATH"
echo "Results: $TRIALS_DIR"
echo ""

run_trials() {
    local model_name="$1"
    local api_base="$2"
    local n="$3"
    local label="$4"

    echo "[$(date -u +%H:%M:%S)] Starting $n trials with $label ($model_name @ $api_base)"

    for i in $(seq 1 "$n"); do
        trial_id="${label}_$(openssl rand -hex 4)"
        trial_dir="$TRIALS_DIR/${trial_id}"
        mkdir -p "$trial_dir/agent" "$trial_dir/verifier"

        sg docker -c "cd $SKYRL_DIR && uv run --isolated --extra harbor python -c \"
import asyncio
from harbor.trial.trial import Trial
from harbor.models.trial.config import TrialConfig
import json, copy

config = {
    'trial_name': '${trial_id}',
    'task': {'path': '${TASK_PATH}'},
    'agent': {
        'name': 'terminus-2',
        'override_timeout_sec': 3600,
        'kwargs': {
            'parser_name': 'xml',
            'interleaved_thinking': True,
            'max_turns': 64,
            'suppress_max_turns_warning': True,
            'enable_summarize': False,
            'temperature': 1.0,
            'model_info': {'max_input_tokens': 55000, 'max_output_tokens': 8192,
                           'input_cost_per_token': 0.0, 'output_cost_per_token': 0.0},
            'model_name': 'hosted_vllm/${model_name}',
            'api_base': '${api_base}',
            'session_id': '${trial_id}',
            'llm_kwargs': {'timeout': 900, 'max_retries': 0, 'top_p': 1.0}
        }
    },
    'environment': {'type': 'docker', 'override_cpus': 3,
                    'override_memory_mb': 16384, 'override_storage_mb': 10240,
                    'delete': True, 'suppress_override_warnings': True},
    'verifier': {'disable': False},
    'log_dir': '${trial_dir}',
    'agent_artifacts_dir': '${trial_dir}/agent',
    'verifier_artifacts_dir': '${trial_dir}/verifier',
}

async def run():
    trial = await Trial.create(TrialConfig.model_validate(config))
    result = await trial.run()
    print('reward:', result.verifier_result.rewards if result.verifier_result else 'none')
    print('exception:', result.exception_info.exception_type if result.exception_info else 'none')

asyncio.run(run())
\"" 2>&1 | tee "$trial_dir/trial.log" || true
        echo "  Trial $i/$n done: $trial_id"
    done
}

# Run trained model trials
run_trials "qwen3.5-4b-trained" "http://127.0.0.1:8124/v1" "$N_TRAINED" "trained"

# Summary
echo ""
echo "=== Results ==="
python3 - "$TRIALS_DIR" "trained" << 'PYEOF'
import sys, json
from pathlib import Path

trials_dir = Path(sys.argv[1])
label = sys.argv[2]

rewards = []
for d in trials_dir.iterdir():
    if not d.is_dir() or not d.name.startswith(label): continue
    rj = d / 'verifier' / 'reward.json'
    if rj.exists():
        try:
            data = json.loads(rj.read_text())
            rt = d / 'verifier' / 'reward.txt'
            r = float(rt.read_text().strip()) if rt.exists() else data.get('lsv_mean_speedup', 0)
            rewards.append(r)
        except: pass

if rewards:
    import statistics
    nonzero = [r for r in rewards if r > 0]
    print(f"{label}: {len(rewards)} trials, {len(nonzero)} nonzero")
    print(f"  mean={statistics.mean(rewards):.4f}  std={statistics.stdev(rewards):.4f}")
    print(f"  rewards={[round(r,4) for r in sorted(rewards)]}")
else:
    print(f"{label}: no rewards yet")
PYEOF

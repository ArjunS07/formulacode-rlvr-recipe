#!/usr/bin/env bash
# Run a single agent trial for a Harbor task using model.formulacode.org.
# Usage: bash run_agent.sh <task_name> <task_path> [model]
# Default model: qwen3.5-4b
# Results land in: results/trials/<task_name>_agent_<id>/
set -uo pipefail

TASK_NAME="${1:?Usage: run_agent.sh <task_name> <task_path> [model]}"
TASK_PATH="${2:?}"
MODEL="${3:-qwen3.5-4b}"

TRIALS_DIR="/home/arjun/formulacode-rlvr-recipe/results/trials"
SKYRL_DIR="/home/arjun/formulacode-rlvr-recipe/SkyRL"
TOKENS_ENV="/home/arjun/datasmith/tokens.env"

# Load tokens (for LITELLM_MASTER_KEY)
set -a && source "$TOKENS_ENV" && set +a

TRIAL_ID="${TASK_NAME}_agent_$(openssl rand -hex 4)"
TRIAL_DIR="$TRIALS_DIR/$TRIAL_ID"

# Pre-create trial dir so tee can write the log before Harbor creates it internally
mkdir -p "$TRIAL_DIR"

echo "[$(date -u +%H:%M:%S)] Starting agent trial: $TRIAL_ID"
echo "  task:   $TASK_NAME @ $TASK_PATH"
echo "  model:  $MODEL @ https://model.formulacode.org/v1"
echo "  output: $TRIAL_DIR"

sg docker -c "cd $SKYRL_DIR && LITELLM_MASTER_KEY='$LITELLM_MASTER_KEY' OPENAI_API_KEY='$LITELLM_MASTER_KEY' uv run --frozen --extra harbor python -c \"
import asyncio, os
from harbor.trial.trial import Trial
from harbor.models.trial.config import TrialConfig

api_key = os.environ.get('LITELLM_MASTER_KEY', '')

config = {
    'trial_name': '${TRIAL_ID}',
    'task': {'path': '${TASK_PATH}'},
    'trials_dir': '${TRIALS_DIR}',
    'agent': {
        'name': 'terminus-2',
        'model_name': 'openai/${MODEL}',
        'override_timeout_sec': 3600,
        'kwargs': {
            'parser_name': 'xml',
            'interleaved_thinking': True,
            'max_turns': 64,
            'suppress_max_turns_warning': True,
            'enable_summarize': False,
            'temperature': 1.0,
            'model_info': {
                'max_input_tokens': 55000,
                'max_output_tokens': 8192,
                'input_cost_per_token': 0.0,
                'output_cost_per_token': 0.0,
            },
            'api_base': 'https://model.formulacode.org/v1',
            'session_id': '${TRIAL_ID}',
            'llm_kwargs': {
                'timeout': 900,
                'max_retries': 0,
                'top_p': 1.0,
                'api_key': api_key,
            },
        },
    },
    'environment': {
        'type': 'docker',
        'override_cpus': 3,
        'override_memory_mb': 16384,
        'override_storage_mb': 10240,
        'delete': True,
        'suppress_override_warnings': True,
    },
    'verifier': {'disable': False},
}

async def run():
    trial = await Trial.create(TrialConfig.model_validate(config))
    result = await trial.run()
    if result.verifier_result:
        print('reward:', result.verifier_result.rewards)
    exc = result.exception_info.exception_type if result.exception_info else 'none'
    print('exception:', exc)

asyncio.run(run())
\"" 2>&1 | tee "$TRIAL_DIR/trial.log" || true

echo "[$(date -u +%H:%M:%S)] Trial done: $TRIAL_ID"

python3 -c "
import json, sys
from pathlib import Path
rj = Path('${TRIAL_DIR}/verifier/reward.json')
if rj.exists():
    d = json.loads(rj.read_text())
    print('lsv_mean_speedup    :', d.get('lsv_mean_speedup'))
    print('tests_passed        :', d.get('tests_passed'))
    print('num_valid_benchmarks:', d.get('num_valid_benchmarks'))
    print('lsv_error           :', d.get('lsv_error'))
    rt = Path('${TRIAL_DIR}/verifier/reward.txt')
    if rt.exists(): print('reward.txt          :', rt.read_text().strip())
else:
    print('reward.json not found — check $TRIAL_DIR/trial.log', file=sys.stderr)
"

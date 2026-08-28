#!/usr/bin/env bash
# pass_k.sh — run k agent trials for a Harbor task and report pass statistics.
# Usage: bash pass_k.sh <task_name> <task_path> [k=16] [parallelism=2] [model=qwen3.5-4b]
# Results: results/passk_<task_name>_<timestamp>.jsonl  (one JSON line per trial + summary)
set -uo pipefail

TASK_NAME="${1:?Usage: pass_k.sh <task_name> <task_path> [k] [parallelism] [model]}"
TASK_PATH="${2:?}"
K="${3:-16}"
PARALLELISM="${4:-2}"
MODEL="${5:-qwen3.5-4b}"

TRIALS_DIR="/home/arjun/formulacode-rlvr-recipe/results/trials"
SKYRL_DIR="/home/arjun/formulacode-rlvr-recipe/SkyRL"
TOKENS_ENV="/home/arjun/datasmith/tokens.env"
RESULTS_DIR="/home/arjun/formulacode-rlvr-recipe/results/passk/${TASK_NAME}"
JSONL_FILE="$RESULTS_DIR/passk_${TASK_NAME}_$(date -u +%Y%m%dT%H%M%S).jsonl"

set -a && source "$TOKENS_ENV" && set +a
mkdir -p "$TRIALS_DIR" "$RESULTS_DIR"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " pass@k=$K  task=$TASK_NAME  model=$MODEL  parallel=$PARALLELISM"
echo " results → $JSONL_FILE"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# ── Single trial runner (runs in background) ─────────────────────────────────
run_one() {
    local idx="$1"
    local TRIAL_ID="${TASK_NAME}_agent_$(openssl rand -hex 4)"
    local TRIAL_DIR="$TRIALS_DIR/$TRIAL_ID"
    mkdir -p "$TRIAL_DIR"

    echo "[$(date -u +%H:%M:%S)] [$idx/$K] Starting $TRIAL_ID"

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
            'max_turns': 96,
            'suppress_max_turns_warning': True,
            'enable_summarize': False,
            'temperature': 1.0,
            'model_info': {
                'max_input_tokens': 55000, 'max_output_tokens': 8192,
                'input_cost_per_token': 0.0, 'output_cost_per_token': 0.0,
            },
            'api_base': 'https://model.formulacode.org/v1',
            'session_id': '${TRIAL_ID}',
            'llm_kwargs': {
                'timeout': 900, 'max_retries': 0, 'top_p': 1.0,
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
        'env': {
            'DATASMITH_CF_ACCESS_CLIENT_ID': '${DATASMITH_CF_ACCESS_CLIENT_ID}',
            'DATASMITH_CF_ACCESS_CLIENT_SECRET': '${DATASMITH_CF_ACCESS_CLIENT_SECRET}',
        },
    },
    'verifier': {'disable': False},
}

async def run():
    trial = await Trial.create(TrialConfig.model_validate(config))
    result = await trial.run()

asyncio.run(run())
\"" >> "$TRIAL_DIR/trial.log" 2>&1

    # Extract metrics and write JSONL record
    python3 - "$TRIAL_DIR" "$TRIAL_ID" "$idx" >> "$JSONL_FILE" << 'PYEOF'
import json, sys, math
from pathlib import Path

trial_dir, trial_id, idx = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3])
rj = trial_dir / "verifier" / "reward.json"
rt = trial_dir / "verifier" / "reward.txt"

if not rj.exists():
    rec = {"trial_id": trial_id, "idx": idx, "error": "reward.json missing"}
    print(json.dumps(rec))
    sys.exit(0)

d = json.loads(rj.read_text())
reward_txt = float(rt.read_text().strip()) if rt.exists() else None

rec = {
    "trial_id": trial_id,
    "idx": idx,
    "tests_passed": d.get("tests_passed"),
    "lsv_mean_speedup": d.get("lsv_mean_speedup"),
    "num_valid_benchmarks": d.get("num_valid_benchmarks"),
    "lsv_error": d.get("lsv_error"),
    "reward": reward_txt,
    "patch_files": d.get("patch", {}).get("files"),
    "patch_added": d.get("patch", {}).get("added_lines"),
    "patch_removed": d.get("patch", {}).get("removed_lines"),
    "per_benchmark_speedups": d.get("per_benchmark_speedups", {}),
}
print(json.dumps(rec))
PYEOF

    echo "[$(date -u +%H:%M:%S)] [$idx/$K] Done    $TRIAL_ID"
}

# ── Run k trials with bounded parallelism ────────────────────────────────────
declare -a PIDS=()

for i in $(seq 1 "$K"); do
    run_one "$i" &
    PIDS+=($!)

    # When we've hit the parallelism cap, wait for one to finish before launching more
    if (( ${#PIDS[@]} >= PARALLELISM )); then
        wait "${PIDS[0]}"
        PIDS=("${PIDS[@]:1}")
    fi
done

# Wait for remaining
for pid in "${PIDS[@]}"; do
    wait "$pid"
done

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " All $K trials complete. Computing summary..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# ── Summary ──────────────────────────────────────────────────────────────────
python3 - "$JSONL_FILE" "$K" "$TASK_NAME" << 'PYEOF'
import json, math, sys
from pathlib import Path

jsonl_path, k, task_name = Path(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
lines = [l for l in jsonl_path.read_text().splitlines() if l.strip()]
records = []
for l in lines:
    try:
        r = json.loads(l)
        if "error" not in r:
            records.append(r)
    except Exception:
        pass

n = len(records)
errors = k - n

tests_passed   = [r for r in records if r.get("tests_passed")]
speedup_gt1    = [r for r in records if (r.get("lsv_mean_speedup") or 0) > 1.0]
reward_gt1     = [r for r in records if (r.get("reward") or 0) > 1.0]
valid_bm       = [r for r in records if (r.get("num_valid_benchmarks") or 0) > 0]

lsvs    = [r["lsv_mean_speedup"] for r in records if r.get("lsv_mean_speedup") is not None]
rewards = [r["reward"] for r in records if r.get("reward") is not None]

def stats(vals):
    if not vals: return "n/a"
    s = sorted(vals)
    med = s[len(s)//2]
    return f"min={s[0]:.4f}  med={med:.4f}  max={s[-1]:.4f}"

print(f"\n{'─'*55}")
print(f"  Task:              {task_name}")
print(f"  k:                 {k}  (completed: {n}  errors: {errors})")
print(f"{'─'*55}")
print(f"  tests_passed:      {len(tests_passed)}/{n}  ({100*len(tests_passed)/n:.0f}%)" if n else "  no data")
print(f"  valid benchmarks:  {len(valid_bm)}/{n}  ({100*len(valid_bm)/n:.0f}%)" if n else "")
print(f"  lsv > 1.0:         {len(speedup_gt1)}/{n}  ({100*len(speedup_gt1)/n:.0f}%)" if n else "")
print(f"  reward > 1.0:      {len(reward_gt1)}/{n}  ({100*len(reward_gt1)/n:.0f}%)" if n else "")
print(f"{'─'*55}")
print(f"  lsv distribution:  {stats(lsvs)}")
print(f"  reward dist:       {stats(rewards)}")
print(f"{'─'*55}")

# Per-trial table
print(f"\n  {'#':>3}  {'trial_id':<28}  {'tests':>5}  {'lsv':>8}  {'reward':>8}  {'#bm':>3}")
print(f"  {'─'*3}  {'─'*28}  {'─'*5}  {'─'*8}  {'─'*8}  {'─'*3}")
for r in sorted(records, key=lambda x: x["idx"]):
    tests = "✓" if r.get("tests_passed") else "✗"
    lsv   = f"{r['lsv_mean_speedup']:.4f}" if r.get("lsv_mean_speedup") is not None else "  n/a "
    rew   = f"{r['reward']:.4f}" if r.get("reward") is not None else "  n/a "
    nbm   = str(r.get("num_valid_benchmarks") or 0)
    print(f"  {r['idx']:>3}  {r['trial_id']:<28}  {tests:>5}  {lsv:>8}  {rew:>8}  {nbm:>3}")

# Append summary as final metadata line
summary = {
    "__summary__": True,
    "k": k, "n_completed": n, "n_errors": errors,
    "pass_tests": len(tests_passed), "pass_lsv_gt1": len(speedup_gt1),
    "pass_reward_gt1": len(reward_gt1),
    "lsv_stats": stats(lsvs), "reward_stats": stats(rewards),
}
print()
with open(jsonl_path, "a") as f:
    f.write(json.dumps(summary) + "\n")
print(f"  Results written to: {jsonl_path}")
PYEOF

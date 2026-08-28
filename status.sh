#!/usr/bin/env bash
# FormulaCode RLVR — quick status report
# Usage: bash status.sh [run_name]

set -uo pipefail

RUN_NAME="${1:-formulacode-grpo-g8-2gpu}"
STORAGE_ROOT="/home/arjun/formulacode-rlvr-recipe/results/$RUN_NAME"
TRIALS_DIR="/home/arjun/formulacode-rlvr-recipe/results/trials"
LOGFILE="$STORAGE_ROOT/run.log"

strip_ansi() { sed 's/\x1b\[[0-9;]*[mABCDJKHf]//g'; }

echo "========================================"
echo " FormulaCode RLVR — $RUN_NAME"
echo " $(date)"
echo "========================================"

# --- Process ---
echo ""
echo "[ Process ]"
if pgrep -f "main_formulacode" > /dev/null 2>&1; then
    pid=$(pgrep -f "main_formulacode" | head -1)
    started=$(ps -p "$pid" -o lstart= 2>/dev/null | xargs || echo "unknown")
    echo "  RUNNING  pid=$pid  started: $started"
else
    echo "  DEAD — training process not found"
fi

# --- GPUs ---
echo ""
echo "[ GPUs 2,3 ]"
nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu \
    --format=csv,noheader 2>/dev/null \
    | grep -E "^[23]," \
    | awk -F', ' '{printf "  GPU %s: %s used, %s free, %s util\n", $1, $2, $3, $4}'

# --- Run log progress ---
echo ""
echo "[ Training Progress ]"

if [ ! -f "$LOGFILE" ]; then
    echo "  Log not found: $LOGFILE"
else
    total=$(grep -oP "Total steps: \K[0-9]+" "$LOGFILE" 2>/dev/null | tail -1 || echo "?")
    echo "  Total steps: $total"

    echo "  Step events:"
    grep -E "Started: 'step'|Finished: 'step'" "$LOGFILE" 2>/dev/null \
        | strip_ansi \
        | grep -oP "\d{2}:\d{2}:\d{2}.*" \
        | sed 's/^/    /' \
        || true

    echo "  Trajectory progress:"
    grep "Generating Trajectories:" "$LOGFILE" 2>/dev/null \
        | strip_ansi \
        | tail -1 \
        | sed 's/.*Generating Trajectories:/    /' \
        | tr -d '\r' \
        || true

    echo ""
    echo "  Completed step metrics (overall avg):"
    grep -P "avg_final_rewards:" "$LOGFILE" 2>/dev/null \
        | strip_ansi \
        | sed 's/.*skyrl_entrypoint[^)]*) //' \
        | sed 's/^/    /' \
        || true
fi

# --- Per-task reward summary ---
echo ""
echo "[ Per-Task Rewards (this run) ]"

if [ ! -f "$LOGFILE" ]; then
    echo "  Log not found"
else
    python3 - "$TRIALS_DIR" "$LOGFILE" << 'PYEOF'
import sys, json, re
from pathlib import Path

trials_dir = Path(sys.argv[1])
logfile = Path(sys.argv[2])

# Parse run start time from first timestamp in log
# Handles ISO format and `date` output: "Mon Jun  1 01:03:38 AM UTC 2026"
run_start_ts = None
try:
    from datetime import datetime
    MONTHS = {'Jan':1,'Feb':2,'Mar':3,'Apr':4,'May':5,'Jun':6,'Jul':7,'Aug':8,'Sep':9,'Oct':10,'Nov':11,'Dec':12}
    with open(logfile) as f:
        for line in f:
            m = re.search(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', line)
            if m:
                run_start_ts = datetime.fromisoformat(m.group(1)).timestamp()
                break
            m = re.search(r'starting at \w+ (\w+)\s+(\d+) (\d+):(\d+):(\d+) (AM|PM) \S+ (\d{4})', line)
            if m:
                mon, day, h, mi, s, ampm, yr = m.groups()
                h, mi, s, day, yr = int(h), int(mi), int(s), int(day), int(yr)
                if ampm == 'PM' and h != 12: h += 12
                elif ampm == 'AM' and h == 12: h = 0
                run_start_ts = datetime(yr, MONTHS.get(mon, 1), day, h, mi, s).timestamp()
                break
except Exception:
    pass

# Get tasks from the log
tasks_in_run = set()
try:
    with open(logfile) as f:
        for line in f:
            for m in re.finditer(r'harbor-tasks-[^/]+/([^\s\'\",\]]+)', line):
                tasks_in_run.add(m.group(1))
except Exception:
    pass

# trial_name -> {reward, patch_applied, tests_passed, lsv_error, num_benchmarks}
trials_data = {}
# task -> list of trial_names
task_trials = {}
# task -> [total, agent_done, reward_done]
counts = {}

for d in trials_dir.iterdir():
    if not d.is_dir():
        continue
    name = d.name
    parts = name.rsplit('__', 1)
    task = parts[0] if len(parts) == 2 else name

    if tasks_in_run and task not in tasks_in_run:
        continue

    # Skip old trials (filter by config.json mtime, or directory mtime if no config yet)
    if run_start_ts:
        cfg = d / 'config.json'
        try:
            anchor = cfg.stat().st_mtime if cfg.exists() else d.stat().st_mtime
            if anchor < run_start_ts - 5:
                continue
        except OSError:
            pass

    if task not in counts:
        counts[task] = [0, 0, 0]
        task_trials[task] = []

    counts[task][0] += 1
    task_trials[task].append(name)  # track all trials for verifier check

    # Agent done?
    if (d / 'agent' / 'trajectory.json').exists():
        counts[task][1] += 1

    # Reward done?
    reward_json = d / 'verifier' / 'reward.json'
    reward_txt = d / 'verifier' / 'reward.txt'
    if reward_json.exists():
        counts[task][2] += 1
        try:
            with open(reward_json) as f:
                rdata = json.load(f)
            reward = float(open(reward_txt).read().strip()) if reward_txt.exists() else rdata.get('lsv_mean_speedup', 0.0)
            trials_data[name] = {
                'task': task,
                'reward': reward,
                'patch_applied': rdata.get('patch', {}).get('applied', False),
                'tests_passed': rdata.get('tests_passed', None),
                'lsv_error': rdata.get('lsv_error'),
                'num_benchmarks': rdata.get('num_valid_benchmarks', 0),
                'per_benchmark_speedups': rdata.get('per_benchmark_speedups', {}),
                'max_speedup': rdata.get('max_speedup', 0),
            }
        except Exception as e:
            pass

if not counts:
    print("  No trials started yet for this run")
else:
    for task in sorted(counts):
        total, agent_done, reward_done = counts[task]
        bar_char = lambda d, t: '#' * d + '-' * (t - d)
        print(f"\n  {task}")
        print(f"    Progress: {agent_done}/{total} agent done, {reward_done}/{total} reward computed")

        if reward_done == 0:
            # Check if verifier is running (test-stdout.txt exists)
            # Count verifiers in trials already filtered for this run
            verif_count = sum(
                1 for tn in task_trials.get(task, [])
                if (trials_dir / tn / 'verifier' / 'test-stdout.txt').exists()
            )
            if verif_count > 0:
                print(f"    Verifier running for {verif_count} trial(s)...")
            continue

        # Get reward stats
        task_rewards = [trials_data[tn]['reward'] for tn in task_trials.get(task, []) if tn in trials_data]
        if not task_rewards:
            continue

        nonzero = [r for r in task_rewards if r > 0]
        speedups = [r for r in task_rewards if r > 1.0]
        slowdowns = [r for r in task_rewards if 0 < r < 1.0]
        zeros = [r for r in task_rewards if r == 0.0]

        avg = sum(task_rewards) / len(task_rewards)
        avg_nz = sum(nonzero) / len(nonzero) if nonzero else None
        mx = max(task_rewards)
        mn = min(task_rewards)

        print(f"    Rewards ({reward_done} trials): avg={avg:.4f}  min={mn:.4f}  max={mx:.4f}")
        print(f"    Speedups: {len(speedups)}  Slowdowns: {len(slowdowns)}  Zero: {len(zeros)}")
        if avg_nz is not None and avg_nz != avg:
            print(f"    Nonzero avg: {avg_nz:.4f}")

        # Flags
        flags = []
        if len(zeros) > len(task_rewards) // 2:
            flags.append("WARN: majority zero rewards")
        if all(r < 1.0 for r in nonzero) if nonzero else False:
            flags.append("WARN: all nonzero are slowdowns (model making code worse)")
        if any(trials_data[tn].get('lsv_error') for tn in task_trials.get(task, []) if tn in trials_data):
            flags.append("WARN: LSV errors detected")
        if any(not trials_data[tn].get('tests_passed', True) for tn in task_trials.get(task, []) if tn in trials_data):
            flags.append("WARN: test failures detected")
        patch_rates = [trials_data[tn]['patch_applied'] for tn in task_trials.get(task, []) if tn in trials_data]
        if patch_rates and sum(patch_rates) == 0:
            flags.append("WARN: no patches applied (agent not coding)")
        for f in flags:
            print(f"    ** {f}")
PYEOF
fi

echo ""
echo "========================================"

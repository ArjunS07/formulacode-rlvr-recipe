#!/usr/bin/env bash
# oracle_all.sh — orchestrates all 4 oracle trials
# Order: h11 (solo) → pvlib + networkx (parallel) → joblib (last, ≤3 retries if lsv > 1.5)
# Results echoed live and written to results/oracle_h_values.json

set -uo pipefail

TRIALS_DIR="/home/arjun/formulacode-rlvr-recipe/results/oracle-runs"
SKYRL_DIR="/home/arjun/formulacode-rlvr-recipe/SkyRL"
RESULTS_FILE="/home/arjun/formulacode-rlvr-recipe/results/oracle_h_values.json"

mkdir -p "$TRIALS_DIR" "$(dirname "$RESULTS_FILE")"
[ -f "$RESULTS_FILE" ] || echo '{}' > "$RESULTS_FILE"

# ── Single trial runner ───────────────────────────────────────────────────────
# Writes trial_dir to /tmp/oracle_trial_dir_<task_name> on completion.
run_trial() {
    local task_name="$1"
    local task_path="$2"
    local trial_id="${task_name}_oracle_$(openssl rand -hex 4)"
    local trial_dir="$TRIALS_DIR/$trial_id"

    echo "[$(date -u +%H:%M:%S)] Starting: $trial_id"
    mkdir -p "$trial_dir"

    sg docker -c "cd $SKYRL_DIR && uv run --isolated --extra harbor python -c \"
import asyncio, os, sys
sys.path.insert(0, '/home/arjun/datasmith/src')
os.environ.setdefault('DATASMITH_LSV_ROUNDS', '5')
from harbor.trial.trial import Trial
from harbor.models.trial.config import TrialConfig

config = {
    'trial_name': '${trial_id}',
    'task': {'path': '${task_path}'},
    'trials_dir': '${TRIALS_DIR}',
    'agent': {'name': 'oracle'},
    'environment': {
        'type': 'docker',
        'override_cpus': 2,
        'override_memory_mb': 12288,
        'override_storage_mb': 10240,
        'delete': True,
        'suppress_override_warnings': True,
    },
    'verifier': {'disable': False},
}

async def run():
    trial = await Trial.create(TrialConfig.model_validate(config))
    result = await trial.run()
    exc = result.exception_info.exception_type if result.exception_info else 'none'
    print('exception:', exc)

asyncio.run(run())
\"" 2>&1 | tee "$trial_dir/trial.log"

    echo "[$(date -u +%H:%M:%S)] Done: $trial_id"

    python3 -c "
import json, sys
from pathlib import Path
rj = Path('${trial_dir}/verifier/reward.json')
if rj.exists():
    d = json.loads(rj.read_text())
    print('  lsv_mean_speedup    :', d.get('lsv_mean_speedup'))
    print('  num_valid_benchmarks:', d.get('num_valid_benchmarks'))
    print('  tests_passed        :', d.get('tests_passed'))
    print('  selected_count      :', d.get('selected_count'))
else:
    print('  reward.json not found — check ${trial_dir}/trial.log', file=sys.stderr)
"

    echo "$trial_dir" > "/tmp/oracle_trial_dir_${task_name}"
}

# ── Record h value into results JSON ─────────────────────────────────────────
record_h() {
    local task_id="$1"
    local trial_dir="$2"
    python3 - "$task_id" "$trial_dir" "$RESULTS_FILE" << 'EOF'
import json, sys
from pathlib import Path

task_id, trial_dir, results_file = sys.argv[1], sys.argv[2], sys.argv[3]
rj = Path(trial_dir) / 'verifier' / 'reward.json'
rf = Path(results_file)

d = json.loads(rf.read_text()) if rf.exists() else {}

if rj.exists():
    reward = json.loads(rj.read_text())
    h = reward.get('lsv_mean_speedup')
    d[task_id] = {
        'lsv_mean_speedup': h,
        'num_valid_benchmarks': reward.get('num_valid_benchmarks'),
        'tests_passed': reward.get('tests_passed'),
        'trial_dir': trial_dir,
    }
    rf.write_text(json.dumps(d, indent=2))
    print(f'[RESULT] {task_id}: h={h}  →  {results_file}')
else:
    d[task_id] = {'lsv_mean_speedup': None, 'error': 'reward.json missing', 'trial_dir': trial_dir}
    rf.write_text(json.dumps(d, indent=2))
    print(f'[RESULT] {task_id}: FAILED — reward.json missing', file=sys.stderr)
EOF
}

# ── Helper: read lsv from a trial_dir ────────────────────────────────────────
get_lsv() {
    local trial_dir="$1"
    python3 -c "
import json
from pathlib import Path
rj = Path('${trial_dir}/verifier/reward.json')
if rj.exists():
    d = json.loads(rj.read_text())
    h = d.get('lsv_mean_speedup')
    print(h if h is not None else 'null')
else:
    print('null')
" 2>/dev/null || echo 'null'
}

# ═══════════════════════════════════════════════════════════════════════════════
echo ""
echo "══════════════════════════════════════════════"
echo " Phase 1: h11 (solo — baseline cache cleared)"
echo "══════════════════════════════════════════════"
run_trial "h11" "/home/arjun/harbor-tasks-may18/python-hyper_h11_34"
record_h "python-hyper_h11_34" "$(cat /tmp/oracle_trial_dir_h11)"

# ═══════════════════════════════════════════════════════════════════════════════
echo ""
echo "══════════════════════════════════════════════"
echo " Phase 2: pvlib + networkx (parallel)"
echo "══════════════════════════════════════════════"

run_trial "pvlib"    "/home/arjun/harbor-tasks-may18/pvlib_pvlib-python_369" &
PID_PVLIB=$!

run_trial "networkx" "/home/arjun/harbor-tasks-may18/networkx_networkx_8148" &
PID_NX=$!

PVLIB_OK=0
NX_OK=0

wait $PID_PVLIB && PVLIB_OK=1 || echo "[WARN] pvlib trial process exited non-zero"
record_h "pvlib_pvlib-python_369" "$(cat /tmp/oracle_trial_dir_pvlib 2>/dev/null || echo '')"

wait $PID_NX && NX_OK=1 || echo "[WARN] networkx trial process exited non-zero"
record_h "networkx_networkx_8148" "$(cat /tmp/oracle_trial_dir_networkx 2>/dev/null || echo '')"

# ═══════════════════════════════════════════════════════════════════════════════
echo ""
echo "══════════════════════════════════════════════"
echo " Phase 3: joblib (last, ≤3 retries if lsv > 1.5)"
echo "══════════════════════════════════════════════"

JOBLIB_FINAL_DIR=""
JOBLIB_VALID=0

for attempt in 1 2 3; do
    echo "[joblib] Attempt $attempt/3"
    run_trial "joblib" "/home/arjun/harbor-tasks-may18/joblib_joblib_484" || true
    JOBLIB_TRIAL_DIR="$(cat /tmp/oracle_trial_dir_joblib 2>/dev/null || echo '')"
    JOBLIB_FINAL_DIR="$JOBLIB_TRIAL_DIR"

    if [ -z "$JOBLIB_TRIAL_DIR" ]; then
        echo "[joblib] Attempt $attempt: trial_dir not written — trial likely crashed"
        continue
    fi

    LSV="$(get_lsv "$JOBLIB_TRIAL_DIR")"

    if python3 -c "
h = '$LSV'
if h == 'null': raise SystemExit(1)
raise SystemExit(0 if float(h) <= 1.5 else 1)
" 2>/dev/null; then
        echo "[joblib] lsv=$LSV — valid (≤1.5), done."
        JOBLIB_VALID=1
        break
    else
        echo "[joblib] lsv=$LSV — cold-cache outlier or error (attempt $attempt/3)"
        [ $attempt -eq 3 ] && echo "[joblib] WARNING: all 3 attempts gave outlier/error. Using last value anyway."
    fi
done

record_h "joblib_joblib_484" "$JOBLIB_FINAL_DIR"

# ═══════════════════════════════════════════════════════════════════════════════
echo ""
echo "══════════════════════════════════════════════"
echo " All oracle runs complete"
echo " Results file: $RESULTS_FILE"
echo "══════════════════════════════════════════════"
python3 -c "
import json
from pathlib import Path
d = json.loads(Path('$RESULTS_FILE').read_text())
for task, vals in d.items():
    h = vals.get('lsv_mean_speedup') if isinstance(vals, dict) else vals
    print(f'  {task}: h={h}')
"

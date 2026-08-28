# Oracle Runs Plan — joblib, pvlib, h11, networkx

> **⚠ Superseded by `rl-prep/` (2026-07-18) — kept for history.** The oracle runs are done;
> confirmed H values (pvlib 22.563 / joblib 1.952 / networkx 1.323 / h11 1.203 / shapely
> 2.340) and per-task status live in `rl-prep/containers/`. Pre-measurement H guesses in
> this doc (e.g. pvlib "~1.65×") are wrong.

## Purpose

Run oracle trials for 4 Harbor tasks to establish ground-truth speedup values (h) for
the RL reward formula. The oracle applies the human-authored GT solution patch and
measures the resulting speedup via LSV (LightSpeed Verifier). The key output per task
is `lsv_mean_speedup` from `reward.json` — the raw geometric mean speedup of the oracle
solution vs. the unmodified baseline.

These h values will be hard-coded into the shaped RL reward formula in
`datasmith/src/datasmith/harbor_adapter/template/parser.py` (on the `rl` branch), which
currently uses `h = 1.08` as a placeholder.

## Resource Config — CRITICAL

**2 CPUs, 12 GB RAM** — must be identical for oracle and all future agent runs, because
the LSV baseline cache is keyed on `(cpu_count, mem_bytes, machine_class, docker_host_id,
detected_cpu_model)`. A mismatch between oracle and agent config = cache miss = baseline
re-measured = potentially different environment = wrong lsv ratio.

Do not deviate from this config without re-running all oracle trials.

## Tasks

| Task | Directory | Docker image | Notes |
|------|-----------|--------------|-------|
| joblib | `/home/arjun/harbor-tasks-may18/joblib_joblib_484` | `formulacode/joblib_joblib_484:latest` | Valid existing oracle; run to establish 2CPU/12GB cache entry |
| pvlib | `/home/arjun/harbor-tasks-may18/pvlib_pvlib-python_369` | `formulacode/pvlib_pvlib-python_369:latest` (e7a0e2ba) | Docker image just rebuilt (2026-06-07) with benchmark restore fix |
| h11 | `/home/arjun/harbor-tasks-may18/python-hyper_h11_34` | `formulacode/python-hyper_h11_34:latest` | LSV baseline cache deliberately cleared; first run re-measures baseline |
| networkx | `/home/arjun/harbor-tasks-may18/networkx_networkx_8148` | `formulacode/networkx_networkx_8148:latest` | First oracle run ever for this task |

### What the oracle patches do

- **joblib**: Adds `supports_timeout` class attribute to backend classes, removing per-job
  `getfullargspec()` introspection. Patches `joblib/parallel.py` + `_parallel_backends.py`.
- **pvlib**: Vectorizes `lookup_linke_turbidity()` in `pvlib/clearsky.py`, replacing
  per-row `calendar.isleap` loop with numpy operations. Expected ~1.65x speedup.
- **h11**: Optimizes `_util.py` (bytesify), `_headers.py`, `_connection.py` — changes
  string comparisons to bytes (`b"connection"` lowercase), reduces per-header work.
- **networkx**: Replaces buggy DFS in `networkx/algorithms/dag.py:is_aperiodic` with a
  correct and faster algorithm. Also adds `benchmarks/benchmarks/benchmark_aperiodic.py`
  (the benchmark file survives git clean via `/opt/` staging in the Docker image).

## Results Location

```
/home/arjun/formulacode-rlvr-recipe/results/oracle-runs/
  joblib_oracle_<trial_id>/
    agent/
    verifier/
      reward.json        ← read lsv_mean_speedup from here
      reward.txt         ← shaped reward (= raw speedup for oracle); secondary
  pvlib_oracle_<trial_id>/
  h11_oracle_<trial_id>/
  networkx_oracle_<trial_id>/
```

## LSV Rounds

The adapter default is now 5 rounds (`DATASMITH_LSV_ROUNDS=5` in `tokens.env`, or set
via env var). Both `lsv_init.py` (baseline measurement) and `lsv_measure.py` (agent
measurement) take the min across rounds. This is set in the datasmith `rl` branch.

If running via the standalone Harbor Python API and `DATASMITH_LSV_ROUNDS` is not picked
up automatically, pass `rounds=5` explicitly when constructing the trial config or set
the env var before running.

## Oracle Agent Behavior

The `oracle` agent type (`agent.name: oracle`) does NOT use an LLM. It:
1. Uploads `solution/` directory from the task dir into the container
2. Runs `solution/solve.sh` which applies the GT patch via `patch -p1`
3. Exits — the Harbor verifier then runs setup.sh (LSV init) and test.sh (LSV measure)

No model server or API key needed for oracle runs.

## Running on Local Docker

Use the pattern from `eval_optuna.sh`. Run from the SkyRL directory with `sg docker`:

```bash
TRIALS_DIR="/home/arjun/formulacode-rlvr-recipe/results/oracle-runs"
mkdir -p "$TRIALS_DIR"
SKYRL_DIR="/home/arjun/formulacode-rlvr-recipe/SkyRL"

run_oracle() {
    local task_name="$1"
    local task_path="$2"

    trial_id="${task_name}_oracle_$(openssl rand -hex 4)"
    trial_dir="$TRIALS_DIR/${trial_id}"
    mkdir -p "$trial_dir/agent" "$trial_dir/verifier"

    echo "[$(date -u +%H:%M:%S)] Starting oracle trial: $trial_id"

    sg docker -c "cd $SKYRL_DIR && uv run --isolated --extra harbor python -c \"
import asyncio
from harbor.trial.trial import Trial
from harbor.models.trial.config import TrialConfig

config = {
    'trial_name': '${trial_id}',
    'task': {'path': '${task_path}'},
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
    'log_dir': '${trial_dir}',
    'agent_artifacts_dir': '${trial_dir}/agent',
    'verifier_artifacts_dir': '${trial_dir}/verifier',
}

async def run():
    trial = await Trial.create(TrialConfig.model_validate(config))
    result = await trial.run()
    print('done:', result.verifier_result.rewards if result.verifier_result else 'no verifier result')
    print('exception:', result.exception_info.exception_type if result.exception_info else 'none')

asyncio.run(run())
\"" 2>&1 | tee "$trial_dir/trial.log"

    echo "[$(date -u +%H:%M:%S)] Trial complete: $trial_id"
    # Extract h value
    python3 -c "
import json; from pathlib import Path
rj = Path('${trial_dir}/verifier/reward.json')
if rj.exists():
    d = json.loads(rj.read_text())
    print('lsv_mean_speedup:', d.get('lsv_mean_speedup'))
    print('num_valid_benchmarks:', d.get('num_valid_benchmarks'))
    print('tests_passed:', d.get('tests_passed'))
else:
    print('reward.json not found')
"
}

run_oracle "joblib"   "/home/arjun/harbor-tasks-may18/joblib_joblib_484"
run_oracle "pvlib"    "/home/arjun/harbor-tasks-may18/pvlib_pvlib-python_369"
run_oracle "h11"      "/home/arjun/harbor-tasks-may18/python-hyper_h11_34"
run_oracle "networkx" "/home/arjun/harbor-tasks-may18/networkx_networkx_8148"
```

## Running on Daytona

First-ever Daytona trial for this project. Prerequisites:
- `tokens.env` in `/home/arjun/datasmith/` must have `DAYTONA_API_KEY` set ✓
- The Daytona workspace image must be accessible (it uses the task's Docker image,
  which must be pushed to DockerHub or a registry Daytona can pull from)

**Important**: Check whether the task Docker images are publicly available on DockerHub
(`formulacode/pvlib_pvlib-python_369`, etc.) before running Daytona trials. The local
`e7a0e2ba` pvlib image is NOT on DockerHub yet — push it first:
```bash
sg docker -c "docker push formulacode/pvlib_pvlib-python_369:latest"
```

Daytona trial config changes from Docker:
- `environment.type`: `"daytona"` instead of `"docker"`
- Remove `override_cpus` / `override_memory_mb` / `override_storage_mb` — Daytona uses
  its own `Resources` object derived from these, but the harbor config may pass them
  differently. Check `harbor/environments/daytona*.py` for the exact field names.
- The `DAYTONA_API_KEY` env var must be set (loaded from `tokens.env` via datasmith's
  `dotenv.load_dotenv` at import time, or export it manually before running)

Daytona trial template (adjust environment block):
```python
'environment': {
    'type': 'daytona',
    'override_cpus': 2,
    'override_memory_mb': 12288,
    'override_storage_mb': 10240,
    'suppress_override_warnings': True,
},
```

Run from the datasmith directory (so `tokens.env` is loaded) or export vars explicitly:
```bash
cd /home/arjun/datasmith
set -a && source tokens.env && set +a
cd /home/arjun/formulacode-rlvr-recipe/SkyRL
sg docker -c "... python -c '...'"
```

## Reading Results

After each trial, the key value is:
```python
import json
from pathlib import Path

reward = json.loads(Path("verifier/reward.json").read_text())
h = reward["lsv_mean_speedup"]   # raw geomean speedup — use this for RL formula
# NOT reward.txt (that now contains the shaped reward value)
```

Once all 4 oracle h values are known, update `parser.py` on the `rl` branch:
```python
# In write_reward(), replace:
h = 1.08  # TODO: fetch oracle speedup from DB

# With a per-task lookup, e.g.:
_ORACLE_H = {
    "joblib_joblib_484": <measured>,
    "pvlib_pvlib-python_369": <measured>,
    "python-hyper_h11_34": <measured>,
    "networkx_networkx_8148": <measured>,
}
task_id = os.environ.get("TASK_ID", os.environ.get("LSV_TASK_ID", ""))
h = _ORACLE_H.get(task_id, 1.08)
```

## Known Issues / Gotchas

- **pvlib**: Docker image `e7a0e2ba` has the benchmark restore fix (benchmark_clearsky.py
  survives git clean). Older image `d1f6557ccb91` does NOT. Always use `latest` tag.
- **h11**: LSV baseline cache was cleared from Datasmith DB on 2026-06-07. The first
  oracle trial will remeasure the baseline (slow, ~5 rounds of ASV). All subsequent
  agent trials will use that fresh baseline. If the oracle trial fails before writing
  baselines back to Datasmith, re-run it.
- **networkx**: No prior oracle run exists. The benchmark file is pre-staged in the
  Docker image and restored by entrypoint.sh after git clean.
- **Cold-cache outliers**: joblib oracle occasionally shows lsv=2.5-3.4x due to cold
  `getfullargspec` cache at baseline measurement time. If lsv > 1.5 for joblib, the
  measurement is a fluke — re-run. Expected oracle lsv is ~1.0-1.05 for joblib
  (the optimization is meaningful but the benchmark measures n_jobs=1 sequential work).
- **Daytona first run**: Harbor's Daytona integration has not been exercised for these
  tasks before. If the Daytona workspace fails to start, check:
  1. Docker image is accessible from Daytona's registry
  2. `DAYTONA_API_KEY` is set and valid
  3. Harbor's Daytona environment fields match what the Daytona SDK expects

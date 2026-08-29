#!/usr/bin/env python3
"""Run oracle trials for the FormulaCode gold RL tasks and build the oracle table.

The oracle agent (``agent.name = oracle``) applies each task's ground-truth
``solution/`` patch and lets the harbor verifier measure per-benchmark speedups via
LSV. We read ``per_benchmark_speedups`` from each trial's ``reward.json`` and derive
the fields the custom RL reward needs:

  * which benchmarks the oracle moved beyond the noise floor (the summed set), and
  * whether the oracle regressed each benchmark (selects the reward branch).

CRITICAL: resources here (2 CPU / 12 GB, 5 LSV rounds) must match every agent trial,
because the LSV baseline cache is keyed on the machine/resource fingerprint. Running
the oracle first also warms that local baseline cache for the agent runs.

Usage (must run under the docker group; GPUs are not needed — oracle uses no LLM):
    sg docker -c ".../skyrl-formulacode/.venv/bin/python rl/oracle_gold.py"
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# ── Config ──────────────────────────────────────────────────────────────────
TASKS_ROOT = Path("/home/arjun/formulacode-verified-rl/initial_survey/tasks")
RESULTS_DIR = Path("/home/arjun/formulacode-rlvr-recipe/results/oracle-runs")
ORACLE_TABLE = Path(
    "/home/arjun/formulacode-verified-rl/initial_survey/oracle/gold_oracle_benchmarks.json"
)

NOISE_FLOOR = 1.05
LSV_ROUNDS = 5
OVERRIDE_CPUS = 2
OVERRIDE_MEMORY_MB = 12288
OVERRIDE_STORAGE_MB = 10240
MAX_CONCURRENCY = 3  # oracle trials in flight at once (CPU-bound)

# task key -> (task dir name, "owner/repo#issue"). Superset (gold-8 + breadth); the RL training
# set is chosen downstream in run_local_train_qwen.sh, not here.
GOLD_TASKS: dict[str, tuple[str, str]] = {
    # ── gold-8 ───────────────────────────────────────────────────────────────
    "shapely#1307": ("shapely__shapely__1307", "shapely/shapely#1307"),
    "geopandas#3345": ("geopandas__geopandas__3345", "geopandas/geopandas#3345"),
    "uxarray#1118": ("UXARRAY__uxarray__1118", "UXARRAY/uxarray#1118"),
    "flox#176": ("xarray-contrib__flox__176", "xarray-contrib/flox#176"),
    "flox#172": ("xarray-contrib__flox__172", "xarray-contrib/flox#172"),
    "numpy-financial#96": ("numpy__numpy-financial__96", "numpy/numpy-financial#96"),
    "networkx#6337": ("networkx__networkx__6337", "networkx/networkx#6337"),
    "flox#53": ("xarray-contrib__flox__53", "xarray-contrib/flox#53"),
    # ── breadth (survey 18-task mixed-sign candidates) ───────────────────────
    "tiledb-py#1005": ("TileDB-Inc__TileDB-Py__1005", "TileDB-Inc/TileDB-Py#1005"),
    "uxarray#1112": ("UXARRAY__uxarray__1112", "UXARRAY/uxarray#1112"),
    "tiled#1169": ("bluesky__tiled__1169", "bluesky/tiled#1169"),
    "tiled#1213": ("bluesky__tiled__1213", "bluesky/tiled#1213"),
    "tiled#982": ("bluesky__tiled__982", "bluesky/tiled#982"),
    "joblib#484": ("joblib__joblib__484", "joblib/joblib#484"),
    "networkx#8138": ("networkx__networkx__8138", "networkx/networkx#8138"),
    "numpy-financial#47": ("numpy__numpy-financial__47", "numpy/numpy-financial#47"),
    "bottleneck#298": ("pydata__bottleneck__298", "pydata/bottleneck#298"),
    "bottleneck#305": ("pydata__bottleneck__305", "pydata/bottleneck#305"),
    "shapely#2359": ("shapely__shapely__2359", "shapely/shapely#2359"),
    "flox#70": ("xarray-contrib__flox__70", "xarray-contrib/flox#70"),
    "xdsl#1159": ("xdslproject__xdsl__1159", "xdslproject/xdsl#1159"),
    "xdsl#1332": ("xdslproject__xdsl__1332", "xdslproject/xdsl#1332"),
    # ── second breadth wave (mixed-sign diversity candidates) ────────────────
    "pybop#335": ("pybop-team__PyBOP__335", "pybop-team/PyBOP#335"),
    "bottleneck#285": ("pydata__bottleneck__285", "pydata/bottleneck#285"),
    "tiledb-py#467": ("TileDB-Inc__TileDB-Py__467", "TileDB-Inc/TileDB-Py#467"),
    "xdsl#1567": ("xdslproject__xdsl__1567", "xdslproject/xdsl#1567"),
    "geopandas#2939": ("geopandas__geopandas__2939", "geopandas/geopandas#2939"),
    "shapely#1562": ("shapely__shapely__1562", "shapely/shapely#1562"),
    "tiled#1283": ("bluesky__tiled__1283", "bluesky/tiled#1283"),
    "networkx#8148": ("networkx__networkx__8148", "networkx/networkx#8148"),
    "networkx#8218": ("networkx__networkx__8218", "networkx/networkx#8218"),
    "uxarray#61": ("UXARRAY__uxarray__61", "UXARRAY/uxarray#61"),
    "bottleneck#327": ("pydata__bottleneck__327", "pydata/bottleneck#327"),
    "xdsl#1111": ("xdslproject__xdsl__1111", "xdslproject/xdsl#1111"),
    "xdsl#1460": ("xdslproject__xdsl__1460", "xdslproject/xdsl#1460"),
    "tiledb-py#834": ("TileDB-Inc__TileDB-Py__834", "TileDB-Inc/TileDB-Py#834"),
    "networkx#4830": ("networkx__networkx__4830", "networkx/networkx#4830"),
    "pybop#256": ("pybop-team__PyBOP__256", "pybop-team/PyBOP#256"),
    "pybamm#465": ("pybamm-team__PyBaMM__465", "pybamm-team/PyBaMM#465"),
    "networkx#4909": ("networkx__networkx__4909", "networkx/networkx#4909"),
}


def build_config(task_key: str, task_dir: Path) -> dict:
    trial_name = f"{task_key.replace('#', '_').replace('/', '_')}_oracle"
    return {
        "trial_name": trial_name,
        "task": {"path": str(task_dir)},
        "trials_dir": str(RESULTS_DIR),
        "agent": {"name": "oracle"},
        "environment": {
            "type": "docker",
            "override_cpus": OVERRIDE_CPUS,
            "override_memory_mb": OVERRIDE_MEMORY_MB,
            "override_storage_mb": OVERRIDE_STORAGE_MB,
            "delete": False,  # keep the trial dir for inspection + reward.json
            "suppress_override_warnings": True,
        },
        "verifier": {"disable": False},
    }


async def run_one(task_key: str, sem: asyncio.Semaphore, collect_only: bool = False) -> dict:
    """Run one oracle trial (or just collect its result); return a status record.

    ``collect_only`` skips execution and reads an existing trial dir — used to
    rebuild the authoritative oracle table after trials finish, without racing the
    running processes or re-measuring.
    """
    from harbor.trial.trial import Trial
    from harbor.models.trial.config import TrialConfig

    dir_name, label = GOLD_TASKS[task_key]
    task_dir = TASKS_ROOT / dir_name
    rec: dict = {"task": label, "task_key": task_key, "task_dir": dir_name, "ok": False}
    if not (task_dir / "instruction.md").exists():
        rec["error"] = f"task dir missing/invalid: {task_dir}"
        return rec
    cfg = build_config(task_key, task_dir)
    if collect_only:
        rec["trial_dir"] = str(RESULTS_DIR / cfg["trial_name"])
        rec.update(_read_reward(Path(rec["trial_dir"])))
        return rec
    async with sem:
        ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
        print(f"[{ts}] START oracle {label}", flush=True)
        try:
            trial = await Trial.create(TrialConfig.model_validate(cfg))
            result = await trial.run()
            rec["exception"] = (
                result.exception_info.exception_type if result.exception_info else None
            )
        except Exception as e:  # noqa: BLE001 — one bad trial must not kill the batch
            rec["error"] = f"trial raised: {type(e).__name__}: {e}"
            print(f"[{task_key}] ERROR {rec['error']}", flush=True)
            return rec
    rec["trial_dir"] = str(RESULTS_DIR / cfg["trial_name"])
    rec.update(_read_reward(Path(rec["trial_dir"])))
    # Publish the oracle baseline snapshot to the prod store (durable, host-side).
    try:
        from publish_snapshots import publish_snapshot
        owner, rest = label.split("/", 1)
        repo, issue = rest.split("#", 1)
        snap = Path(rec["trial_dir"]) / "artifacts" / ".snapshots"
        st, _ = publish_snapshot(owner, repo, issue, snap)
        rec["snapshot_published"] = st
    except Exception as e:  # noqa: BLE001 — publishing is best-effort
        rec["snapshot_published"] = f"error: {e}"
    done = datetime.now(timezone.utc).strftime("%H:%M:%S")
    print(
        f"[{done}] DONE  {label}: benches={rec.get('n_benchmarks')} "
        f"impacted={rec.get('n_impacted')} H={rec.get('lsv_mean_speedup')}",
        flush=True,
    )
    return rec


def _read_reward(trial_dir: Path) -> dict:
    rj = trial_dir / "verifier" / "reward.json"
    if not rj.exists():
        return {"error": f"no reward.json at {rj}"}
    try:
        d = json.loads(rj.read_text())
    except Exception as e:  # noqa: BLE001
        return {"error": f"reward.json unreadable: {e}"}
    per = d.get("per_benchmark_speedups") or {}
    benchmarks = {}
    n_impacted = 0
    for name, g in per.items():
        entry = _derive(g)
        if entry["impacted_beyond_noise_floor"]:
            n_impacted += 1
        benchmarks[name] = entry
    return {
        "ok": bool(d.get("tests_passed")) and d.get("lsv_error") is None,
        "tests_passed": d.get("tests_passed"),
        "lsv_error": d.get("lsv_error"),
        "lsv_mean_speedup": d.get("lsv_mean_speedup"),  # oracle H
        "n_benchmarks": len(per),
        "n_impacted": n_impacted,
        "benchmarks": benchmarks,
    }


def _derive(oracle_speedup) -> dict:
    """Derive reward-relevant fields for one oracle benchmark speedup (base/oracle)."""
    try:
        g = float(oracle_speedup)
    except (TypeError, ValueError):
        g = None
    if g is None or not math.isfinite(g) or g <= 0:
        return {
            "oracle_speedup": oracle_speedup,
            "oracle_log_speedup": None,
            "oracle_regressed": None,
            "impacted_beyond_noise_floor": False,
        }
    log = math.log(g)
    return {
        "oracle_speedup": g,
        "oracle_log_speedup": log,
        "oracle_regressed": log < 0.0,
        "impacted_beyond_noise_floor": abs(log) > math.log(NOISE_FLOOR),
    }


async def main_async(task_keys: list[str], collect_only: bool = False) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ORACLE_TABLE.parent.mkdir(parents=True, exist_ok=True)
    sem = asyncio.Semaphore(MAX_CONCURRENCY)
    recs = await asyncio.gather(*(run_one(k, sem, collect_only) for k in task_keys))

    # Merge into any existing table so a partial re-run (--tasks subset) updates only those keys
    # instead of truncating the 40-entry gold table.
    existing_tasks: dict = {}
    if ORACLE_TABLE.exists():
        try:
            existing_tasks = json.loads(ORACLE_TABLE.read_text()).get("tasks", {})
        except (json.JSONDecodeError, OSError):
            existing_tasks = {}
    table = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": "local oracle trials (agent.name=oracle) on this machine",
        "noise_floor": NOISE_FLOOR,
        "lsv_rounds": LSV_ROUNDS,
        "resources": {"cpus": OVERRIDE_CPUS, "memory_mb": OVERRIDE_MEMORY_MB},
        "tasks": dict(existing_tasks),
    }
    for r in recs:
        table["tasks"][r["task"]] = {
            k: r.get(k)
            for k in (
                "task_key", "task_dir", "ok", "tests_passed", "lsv_error",
                "lsv_mean_speedup", "n_benchmarks", "n_impacted", "trial_dir",
                "exception", "error", "benchmarks",
            )
        }
    ORACLE_TABLE.write_text(json.dumps(table, indent=2, sort_keys=True) + "\n")
    print(f"\nWrote oracle table -> {ORACLE_TABLE}", flush=True)

    print("\n=== SUMMARY ===", flush=True)
    for r in recs:
        flag = "OK " if r.get("ok") and r.get("n_impacted", 0) >= 1 else "!! "
        print(
            f"  {flag}{r['task']}: impacted={r.get('n_impacted')} "
            f"H={r.get('lsv_mean_speedup')} err={r.get('error') or r.get('exception')}",
            flush=True,
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="*", default=list(GOLD_TASKS),
                    help="task keys to run (default: all gold)")
    ap.add_argument("--collect-only", action="store_true",
                    help="skip running; rebuild the oracle table from existing trial dirs")
    os.environ.setdefault("DATASMITH_LSV_ROUNDS", str(LSV_ROUNDS))
    os.environ.setdefault("FORMULACODE_NO_UPLOAD", "1")  # keep oracle data local
    args = ap.parse_args()
    unknown = [t for t in args.tasks if t not in GOLD_TASKS]
    if unknown:
        sys.exit(f"unknown task keys: {unknown}; valid: {list(GOLD_TASKS)}")
    asyncio.run(main_async(args.tasks, collect_only=args.collect_only))


if __name__ == "__main__":
    main()

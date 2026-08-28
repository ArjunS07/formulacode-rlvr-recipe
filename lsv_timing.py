#!/usr/bin/env python3
"""
lsv_timing.py — Extract lsv_init and lsv_measure timing from all trial and oracle runs.

Outputs a summary table and a JSON file suitable for later plotting.

Usage:
    python3 lsv_timing.py [--json output.json]
"""
import argparse
import json
from pathlib import Path

TRIALS_DIR = Path("/home/arjun/formulacode-rlvr-recipe/results/trials")
ORACLE_DIR = Path("/home/arjun/formulacode-rlvr-recipe/results/oracle-runs")

# Map directory-name prefixes to canonical short task names
_TASK_ALIASES = {
    "joblib_joblib_484": "joblib",
    "pvlib_pvlib-python_369": "pvlib",
    "python-hyper_h11_34": "h11",
    "h11_agent": "h11",
    "joblib_agent": "joblib",
    "pvlib_agent": "pvlib",
    "networkx_agent": "networkx",
}


def _canonical_task(dir_name: str) -> str:
    """Return a normalized task name from a trial/oracle directory name."""
    # Oracle pattern: <task>_oracle_<hash>
    if "_oracle_" in dir_name:
        return dir_name.rsplit("_oracle_", 1)[0]
    # Agent trial patterns from pass_k / run_agent:
    #   h11_agent_<hash>
    #   joblib_agent_<hash>
    for prefix, alias in _TASK_ALIASES.items():
        if dir_name.startswith(prefix):
            return alias
    # RL training trial pattern: <task_id>__<run_id>  (double underscore)
    if "__" in dir_name:
        return dir_name.split("__")[0]
    return dir_name


def extract_timings(trial_dir: Path, kind: str) -> dict | None:
    artifacts = trial_dir / "artifacts"

    lsv_init_s = None
    lsv_measure_s = None
    lsv_mean_speedup = None
    num_valid_benchmarks = None
    tests_passed = None

    # ── lsv_init time ─────────────────────────────────────────────────────────
    # setup_timings.json has the real wall-clock lsv_init time
    setup_timings = artifacts / "setup_timings.json"
    if setup_timings.exists():
        try:
            st = json.loads(setup_timings.read_text())
            lsv_init_s = st.get("lsv_init_s")
        except Exception:
            pass

    # ── lsv_measure time ──────────────────────────────────────────────────────
    lsv_measure_results = artifacts / "lsv" / "lsv_measure_results.json"
    if lsv_measure_results.exists():
        try:
            mr = json.loads(lsv_measure_results.read_text())
            t = mr.get("timing", {})
            lsv_measure_s = t.get("total_s")
        except Exception:
            pass

    if lsv_init_s is None and lsv_measure_s is None:
        return None

    # ── reward / outcome ──────────────────────────────────────────────────────
    reward_json = trial_dir / "verifier" / "reward.json"
    if reward_json.exists():
        try:
            rj = json.loads(reward_json.read_text())
            lsv_mean_speedup = rj.get("lsv_mean_speedup")
            num_valid_benchmarks = rj.get("num_valid_benchmarks")
            tests_passed = rj.get("tests_passed")
        except Exception:
            pass

    return {
        "trial_id": trial_dir.name,
        "task_name": _canonical_task(trial_dir.name),
        "kind": kind,
        "lsv_init_s": lsv_init_s,
        "lsv_measure_s": lsv_measure_s,
        "lsv_mean_speedup": lsv_mean_speedup,
        "num_valid_benchmarks": num_valid_benchmarks,
        "tests_passed": tests_passed,
    }


def collect_all() -> list[dict]:
    records = []
    if TRIALS_DIR.exists():
        for d in sorted(TRIALS_DIR.iterdir()):
            if d.is_dir():
                rec = extract_timings(d, kind="agent")
                if rec:
                    records.append(rec)
    if ORACLE_DIR.exists():
        for d in sorted(ORACLE_DIR.iterdir()):
            if d.is_dir():
                rec = extract_timings(d, kind="oracle")
                if rec:
                    records.append(rec)
    return records


def fmt_s(v) -> str:
    if v is None:
        return "n/a"
    if v >= 60:
        return f"{v/60:.1f}m"
    return f"{v:.1f}s"


def print_table(records: list[dict]) -> None:
    if not records:
        print("No records found.")
        return

    by_task: dict[str, dict[str, list]] = {}
    for r in records:
        task = r["task_name"]
        kind = r["kind"]
        by_task.setdefault(task, {"agent": [], "oracle": []})
        by_task[task][kind].append(r)

    print(f"\n{'─'*82}")
    print(f"  {'Task':<26}  {'Kind':<6}  {'N':>4}  {'Init avg':>9}  {'Measure avg':>11}  {'lsv avg':>8}")
    print(f"{'─'*82}")

    for task in sorted(by_task):
        for kind in ("oracle", "agent"):
            recs = by_task[task][kind]
            if not recs:
                continue
            inits    = [r["lsv_init_s"]      for r in recs if r["lsv_init_s"]      is not None]
            measures = [r["lsv_measure_s"]    for r in recs if r["lsv_measure_s"]   is not None]
            lsvs     = [r["lsv_mean_speedup"] for r in recs if r["lsv_mean_speedup"] is not None]

            init_avg    = fmt_s(sum(inits)/len(inits))       if inits    else "n/a"
            measure_avg = fmt_s(sum(measures)/len(measures)) if measures else "n/a"
            lsv_avg     = f"{sum(lsvs)/len(lsvs):.4f}"      if lsvs     else "n/a"

            print(f"  {task:<26}  {kind:<6}  {len(recs):>4}  {init_avg:>9}  {measure_avg:>11}  {lsv_avg:>8}")

    print(f"{'─'*82}")

    # Per-oracle detail (these are few enough to show individually)
    oracle_recs = [r for r in records if r["kind"] == "oracle"]
    if oracle_recs:
        print(f"\nOracle runs:")
        print(f"  {'Trial':<45}  {'Init':>9}  {'Measure':>9}  {'lsv':>7}")
        for r in oracle_recs:
            print(f"  {r['trial_id']:<45}  {fmt_s(r['lsv_init_s']):>9}  {fmt_s(r['lsv_measure_s']):>9}  {r['lsv_mean_speedup'] or 'n/a':>7}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", metavar="PATH")
    args = parser.parse_args()

    records = collect_all()
    print_table(records)
    n_agent  = sum(1 for r in records if r["kind"] == "agent")
    n_oracle = sum(1 for r in records if r["kind"] == "oracle")
    print(f"\nTotal: {len(records)}  (agent: {n_agent}, oracle: {n_oracle})")

    if args.json:
        Path(args.json).write_text(json.dumps(records, indent=2))
        print(f"JSON written to: {args.json}")


if __name__ == "__main__":
    main()

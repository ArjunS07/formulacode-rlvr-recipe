#!/usr/bin/env python3
"""Generalized analysis for a FormulaCode GRPO run — no hardcoded tasks.

Reads a run directory produced by ``run_local_train.sh`` and emits:
  * ``analysis/metrics.csv``   — per-step trainer + generate + reward metrics (long form).
  * ``analysis/stats.json``    — headline numbers + per-task summaries.
  * ``analysis/report.txt``    — human-readable audit (gradients, masking, diversity).
  * ``analysis/*.png``         — reward curve, entropy, grad_norm, response_length,
                                 masking/verify rates, per-task intra-group std (diversity),
                                 speedup distribution.

Everything is derived from the run itself:
  * ``run.log``                — the console logger's per-step metric dicts.
  * ``trials/<name>/verifier/reward.json``   — per-trial speedups / tests / snapshots.
  * ``trials/<name>/artifacts/summary_*.json`` — snapshot-tool verify verdict.

Task identities come from the trial directory names, so this works for any task set.

Usage:
  python analyze_run.py results/fc-grpo-9b-6task-v3
  python analyze_run.py results/fc-grpo-9b-6task-v3 --oracle-table \
      /home/arjun/formulacode-verified-rl/initial_survey/oracle/gold_oracle_benchmarks.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev

# ── run.log parsing ───────────────────────────────────────────────────────────
# The console logger prints metric dicts like  {'policy_entropy': 0.1, 'grad_norm': 0.4, ...}
# and single lines like  "avg_final_rewards: 0.3, avg_response_length: 812.0".
# We extract every  key: value  / 'key': value  numeric pair generically, so new metric
# keys are picked up automatically without code changes.
_PAIR = re.compile(r"'?([A-Za-z0-9_/]+)'?\s*[:=]\s*(-?\d+\.?\d*(?:[eE][-+]?\d+)?)")
_ANSI = re.compile(r"\x1b\[[0-9;]*m")
# Keys worth trending. Anything matching is captured; this list only orders/plots them.
_TREND_KEYS = [
    "avg_final_rewards", "reward", "response_length", "avg_response_length",
    "policy_entropy", "grad_norm", "policy_loss", "final_loss",
    "loss_metrics/clip_ratio", "kl", "policy_lr",
    "generate/num_error_trajectories", "generate/num_timeout_trajectories",
    "generate/num_masked_instances", "generate/avg_num_turns",
    "reward/fc_correctness_rate", "reward/fc_failure_rate",
    "reward/fc_positive_reward_rate", "reward/fc_agent_geomean_speedup",
    "reward/fc_mean_group_std", "reward/fc_zero_diversity_tasks",
    "reward/fc_snapshot_verify_rate",
]


def parse_run_log(run_log: Path) -> list[dict]:
    """Return a list of per-step metric dicts, in log order.

    A "step" is a cluster of metric lines. We split on the trainer's per-step dict
    (the line carrying ``grad_norm``) so each returned dict aggregates the metrics logged
    around one optimizer step. Robust to missing keys.
    """
    if not run_log.exists():
        return []
    steps: list[dict] = []
    cur: dict = {}
    for raw in run_log.read_text(errors="replace").splitlines():
        line = _ANSI.sub("", raw)
        pairs = dict((k, float(v)) for k, v in _PAIR.findall(line))
        if not pairs:
            continue
        # Ignore the one-time config dump (huge, non-metric); heuristic: config lines are
        # single "key: value" with keys we never trend and appear before any grad_norm.
        cur.update(pairs)
        if "grad_norm" in pairs:            # end-of-step marker
            steps.append(cur)
            cur = {}
    if cur.get("avg_final_rewards") is not None or cur.get("reward/fc_correctness_rate") is not None:
        steps.append(cur)                    # trailing partial step (generate metrics, no update yet)
    return steps


# ── per-trial parsing ─────────────────────────────────────────────────────────
def _task_of(trial_name: str) -> str:
    """geopandas__geopandas__3345__aB3xQ  ->  geopandas__geopandas__3345 (drop trial suffix)."""
    parts = trial_name.split("__")
    return "__".join(parts[:3]) if len(parts) >= 3 else trial_name


def parse_trials(trials_dir: Path) -> dict[str, list[dict]]:
    """task_id -> list of per-trial records {reward.json fields + snapshot verdict}."""
    by_task: dict[str, list[dict]] = defaultdict(list)
    if not trials_dir.exists():
        return by_task
    for trial in sorted(trials_dir.iterdir()):
        if not trial.is_dir():
            continue
        rj = trial / "verifier" / "reward.json"
        if not rj.exists():
            continue
        try:
            data = json.loads(rj.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        # snapshot verify verdict from artifacts/summary_*.json
        verify_ran, snap_passed = False, None
        for s in (trial / "artifacts").glob("summary_*.json"):
            verify_ran = True
            try:
                if json.loads(s.read_text()).get("passed") is False:
                    snap_passed = False
            except (json.JSONDecodeError, OSError):
                pass
        if verify_ran and snap_passed is None:
            snap_passed = True
        rec = {
            "trial": trial.name,
            "lsv_mean_speedup": data.get("lsv_mean_speedup"),
            "per_benchmark_speedups": data.get("per_benchmark_speedups") or {},
            "num_valid_benchmarks": data.get("num_valid_benchmarks"),
            "tests_passed": data.get("tests_passed"),
            "snapshots_passed_baked": data.get("snapshots_passed"),
            "snapshot_verify_ran": verify_ran,
            "snapshot_passed_real": snap_passed,
        }
        by_task[_task_of(trial.name)].append(rec)
    return by_task


# ── plotting ──────────────────────────────────────────────────────────────────
def make_plots(steps: list[dict], by_task: dict, out: Path) -> list[str]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # noqa: BLE001
        print(f"[analyze] matplotlib unavailable ({e}); skipping plots.")
        return []
    made = []

    def series(key):
        xs, ys = [], []
        for i, s in enumerate(steps):
            if key in s:
                xs.append(i)
                ys.append(s[key])
        return xs, ys

    panels = [
        ("avg_final_rewards", "mean reward"),
        ("policy_entropy", "policy entropy"),
        ("grad_norm", "grad norm"),
        ("response_length", "response length (tokens)"),
        ("reward/fc_mean_group_std", "intra-group reward std (GRPO diversity)"),
        ("reward/fc_snapshot_verify_rate", "snapshot verify rate (gate active)"),
        ("reward/fc_positive_reward_rate", "positive-reward rate"),
        ("generate/num_error_trajectories", "errored trajectories"),
    ]
    fig, axes = plt.subplots(4, 2, figsize=(13, 15))
    for ax, (key, label) in zip(axes.flat, panels):
        xs, ys = series(key)
        if xs:
            ax.plot(xs, ys, marker="o", ms=3)
        ax.set_title(label, fontsize=10)
        ax.set_xlabel("step")
        ax.grid(alpha=0.3)
    fig.suptitle(f"{out.parent.name} — training audit", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    p = out / "training_curves.png"
    fig.savefig(p, dpi=110)
    plt.close(fig)
    made.append(str(p))

    # Per-task speedup distribution (intra-group diversity of the outcome).
    if by_task:
        fig, ax = plt.subplots(figsize=(11, 6))
        labels, data = [], []
        for task, recs in sorted(by_task.items()):
            sp = [r["lsv_mean_speedup"] for r in recs
                  if isinstance(r["lsv_mean_speedup"], (int, float))]
            if sp:
                labels.append(task.replace("__", "/")[:28])
                data.append(sp)
        if data:
            ax.boxplot(data, labels=labels, showfliers=True)
            ax.axhline(1.0, color="grey", ls="--", lw=1, label="no change")
            ax.axhline(1.05, color="orange", ls=":", lw=1, label="noise floor 1.05x")
            ax.set_ylabel("agent geomean speedup (x)")
            ax.set_title("Per-task agent speedup distribution (within-group spread)")
            ax.legend(fontsize=8)
            plt.setp(ax.get_xticklabels(), rotation=25, ha="right", fontsize=8)
            fig.tight_layout()
            p = out / "per_task_speedup.png"
            fig.savefig(p, dpi=110)
            plt.close(fig)
            made.append(str(p))
    return made


# ── report ────────────────────────────────────────────────────────────────────
def build_stats(steps: list[dict], by_task: dict) -> dict:
    last = steps[-1] if steps else {}
    stats: dict = {
        "n_steps_logged": len(steps),
        "final_step_metrics": {k: last[k] for k in _TREND_KEYS if k in last},
    }
    # learning-signal health: is the gradient real?
    gn = [s["grad_norm"] for s in steps if "grad_norm" in s]
    rl = [s.get("response_length") for s in steps if s.get("response_length") is not None]
    stats["learning_signal"] = {
        "grad_norm_mean": mean(gn) if gn else None,
        "grad_norm_all_zero": bool(gn) and all(g == 0 for g in gn),
        "response_length_mean": mean(rl) if rl else None,
        "response_length_collapsed": bool(rl) and all(r <= 1 for r in rl),
    }
    per_task = {}
    for task, recs in sorted(by_task.items()):
        sp = [r["lsv_mean_speedup"] for r in recs
              if isinstance(r["lsv_mean_speedup"], (int, float))]
        per_task[task] = {
            "n_trials": len(recs),
            "speedup_mean": mean(sp) if sp else None,
            "speedup_max": max(sp) if sp else None,
            "speedup_std": pstdev(sp) if len(sp) >= 2 else 0.0,
            "n_beat_noise_floor": sum(1 for s in sp if s > 1.05),
            "verify_ran_rate": mean([1.0 if r["snapshot_verify_ran"] else 0.0 for r in recs]) if recs else 0.0,
            "n_regressions": sum(1 for r in recs if r["snapshot_passed_real"] is False),
        }
    stats["per_task"] = per_task
    return stats


def write_report(stats: dict, out: Path) -> None:
    L = []
    L.append(f"FormulaCode GRPO run audit — {out.parent.name}")
    L.append("=" * 60)
    ls = stats["learning_signal"]
    L.append("\nLEARNING SIGNAL")
    L.append(f"  steps logged            : {stats['n_steps_logged']}")
    L.append(f"  grad_norm mean          : {ls['grad_norm_mean']}")
    if ls["grad_norm_all_zero"]:
        L.append("  !! grad_norm is ZERO on every step — NO learning is happening.")
    if ls["response_length_collapsed"]:
        L.append("  !! response_length collapsed to <=1 — trajectories are placeholders (masked).")
    L.append(f"  response_length mean    : {ls['response_length_mean']}")
    fsm = stats["final_step_metrics"]
    if fsm:
        L.append("\nFINAL-STEP METRICS")
        for k, v in fsm.items():
            L.append(f"  {k:<40}: {v}")
    L.append("\nPER-TASK (speedup + correctness gate)")
    for task, s in stats["per_task"].items():
        L.append(f"  {task}")
        L.append(f"     trials={s['n_trials']}  speedup mean={s['speedup_mean']}  max={s['speedup_max']}"
                 f"  std={s['speedup_std']:.4f}" if s["speedup_mean"] is not None else
                 f"     trials={s['n_trials']}  (no numeric speedups)")
        L.append(f"     beat 1.05x: {s['n_beat_noise_floor']}/{s['n_trials']}"
                 f"   verify_ran_rate={s['verify_ran_rate']:.2f}"
                 f"   regressions={s['n_regressions']}")
    (out / "report.txt").write_text("\n".join(L) + "\n")


def write_metrics_csv(steps: list[dict], out: Path) -> None:
    keys = []
    for s in steps:
        for k in s:
            if k not in keys:
                keys.append(k)
    lines = ["step," + ",".join(keys)]
    for i, s in enumerate(steps):
        lines.append(str(i) + "," + ",".join(str(s.get(k, "")) for k in keys))
    (out / "metrics.csv").write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", help="path to results/<run_name>")
    ap.add_argument("--oracle-table", default=None, help="gold_oracle_benchmarks.json (optional context)")
    args = ap.parse_args()

    run_dir = Path(args.run_dir).resolve()
    if not run_dir.exists():
        sys.exit(f"run dir not found: {run_dir}")
    out = run_dir / "analysis"
    out.mkdir(exist_ok=True)

    steps = parse_run_log(run_dir / "run.log")
    by_task = parse_trials(run_dir / "trials")
    print(f"[analyze] {len(steps)} steps, {sum(len(v) for v in by_task.values())} trials "
          f"across {len(by_task)} tasks")

    write_metrics_csv(steps, out)
    stats = build_stats(steps, by_task)
    (out / "stats.json").write_text(json.dumps(stats, indent=2))
    write_report(stats, out)
    made = make_plots(steps, by_task, out)
    print(f"[analyze] wrote: metrics.csv, stats.json, report.txt, {len(made)} plots -> {out}")
    print((out / "report.txt").read_text())


if __name__ == "__main__":
    main()

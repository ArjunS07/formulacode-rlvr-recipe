#!/usr/bin/env python3
"""
RL training eval script for formulacode-grpo-3task-g8.

Usage:
    uv run --with matplotlib --with seaborn --with scipy python eval_training.py \
        results/formulacode-grpo-3task-g8/ [--trajectories]

Generates plots to results/formulacode-grpo-3task-g8/analysis/plots/
and a text report to results/formulacode-grpo-3task-g8/analysis/report.txt

Reads:
  - run.log  : per-step aggregate metrics from SkyRL trainer
  - results/trials/<task>_agent_* : per-trial reward.json + trajectory

Plots:
  P1  Reward curve + response length (from log)
  P2  Entropy + overlong filtering rate (from log)
  P3  Timing breakdown per step (from log)
  P4  Per-task reward over steps (from trials, grouped by step)
  P5  GRPO signal quality: within-group reward std per task/step
  P6  Reward distribution per task (all trials, violin)
  P7  Reward taxonomy over steps (stacked bar)
  P8  Patch application rate + patch size vs reward (from trials)
  P9  Agent turn count distribution (from trials)
"""

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import seaborn as sns

TASKS = [
    "pvlib_pvlib-python_369",
    "joblib_joblib_484",
    "networkx_networkx_8148",
]
TASK_SHORT = {
    "pvlib_pvlib-python_369":   "pvlib",
    "joblib_joblib_484":        "joblib",
    "networkx_networkx_8148":   "networkx",
}
ORACLE_H = {
    "pvlib":    22.563,
    "joblib":   1.9524,
    "networkx": 1.3229,
}
REWARD_FORMULA_H = {k: v for k, v in ORACLE_H.items()}

PALETTE = sns.color_palette("colorblind")
TASK_COLOR = {t: PALETTE[i] for i, t in enumerate(TASKS)}
TASK_SHORT_COLOR = {TASK_SHORT[t]: PALETTE[i] for i, t in enumerate(TASKS)}

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
TS_RE   = re.compile(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+)")

sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams.update({"figure.dpi": 130, "savefig.dpi": 200, "savefig.bbox": "tight"})


# ── Log parsing ───────────────────────────────────────────────────────────────

def parse_run_log(logfile: Path) -> list[dict]:
    """Parse SkyRL run.log → list of dicts, one per completed step."""
    rows = []
    seen: set = set()
    pending: dict = {}
    waiting: list = []

    pat = {
        "start":       re.compile(r"Started: 'step'"),
        "finish":      re.compile(r"Finished: 'step', time cost: ([\d.]+)s"),
        "global_step": re.compile(r"'trainer/global_step': (\d+)"),
        "avg_reward":  re.compile(r"'reward/avg_final_rewards': '([\d.]+)'"),
        "raw_reward":  re.compile(r"'reward/avg_raw_reward': '([\d.]+)'"),
        "pass_at_k":   re.compile(r"'reward/avg_pass_at_(\d+)': '([\d.]+)'"),
        "entropy":     re.compile(r"'policy/policy_entropy': '([\d.]+)'"),
        "resp_len":    re.compile(r"'policy/response_length': '([\d.]+)'"),
        "gen_time":    re.compile(r"'timing/generate': '([\d.]+)'"),
        "train_time":  re.compile(r"'timing/policy_train': '([\d.]+)'"),
        "fwd_time":    re.compile(r"'timing/fwd_logprobs_values_reward': '([\d.]+)'"),
        "num_seq":     re.compile(r"'trainer/num_seq_after_merge': '(\d+)'"),
        "total_traj":  re.compile(r"total trajectories before merge: (\d+)"),
        "context_exc": re.compile(r"(\d+) trajectories exceeded context length"),
    }

    for raw in logfile.read_text(errors="replace").splitlines():
        line = ANSI_RE.sub("", raw)
        key = line.strip()
        if key in seen:
            seen.clear()
            continue
        seen = {key}

        ts_m = TS_RE.search(line)
        ts = ts_m.group(1) if ts_m else None

        if pat["start"].search(line):
            pending = {"step_start": ts}

        for k, p in pat.items():
            if k in ("start", "finish", "pass_at_k"):
                continue
            m = p.search(line)
            if m:
                pending[k] = float(m.group(1)) if "." in m.group(1) else int(m.group(1))

        for m in pat["pass_at_k"].finditer(line):
            pending[f"pass_at_{m.group(1)}"] = float(m.group(2))

        m = pat["finish"].search(line)
        if m:
            pending["step_time_s"] = float(m.group(1))
            pending["step_end"] = ts
            if "global_step" in pending:
                rows.append(dict(pending))
                pending = {}
            else:
                waiting.append(dict(pending))
                pending = {}

        m = pat["global_step"].search(line)
        if m:
            step_num = int(m.group(1))
            if "global_step" not in pending:
                pending["global_step"] = step_num
            if waiting:
                waiting[-1]["global_step"] = step_num
                rows.append(waiting.pop())

    for row in waiting:
        if "global_step" not in row:
            row["global_step"] = len(rows) + 1
        rows.append(row)

    seen_steps: set = set()
    deduped = []
    for r in rows:
        s = r.get("global_step", -1)
        if s not in seen_steps:
            seen_steps.add(s)
            deduped.append(r)
    return sorted(deduped, key=lambda r: r.get("global_step", 0))


# ── Trial loading ─────────────────────────────────────────────────────────────

def load_trials(trials_dir: Path, run_start: datetime) -> list[dict]:
    rows = []
    for td in sorted(trials_dir.iterdir()):
        rj = td / "result.json"
        if not rj.exists():
            continue
        try:
            r = json.loads(rj.read_text())
        except Exception:
            continue

        task_name = r.get("task_name", "")
        if task_name not in TASKS:
            continue

        started_str = r.get("started_at", "")
        if not started_str:
            continue
        try:
            started = datetime.fromisoformat(started_str.replace("Z", "+00:00"))
        except Exception:
            continue
        if started < run_start:
            continue

        reward = None
        rw = td / "verifier" / "reward.txt"
        if rw.exists():
            try:
                reward = float(rw.read_text().strip())
            except Exception:
                pass
        if reward is None:
            reward = r.get("reward")

        speedup = None
        lsv_error = None
        rj2 = td / "verifier" / "reward.json"
        if rj2.exists():
            try:
                rdata = json.loads(rj2.read_text())
                speedup = rdata.get("lsv_mean_speedup")
                lsv_error = rdata.get("lsv_error")
            except Exception:
                pass

        patch_applied = False
        patch_added = patch_removed = 0
        patch_path = td / "artifacts" / "patch.diff"
        if patch_path.exists():
            txt = patch_path.read_text(errors="replace")
            patch_applied = bool(txt.strip())
            for line in txt.splitlines():
                if line.startswith("+") and not line.startswith("+++"):
                    patch_added += 1
                elif line.startswith("-") and not line.startswith("---"):
                    patch_removed += 1

        n_turns = 0
        traj_path = td / "agent" / "trajectory.json"
        if traj_path.exists():
            try:
                traj = json.loads(traj_path.read_text())
                n_turns = len(traj.get("steps", []))
            except Exception:
                pass

        rows.append({
            "trial_id":     td.name,
            "task":         task_name,
            "short":        TASK_SHORT[task_name],
            "started":      started,
            "reward":       reward,
            "speedup":      speedup,
            "lsv_error":    lsv_error,
            "patch":        patch_applied,
            "added":        patch_added,
            "removed":      patch_removed,
            "n_turns":      n_turns,
        })

    return sorted(rows, key=lambda r: r["started"])


def assign_step(trials: list[dict], log_rows: list[dict]) -> list[dict]:
    """Tag each trial with a training step number using log timing."""
    if not log_rows:
        for i, t in enumerate(trials):
            t["step"] = None
        return trials

    windows = []
    for row in log_rows:
        s, e = row.get("step_start"), row.get("step_end")
        step = row.get("global_step")
        if s and e and step is not None:
            windows.append((step,
                            datetime.fromisoformat(s),
                            datetime.fromisoformat(e)))

    for t in trials:
        ts = t["started"].replace(tzinfo=None)
        t["step"] = None
        for step, s, e in windows:
            if s <= ts <= e:
                t["step"] = step
                break

    return trials


def taxonomy(trial: dict) -> str:
    r = trial.get("reward")
    if r is None:
        return "error"
    if not trial.get("patch"):
        return "no_source_edit"
    if r <= 0:
        return "broken_tests"
    sp = trial.get("speedup") or 0
    if sp <= 1.0:
        return "no_speedup"
    short = trial["short"]
    h = ORACLE_H.get(short, 1.5)
    if sp >= h * 0.95:
        return "oracle_level"
    return "partial"


# ── Plots ─────────────────────────────────────────────────────────────────────

TAX_ORDER  = ["oracle_level", "partial", "no_speedup", "broken_tests", "no_source_edit", "error"]
TAX_COLORS = {
    "oracle_level":  PALETTE[2],
    "partial":       PALETTE[0],
    "no_speedup":    PALETTE[3],
    "broken_tests":  PALETTE[1],
    "no_source_edit": PALETTE[4],
    "error":         "gray",
}
TAX_LABELS = {
    "oracle_level":  "Oracle-level",
    "partial":       "Partial speedup",
    "no_speedup":    "No speedup",
    "broken_tests":  "Broken tests",
    "no_source_edit": "No source edit",
    "error":         "Error/timeout",
}


def p1_reward_curve(log_rows: list[dict], out: Path):
    if not log_rows or "avg_reward" not in log_rows[0]:
        return
    steps = [r["global_step"] for r in log_rows]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    ax = axes[0]
    rewards = [r.get("avg_reward") for r in log_rows]
    raw = [r.get("raw_reward") for r in log_rows]
    ax.plot(steps, rewards, marker="o", color="steelblue", lw=2, ms=6, label="avg_final_reward")
    if any(v is not None for v in raw):
        ax.plot(steps, raw, marker="s", color="darkorange", lw=1.5, ms=5,
                ls="--", label="avg_raw_reward")
    ax.axhline(1.0, color="gray", ls="--", lw=1, label="g=1 (no speedup)")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Reward")
    ax.set_title("Training reward (GRPO, 3 tasks)")
    ax.legend()
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    ax = axes[1]
    resp = [r.get("resp_len") for r in log_rows]
    if any(v is not None for v in resp):
        ax.plot(steps, resp, marker="s", color="purple", lw=2, ms=5)
        ax.axhline(65536, color="red", ls="--", lw=1, label="max_seq_len (65536)")
        ax.set_xlabel("Training step")
        ax.set_ylabel("Avg response length (tokens)")
        ax.set_title("Response length over training")
        ax.legend()
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    plt.tight_layout()
    fig.savefig(out / "P1_reward_curve.png")
    plt.close(fig)


def p2_entropy_overlong(log_rows: list[dict], out: Path):
    if not log_rows:
        return
    steps = [r["global_step"] for r in log_rows]
    entropy = [r.get("entropy") for r in log_rows]
    num_seq = [r.get("num_seq") for r in log_rows]
    total   = [r.get("total_traj") for r in log_rows]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    ax = axes[0]
    if any(v is not None for v in entropy):
        ax.plot(steps, entropy, marker="o", color="teal", lw=2, ms=6)
        ax.axhline(0, color="red", ls="--", lw=1, label="entropy=0 (mode collapse)")
        ax.set_xlabel("Training step")
        ax.set_ylabel("Policy entropy")
        ax.set_title("Policy entropy over training")
        ax.legend()
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    ax = axes[1]
    filt_rates = []
    for ns, tot in zip(num_seq, total):
        if ns is not None and tot is not None and tot > 0:
            filt_rates.append(1.0 - ns / tot)
        else:
            filt_rates.append(None)
    if any(v is not None for v in filt_rates):
        ax.plot(steps, filt_rates, marker="s", color="crimson", lw=2, ms=5)
        ax.set_xlabel("Training step")
        ax.set_ylabel("Fraction of trajectories filtered (overlong)")
        ax.set_title("Overlong filtering rate")
        ax.set_ylim(-0.05, 1.05)
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    plt.tight_layout()
    fig.savefig(out / "P2_entropy_overlong.png")
    plt.close(fig)


def p3_timing(log_rows: list[dict], out: Path):
    has_timing = any(r.get("gen_time") is not None for r in log_rows)
    if not log_rows or not has_timing:
        return
    steps = [r["global_step"] for r in log_rows]
    gen   = [r.get("gen_time", 0) / 60 for r in log_rows]
    train = [r.get("train_time", 0) / 60 for r in log_rows]
    fwd   = [r.get("fwd_time", 0) / 60 for r in log_rows]

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.stackplot(steps, fwd, train, gen,
                 labels=["fwd pass (min)", "policy train (min)", "generate (min)"],
                 colors=[PALETTE[0], PALETTE[2], PALETTE[1]], alpha=0.8)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Time (minutes)")
    ax.set_title("Per-step timing breakdown")
    ax.legend(loc="upper right")
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    plt.tight_layout()
    fig.savefig(out / "P3_timing.png")
    plt.close(fig)


def p4_per_task_reward(trials: list[dict], out: Path):
    df_rows = [t for t in trials if t["reward"] is not None and t["step"] is not None]
    if not df_rows:
        return
    fig, ax = plt.subplots(figsize=(12, 5))
    for task in TASKS:
        td = [r for r in df_rows if r["task"] == task]
        if not td:
            continue
        by_step = defaultdict(list)
        for r in td:
            by_step[r["step"]].append(r["reward"])
        steps_s = sorted(by_step)
        means = [np.mean(by_step[s]) for s in steps_s]
        stds  = [np.std(by_step[s]) for s in steps_s]
        color = TASK_COLOR[task]
        ax.plot(steps_s, means, marker="o", label=TASK_SHORT[task], color=color, lw=2, ms=6)
        ax.fill_between(steps_s,
                        [m - s for m, s in zip(means, stds)],
                        [m + s for m, s in zip(means, stds)],
                        alpha=0.15, color=color)
    ax.axhline(1.0, color="gray", ls="--", lw=1)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Mean reward ± 1 std")
    ax.set_title("Per-task reward over training")
    ax.legend()
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    plt.tight_layout()
    fig.savefig(out / "P4_per_task_reward.png")
    plt.close(fig)


def p5_grpo_signal(trials: list[dict], out: Path):
    """Within-group reward variance per task/step — GRPO needs this > 0."""
    df_rows = [t for t in trials if t["reward"] is not None and t["step"] is not None]
    if not df_rows:
        return
    fig, ax = plt.subplots(figsize=(12, 5))
    for task in TASKS:
        td = [r for r in df_rows if r["task"] == task]
        if not td:
            continue
        by_step = defaultdict(list)
        for r in td:
            by_step[r["step"]].append(r["reward"])
        steps_s = sorted(by_step)
        stds = [np.std(by_step[s]) if len(by_step[s]) >= 2 else 0 for s in steps_s]
        ax.plot(steps_s, stds, marker="o", label=TASK_SHORT[task],
                color=TASK_COLOR[task], lw=2, ms=6)
    ax.axhline(0, color="red", ls="--", lw=1, label="zero variance (no GRPO signal)")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Within-group reward std (n=8)")
    ax.set_title("GRPO signal quality: reward variance per task/step")
    ax.legend()
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    plt.tight_layout()
    fig.savefig(out / "P5_grpo_signal.png")
    plt.close(fig)


def p6_reward_dist(trials: list[dict], out: Path):
    df_rows = [t for t in trials if t["reward"] is not None]
    if not df_rows:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    order = [TASK_SHORT[t] for t in TASKS]
    data_by_task = {TASK_SHORT[t]: [r["reward"] for r in df_rows if r["task"] == t]
                    for t in TASKS}
    parts = ax.violinplot([data_by_task[o] for o in order if data_by_task[o]],
                          positions=range(len(order)),
                          showmedians=True, showextrema=False)
    for i, (patch, task) in enumerate(zip(parts["bodies"], order)):
        patch.set_facecolor(TASK_SHORT_COLOR.get(task, "gray"))
        patch.set_alpha(0.6)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order)
    ax.axhline(1.0, color="gray", ls="--", lw=1, label="g=1 (no speedup)")
    ax.axhline(3.5, color="green", ls=":", lw=1, label="reward at g=H (oracle)")
    ax.set_xlabel("Task")
    ax.set_ylabel("Reward")
    ax.set_title("Reward distribution per task (all trials)")
    ax.legend()
    plt.tight_layout()
    fig.savefig(out / "P6_reward_dist.png")
    plt.close(fig)


def p7_taxonomy_over_steps(trials: list[dict], out: Path):
    df_rows = [t for t in trials if t["step"] is not None]
    if not df_rows:
        return
    steps = sorted(set(r["step"] for r in df_rows))
    tax_by_step = defaultdict(lambda: defaultdict(int))
    for r in df_rows:
        tax_by_step[r["step"]][taxonomy(r)] += 1

    bottoms = {k: [0] * len(steps) for k in TAX_ORDER}
    fig, ax = plt.subplots(figsize=(max(10, len(steps) * 0.8), 5))
    prev = [0] * len(steps)
    for cat in TAX_ORDER:
        vals = [tax_by_step[s].get(cat, 0) for s in steps]
        ax.bar(steps, vals, bottom=prev, label=TAX_LABELS[cat],
               color=TAX_COLORS[cat], alpha=0.85)
        prev = [p + v for p, v in zip(prev, vals)]
    ax.set_xlabel("Training step")
    ax.set_ylabel("Trial count")
    ax.set_title("Trial taxonomy over training steps")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    plt.tight_layout()
    fig.savefig(out / "P7_taxonomy.png")
    plt.close(fig)


def p8_patch_analysis(trials: list[dict], out: Path):
    patched = [t for t in trials if t["patch"] and t["reward"] is not None]
    if not patched:
        return
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    ax = axes[0]
    for task in TASKS:
        td = [r for r in trials if r["task"] == task and r["step"] is not None]
        if not td:
            continue
        by_step = defaultdict(list)
        for r in td:
            by_step[r["step"]].append(int(r["patch"]))
        steps_s = sorted(by_step)
        rates = [np.mean(by_step[s]) for s in steps_s]
        ax.plot(steps_s, rates, marker="o", label=TASK_SHORT[task],
                color=TASK_COLOR[task], lw=2)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Fraction of trials that patched source")
    ax.set_title("Patch application rate over training")
    ax.set_ylim(-0.05, 1.05)
    ax.legend()
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    ax = axes[1]
    for task in TASKS:
        td = [r for r in patched if r["task"] == task]
        if not td:
            continue
        sizes = [r["added"] + r["removed"] for r in td]
        rewards = [r["reward"] for r in td]
        ax.scatter(sizes, rewards, alpha=0.4, s=18,
                   label=TASK_SHORT[task], color=TASK_COLOR[task])
    ax.axhline(1.0, color="gray", ls="--", lw=1)
    ax.set_xlabel("Lines changed (added + removed)")
    ax.set_ylabel("Reward")
    ax.set_title("Patch size vs reward")
    ax.legend()

    plt.tight_layout()
    fig.savefig(out / "P8_patch_analysis.png")
    plt.close(fig)


def p9_turn_count(trials: list[dict], out: Path):
    with_turns = [t for t in trials if t["n_turns"] > 0]
    if not with_turns:
        return
    fig, ax = plt.subplots(figsize=(10, 4.5))
    order = [TASK_SHORT[t] for t in TASKS]
    data = {TASK_SHORT[t]: [r["n_turns"] for r in with_turns if r["task"] == t] for t in TASKS}
    positions = [i for i, o in enumerate(order) if data[o]]
    labels    = [o for o in order if data[o]]
    vals      = [data[o] for o in order if data[o]]
    parts = ax.violinplot(vals, positions=positions, showmedians=True, showextrema=False)
    for patch, lbl in zip(parts["bodies"], labels):
        patch.set_facecolor(TASK_SHORT_COLOR.get(lbl, "gray"))
        patch.set_alpha(0.6)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.axhline(96, color="red", ls="--", lw=1, label="max_turns=96")
    ax.set_xlabel("Task")
    ax.set_ylabel("Agent turns per episode")
    ax.set_title("Agent trajectory length per task")
    ax.legend()
    plt.tight_layout()
    fig.savefig(out / "P9_turn_count.png")
    plt.close(fig)


# ── Report ────────────────────────────────────────────────────────────────────

def write_report(log_rows: list[dict], trials: list[dict], out: Path) -> Path:
    lines = [
        "FormulaCode RLVR — 3-Task Training Report",
        f"Generated: {datetime.now().isoformat()}",
        "=" * 70,
    ]

    lines.append("\n## Training progress (from run.log)")
    if log_rows:
        latest = log_rows[-1]
        first  = log_rows[0]
        lines.append(f"  Steps completed: {latest['global_step']}")
        lines.append(f"  Avg reward  — step 1: {first.get('avg_reward', '?'):.4f}  "
                     f"latest: {latest.get('avg_reward', '?'):.4f}")
        if "entropy" in latest:
            lines.append(f"  Entropy     — step 1: {first.get('entropy', '?'):.4f}  "
                         f"latest: {latest.get('entropy', '?'):.4f}")
        if "resp_len" in latest:
            lines.append(f"  Resp length — step 1: {first.get('resp_len', '?'):.0f}  "
                         f"latest: {latest.get('resp_len', '?'):.0f}  (max_seq=65536)")
        num_s = latest.get("num_seq")
        tot   = latest.get("total_traj")
        if num_s and tot:
            lines.append(f"  Overlong filter (latest): {1 - num_s/tot:.1%} of trajectories excluded")
        if "gen_time" in latest:
            lines.append(f"  Latest step timing: generate={latest['gen_time']/60:.1f}m  "
                         f"train={latest.get('train_time', 0)/60:.1f}m")
    else:
        lines.append("  No completed steps yet.")

    lines.append("\n## Per-task summary (all trials in run)")
    for task in TASKS:
        td = [r for r in trials if r["task"] == task]
        if not td:
            continue
        rewards = [r["reward"] for r in td if r["reward"] is not None]
        patches = sum(1 for r in td if r["patch"])
        oracle_h = ORACLE_H[TASK_SHORT[task]]
        oracle_level = sum(1 for r in td if r.get("speedup") and r["speedup"] >= oracle_h * 0.95)
        lines.append(f"\n  {TASK_SHORT[task]} (H={oracle_h:.3f}):")
        lines.append(f"    Total trials: {len(td)}  patched: {patches}  "
                     f"oracle-level: {oracle_level} ({oracle_level/max(len(td),1)*100:.0f}%)")
        if rewards:
            lines.append(f"    Reward — mean: {np.mean(rewards):.3f}  "
                         f"std: {np.std(rewards):.3f}  "
                         f"min: {min(rewards):.3f}  max: {max(rewards):.3f}")
        tax_counts = defaultdict(int)
        for r in td:
            tax_counts[taxonomy(r)] += 1
        lines.append(f"    Taxonomy: " +
                     "  ".join(f"{k}={v}" for k, v in tax_counts.items() if v > 0))

    lines.append("\n## GRPO signal quality (within-group reward std per step)")
    df_rows = [t for t in trials if t["reward"] is not None and t["step"] is not None]
    if df_rows:
        for task in TASKS:
            td = [r for r in df_rows if r["task"] == task]
            if not td:
                continue
            by_step = defaultdict(list)
            for r in td:
                by_step[r["step"]].append(r["reward"])
            stds = [np.std(v) for v in by_step.values() if len(v) >= 2]
            if stds:
                lines.append(f"  {TASK_SHORT[task]:12s}: "
                             f"mean std={np.mean(stds):.3f}  "
                             f"steps with std>0.1: {sum(s > 0.1 for s in stds)}/{len(stds)}")
    else:
        lines.append("  Not enough data (need step assignments).")

    path = out / "report.txt"
    path.write_text("\n".join(lines))
    print("\n".join(lines))
    return path


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", help="Path to run directory (e.g. results/formulacode-grpo-3task-g8/)")
    parser.add_argument("--trials-dir", default=None, help="Override trials dir (default: results/trials/)")
    parser.add_argument("--run-start", default=None, help="ISO datetime for run start filter")
    args = parser.parse_args()

    run_dir   = Path(args.run_dir)
    repo_root = Path(__file__).parent
    trials_dir = Path(args.trials_dir) if args.trials_dir else repo_root / "results" / "trials"
    logfile   = run_dir / "run.log"
    out_dir   = run_dir / "analysis"
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    print(f"Run dir:    {run_dir}")
    print(f"Log:        {logfile} ({'exists' if logfile.exists() else 'MISSING'})")
    print(f"Trials dir: {trials_dir}")
    print(f"Output:     {out_dir}")

    log_rows = parse_run_log(logfile) if logfile.exists() else []
    print(f"Completed steps in log: {len(log_rows)}")

    if args.run_start:
        run_start = datetime.fromisoformat(args.run_start).replace(tzinfo=timezone.utc)
    elif logfile.exists() and logfile.stat().st_size:
        first_line = logfile.read_text().splitlines()[0]
        m = re.search(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", first_line)
        run_start = (datetime.fromisoformat(m.group(1)).replace(tzinfo=timezone.utc)
                     if m else datetime(2026, 6, 7, tzinfo=timezone.utc))
    else:
        run_start = datetime(2026, 6, 7, tzinfo=timezone.utc)

    print(f"Run start: {run_start.isoformat()}")

    trials = load_trials(trials_dir, run_start) if trials_dir.exists() else []
    trials = assign_step(trials, log_rows)
    print(f"Trials loaded: {len(trials)}")

    print("\nGenerating plots...")
    p1_reward_curve(log_rows, plots_dir)
    p2_entropy_overlong(log_rows, plots_dir)
    p3_timing(log_rows, plots_dir)
    p4_per_task_reward(trials, plots_dir)
    p5_grpo_signal(trials, plots_dir)
    p6_reward_dist(trials, plots_dir)
    p7_taxonomy_over_steps(trials, plots_dir)
    p8_patch_analysis(trials, plots_dir)
    p9_turn_count(trials, plots_dir)
    print(f"Plots saved to {plots_dir}/")

    report_path = write_report(log_rows, trials, out_dir)
    print(f"\nReport: {report_path}")


if __name__ == "__main__":
    main()

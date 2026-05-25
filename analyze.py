#!/usr/bin/env python3
"""
FormulaCode RLVR — scientific analysis of a training run.

Usage:
    uv run --with matplotlib --with seaborn --with scipy \
        python /mnt/sdd3/asharma/formulacode-rlvr/analyze.py \
        --run formulacode-grpo-g8-big \
        --out /mnt/sdd3/asharma/formulacode-rlvr/analysis

Generates:
  - plots/  : PNG figures
  - report.txt : structured text report
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
import pandas as pd
import seaborn as sns

# ── Config ───────────────────────────────────────────────────────────────────

RESULTS_ROOT = Path("/mnt/sdd3/asharma/formulacode-rlvr/results")
TRAIN_TASKS = [
    "shapely_shapely_2359",
    "pvlib_pvlib-python_369",
    "optuna_optuna_4540",
    "joblib_joblib_484",
    "microsoft_Qcodes_7669",
]
TASK_SHORT = {
    "shapely_shapely_2359":     "shapely",
    "pvlib_pvlib-python_369":   "pvlib",
    "optuna_optuna_4540":       "optuna",
    "joblib_joblib_484":        "joblib",
    "microsoft_Qcodes_7669":    "Qcodes",
}
PALETTE = sns.color_palette("tab10", len(TRAIN_TASKS))
TASK_COLOR = {t: PALETTE[i] for i, t in enumerate(TRAIN_TASKS)}

sns.set_theme(style="whitegrid", font_scale=1.15)

# ── Parsing helpers ──────────────��──────────────────────��─────────────────────

def parse_run_log(logfile: Path) -> pd.DataFrame:
    """Parse SkyRL run.log → DataFrame with one row per completed step.

    Handles two log formats:
    - New: 'trainer/global_step' appears inside the step (before Finished)
    - Old: 'trainer/global_step' appears after Finished in the metrics dict
    In both cases, deduplicates Ray-doubled log lines.
    """
    rows = []
    seen_lines: set = set()

    reward_re  = re.compile(r"avg_final_rewards: ([\d.]+).*avg_response_length: ([\d.]+)")
    passat_re  = re.compile(r"reward/avg_pass_at_(\d+): ([\d.]+)")
    raw_re     = re.compile(r"reward/avg_raw_reward: ([\d.]+)")
    step_re    = re.compile(r"'trainer/global_step': (\d+)")
    start_re   = re.compile(r"Started: 'step'")
    finish_re  = re.compile(r"Finished: 'step', time cost: ([\d.]+)s")
    ts_re      = re.compile(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+)")
    # strip ANSI escape codes
    ansi_re    = re.compile(r"\x1b\[[0-9;]*m")

    pending = {}
    # buffer rows that finished but haven't received global_step yet
    waiting_for_step: list = []

    for raw_line in logfile.read_text(errors="replace").splitlines():
        line = ansi_re.sub("", raw_line)
        # deduplicate Ray-doubled lines (same content seen twice in a row)
        key = line.strip()
        if key in seen_lines:
            seen_lines.clear()
            continue
        seen_lines = {key}

        ts_m = ts_re.search(line)
        ts = datetime.fromisoformat(ts_m.group(1)) if ts_m else None

        if start_re.search(line):
            pending = {"step_start": ts}

        m = reward_re.search(line)
        if m:
            pending["avg_final_reward"] = float(m.group(1))
            pending["avg_response_length"] = float(m.group(2))

        m = raw_re.search(line)
        if m:
            pending["avg_raw_reward"] = float(m.group(1))

        for m in passat_re.finditer(line):
            pending[f"pass_at_{m.group(1)}"] = float(m.group(2))

        m = step_re.search(line)
        if m:
            step_num = int(m.group(1))
            pending["global_step"] = step_num
            # also patch any row that finished before seeing global_step
            if waiting_for_step:
                waiting_for_step[-1]["global_step"] = step_num
                rows.append(waiting_for_step.pop())

        m = finish_re.search(line)
        if m:
            pending["step_time_s"] = float(m.group(1))
            pending["step_end"] = ts
            if "global_step" in pending:
                rows.append(dict(pending))
                pending = {}
            else:
                # global_step comes after Finished in old-format logs
                waiting_for_step.append(dict(pending))
                pending = {}

    # flush any waiting rows — assign sequential numbers if still missing
    for row in waiting_for_step:
        if "global_step" not in row:
            row["global_step"] = len(rows) + 1
        rows.append(row)

    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    # assign sequential step index if global_step has gaps or duplicates
    df = df.drop_duplicates("global_step").sort_values("global_step").reset_index(drop=True)
    if df["global_step"].iloc[0] != 1:
        df["global_step"] = range(1, len(df) + 1)
    return df


def load_trials(trials_dir: Path, run_start: datetime) -> pd.DataFrame:
    """
    Scan trials_dir for all trials belonging to this run (started after run_start).
    Returns DataFrame with per-trial reward, task, patch stats, agent stats.
    """
    rows = []
    for td in sorted(trials_dir.iterdir()):
        rj = td / "result.json"
        if not rj.exists():
            continue
        try:
            r = json.loads(rj.read_text())
        except Exception:
            continue

        started_at_str = r.get("started_at", "")
        if not started_at_str:
            continue
        try:
            started_at = datetime.fromisoformat(started_at_str.replace("Z", "+00:00"))
        except Exception:
            continue

        # Only include trials from this run
        if started_at < run_start:
            continue

        task_name = r.get("task_name", "unknown")
        if task_name not in TRAIN_TASKS:
            continue

        # Reward: prefer verifier/reward.txt
        reward = None
        rw_path = td / "verifier" / "reward.txt"
        if rw_path.exists():
            try:
                reward = float(rw_path.read_text().strip())
            except Exception:
                pass
        if reward is None:
            reward = r.get("reward")

        # Patch stats
        patch_path = td / "artifacts" / "patch.diff"
        patch_applied = False
        patch_files = 0
        patch_added = 0
        patch_removed = 0
        if patch_path.exists():
            patch_text = patch_path.read_text(errors="replace")
            patch_applied = len(patch_text.strip()) > 0
            for line in patch_text.splitlines():
                if line.startswith("diff --git"):
                    patch_files += 1
                elif line.startswith("+") and not line.startswith("+++"):
                    patch_added += 1
                elif line.startswith("-") and not line.startswith("---"):
                    patch_removed += 1

        # Agent stats from trajectory
        n_turns = 0
        traj_path = td / "agent" / "trajectory.json"
        if traj_path.exists():
            try:
                traj = json.loads(traj_path.read_text())
                n_turns = len(traj.get("steps", []))
            except Exception:
                pass

        # LSV stats from test-stdout
        n_benchmarks = 0
        max_speedup = None
        stdout_path = td / "verifier" / "test-stdout.txt"
        if stdout_path.exists():
            txt = stdout_path.read_text(errors="replace")
            speedups = re.findall(r"\(([+-][\d.]+)%\)", txt)
            if speedups:
                pcts = [float(x) / 100 for x in speedups]
                max_speedup = max(pcts)
                n_benchmarks = len(pcts)

        # Exception?
        exc_path = td / "exception.txt"
        had_exception = exc_path.exists() and exc_path.stat().st_size > 0

        rows.append({
            "trial_id":       td.name,
            "task_name":      task_name,
            "task_short":     TASK_SHORT.get(task_name, task_name),
            "started_at":     started_at,
            "reward":         reward,
            "patch_applied":  patch_applied,
            "patch_files":    patch_files,
            "patch_added":    patch_added,
            "patch_removed":  patch_removed,
            "n_turns":        n_turns,
            "n_benchmarks":   n_benchmarks,
            "max_speedup":    max_speedup,
            "had_exception":  had_exception,
        })

    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).sort_values("started_at").reset_index(drop=True)
    return df


def assign_steps(trials_df: pd.DataFrame, log_df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign a training step number to each trial using log step timing.
    A trial belongs to step N if its started_at falls within step N's generation window.
    Falls back to ordinal grouping by task if log data is sparse.
    """
    if trials_df.empty:
        return trials_df

    if log_df.empty or "step_start" not in log_df.columns:
        # Ordinal: sort by time, assign 1-indexed groups of 40
        trials_df = trials_df.copy().sort_values("started_at").reset_index(drop=True)
        trials_df["step"] = (trials_df.index // 40) + 1
        return trials_df

    # Build step windows from log
    step_windows = []
    for _, row in log_df.iterrows():
        if pd.notna(row.get("step_start")) and pd.notna(row.get("step_end")):
            step_windows.append((row["global_step"], row["step_start"], row["step_end"]))

    def find_step(ts):
        for step, s, e in step_windows:
            if s.replace(tzinfo=None) <= ts.replace(tzinfo=None) <= e.replace(tzinfo=None):
                return step
        return None

    trials_df = trials_df.copy()
    trials_df["step"] = trials_df["started_at"].apply(find_step)
    # Fill gaps with ordinal
    if trials_df["step"].isna().all():
        trials_df["step"] = (trials_df.sort_values("started_at").index // 40) + 1
    return trials_df


# ── Plots ─────────────────────────────────────────���───────────────────────────

def plot_reward_curve(log_df: pd.DataFrame, out: Path):
    """Training reward curve from SkyRL log."""
    if log_df.empty or "avg_final_reward" not in log_df.columns:
        return
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    ax = axes[0]
    ax.plot(log_df["global_step"], log_df["avg_final_reward"],
            marker="o", color="steelblue", linewidth=2, markersize=6)
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1, label="baseline (1.0)")
    ax.set_xlabel("Training step")
    ax.set_ylabel("avg_final_reward")
    ax.set_title("Training reward (all tasks, GRPO)")
    ax.legend()
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    ax = axes[1]
    if "avg_response_length" in log_df.columns:
        ax.plot(log_df["global_step"], log_df["avg_response_length"],
                marker="s", color="darkorange", linewidth=2, markersize=6)
        ax.set_xlabel("Training step")
        ax.set_ylabel("avg response length (tokens)")
        ax.set_title("Response length over training")
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    plt.tight_layout()
    fig.savefig(out / "01_reward_curve.png", dpi=150)
    plt.close(fig)


def plot_per_task_rewards(trials_df: pd.DataFrame, out: Path):
    """Per-task reward over training steps."""
    df = trials_df[trials_df["reward"].notna() & trials_df["step"].notna()].copy()
    if df.empty:
        return

    fig, ax = plt.subplots(figsize=(12, 5))
    for task in TRAIN_TASKS:
        td = df[df["task_name"] == task]
        if td.empty:
            continue
        # mean reward per step for this task
        gd = td.groupby("step")["reward"].agg(["mean", "std"]).reset_index()
        ax.plot(gd["step"], gd["mean"], marker="o", label=TASK_SHORT[task],
                color=TASK_COLOR[task], linewidth=1.8)
        ax.fill_between(gd["step"],
                        gd["mean"] - gd["std"].fillna(0),
                        gd["mean"] + gd["std"].fillna(0),
                        alpha=0.15, color=TASK_COLOR[task])
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Mean reward ± 1 std (group of 8)")
    ax.set_title("Per-task reward over training")
    ax.legend(loc="upper left")
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    plt.tight_layout()
    fig.savefig(out / "02_per_task_rewards.png", dpi=150)
    plt.close(fig)


def plot_reward_distribution(trials_df: pd.DataFrame, out: Path):
    """Reward distribution per task (violin + strip)."""
    df = trials_df[trials_df["reward"].notna()].copy()
    if df.empty:
        return

    df["task_short"] = df["task_name"].map(TASK_SHORT)
    order = [TASK_SHORT[t] for t in TRAIN_TASKS if TASK_SHORT[t] in df["task_short"].values]

    fig, ax = plt.subplots(figsize=(11, 5))
    sns.violinplot(data=df, x="task_short", y="reward", order=order, hue="task_short",
                   palette={TASK_SHORT[t]: TASK_COLOR[t] for t in TRAIN_TASKS},
                   inner=None, ax=ax, alpha=0.6, legend=False)
    sns.stripplot(data=df, x="task_short", y="reward", order=order, hue="task_short",
                  palette={TASK_SHORT[t]: TASK_COLOR[t] for t in TRAIN_TASKS},
                  size=3, alpha=0.5, jitter=True, ax=ax, legend=False)
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1, label="baseline")
    ax.set_xlabel("Task")
    ax.set_ylabel("Reward")
    ax.set_title("Reward distribution per task (all trials)")
    ax.legend()
    plt.tight_layout()
    fig.savefig(out / "03_reward_distribution.png", dpi=150)
    plt.close(fig)


def plot_grpo_signal_quality(trials_df: pd.DataFrame, out: Path):
    """
    Within-group reward variance per step per task.
    GRPO learns nothing when all 8 samples have the same reward (std=0).
    """
    df = trials_df[trials_df["reward"].notna() & trials_df["step"].notna()].copy()
    if df.empty:
        return

    rows = []
    for (step, task), g in df.groupby(["step", "task_name"]):
        rewards = g["reward"].dropna().values
        if len(rewards) < 2:
            continue
        rows.append({
            "step": step,
            "task_short": TASK_SHORT.get(task, task),
            "reward_std": np.std(rewards),
            "reward_mean": np.mean(rewards),
            "n": len(rewards),
        })
    if not rows:
        return
    gdf = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(12, 5))
    for task_short in gdf["task_short"].unique():
        td = gdf[gdf["task_short"] == task_short].sort_values("step")
        task_full = {v: k for k, v in TASK_SHORT.items()}.get(task_short, task_short)
        ax.plot(td["step"], td["reward_std"], marker="o",
                label=task_short, color=TASK_COLOR.get(task_full), linewidth=1.8)
    ax.axhline(0, color="gray", linestyle="--", linewidth=1, label="zero variance (no signal)")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Within-group reward std (n=8)")
    ax.set_title("GRPO signal quality: within-group reward variance per task")
    ax.legend()
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    plt.tight_layout()
    fig.savefig(out / "04_grpo_signal_quality.png", dpi=150)
    plt.close(fig)


def plot_patch_analysis(trials_df: pd.DataFrame, out: Path):
    """Patch application rate and patch size over training."""
    df = trials_df[trials_df["step"].notna()].copy()
    if df.empty:
        return

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    # Patch application rate per step per task
    ax = axes[0]
    for task in TRAIN_TASKS:
        td = df[df["task_name"] == task]
        if td.empty:
            continue
        gd = td.groupby("step")["patch_applied"].mean().reset_index()
        ax.plot(gd["step"], gd["patch_applied"], marker="o",
                label=TASK_SHORT[task], color=TASK_COLOR[task], linewidth=1.8)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Fraction of trials that applied a patch")
    ax.set_title("Patch application rate")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=9)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    # Patch size (lines changed) distribution
    ax = axes[1]
    patched = df[df["patch_applied"] == True].copy()
    patched["lines_changed"] = patched["patch_added"] + patched["patch_removed"]
    if not patched.empty:
        patched["task_short"] = patched["task_name"].map(TASK_SHORT)
        order = [TASK_SHORT[t] for t in TRAIN_TASKS if TASK_SHORT[t] in patched["task_short"].values]
        sns.boxplot(data=patched, x="task_short", y="lines_changed", order=order,
                    hue="task_short",
                    palette={TASK_SHORT[t]: TASK_COLOR[t] for t in TRAIN_TASKS},
                    ax=ax, legend=False)
        ax.set_xlabel("Task")
        ax.set_ylabel("Lines changed (added + removed)")
        ax.set_title("Patch size distribution")

    # Patch size vs reward scatter
    ax = axes[2]
    pr = df[df["patch_applied"] == True & df["reward"].notna()].copy()
    pr["lines_changed"] = pr["patch_added"] + pr["patch_removed"]
    for task in TRAIN_TASKS:
        td = pr[pr["task_name"] == task]
        if td.empty:
            continue
        ax.scatter(td["lines_changed"], td["reward"], alpha=0.4, s=18,
                   label=TASK_SHORT[task], color=TASK_COLOR[task])
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel("Lines changed")
    ax.set_ylabel("Reward")
    ax.set_title("Patch size vs reward")
    ax.legend(fontsize=9)

    plt.tight_layout()
    fig.savefig(out / "05_patch_analysis.png", dpi=150)
    plt.close(fig)


def plot_agent_behavior(trials_df: pd.DataFrame, out: Path):
    """Agent turn count and speedup distributions."""
    df = trials_df.copy()
    if df.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Turn count distribution per task
    ax = axes[0]
    df_turns = df[df["n_turns"] > 0].copy()
    if not df_turns.empty:
        df_turns["task_short"] = df_turns["task_name"].map(TASK_SHORT)
        order = [TASK_SHORT[t] for t in TRAIN_TASKS if TASK_SHORT[t] in df_turns["task_short"].values]
        sns.violinplot(data=df_turns, x="task_short", y="n_turns", order=order,
                       hue="task_short",
                       palette={TASK_SHORT[t]: TASK_COLOR[t] for t in TRAIN_TASKS},
                       inner="box", ax=ax, alpha=0.7, legend=False)
        ax.set_xlabel("Task")
        ax.set_ylabel("Number of agent turns")
        ax.set_title("Agent trajectory length per task")

    # Benchmark speedup distribution
    ax = axes[1]
    df_sp = df[df["max_speedup"].notna()].copy()
    if not df_sp.empty:
        for task in TRAIN_TASKS:
            td = df_sp[df_sp["task_name"] == task]["max_speedup"]
            if td.empty:
                continue
            ax.hist(td * 100, bins=20, alpha=0.5, label=TASK_SHORT[task],
                    color=TASK_COLOR[task])
        ax.axvline(0, color="gray", linestyle="--", linewidth=1)
        ax.set_xlabel("Max benchmark speedup (%)")
        ax.set_ylabel("Count")
        ax.set_title("Best benchmark improvement per trial")
        ax.legend(fontsize=9)

    plt.tight_layout()
    fig.savefig(out / "06_agent_behavior.png", dpi=150)
    plt.close(fig)


def plot_reward_vs_patch(trials_df: pd.DataFrame, out: Path):
    """
    Does applying a patch actually increase reward?
    Compares reward distribution for patch_applied=True vs False per task.
    """
    df = trials_df[trials_df["reward"].notna()].copy()
    if df.empty:
        return

    fig, axes = plt.subplots(1, len(TRAIN_TASKS), figsize=(16, 4), sharey=True)
    for i, task in enumerate(TRAIN_TASKS):
        ax = axes[i]
        td = df[df["task_name"] == task]
        if td.empty:
            ax.set_visible(False)
            continue
        patched = td[td["patch_applied"] == True]["reward"].dropna()
        unpatched = td[td["patch_applied"] == False]["reward"].dropna()
        if patched.empty and unpatched.empty:
            ax.set_visible(False)
            continue
        if not unpatched.empty:
            ax.hist(unpatched, bins=15, alpha=0.6, label="no patch", color="gray")
        if not patched.empty:
            ax.hist(patched, bins=15, alpha=0.6, label="patched", color=TASK_COLOR[task])
        ax.axvline(1.0, color="black", linestyle="--", linewidth=1)
        ax.set_title(TASK_SHORT[task])
        ax.set_xlabel("Reward")
        if i == 0:
            ax.set_ylabel("Count")
        if i == len(TRAIN_TASKS) - 1:
            ax.legend(fontsize=8)
    plt.suptitle("Reward: patched vs unpatched trials", y=1.02)
    plt.tight_layout()
    fig.savefig(out / "07_reward_vs_patch.png", dpi=150)
    plt.close(fig)


# ── Trajectory inspection ───────────���─────────────────────────────────────────

def summarize_trajectory(td: Path) -> dict:
    """Extract key facts from one trial directory."""
    out = {"trial_id": td.name}

    rj = td / "result.json"
    if rj.exists():
        r = json.loads(rj.read_text())
        out["task_name"] = r.get("task_name", "?")
        out["started_at"] = r.get("started_at", "?")

    rw = td / "verifier" / "reward.txt"
    out["reward"] = float(rw.read_text().strip()) if rw.exists() else None

    patch = td / "artifacts" / "patch.diff"
    out["patch"] = patch.read_text(errors="replace") if patch.exists() else ""
    out["patch_applied"] = len(out["patch"].strip()) > 0

    traj = td / "agent" / "trajectory.json"
    out["steps"] = []
    if traj.exists():
        try:
            tj = json.loads(traj.read_text())
            for s in tj.get("steps", []):
                out["steps"].append({
                    "step_id": s.get("step_id"),
                    "source": s.get("source"),
                    "message": str(s.get("message", ""))[:800],
                })
        except Exception:
            pass

    stdout = td / "verifier" / "test-stdout.txt"
    if stdout.exists():
        txt = stdout.read_text(errors="replace")
        speedups = re.findall(r"time_\S+: [\d.]+\w+ -> [\d.]+\w+ \([+-][\d.]+%\)", txt)
        out["speedup_lines"] = speedups

    return out


def print_trajectory_report(summary: dict, file=sys.stdout):
    print(f"\n{'='*70}", file=file)
    print(f"Trial: {summary['trial_id']}", file=file)
    print(f"Task:  {summary.get('task_name','?')}", file=file)
    print(f"Reward: {summary.get('reward')}", file=file)
    print(f"Patch applied: {summary['patch_applied']}", file=file)
    print(f"Agent turns: {len(summary['steps'])}", file=file)

    if summary.get("speedup_lines"):
        print("\nBenchmark results:", file=file)
        for line in summary["speedup_lines"]:
            print(f"  {line}", file=file)

    if summary["patch_applied"] and summary["patch"]:
        print("\nPatch:", file=file)
        for line in summary["patch"].splitlines()[:60]:
            print(f"  {line}", file=file)
        if len(summary["patch"].splitlines()) > 60:
            print(f"  ... ({len(summary['patch'].splitlines())} lines total)", file=file)

    if summary["steps"]:
        print("\nAgent trajectory (last 5 turns):", file=file)
        for s in summary["steps"][-5:]:
            print(f"\n  [{s['source']}] step {s['step_id']}:", file=file)
            print(f"    {s['message'][:400]}", file=file)


# ── Text report ───────────────��───────────────────────────���───────────────────

def write_report(log_df: pd.DataFrame, trials_df: pd.DataFrame, out: Path):
    lines = []
    lines.append(f"FormulaCode RLVR — Analysis Report")
    lines.append(f"Generated: {datetime.now().isoformat()}")
    lines.append("=" * 70)

    lines.append(f"\n## Training progress")
    if not log_df.empty:
        lines.append(f"  Steps completed: {log_df['global_step'].max()}")
        lines.append(f"  Avg reward (latest step): {log_df['avg_final_reward'].iloc[-1]:.4f}")
        lines.append(f"  Avg reward (first step):  {log_df['avg_final_reward'].iloc[0]:.4f}")
        if len(log_df) > 1:
            delta = log_df['avg_final_reward'].iloc[-1] - log_df['avg_final_reward'].iloc[0]
            lines.append(f"  Reward delta:             {delta:+.4f}")
        lines.append(f"  Avg step time:  {log_df['step_time_s'].mean()/60:.1f} min")
    else:
        lines.append("  No completed steps yet.")

    lines.append(f"\n## Trials")
    if not trials_df.empty:
        lines.append(f"  Total trials collected: {len(trials_df)}")
        lines.append(f"  Trials with reward:     {trials_df['reward'].notna().sum()}")
        lines.append(f"  Patch applied rate:     {trials_df['patch_applied'].mean()*100:.1f}%")
        lines.append(f"  Exception rate:         {trials_df['had_exception'].mean()*100:.1f}%")
        lines.append(f"\n  Per-task summary:")
        for task in TRAIN_TASKS:
            td = trials_df[trials_df["task_name"] == task]
            if td.empty:
                continue
            r = td["reward"].dropna()
            lines.append(f"    {TASK_SHORT[task]:10s}: "
                         f"n={len(td):3d}  "
                         f"reward={r.mean():.3f}±{r.std():.3f}  "
                         f"patch={td['patch_applied'].mean()*100:.0f}%  "
                         f"exc={td['had_exception'].mean()*100:.0f}%")
    else:
        lines.append("  No trials from this run yet.")

    lines.append(f"\n## GRPO signal quality (within-group reward std)")
    if not trials_df.empty and "step" in trials_df.columns:
        df = trials_df[trials_df["reward"].notna() & trials_df["step"].notna()]
        for task in TRAIN_TASKS:
            td = df[df["task_name"] == task]
            if td.empty:
                continue
            stds = td.groupby("step")["reward"].std().dropna()
            if stds.empty:
                continue
            lines.append(f"    {TASK_SHORT[task]:10s}: mean std={stds.mean():.3f}  "
                         f"steps with std>0.05: {(stds > 0.05).sum()}/{len(stds)}")

    lines.append(f"\n## Best and worst trajectories")
    if not trials_df.empty:
        df_r = trials_df[trials_df["reward"].notna()]
        for task in TRAIN_TASKS:
            td = df_r[df_r["task_name"] == task]
            if td.empty:
                continue
            best = td.loc[td["reward"].idxmax()]
            worst = td.loc[td["reward"].idxmin()]
            lines.append(f"    {TASK_SHORT[task]:10s}: "
                         f"best={best['reward']:.3f} ({best['trial_id']})  "
                         f"worst={worst['reward']:.3f} ({worst['trial_id']})")

    report_path = out / "report.txt"
    report_path.write_text("\n".join(lines))
    print("\n".join(lines))
    return report_path


# ── Main ─────────────────────────────��───────────────────────────���───────────

def main():
    parser = argparse.ArgumentParser(description="FormulaCode RLVR analysis")
    parser.add_argument("--run", default="formulacode-grpo-g8-big")
    parser.add_argument("--out", default=None)
    parser.add_argument("--trajectories", action="store_true",
                        help="Print full trajectory reports for best trial per task")
    parser.add_argument("--run-start", default=None,
                        help="ISO datetime for run start (default: infer from run.log)")
    args = parser.parse_args()

    run_dir = RESULTS_ROOT / args.run
    trials_dir = RESULTS_ROOT / "trials"
    out_dir = Path(args.out) if args.out else run_dir / "analysis"
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    logfile = run_dir / "run.log"
    print(f"Run:    {args.run}")
    print(f"Log:    {logfile} ({'exists' if logfile.exists() else 'MISSING'})")
    print(f"Output: {out_dir}")

    # Parse training log
    log_df = parse_run_log(logfile) if logfile.exists() else pd.DataFrame()
    print(f"Completed steps in log: {len(log_df)}")

    # Infer run start time
    if args.run_start:
        run_start = datetime.fromisoformat(args.run_start).replace(tzinfo=timezone.utc)
    else:
        # Use first timestamp in run.log, or fallback to file mtime
        run_start = None
        if logfile.exists():
            first_line = logfile.read_text().splitlines()[0] if logfile.stat().st_size else ""
            # Matches "Mon May 25 06:26:39 AM UTC 2026" or "Mon May 25 06:26:39 UTC 2026"
            m = re.search(r"(\w{3} \w{3} +\d+ \d{2}:\d{2}:\d{2}(?: [AP]M)? \w+ \d{4})", first_line)
            if m:
                for fmt in ("%a %b %d %I:%M:%S %p %Z %Y", "%a %b %d %H:%M:%S %Z %Y"):
                    try:
                        run_start = datetime.strptime(m.group(1), fmt).replace(tzinfo=timezone.utc)
                        break
                    except Exception:
                        pass
        if run_start is None:
            run_start = datetime(2026, 5, 25, 6, 26, 0, tzinfo=timezone.utc)

    print(f"Run start: {run_start.isoformat()}")

    # Load trials
    trials_df = load_trials(trials_dir, run_start)
    trials_df = assign_steps(trials_df, log_df)
    print(f"Trials loaded: {len(trials_df)}")

    # Generate plots
    print("\nGenerating plots...")
    plot_reward_curve(log_df, plots_dir)
    plot_per_task_rewards(trials_df, plots_dir)
    plot_reward_distribution(trials_df, plots_dir)
    plot_grpo_signal_quality(trials_df, plots_dir)
    plot_patch_analysis(trials_df, plots_dir)
    plot_agent_behavior(trials_df, plots_dir)
    plot_reward_vs_patch(trials_df, plots_dir)
    print(f"Plots saved to {plots_dir}/")

    # Text report
    print("\n")
    report_path = write_report(log_df, trials_df, out_dir)
    print(f"\nReport saved to {report_path}")

    # Trajectory deep-dive
    if args.trajectories and not trials_df.empty:
        traj_out = out_dir / "trajectories.txt"
        with open(traj_out, "w") as f:
            df_r = trials_df[trials_df["reward"].notna()]
            for task in TRAIN_TASKS:
                td = df_r[df_r["task_name"] == task]
                if td.empty:
                    continue
                # Best and worst
                for label, idx in [("BEST", td["reward"].idxmax()), ("WORST", td["reward"].idxmin())]:
                    trial_id = td.loc[idx, "trial_id"]
                    td_path = trials_dir / trial_id
                    summary = summarize_trajectory(td_path)
                    print(f"\n[{label} — {TASK_SHORT[task]}]", file=f)
                    print_trajectory_report(summary, file=f)
        print(f"Trajectory report saved to {traj_out}")


if __name__ == "__main__":
    main()

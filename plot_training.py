#!/usr/bin/env python3
"""Extract training curves from FormulaCode RLVR trial results and run.log files."""

import json
import math
import os
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

RESULTS = Path("/home/arjun/formulacode-rlvr-recipe/results")
TRIALS  = RESULTS / "trials"
RUN_LOGS = [
    RESULTS / "formulacode-grpo-g8-2gpu"  / "run.log",
    RESULTS / "formulacode-grpo-g16-2gpu" / "run.log",
]

# Reward formula constants
PENALTY     = 1.0
FLOOR       = 0.25
ALPHA       = 2.5
DENOM_FLOOR = math.log(1.08)   # ≈ 0.0770
C           = 2.0

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")

def strip_ansi(s):
    return ANSI_RE.sub("", s)

# ── 1. Parse run.log for step-level metrics and boundary timestamps ──────────

STEP_HEADER_RE = re.compile(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+).*skyrl\.train\.utils\.tracking.*Step (\d+):")
METRIC_RE      = re.compile(r"'([\w/]+)':\s*'?([-\d.e+]+)'?")
PREFIX_RE      = re.compile(r"^\([^)]+\)\s*")   # strip "(pid=...) " prefix

def parse_run_logs(paths):
    """Return (step_metrics dict, step_end_times dict)."""
    step_metrics   = {}
    step_end_times = {}

    for path in paths:
        lines = path.read_text(errors="replace").splitlines()
        i = 0
        while i < len(lines):
            line = strip_ansi(lines[i])
            m = STEP_HEADER_RE.search(line)
            if m:
                ts_str, step_str = m.groups()
                step = int(step_str)
                ts = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S.%f").replace(tzinfo=timezone.utc)
                step_end_times[step] = ts

                # Collect dict lines until a line with just "}" (end of dict)
                metrics = {}
                i += 1
                while i < len(lines):
                    dline = strip_ansi(lines[i])
                    dline = PREFIX_RE.sub("", dline).strip()
                    if not dline or dline == "}":
                        break
                    if not dline.startswith("'") and not dline.startswith("{"):
                        break   # hit something that's not dict content
                    for km in METRIC_RE.finditer(dline):
                        key, val = km.groups()
                        try:
                            metrics[key] = float(val)
                        except ValueError:
                            pass
                    i += 1
                step_metrics[step] = metrics
            else:
                i += 1

    return step_metrics, step_end_times

step_metrics, step_end_times = parse_run_logs(RUN_LOGS)

# Build ordered list of (step, end_ts) for trial assignment
sorted_steps = sorted(step_end_times.items(), key=lambda x: x[0])
print(f"Found {len(sorted_steps)} steps: {[s for s,_ in sorted_steps]}")

# Training start: first timestamp from g8 run.log (pre-training trials are excluded)
TRAINING_START = datetime(2026, 6, 2, 7, 6, 25, tzinfo=timezone.utc)
# Continuation start: first timestamp from g16 run.log
CONTINUATION_START = datetime(2026, 6, 3, 3, 0, 0, tzinfo=timezone.utc)

# Step boundaries: (start, end) for each step
# Step N starts after step N-1 ends; step 1 starts at TRAINING_START
step_windows = {}
for i, (step, end_ts) in enumerate(sorted_steps):
    if i == 0:
        start_ts = TRAINING_START
    else:
        # Check for the gap between runs (steps 8 → 10)
        prev_step, prev_end = sorted_steps[i - 1]
        start_ts = CONTINUATION_START if step == 10 else prev_end
    step_windows[step] = (start_ts, end_ts)

def assign_step(finished_at_str):
    """Return the step number a trial belongs to based on its finished_at timestamp."""
    try:
        dt = datetime.fromisoformat(finished_at_str.replace("Z", "+00:00"))
    except Exception:
        return None
    if dt < TRAINING_START:
        return None   # pre-training trial — exclude
    for step, (start_ts, end_ts) in step_windows.items():
        if start_ts <= dt <= end_ts:
            return step
    return None

# ── 2. Reward formula ────────────────────────────────────────────────────────

def compute_reward(tests_passed, lsv):
    """Compute reward from the formula. Uses DENOM_FLOOR (oracle unknown)."""
    if not tests_passed:
        return -PENALTY
    g = lsv if lsv is not None else 0.0
    if g <= 0:
        return FLOOR
    if g < 1:
        return FLOOR + (1 - FLOOR) * g
    # g >= 1
    denom = DENOM_FLOOR   # conservative: no oracle speedup → floor denominator
    ratio = math.log(g) / denom
    return 1 + ALPHA * min(ratio, C)

# ── 3. Walk trial directories ────────────────────────────────────────────────

TASK_LABELS = {
    "joblib_joblib_484":        "joblib",
    "python-hyper_h11_34":      "h11",
    "pvlib_pvlib-python_369":   "pvlib",
}

# Per step, per task: lists of (reward, lsv, n_output_tokens)
data = defaultdict(lambda: defaultdict(lambda: {"rewards": [], "lsvs": [], "tokens": []}))

skipped = 0
for trial_dir in TRIALS.iterdir():
    if not trial_dir.is_dir():
        continue

    result_path  = trial_dir / "result.json"
    reward_path  = trial_dir / "verifier" / "reward.json"
    if not result_path.exists() or not reward_path.exists():
        skipped += 1
        continue

    try:
        result = json.loads(result_path.read_text())
        reward_data = json.loads(reward_path.read_text())
    except Exception:
        skipped += 1
        continue

    # Identify task
    trial_name = result.get("trial_name", trial_dir.name)
    task_key = "__".join(trial_name.split("__")[:-1])   # strip trailing __ID
    task_label = TASK_LABELS.get(task_key)
    if task_label is None:
        continue   # not one of the three training tasks

    # Assign to step
    finished_at = result.get("finished_at", "")
    step = assign_step(finished_at)
    if step is None:
        skipped += 1
        continue

    # Extract metrics
    tests_passed = reward_data.get("tests_passed", False)
    lsv          = reward_data.get("lsv_mean_speedup")
    if lsv is None:
        lsv = 0.0

    n_out_tokens = result.get("agent_result", {}).get("n_output_tokens", 0)

    reward = compute_reward(tests_passed, lsv)

    d = data[step][task_label]
    d["rewards"].append(reward)
    d["lsvs"].append(lsv)
    d["tokens"].append(n_out_tokens)

print(f"Skipped {skipped} trials (missing files / unknown task)")
print(f"Loaded data for steps: {sorted(data.keys())}")
for step in sorted(data.keys()):
    counts = {t: len(data[step][t]["rewards"]) for t in data[step]}
    print(f"  step {step:2d}: {counts}")

# ── 4. Aggregate per step per task ──────────────────────────────────────────

tasks = ["joblib", "h11", "pvlib"]
colors = {"joblib": "#1f77b4", "h11": "#ff7f0e", "pvlib": "#2ca02c"}

all_steps = sorted(step_metrics.keys())

def agg(data, step, task, key):
    vals = data[step][task][key]
    if not vals:
        return np.nan, np.nan
    return np.mean(vals), np.std(vals)

# ── 5. Step-level metrics from run.log ──────────────────────────────────────

entropy_steps = []
entropy_vals  = []
avg_reward_steps = []
avg_reward_vals  = []
avg_tokens_steps = []
avg_tokens_vals  = []

for step, metrics in sorted(step_metrics.items()):
    if "policy/policy_entropy" in metrics:
        entropy_steps.append(step)
        entropy_vals.append(metrics["policy/policy_entropy"])
    if "loss/avg_final_rewards" in metrics:
        avg_reward_steps.append(step)
        avg_reward_vals.append(metrics["loss/avg_final_rewards"])
    tok_key = "generate/avg_num_tokens"
    if tok_key in metrics:
        avg_tokens_steps.append(step)
        avg_tokens_vals.append(metrics[tok_key])

# ── 6. Plot ──────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(3, 2, figsize=(14, 12))
fig.suptitle("FormulaCode RLVR Training Curves", fontsize=14, fontweight="bold")

# ── 6a. Entropy (from run.log, aggregate) ───────────────────────────────────
ax = axes[0, 0]
ax.plot(entropy_steps, entropy_vals, "ko-", lw=2, ms=5, label="policy entropy")
ax.set_title("Policy Entropy (aggregate)")
ax.set_xlabel("Training step")
ax.set_ylabel("Entropy")
ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
ax.grid(alpha=0.3)
ax.legend()

# ── 6b. Avg final reward from run.log (aggregate) ───────────────────────────
ax = axes[0, 1]
ax.plot(avg_reward_steps, avg_reward_vals, "rs-", lw=2, ms=5)
ax.set_title("Avg Final Reward (aggregate, from run.log)")
ax.set_xlabel("Training step")
ax.set_ylabel("Reward")
ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
ax.grid(alpha=0.3)

# ── 6c. Reward per task per step ─────────────────────────────────────────────
ax = axes[1, 0]
for task in tasks:
    means, stds, steps_used = [], [], []
    for step in all_steps:
        mu, sigma = agg(data, step, task, "rewards")
        if not np.isnan(mu):
            steps_used.append(step)
            means.append(mu)
            stds.append(sigma)
    means = np.array(means); stds = np.array(stds)
    ax.plot(steps_used, means, "o-", color=colors[task], lw=2, ms=4, label=task)
    ax.fill_between(steps_used, means - stds, means + stds, alpha=0.15, color=colors[task])
ax.set_title("Reward per Task (mean ± 1σ)")
ax.set_xlabel("Training step")
ax.set_ylabel("Reward")
ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
ax.legend()
ax.grid(alpha=0.3)

# ── 6d. LSV (raw speedup) per task per step ──────────────────────────────────
ax = axes[1, 1]
for task in tasks:
    means, stds, steps_used = [], [], []
    for step in all_steps:
        mu, sigma = agg(data, step, task, "lsvs")
        if not np.isnan(mu):
            steps_used.append(step)
            means.append(mu)
            stds.append(sigma)
    means = np.array(means); stds = np.array(stds)
    ax.plot(steps_used, means, "o-", color=colors[task], lw=2, ms=4, label=task)
    ax.fill_between(steps_used, means - stds, means + stds, alpha=0.15, color=colors[task])
ax.axhline(1.0, color="gray", ls="--", lw=1, label="speedup = 1.0")
ax.set_title("LSV Mean Speedup per Task (mean ± 1σ)")
ax.set_xlabel("Training step")
ax.set_ylabel("Speedup (×)")
ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
ax.legend()
ax.grid(alpha=0.3)

# ── 6e. Sequence length per task per step ────────────────────────────────────
ax = axes[2, 0]
for task in tasks:
    means, stds, steps_used = [], [], []
    for step in all_steps:
        mu, sigma = agg(data, step, task, "tokens")
        if not np.isnan(mu):
            steps_used.append(step)
            means.append(mu / 1000)   # show in thousands
            stds.append(sigma / 1000)
    means = np.array(means); stds = np.array(stds)
    ax.plot(steps_used, means, "o-", color=colors[task], lw=2, ms=4, label=task)
    ax.fill_between(steps_used, means - stds, means + stds, alpha=0.15, color=colors[task])
ax.set_title("Output Tokens per Task (mean ± 1σ)")
ax.set_xlabel("Training step")
ax.set_ylabel("Output tokens (k)")
ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
ax.legend()
ax.grid(alpha=0.3)

# ── 6f. Aggregate avg_num_tokens from run.log ────────────────────────────────
ax = axes[2, 1]
avg_tokens_k = [t / 1000 for t in avg_tokens_vals]
ax.plot(avg_tokens_steps, avg_tokens_k, "g^-", lw=2, ms=5)
ax.set_title("Avg Sequence Length (aggregate, from run.log)")
ax.set_xlabel("Training step")
ax.set_ylabel("Avg tokens per rollout (k)")
ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
ax.grid(alpha=0.3)

plt.tight_layout()
out_path = RESULTS / "training_curves.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"\nSaved → {out_path}")
plt.close()

# ── 7. Also save a CSV of the per-trial data for later analysis ──────────────
import csv
csv_path = RESULTS / "trial_data.csv"
with open(csv_path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["step", "task", "trial_name", "reward", "lsv", "tests_passed", "n_output_tokens"])
    for step in sorted(data.keys()):
        for task in tasks:
            d = data[step][task]
            n = len(d["rewards"])
            for i in range(n):
                w.writerow([step, task, "", d["rewards"][i], d["lsvs"][i], "", d["tokens"][i]])
print(f"Saved CSV → {csv_path}")

#!/usr/bin/env python3
"""
Analyze context lengths across all completed pass@k trials.
Reads trajectory.json for each trial and reports prompt/completion token usage.
"""
import json
import os
import glob

TRIALS_DIR = "/home/arjun/formulacode-rlvr-recipe/results/trials"
RESULTS = {
    "h11":      "/home/arjun/formulacode-rlvr-recipe/results/passk_h11_20260607T093422.jsonl",
    "pvlib":    "/home/arjun/formulacode-rlvr-recipe/results/passk_pvlib_20260607T093853.jsonl",
    "networkx": "/home/arjun/formulacode-rlvr-recipe/results/passk/networkx/passk_networkx_20260607T095012.jsonl",
}

def analyze_trial(trial_id):
    traj_path = os.path.join(TRIALS_DIR, trial_id, "agent", "trajectory.json")
    if not os.path.exists(traj_path):
        return None

    with open(traj_path) as f:
        traj = json.load(f)

    steps = [s for s in traj.get("steps", []) if s.get("source") == "agent"]
    if not steps:
        return None

    prompt_tokens    = [s["metrics"]["prompt_tokens"]     for s in steps if s.get("metrics")]
    completion_tokens = [s["metrics"]["completion_tokens"] for s in steps if s.get("metrics")]

    if not prompt_tokens:
        return None

    return {
        "num_turns":          len(steps),
        "max_prompt_tokens":  max(prompt_tokens),       # context at last turn
        "final_prompt_tokens": prompt_tokens[-1],
        "total_completion_tokens": sum(completion_tokens),
        "avg_completion_tokens":   sum(completion_tokens) / len(completion_tokens),
        "max_completion_tokens":   max(completion_tokens),
        "prompt_tokens_by_turn":  prompt_tokens,
        "completion_tokens_by_turn": completion_tokens,
    }

# Load rewards from JSONL files
rewards = {}
for task, path in RESULTS.items():
    if not os.path.exists(path):
        continue
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            tid = d.get("trial_id")
            if tid:
                rewards[tid] = {"task": task, "reward": d.get("reward")}

# Analyze all trial dirs
rows = []
for trial_id in sorted(os.listdir(TRIALS_DIR)):
    stats = analyze_trial(trial_id)
    if stats is None:
        continue
    meta = rewards.get(trial_id, {})
    rows.append({
        "trial_id":   trial_id,
        "task":       meta.get("task", "unknown"),
        "reward":     meta.get("reward"),
        **stats,
    })

# Sort by task then reward
rows.sort(key=lambda r: (r["task"], -(r["reward"] or 0)))

# Print summary table
header = f"{'trial_id':<32} {'task':<10} {'reward':>7} {'turns':>6} {'final_ctx':>10} {'max_ctx':>10} {'total_out':>10} {'avg_out':>8}"
print(header)
print("-" * len(header))
for r in rows:
    reward_str = f"{r['reward']:.3f}" if r['reward'] is not None else "   N/A"
    print(f"{r['trial_id']:<32} {r['task']:<10} {reward_str:>7} {r['num_turns']:>6} "
          f"{r['final_prompt_tokens']:>10,} {r['max_prompt_tokens']:>10,} "
          f"{r['total_completion_tokens']:>10,} {r['avg_completion_tokens']:>8.0f}")

# Summary stats per task
print()
for task in ["h11", "pvlib", "networkx"]:
    task_rows = [r for r in rows if r["task"] == task]
    if not task_rows:
        continue
    max_ctxs = [r["max_prompt_tokens"] for r in task_rows]
    total_outs = [r["total_completion_tokens"] for r in task_rows]
    turns = [r["num_turns"] for r in task_rows]
    print(f"{task}: n={len(task_rows)}  "
          f"max_ctx: min={min(max_ctxs):,} avg={sum(max_ctxs)//len(max_ctxs):,} max={max(max_ctxs):,}  "
          f"total_out: min={min(total_outs):,} avg={sum(total_outs)//len(total_outs):,} max={max(total_outs):,}  "
          f"turns: min={min(turns)} avg={sum(turns)//len(turns)} max={max(turns)}")

# Write JSON output
out_path = "/home/arjun/formulacode-rlvr-recipe/results/context_length_analysis.json"
with open(out_path, "w") as f:
    json.dump(rows, f, indent=2)
print(f"\nFull data written to {out_path}")

#!/usr/bin/env python3
"""
Extract agent reasoning/thinking from a Harbor trial trajectory.
Usage: python3 extract_reasoning.py <trial_id_or_trajectory_path> [output.json]
"""
import json
import sys
import os

TRIALS_DIR = "/home/arjun/formulacode-rlvr-recipe/results/trials"


def extract(traj_path, out_path=None):
    with open(traj_path) as f:
        traj = json.load(f)

    steps = traj.get("steps", [])
    # Step 0 is the system prompt — skip it
    agent_steps = [s for s in steps if s.get("source") == "agent"]

    records = []
    for s in agent_steps:
        step_id = s["step_id"]
        ts = s.get("timestamp", "")
        message = s.get("message", "").strip()
        tool_calls = s.get("tool_calls", [])
        obs = s.get("observation")

        # Extract tool call summaries
        calls = []
        for c in tool_calls:
            fn = c.get("function_name", "")
            args = c.get("arguments", {})
            # For bash commands, grab the keystrokes
            cmd = args.get("keystrokes", args.get("command", ""))
            calls.append({"fn": fn, "cmd": cmd[:300] if cmd else str(args)[:300]})

        # Extract terminal output from observation
        terminal_out = ""
        if obs and isinstance(obs, dict):
            results = obs.get("results", [])
            if results and isinstance(results, list):
                content = results[0].get("content", "") if results else ""
                # Strip the warnings prefix if present
                if "New Terminal Output:" in content:
                    terminal_out = content.split("New Terminal Output:", 1)[1].strip()
                else:
                    terminal_out = content.strip()
                terminal_out = terminal_out[:1000]  # cap at 1000 chars

        records.append({
            "step": step_id,
            "ts": ts,
            "reasoning": message,
            "tool_calls": calls,
            "terminal_out": terminal_out,
        })

    output = {
        "trial_id": traj.get("session_id", os.path.basename(os.path.dirname(traj_path))),
        "total_steps": len(agent_steps),
        "steps": records,
    }

    if out_path:
        with open(out_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"Wrote {len(records)} steps to {out_path}")
    else:
        # Pretty print to stdout
        for r in records:
            print(f"\n{'='*70}")
            print(f"Step {r['step']}  {r['ts']}")
            print(f"{'='*70}")
            if r["reasoning"]:
                print("REASONING:")
                print(r["reasoning"])
            for c in r["tool_calls"]:
                print(f"\n  [{c['fn']}] $ {c['cmd']}")
            if r["terminal_out"]:
                print(f"\n  OUTPUT:\n  {r['terminal_out'][:500]}")

    return output


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: extract_reasoning.py <trial_id | trajectory_path> [output.json]")
        sys.exit(1)

    arg = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else None

    # Accept either a trial_id or a direct path
    if os.path.exists(arg):
        traj_path = arg
    else:
        traj_path = os.path.join(TRIALS_DIR, arg, "agent", "trajectory.json")
        if not os.path.exists(traj_path):
            print(f"No trajectory found at {traj_path}")
            sys.exit(1)

    extract(traj_path, out)

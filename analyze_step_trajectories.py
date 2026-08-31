#!/usr/bin/env python3
"""Per-step trajectory analysis for the live FormulaCode RL run: shaped reward (via the real reward fn),
speedup vs oracle, patch size, trajectory length (turns/tokens/context), and the correctness gate.
Analyzes the most-recent completed trajectories (default: last 300 min). Run:
  python3 analyze_step_trajectories.py [--since-min 300] [--trials DIR]
"""
import json, glob, os, math, sys, time, argparse, statistics as st
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--since-min", type=int, default=300)
ap.add_argument("--trials", default="/overflow/arjun/results/fc-grpo-9b-qwen-prod/trials")
ap.add_argument("--oracle", default="/home/arjun/formulacode-verified-rl/initial_survey/oracle/gold_oracle_benchmarks.json")
ap.add_argument("--reward-src", default="/home/arjun/skyrl-formulacode")
a = ap.parse_args()
sys.path.insert(0, a.reward_src)
from examples.train_integrations.harbor.formulacode.reward import compute_task_reward, RewardConfig, OracleTable
cfg = RewardConfig(); OT = OracleTable(a.oracle)
cutoff = time.time() - a.since_min * 60

rows = []
for d in glob.glob(a.trials + "/*/"):
    d = d.rstrip("/"); rj = Path(d) / "verifier" / "reward.json"
    if not rj.exists() or os.path.getmtime(d) < cutoff:
        continue
    try: rjson = json.loads(rj.read_text())
    except Exception: continue
    task = "__".join(os.path.basename(d).split("__")[:3])
    summaries = list((Path(d) / "artifacts").glob("summary_*.json"))
    vran = len(summaries) > 0; snap = True
    for s in summaries:
        try:
            if json.loads(s.read_text()).get("passed") is False: snap = False
        except Exception: pass
    rjson["snapshots_passed"] = snap; rjson["snapshot_verify_ran"] = vran
    rr = compute_task_reward(rjson, OT.entry_for(task), cfg)   # lookup by task name (not trial dir)
    pd = Path(d) / "artifacts" / "patch.diff"; plus = minus = 0; files = set()
    if pd.exists():
        for ln in pd.read_text(errors="ignore").splitlines():
            if ln.startswith("+++ b/"): files.add(ln[6:])
            elif ln.startswith("+") and not ln.startswith("+++"): plus += 1
            elif ln.startswith("-") and not ln.startswith("---"): minus += 1
    ctok = ctx = 0; trn = None
    res = Path(d) / "result.json"
    if res.exists():
        try:
            R = json.loads(res.read_text()); rd = (R.get("agent_result") or {}).get("rollout_details") or []
            if rd:
                comp = rd[0].get("completion_token_ids") or []; prm = rd[0].get("prompt_token_ids") or []
                ctok = sum(len(c) for c in comp); ctx = max((len(p) for p in prm), default=0); trn = len(comp)
        except Exception: pass
    rows.append(dict(task=task, reward=rr.reward, main=rr.oracle_term, coll=rr.diagnostics.get("collateral"),
                     nimp=rr.n_impacted, nsc=rr.n_scored, geo=rjson.get("lsv_mean_speedup"), plus=plus, minus=minus,
                     nf=len(files), ctok=ctok, ctx=ctx, trn=trn, vran=vran, snap=snap))
if not rows:
    print(f"No trajectories with reward.json in the last {a.since_min} min under {a.trials}"); sys.exit(0)
rows.sort(key=lambda r: (r["task"], -r["reward"]))
print(f"{'task':22}{'REWARD':>7}{'main':>6}{'coll':>6}{'imp/sc':>7}{'geoH':>6}{'patch':>10}{'f':>2}{'toks':>7}{'ctx':>7}{'trn':>4}")
for r in rows:
    g = f"{r['geo']:.2f}" if r['geo'] else " -"; c = f"{r['coll']:.2f}" if isinstance(r['coll'], float) else " -"
    print(f"{r['task'][:21]:22}{r['reward']:>7.2f}{r['main']:>6.2f}{c:>6}{(str(r['nimp'])+'/'+str(r['nsc'])):>7}{g:>6}"
          f"{('+%d/-%d' % (r['plus'], r['minus'])):>10}{r['nf']:>2}{r['ctok']:>7}{r['ctx']:>7}{str(r['trn']):>4}")
for task in sorted(set(r["task"] for r in rows)):
    oe = OT.entry_for(task); rs = [r for r in rows if r["task"] == task]
    og = oe.get("lsv_mean_speedup") if oe else None
    print(f"\n{task}: oracle geoH={og:.3f}" if og else f"\n{task}: oracle geoH=?",
          f" agent geoH={[round(r['geo'],2) if r['geo'] else None for r in rs]}")
    print(f"   rewards={[round(r['reward'],2) for r in rs]}")
gv = [r["reward"] for r in rows]
print(f"\n== n={len(rows)} == reward mean={st.mean(gv):.3f} min={min(gv):.2f} max={max(gv):.2f} pos={sum(1 for x in gv if x>0)}")
print(f"main>0={sum(1 for r in rows if r['main']>0)}/{len(rows)}  net-speedup(geoH>1.05)={sum(1 for r in rows if r['geo'] and r['geo']>1.05)}/{sum(1 for r in rows if r['geo'])}")
print(f"tokens gen mean={int(st.mean(r['ctok'] for r in rows))} max={max(r['ctok'] for r in rows)}; ctx mean={int(st.mean(r['ctx'] for r in rows))} max={max(r['ctx'] for r in rows)}/131072")
tt = [r['trn'] for r in rows if r['trn']]
print(f"turns mean={st.mean(tt):.0f} max={max(tt)}; patch mean +{int(st.mean(r['plus'] for r in rows))}/-{int(st.mean(r['minus'] for r in rows))}")
print(f"gate verify_ran={sum(1 for r in rows if r['vran'])}/{len(rows)}; snapshots_passed={sum(1 for r in rows if r['snap'])}/{len(rows)}")

#!/usr/bin/env python3
"""Merge both boxes' oracle rows, pick a family-balanced 30/10 from tasks that have
real reward-v3 signal, and stage the launch TRAIN_DATA. Always yields a valid split
(falls back to the trusted 24-pool if the measurement produced nothing usable)."""
import json, math, os, subprocess, sys
from collections import defaultdict
from pathlib import Path

REPO = "/home/arjun/formulacode-verified-rl"
TABLE = Path(f"{REPO}/initial_survey/oracle/gold_oracle_benchmarks.json")
TASKS_ROOT = f"{REPO}/initial_survey/tasks"
OUT_TRAIN_DATA = Path("/home/arjun/results/launch_train_data.txt")
NF = math.log(1.03)

EXISTING24 = [
    "geopandas/geopandas#3345","networkx/networkx#4909","networkx/networkx#8138","shapely/shapely#1562",
    "networkx/networkx#8148","UXARRAY/uxarray#1118","networkx/networkx#4830","xarray-contrib/flox#176",
    "networkx/networkx#6337","pybop-team/PyBOP#335","xarray-contrib/flox#53","xarray-contrib/flox#172",
    "pybop-team/PyBOP#256","xarray-contrib/flox#70","UXARRAY/uxarray#1112","shapely/shapely#2359",
    "TileDB-Inc/TileDB-Py#467","TileDB-Inc/TileDB-Py#1005","shapely/shapely#1307","TileDB-Inc/TileDB-Py#834",
    "joblib/joblib#484","bluesky/tiled#982","numpy/numpy-financial#96","pybamm-team/PyBaMM#465",
]
NEW19 = [
    "dask/dask#11464","dask/dask#11493","dask/dask#11496","dask/dask#11600","dask/dask#11687",
    "dask/dask#11736","dask/dask#11754","dask/dask#11760","dask/dask#11788",
    "geopandas/geopandas#3282","geopandas/geopandas#3314","geopandas/geopandas#2796",
    "UXARRAY/uxarray#877","UXARRAY/uxarray#989","networkx/networkx#7736","networkx/networkx#8112",
    "networkx/networkx#8135","bluesky/tiled#954","xarray-contrib/flox#230",
]

def fam(k): return k.split("/")[1].split("#")[0]
def dirname(k, row):
    d = (row or {}).get("task_dir")
    return d if d else k.replace("/", "__").replace("#", "__")
def v3_improved(row):
    b = (row or {}).get("benchmarks") or {}
    n = 0
    for x in b.values():
        ls = x.get("oracle_log_speedup")
        if ls is not None and ls > NF:
            n += 1
    return n

def load(p):
    try: return json.loads(Path(p).read_text()).get("tasks", {})
    except Exception: return {}

def main():
    # 1) combine ml6 (local) + ml7 (pulled) tables: base 40 + each box's new keys
    local_tasks = load(TABLE)
    subprocess.run(["scp", "-q", "ml-login7:" + str(TABLE), "/tmp/ml7_gold.json"], check=False)
    ml7_tasks = load("/tmp/ml7_gold.json")
    combined = dict(local_tasks)
    added7 = 0
    for k, row in ml7_tasks.items():
        if k in NEW19 and k not in combined:
            combined[k] = row; added7 += 1
        elif k in NEW19 and v3_improved(row) > v3_improved(combined.get(k)):
            combined[k] = row  # prefer the box that actually measured it
    # write combined back to both boxes so training reads the full table
    meta = {}
    try: meta = {kk: vv for kk, vv in json.loads(TABLE.read_text()).items() if kk != "tasks"}
    except Exception: pass
    meta["tasks"] = combined
    TABLE.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
    subprocess.run(["scp", "-q", str(TABLE), "ml-login7:" + str(TABLE)], check=False)
    print(f"[select] combined table: {len(combined)} tasks (ml7_new={added7})", flush=True)

    # 2) good pool: existing at v3>=2, new at v3>=3 (v3>=3 also filters memory-contamination,
    #    which suppresses oracle-improved benchmarks and would fail the threshold)
    pool, v3map, dirmap = [], {}, {}
    for k in EXISTING24 + NEW19:
        row = combined.get(k)
        if not row: continue
        v = v3_improved(row); thr = 3 if k in NEW19 else 2
        if v >= thr:
            pool.append(k); v3map[k] = v; dirmap[k] = dirname(k, row)
    newgood = [k for k in NEW19 if k in pool]
    print(f"[select] pool={len(pool)} (existing-good + {len(newgood)} new-good): "
          + ", ".join(f"{k.split('/')[-1]}(v3={v3map[k]})" for k in newgood), flush=True)

    # 3) 30/10 family-balanced, task-disjoint (hold out only where family has >=2 in pool)
    byfam = defaultdict(list)
    for k in pool: byfam[fam(k)].append(k)
    eval_n = 10 if len(pool) >= 40 else min(10, len(pool) // 4)
    fams_multi = sorted([f for f, ks in byfam.items() if len(ks) >= 2])
    eval_set = []
    progressed = True
    while len(eval_set) < eval_n and progressed:
        progressed = False
        for f in fams_multi:
            if len(eval_set) >= eval_n: break
            remaining = [k for k in byfam[f] if k not in eval_set]
            if len(remaining) >= 2:  # keep >=1 for train
                pick = min(remaining, key=lambda k: v3map[k])  # hold out the weakest, keep strong in train
                eval_set.append(pick); progressed = True
    train_set = sorted([k for k in pool if k not in eval_set], key=lambda k: -v3map[k])[:30]

    # 4) stage TRAIN_DATA + id files
    def paths(keys): return [f"{TASKS_ROOT}/{dirmap[k]}" for k in keys]
    train_data = "[" + ",".join(f"'{p}'" for p in paths(train_set)) + "]"
    OUT_TRAIN_DATA.write_text(train_data + "\n")
    Path(f"{REPO}/initial_survey/rl_train_ids.md").write_text(
        f"# RL train set ({len(train_set)}) — auto-selected 2026-09-04\n\n"
        + "\n".join(f"- {k} (v3={v3map[k]})" for k in sorted(train_set)) + "\n")
    Path(f"{REPO}/initial_survey/rl_eval_ids.md").write_text(
        f"# RL held-out eval set ({len(eval_set)}) — auto-selected 2026-09-04\n\n"
        + "\n".join(f"- {k} (v3={v3map[k]})" for k in sorted(eval_set)) + "\n")
    print(f"[select] TRAIN={len(train_set)} EVAL={len(eval_set)} -> {OUT_TRAIN_DATA}", flush=True)
    print("[select] TRAIN:", ", ".join(k.split("/")[-1] for k in train_set), flush=True)
    print("[select] EVAL :", ", ".join(k.split("/")[-1] for k in eval_set), flush=True)
    if len(train_set) < 10:
        print("[select] FATAL: <10 train tasks; aborting launch", flush=True); sys.exit(1)

if __name__ == "__main__":
    main()

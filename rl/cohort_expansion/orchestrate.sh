#!/usr/bin/env bash
# orchestrate.sh (ml-login6, detached) — measure 19 oracle baselines (memory-split across
# ml6+ml7 at conc-2), merge+select a 30/10, then launch the run on ml-login7. Always launches
# (falls back to the trusted 24-pool if the measurement yields nothing usable).
set -uo pipefail
mkdir -p /home/arjun/results
LOG=/home/arjun/results/orchestrator.log
exec >>"$LOG" 2>&1
echo "=================================================================="
echo "[orch $(date -u +%FT%TZ)] START"

ML7_TASKS="geopandas#2796 geopandas#3282 geopandas#3314 uxarray#877 uxarray#989 flox#230"
ML6_TASKS="dask#11464 dask#11493 dask#11496 dask#11600 dask#11687 dask#11736 dask#11754 dask#11760 dask#11788 networkx#7736 networkx#8112 networkx#8135"
GATE_TASK="tiled#954"   # cheap ml7 task, doubles as the invocation gate

# ── Step 1: invocation gate (blocking) ──────────────────────────────────────
echo "[orch] Step 1: invocation gate ($GATE_TASK on ml7)"
timeout 2700 ssh -o BatchMode=yes ml-login7 "bash ~/run_oracle_batch.sh $GATE_TASK" || true
GATE_OK=$(ssh -o BatchMode=yes ml-login7 'python3 - <<PY
import json
try:
    t=json.load(open("/home/arjun/formulacode-verified-rl/initial_survey/oracle/gold_oracle_benchmarks.json"))["tasks"]
    b=(t.get("bluesky/tiled#954") or {}).get("benchmarks") or {}
    print("OK" if len(b)>0 else "FAIL")
except Exception as e:
    print("FAIL")
PY')
echo "[orch] gate result: $GATE_OK"

if [ "$GATE_OK" = "OK" ]; then
  echo "[orch] Step 2: parallel measurement (ml7 memory-sensitive | ml6 dask+networkx)"
  timeout 21600 ssh -o BatchMode=yes ml-login7 "bash ~/run_oracle_batch.sh $ML7_TASKS" > /home/arjun/results/oracle_ml7.log 2>&1 &
  P7=$!
  timeout 21600 bash /home/arjun/run_oracle_batch.sh $ML6_TASKS > /home/arjun/results/oracle_ml6.log 2>&1 &
  P6=$!
  wait $P7; echo "[orch] ml7 batch rc=$?"
  wait $P6; echo "[orch] ml6 batch rc=$?"
  docker container prune -f >/dev/null 2>&1 || true
  ssh -o BatchMode=yes ml-login7 'docker container prune -f >/dev/null 2>&1 || true'
else
  echo "[orch] gate FAILED -> skip measurement, fall back to 24-pool"
fi

# ── Step 3/4: merge tables + select 30/10 (always yields a split) ───────────
echo "[orch] Step 3/4: select_and_stage"
python3 /home/arjun/select_and_stage.py || { echo "[orch] select FAILED -> abort launch"; exit 1; }

# ── Step 5: launch on ml7 ───────────────────────────────────────────────────
RN=fc-grpo-9b-qwen-prod-mll7-v2
echo "[orch] Step 5: ship TRAIN_DATA + launch $RN on ml-login7"
scp -q /home/arjun/results/launch_train_data.txt ml-login7:/home/arjun/results/launch_train_data.txt
scp -q /home/arjun/formulacode-verified-rl/initial_survey/rl_train_ids.md ml-login7:/home/arjun/formulacode-verified-rl/initial_survey/rl_train_ids.md 2>/dev/null || true
ssh -o BatchMode=yes ml-login7 "mkdir -p /home/arjun/results && nohup bash /home/arjun/launch_run.sh > /home/arjun/results/${RN}.boot.log 2>&1 & echo launched pid \$!"
sleep 20
echo "[orch] ml7 boot log tail:"; ssh -o BatchMode=yes ml-login7 "tail -n 15 /home/arjun/results/${RN}.boot.log 2>/dev/null" || true
echo "[orch $(date -u +%FT%TZ)] DONE — launch issued ($RN). Monitor: ml7 ~/results/$RN/run.log"

#!/usr/bin/env bash
# run_oracle_batch.sh <task_key> [<task_key> ...]
# Measure oracle baselines (agent.name=oracle, rounds=5, conc-2) for the given GOLD_TASKS keys.
# Merges into this box's gold_oracle_benchmarks.json. Runs under the docker group.
set -uo pipefail
if [ "$#" -eq 0 ]; then echo "no tasks given"; exit 2; fi
INNER="set -a; . /home/arjun/datasmith/tokens.env 2>/dev/null || true; set +a; \
cd /home/arjun/formulacode-rlvr-recipe/rl && \
ORACLE_SKIP_PUBLISH=1 ORACLE_MAX_CONCURRENCY=2 DATASMITH_LSV_ROUNDS=5 FORMULACODE_NO_UPLOAD=1 \
PYTHONPATH=/home/arjun/datasmith/src PATH=/home/arjun/.local/bin:\$PATH \
/home/arjun/skyrl-formulacode/.venv/bin/python oracle_gold.py --tasks $*"
echo "[run_oracle_batch] $(date -u +%H:%M:%S) tasks: $*"
exec sg docker -c "$INNER"

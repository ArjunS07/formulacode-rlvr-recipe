# Cohort-expansion orchestrator

Autonomous pipeline that grows the FormulaCode RL task cohort and launches training. Built 2026-09-04
to reach a 30 train / 10 eval cohort for `fc-grpo-9b-qwen-prod-mll7-v2`.

**Flow:** measure new oracle baselines (memory-split across ml-login6 + ml-login7 at conc-2) →
merge both boxes' `gold_oracle_benchmarks.json` → select a family-balanced 30/10 by reward-v3 signal
(v3≥3) → launch `run_prod_supervisor.sh` on ml-login7. Always launches (falls back to the trusted
24-task pool if measurement yields nothing usable).

## Files (deploy to `/home/arjun/` to run; paths inside are absolute)
- `orchestrate.sh` — driver (runs detached on ml6; logs to `~/results/orchestrator.log`).
- `run_oracle_batch.sh <task_key>...` — runs `oracle_gold.py --tasks ...` under `sg docker` (both boxes).
- `select_and_stage.py` — merges tables, keeps v3≥3 new / v3≥2 existing, writes `rl_{train,eval}_ids.md`
  + `~/results/launch_train_data.txt`.
- `launch_run.sh` — (ml7) exports the launch env and `sg docker` `run_prod_supervisor.sh`.

## Notes
- Task **env dirs** come from neptune (`asharma@neptune.csres.utexas.edu:/mnt/sdd3/asharma/final-paper-experiments/initial_survey/tasks`);
  **oracle rows are always re-measured on-box** — never copied from neptune (contaminated + different fingerprint).
- New tasks must be added to `GOLD_TASKS` in `../oracle_gold.py` before measuring.
- ml6 and ml7 are hardware-identical (AMD EPYC 7513, 4×A100-80GB), so their oracle floats are interchangeable.
- Full context: `formulacode-verified-rl/TRAINING_HANDOFF.md`.

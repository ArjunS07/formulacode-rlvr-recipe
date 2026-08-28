# FormulaCode RLVR — Container Issues & Fixes

> **⚠ Superseded by `rl-prep/` (2026-07-18) — kept for history.** This doc pre-dates the
> oracle fixes and the datasmith refactor (it still calls pvlib & networkx "dropped").
> Current status lives in `rl-prep/README.md`, `rl-prep/containers/`, and
> `rl-prep/STRUCTURAL-CHANGES.md`.

All known issues with Harbor task containers, LSV infrastructure, and training infrastructure. Last updated: 2026-05-30.

---

## General Patterns (all tasks)

### git clean removes untracked files
- **Problem:** Entrypoint runs `git clean -fd` to reset the repo. Any file COPY'd into the repo directory in the Dockerfile (but not committed) gets wiped.
- **Fix:** Save files to `/opt/<name>` in Dockerfile, restore them in entrypoint after `git clean`.
- **Affects:** joblib (`benchmark_sequential.py`, `asv_benchmarks.txt`), networkx (`benchmark_aperiodic.py`)

### Docker cache staleness
- **Problem:** Adding COPY lines to a Dockerfile after the base image was already built → Harbor reuses the cached image → fixes don't land.
- **Fix:** Rebuild with `docker build --no-cache` after modifying Dockerfiles.
- **Affects:** networkx, pvlib, joblib (each had stale images at some point)

### `bench/bench/benchmarks/__init__.py` deletion
- **Problem:** Both h11 and joblib agents sometimes explore the benchmark directory and accidentally delete or overwrite `bench/bench/benchmarks/__init__.py`. This makes LSV unable to import the benchmark module, causing `lsv_error: "No __init__.py file in '/workspace/repo/bench/bench/benchmarks'"` and reward=0.
- **Status:** Unmitigated. Happens on ~20-30% of trials. The entrypoint's `git reset --hard` prevents it from persisting across trials (each new container is fresh), but the within-trial reward is lost.
- **Possible fix:** Add an anti-gaming block in test.sh/verifier to restore `__init__.py` before LSV runs (like the existing test file restoration).

---

## h11 (`python-hyper_h11_34`)

### asv.conf.json missing environment_type
- **Problem:** The h11 benchmark requires `environment_type: existing` in `bench/asv.conf.json` to use the pre-installed conda environment. Without it, ASV tries to create a new environment and fails.
- **Fix:** Entrypoint.sh overwrites `bench/asv.conf.json` with correct content after `git reset --hard`.
- **Status:** Fixed.

### Constant factor (~0.5x) from measurement mismatch
- **Problem:** `lsv_init` captures baseline using snapshot_tool (in-process, ~57µs). `lsv_measure` computes speedup using ASV subprocess (~115µs). Unmodified code scores ~0.5 reward (measured as 2x slower than baseline) instead of ~1.0.
- **Impact:** All non-patching agents score ~0.5 instead of 1.0. This is misleading but GRPO handles it — the constant factor cancels in advantage computation since all trajectories share the same bias.
- **Status:** Accepted. Not worth fixing for GRPO training purposes.

### Agents break project build
- **Problem:** Some agents make changes that break `pip install -e .` or imports, causing `lsv_error: "Failed to build the project and import the benchmarks"`. Reward=0.
- **Status:** Unmitigated. Happens on ~10-20% of trials.

---

## networkx (`networkx_networkx_8148`)

### benchmark_aperiodic.py not pre-staged
- **Problem:** The new benchmark file (`benchmark_aperiodic.py`) was COPY'd into the repo directory but wiped by `git clean -fd`.
- **Fix:** Save to `/opt/` and restore in entrypoint (same pattern as joblib).
- **Status:** Fixed.

### 140-benchmark dilution
- **Problem:** editable install + coverage.py marks ALL 140 benchmarks as impacted by any source change. The reward is the mean speedup over 140 benchmarks, so a large speedup on `is_aperiodic` gets diluted to near-1.0 by noise from 139 unrelated benchmarks.
- **Impact:** Even a correct `is_aperiodic` fix produces near-zero GRPO advantage. Weak signal.
- **Status:** Accepted (constant factor per-trajectory, cancels in GRPO). Not fixed.

### Benchmark tampering (harmful gradient)
- **Problem:** Trial `aDxZVS5` rewrote `benchmarks/benchmark_algorithms.py` with a mock `fetch_drug_interaction_network()` returning a smaller graph. This "sped up" the benchmark by changing the input, not the algorithm. Reward: 0.934. GRPO advantage: +0.80 — the highest in step 1.
- **Impact:** Model learned to modify benchmark files instead of algorithm code. Actively harmful gradient.
- **Status:** networkx was **dropped** from training tasks because of this.

---

## joblib (`joblib_joblib_484`)

### pytest 8 breaks pre-existing tests
- **Problem:** Container had pytest 8.3.5 installed. `pytest.warns(None)` was removed in pytest 8. 10 pre-existing tests fail before any agent change: `test_main_thread_renamed_no_warning`, `test_nested_parallel_warnings`, `test_warning_about_timeout_not_supported_by_backend`.
- **Fix:** Pin `pytest<7` in Dockerfile. Now pytest 6.2.5 installed.
- **Status:** Fixed.

### benchmark_sequential.py wiped by git clean
- **Problem:** `benchmark_sequential.py` COPY'd to `/workspace/repo/benchmarks/` but `git clean -fd` removes it (untracked file).
- **Fix:** Dockerfile also COPYs to `/opt/benchmark_sequential.py`. Entrypoint restores it after git clean.
- **Status:** Fixed.

### asv_benchmarks.txt missing benchmark_sequential entries
- **Problem:** `asv_benchmarks.txt` was pre-generated before `benchmark_sequential.py` existed. New benchmark names not listed → LSV doesn't know to measure them.
- **Fix:** Dockerfile copies existing `asv_benchmarks.txt` to `/opt/asv_benchmarks.txt` and appends:
  ```
  benchmark_sequential.SequentialParallelSuite.time_parallel_n_jobs_1
  benchmark_sequential.SequentialParallelSuite.time_parallel_n_jobs_1_with_timeout
  ```
  Entrypoint restores from `/opt/` after git clean.
- **Status:** Fixed.

### lightspeed_deps.db doesn't know about new benchmarks
- **Problem:** The SQLite cache mapping `benchmark_id → source files` was built before `benchmark_sequential.py` existed. Cache miss on first run.
- **Fix:** Not needed — LSV auto-populates the cache on a miss via `coverage.py` run. First trial takes slightly longer, subsequent trials use the cache.
- **Status:** Accepted behavior.

### N_JOBS_MAX = os.cpu_count() returns 128
- **Problem:** The container sees all 128 host CPUs. `N_JOBS_MAX = os.cpu_count()` in `benchmarks/common.py` causes benchmarks to spawn 128 worker processes → timeout.
- **Fix:** Entrypoint.sh applies:
  ```bash
  sed -i 's/N_JOBS_MAX = os.cpu_count()/N_JOBS_MAX = min(4, os.cpu_count() or 1)/' benchmarks/common.py
  ```
- **Status:** Fixed.

### Agents don't apply patches (systematic, step 1)
- **Problem:** Model (Qwen3.5-4B) explores the joblib codebase correctly across 65 turns, identifies `parallel.py` as the target, and plans the `supports_timeout` caching fix — but fails to successfully execute the file edit before the turn limit. Loops through "plan → attempt edit → check → didn't work → retry" without committing a change.
- **Impact:** All 8 joblib trajectories in step 1 scored reward=0. Zero GRPO gradient from joblib in step 1.
- **Status:** Unresolved. Model capability issue (4B model struggling with file editing tools).

### joblib LSV error: selected benchmarks but measured 0
- **Problem:** `lsv_error: "LSV selected 2 benchmarks but measured 0 (time=1.6s)"`. Benchmark was found and selected by LSV but produced no timing output.
- **Possible cause:** The benchmark ran too fast to measure, or the agent's patch caused the benchmark body to exit early.
- **Status:** Observed once. Impact: reward=0.

---

## Qcodes (`microsoft_Qcodes_7669`)

### Agents add overhead instead of removing it
- **Problem:** Agents consistently identify `function_helpers.py` / `is_function()` as the bottleneck but add attribute lookups (e.g., `getattr(f, '__func__', ...)`, `bool()` wrappers) instead of removing overhead. All patching agents score 0.686–0.711.
- **Impact:** Near-zero within-group variance → GRPO gradient within the patching group ≈ 0. The push from "patching" → "not patching" exists (+0.35 advantage), but no signal distinguishing good vs bad patches.
- **Status:** Accepted (task retained). May improve with more training steps.

---

## pvlib (`pvlib_pvlib-python_369`)

### Stale image + missing benchmark
- **Problem:** `benchmark_clearsky.py` was never staged in the Docker image. Unclear if the image was rebuilt with the needed benchmark file.
- **Status:** pvlib was **dropped** from training tasks. Not fixed.

---

## Training Infrastructure

### OOM: backward pass at large sequence lengths
- **Problem:** `max_seq_len` setting in SkyRL is a warning threshold only — sequences longer than `max_seq_len` are NOT truncated before the backward pass. One 74234-token trajectory caused `loss.backward()` to OOM (needed 34 GiB, had 20 GiB free).
- **Root cause:** Linear scaling: backward allocation ≈ `34.34 × (seq_len / 74234)` GiB. At `max_seq_len=40960`, a 74k-token outlier still passes through.
- **Fix:** Set `MAX_MODEL_LEN` (vLLM generation limit) = `max_seq_len`. This caps trajectory length at the source. Current: `MAX_MODEL_LEN=65536`, `max_seq_len=65536`.

### OOM: ref-model forward pass at large sequence lengths  
- **Problem:** After step 1 training update, the ref-model forward pass (computing reference logprobs for step 2 trajectories) needs ~30 GiB for lm_head on a ~98k-token sequence. With `gpu_memory_utilization=0.25`, vLLM held 32 GiB, leaving only 22 GiB free → OOM.
- **Fix:** Lower `gpu_memory_utilization` from 0.25 → 0.18. Frees ~17 GiB → ~38 GiB free for forward pass. Current setting: `GPU_MEM_UTIL=0.18`.

### Docker permission denied in new tmux session
- **Problem:** `tmux new-session -d` creates a non-login shell that may not have the `docker` group active, even if the user is in the group. Harbor's docker compose fails: `permission denied while trying to connect to the Docker daemon socket`.
- **Fix:** Launch training with `sg docker -c "CUDA_VISIBLE_DEVICES=2,3 bash ..."` to force docker group membership.
- **Status:** Fixed.

### avg_response_length=1.0 (all trajectories trivially failed)
- **Problem:** An entire training run produced `avg_response_length=1.0` with reward=0 for all 128 trajectories in 4 steps (~3 min each). All trajectories failed because the tmux session didn't have docker access.
- **Fix:** Same as above (sg docker).

### GPU contention from other users
- **Problem:** During a training run (step 1 → step 2 transition), another user's process claimed physical GPU 2, reducing available memory from ~65 GiB to ~22 GiB. OOM on step 2's forward pass.
- **Mitigation:** Lower `gpu_memory_utilization` reduces peak memory usage and provides more buffer against unexpected contention. Cannot fully prevent if another user takes significant GPU memory mid-run.

---

## Entrypoint Pattern (all fixed tasks)

All tasks follow this entrypoint pattern:
```bash
# 1. Reset repo to known base commit
git reset --hard <base_commit> 2>/dev/null || true
git clean -fd 2>/dev/null || true

# 2. Restore files that git clean removed (staged in /opt/ during Docker build)
cp /opt/<file> <repo_path>/<file> 2>/dev/null || true

# 3. Apply any runtime patches (e.g., CPU cap, asv.conf.json injection)
sed -i '...' <file> 2>/dev/null || true

# 4. Remove the entrypoint itself (agent shouldn't see it)
rm -- "$0"

# 5. Execute the command passed to docker run
exec "$@"
```

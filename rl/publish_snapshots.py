#!/usr/bin/env python3
"""Publish oracle behavioral snapshots to the FormulaCode prod snapshot store.

FormulaCode's regression-relative correctness gate needs the oracle `.snapshots/`
baseline available to agent trials. The prod `snapshots` storage bucket
(db.formulacode.org) is the durable home, keyed `{owner}__{repo}__{issue}/oracle.tar.gz`
with the tarball rooted at `.snapshots` (big `.pkl`/`.pkl.gz` payloads dropped, matching
datasmith's upload filter).

Auth is host-side only (containers are CF-Access-gated): SUPABASE_KEY + the
CF-Access service-token headers from datasmith/tokens.env, plus a browser User-Agent.

Use:
    python publish_snapshots.py                 # publish all local oracle runs
    python publish_snapshots.py --task geopandas/geopandas#3345
or import publish_snapshot(...) from rl/oracle_gold.py to auto-publish each oracle run.
"""
from __future__ import annotations

import argparse
import io
import os
import tarfile
import urllib.error
import urllib.request
from pathlib import Path

ORACLE_RUNS = Path("/home/arjun/formulacode-rlvr-recipe/results/oracle-runs")
TOKENS_ENV = Path("/home/arjun/datasmith/tokens.env")
_UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/120 Safari/537.36"

# "owner/repo#issue" -> local oracle trial dir name (under ORACLE_RUNS, has artifacts/.snapshots).
# Built from oracle_gold.GOLD_TASKS so publish + the gateway's local fallback cover the WHOLE
# cohort, not a hardcoded 6 (a stale 6-task list is exactly why 24/30 train baselines were never
# published to prod). oracle_gold's top-level imports are stdlib-only, so this pulls no heavy deps;
# its trial dir naming (build_config) is `{task_key with # and / -> _}_oracle`. Both consumers
# (publish main, gateway _local_lookup) guard for the dir actually existing on disk.
def _build_gold() -> dict[str, str]:
    try:
        from oracle_gold import GOLD_TASKS
    except Exception:  # noqa: BLE001 — fall back to nothing rather than crash the gateway import
        return {}
    return {
        label: f"{task_key.replace('#', '_').replace('/', '_')}_oracle"
        for task_key, (_dir, label) in GOLD_TASKS.items()
    }


GOLD = _build_gold()


def _load_tokens() -> dict[str, str]:
    env: dict[str, str] = {}
    for line in TOKENS_ENV.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        env[k.strip()] = v.strip().strip('"').strip("'")
    return env


def _auth_headers(env: dict[str, str]) -> dict[str, str]:
    key = env["SUPABASE_KEY"]
    return {
        "apikey": key,
        "Authorization": f"Bearer {key}",
        "User-Agent": _UA,
        "CF-Access-Client-Id": env["DATASMITH_CF_ACCESS_CLIENT_ID"],
        "CF-Access-Client-Secret": env["DATASMITH_CF_ACCESS_CLIENT_SECRET"],
    }


def object_key(owner: str, repo: str, issue: int | str) -> str:
    """Prod snapshot storage key (matches the existing bucket convention)."""
    return f"{owner}__{repo}__{issue}/oracle.tar.gz"


def _tar_snapshots(snapshot_dir: Path) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        tar.add(
            str(snapshot_dir),
            arcname=".snapshots",
            filter=lambda i: None if i.name.endswith((".pkl", ".pkl.gz")) else i,
        )
    return buf.getvalue()


def publish_snapshot(
    owner: str, repo: str, issue: int | str, snapshot_dir: str | Path,
    env: dict[str, str] | None = None,
) -> tuple[int | None, str]:
    """Tar snapshot_dir and upsert to the prod snapshots bucket. Returns (status, key)."""
    env = env or _load_tokens()
    snapshot_dir = Path(snapshot_dir)
    if not (snapshot_dir / "baseline.json").exists():
        return None, f"no baseline.json in {snapshot_dir}"
    base = env["SUPABASE_URL"]
    key = object_key(owner, repo, issue)
    url = f"{base}/storage/v1/object/snapshots/{key}"
    data = _tar_snapshots(snapshot_dir)
    headers = {**_auth_headers(env), "Content-Type": "application/gzip", "x-upsert": "true"}
    for method in ("POST", "PUT"):
        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                return r.status, key
        except urllib.error.HTTPError as e:
            if method == "POST" and e.code in (400, 409):  # exists -> PUT
                continue
            return e.code, f"{key}: {e.read()[:120]!r}"
        except Exception as e:  # noqa: BLE001
            return None, f"{key}: {e}"
    return None, key


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", help="single task 'owner/repo#issue' (default: all gold)")
    args = ap.parse_args()
    env = _load_tokens()
    tasks = [args.task] if args.task else list(GOLD)
    for label in tasks:
        trial = GOLD.get(label)
        if not trial:
            print(f"  ?? unknown task {label}"); continue
        owner, rest = label.split("/", 1)
        repo, issue = rest.split("#", 1)
        snap = ORACLE_RUNS / trial / "artifacts" / ".snapshots"
        st, info = publish_snapshot(owner, repo, issue, snap, env)
        flag = "OK " if st == 200 else "!! "
        print(f"  {flag}{label}: status={st} {info}")


if __name__ == "__main__":
    main()

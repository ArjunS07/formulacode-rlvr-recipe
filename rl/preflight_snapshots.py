#!/usr/bin/env python3
"""Fail-loud preflight for the FormulaCode snapshot correctness gate.

Training rewards are only trustworthy if the snapshot gate can actually fire: every train
task must serve a NON-VACUOUS oracle baseline — a ``baseline.json`` with real ``pass`` entries
AND a ``snapshots.db`` holding the return-value blobs those entries compare against. The entire
prior training history silently violated this: every published oracle baseline had 0 entries
(the baked ``lexer|verifier`` default filter recorded nothing), so ``snapshot-tool verify``
computed no transitions and every trial "passed" correctness for free.

This script fetches each task's oracle tarball from prod over the SAME storage path the gateway
proxies to containers (``snapshots/{owner}__{repo}__{issue}/oracle.tar.gz``, host-side CF-Access
auth), opens it in memory, and reports:

  * ``baseline_pass``  — ``baseline.json`` entries with status ``pass`` (the reproducible set the
    verify transition matrix keys off; this is what makes the gate non-vacuous),
  * ``db_usable``      — ``snapshots.db`` rows with ``capture_failed=0`` (return values present),
  * a verdict: OK | QUARANTINE (below threshold) | EMPTY | MISSING (404) | ERROR.

Exit status is NON-ZERO if any TRAIN task is not OK, and the offending tasks are printed loudly,
so a launch wrapper can abort before wasting a run on an inert gate. QUARANTINE tasks are also
written to ``--quarantine-out`` so they can be dropped from the train set (fragile tasks whose
oracle can't reproducibly capture enough benchmarks).

Usage:
    python preflight_snapshots.py                      # audit the 30 train ids, abort on any bad
    python preflight_snapshots.py --set eval           # audit the held-out eval ids
    python preflight_snapshots.py --min-pass 3         # threshold (default 3, or $SNAPSHOT_MIN_PASS)
    python preflight_snapshots.py --json               # machine-readable report on stdout
"""
from __future__ import annotations

import argparse
import io
import json
import os
import re
import sqlite3
import sys
import tarfile
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

from publish_snapshots import _auth_headers, _load_tokens, object_key

SURVEY = Path("/home/arjun/formulacode-verified-rl/initial_survey")
TRAIN_IDS = SURVEY / "rl_train_ids.md"
EVAL_IDS = SURVEY / "rl_eval_ids.md"
DEFAULT_MIN_PASS = int(os.environ.get("SNAPSHOT_MIN_PASS", "3"))

# "- owner/repo#issue (v3=..)" -> (owner, repo, issue)
_ID_RE = re.compile(r"^-+\s*([^/\s]+)/([^#\s]+)#(\d+)")


def parse_ids(path: Path) -> list[tuple[str, str, str]]:
    out: list[tuple[str, str, str]] = []
    for line in path.read_text().splitlines():
        m = _ID_RE.match(line.strip())
        if m:
            out.append((m.group(1), m.group(2), m.group(3)))
    return out


def _fetch_tarball(base: str, headers: dict, owner: str, repo: str, issue: str) -> bytes | None:
    key = object_key(owner, repo, issue)
    url = f"{base}/storage/v1/object/snapshots/{key}"
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=180) as r:
            return r.read()
    except urllib.error.HTTPError as e:
        # Supabase storage returns 400 with a NoSuchKey/not_found body (not 404) for a
        # missing object. Treat any "object not found" as MISSING, not a hard error.
        body = b""
        try:
            body = e.read()
        except Exception:  # noqa: BLE001
            pass
        if e.code in (400, 404) and (b"not_found" in body or b"NoSuchKey" in body):
            return None
        raise


def _inspect_tarball(data: bytes) -> dict:
    """Return {baseline_total, baseline_pass, baseline_fail, baseline_skip, db_usable, db_rows}."""
    rec = {"baseline_total": 0, "baseline_pass": 0, "baseline_fail": 0, "baseline_skip": 0,
           "db_usable": 0, "db_rows": 0}
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
        # baseline.json — the manifest verify keys its transition matrix off
        try:
            m = tar.getmember(".snapshots/baseline.json")
            bl = json.loads(tar.extractfile(m).read().decode())
            entries = bl.get("entries") or {}
            rec["baseline_total"] = len(entries)
            for st in entries.values():
                if st == "pass":
                    rec["baseline_pass"] += 1
                elif st == "skip":
                    rec["baseline_skip"] += 1
                else:  # fail / failed_to_pass / unknown
                    rec["baseline_fail"] += 1
        except KeyError:
            pass
        # snapshots.db — the return-value blobs verify compares against
        try:
            db_member = tar.getmember(".snapshots/snapshots.db")
            with tempfile.NamedTemporaryFile(suffix=".db") as tf:
                tf.write(tar.extractfile(db_member).read())
                tf.flush()
                con = sqlite3.connect(tf.name)
                rec["db_rows"] = con.execute("select count(*) from snapshots").fetchone()[0]
                rec["db_usable"] = con.execute(
                    "select count(*) from snapshots where capture_failed=0"
                ).fetchone()[0]
                con.close()
        except (KeyError, sqlite3.Error):
            pass
    return rec


def _verdict(rec: dict | None, min_pass: int) -> str:
    if rec is None:
        return "MISSING"
    if rec.get("error"):
        return "ERROR"
    if rec["baseline_pass"] >= min_pass and rec["db_usable"] >= 1:
        return "OK"
    if rec["baseline_total"] == 0 and rec["db_usable"] == 0:
        return "EMPTY"
    return "QUARANTINE"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=["train", "eval"], default="train")
    ap.add_argument("--min-pass", type=int, default=DEFAULT_MIN_PASS,
                    help=f"min reproducible baseline 'pass' entries (default {DEFAULT_MIN_PASS})")
    ap.add_argument("--quarantine-out", type=Path,
                    default=SURVEY / "rl_quarantine.txt",
                    help="write QUARANTINE/EMPTY/MISSING owner/repo#issue here (one per line)")
    ap.add_argument("--json", action="store_true", help="machine-readable report on stdout")
    args = ap.parse_args()

    ids_path = TRAIN_IDS if args.set == "train" else EVAL_IDS
    ids = parse_ids(ids_path)
    env = _load_tokens()
    base = env["SUPABASE_URL"]
    headers = _auth_headers(env)

    rows: list[dict] = []
    for owner, repo, issue in ids:
        label = f"{owner}/{repo}#{issue}"
        try:
            data = _fetch_tarball(base, headers, owner, repo, issue)
            rec = _inspect_tarball(data) if data is not None else None
        except Exception as e:  # noqa: BLE001
            rec = {"error": str(e)[:80]}
        rows.append({"label": label, "verdict": _verdict(rec, args.min_pass), "rec": rec})

    if args.json:
        print(json.dumps(rows, indent=2))
    else:
        print(f"\nSnapshot preflight — {args.set} set ({len(ids)} tasks), "
              f"min_pass={args.min_pass}, prod={base}\n")
        print(f"  {'task':40} {'verdict':11} {'bl_pass':>7} {'bl_skip':>7} {'db_usable':>9} {'db_rows':>7}")
        for r in rows:
            rec = r["rec"] or {}
            print(f"  {r['label']:40} {r['verdict']:11} "
                  f"{rec.get('baseline_pass',''):>7} {rec.get('baseline_skip',''):>7} "
                  f"{rec.get('db_usable',''):>9} {rec.get('db_rows',''):>7}")

    bad = [r for r in rows if r["verdict"] != "OK"]
    quarantine = [r["label"] for r in rows if r["verdict"] in ("QUARANTINE", "EMPTY", "MISSING")]
    if quarantine and args.quarantine_out:
        args.quarantine_out.write_text("\n".join(quarantine) + "\n")

    if bad:
        print(f"\n*** SNAPSHOT GATE PREFLIGHT FAILED: {len(bad)}/{len(rows)} {args.set} tasks "
              f"have a vacuous/missing oracle baseline ***", file=sys.stderr)
        for r in bad:
            print(f"    {r['verdict']:11} {r['label']}", file=sys.stderr)
        print("\n  The correctness gate CANNOT fire for these tasks — training on them rewards "
              "fast-but-wrong\n  agents for free. Re-record their oracle baselines with "
              "FORMULACODE_SNAPSHOT_FILTER='.*'\n  (rl/oracle_gold.py) and republish "
              "(rl/publish_snapshots.py), or drop them from the train set.\n", file=sys.stderr)
        if quarantine:
            print(f"  Wrote {len(quarantine)} tasks to {args.quarantine_out}", file=sys.stderr)
        return 1

    print(f"\nOK — all {len(rows)} {args.set} tasks serve a non-vacuous oracle baseline "
          f"(>= {args.min_pass} reproducible pass entries).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

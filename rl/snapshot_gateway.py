#!/usr/bin/env python3
"""Host-side snapshot gateway: serves oracle snapshots to CF-gated agent containers.

The FormulaCode task images bake a snapshot download that calls, against SUPABASE_URL:
  GET /rest/v1/tasks?owner=eq..&repo=eq..&issue_number=eq..&select=snapshot_storage_url
  GET /storage/v1/object/snapshots/{owner}/{repo}/{issue}/oracle.tar.gz   (fallback)
On this box containers cannot reach prod (Cloudflare Access + DNS), so we point their
SUPABASE_URL at this gateway (reachable over host networking). The gateway:
  * answers the tasks query with a null snapshot_storage_url (forces the fallback path),
  * on the storage GET, remaps the slash key to the prod convention
    `snapshots/{owner}__{repo}__{issue}/oracle.tar.gz`, FETCHES it from prod
    (db.formulacode.org, host-side CF-Access auth), caches it, and serves it,
  * falls back to the local oracle-runs `.snapshots/` if prod misses,
  * no-ops agent result uploads (POST/PUT/PATCH -> 200).

This is durable (prod-backed) and needs no image rebuild. Run:
    python snapshot_gateway.py [--port 8199]
"""
from __future__ import annotations

import argparse
import http.server
import io
import json
import tarfile
import urllib.error
import urllib.request
from pathlib import Path

from publish_snapshots import _auth_headers, _load_tokens, GOLD, ORACLE_RUNS

CACHE = Path("/home/arjun/_snapshot_cache")


def _local_lookup() -> dict[tuple[str, str, str], Path]:
    """(owner_lower, repo_lower, issue) -> local .snapshots dir, for prod-miss fallback."""
    out: dict[tuple[str, str, str], Path] = {}
    for label, trial in GOLD.items():
        owner, rest = label.split("/", 1)
        repo, issue = rest.split("#", 1)
        snap = ORACLE_RUNS / trial / "artifacts" / ".snapshots"
        if snap.is_dir():
            out[(owner.lower(), repo.lower(), issue)] = snap
    return out


class Gateway(http.server.ThreadingHTTPServer):
    def __init__(self, addr, handler, env):
        super().__init__(addr, handler)
        self.env = env
        self.base = env["SUPABASE_URL"]
        self.headers = _auth_headers(env)
        self.local = _local_lookup()
        CACHE.mkdir(parents=True, exist_ok=True)


class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body=b"", ctype="application/json"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if body:
            self.wfile.write(body)

    # ── prod fetch + local fallback ─────────────────────────────────────────
    def _prod_tarball(self, owner, repo, issue) -> bytes | None:
        key = f"{owner}__{repo}__{issue}/oracle.tar.gz"
        cache = CACHE / key
        if cache.is_file():
            return cache.read_bytes()
        url = f"{self.server.base}/storage/v1/object/snapshots/{key}"
        try:
            with urllib.request.urlopen(
                urllib.request.Request(url, headers=self.server.headers), timeout=120
            ) as r:
                data = r.read()
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_bytes(data)
            print(f"[gateway] prod HIT {key} ({len(data)} bytes)")
            return data
        except urllib.error.HTTPError as e:
            print(f"[gateway] prod MISS {key} ({e.code})")
            return None
        except Exception as e:  # noqa: BLE001
            print(f"[gateway] prod ERR {key}: {e}")
            return None

    def _local_tarball(self, owner, repo, issue) -> bytes | None:
        snap = self.server.local.get((owner.lower(), repo.lower(), str(issue)))
        if not snap:
            return None
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tar:
            tar.add(str(snap), arcname=".snapshots",
                    filter=lambda i: None if i.name.endswith((".pkl", ".pkl.gz")) else i)
        print(f"[gateway] local fallback {owner}/{repo}/{issue}")
        return buf.getvalue()

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path.startswith("/rest/v1/tasks"):
            self._send(200, json.dumps([{"snapshot_storage_url": None}]).encode())
            return
        marker = "/storage/v1/object/"
        if marker in path:
            key = path.split(marker, 1)[1].removeprefix("public/")
            # expect: snapshots/{owner}/{repo}/{issue}/oracle.tar.gz
            parts = key.split("/")
            if len(parts) >= 5 and parts[0] == "snapshots" and parts[-1] == "oracle.tar.gz":
                owner, repo, issue = parts[1], parts[2], parts[3]
                data = self._prod_tarball(owner, repo, issue) or self._local_tarball(owner, repo, issue)
                if data:
                    self._send(200, data, "application/gzip")
                else:
                    self._send(404, b'{"error":"snapshot not found"}')
                return
            self._send(404, b'{"error":"bad key"}')
            return
        self._send(200, b"[]")

    def _drain(self):
        n = int(self.headers.get("Content-Length", 0) or 0)
        if n:
            self.rfile.read(n)

    def do_POST(self):
        self._drain(); self._send(201, b'{"ok":true}')

    def do_PUT(self):
        self._drain(); self._send(200, b'{"ok":true}')

    def do_PATCH(self):
        self._drain(); self._send(200, b'{"ok":true}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8199)
    args = ap.parse_args()
    env = _load_tokens()
    srv = Gateway(("0.0.0.0", args.port), Handler, env)
    print(f"[gateway] serving prod-backed snapshots on 0.0.0.0:{args.port} "
          f"(cache {CACHE}, {len(srv.local)} local fallbacks)")
    srv.serve_forever()


if __name__ == "__main__":
    main()

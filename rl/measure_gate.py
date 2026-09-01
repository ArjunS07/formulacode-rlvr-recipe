#!/usr/bin/env python3
"""Host-side measurement-phase semaphore for FormulaCode trials.

LSV benchmark timing is memory-bandwidth sensitive: running many trials' measure
step at once on a multi-socket box saturates memory bandwidth and corrupts the
timings (benchmarks measure many-x too slow). This gate lets trials GENERATE at
high concurrency but serializes the MEASURE step to <=N at a time.

A trial (setup.sh/test.sh) calls, around lsv_init/lsv_measure:
    curl -s -m <big> -X POST http://127.0.0.1:8266/acquire?sid=<uniq>   # blocks until a slot frees
    ... run lsv_measure.py ...
    curl -s -m 5     -X POST http://127.0.0.1:8266/release?sid=<uniq>

Fail-open by design: if the gate is down or acquire times out, the trial proceeds
(never deadlock training). A lease TTL auto-frees a slot whose holder died without
releasing, so a killed/OOM'd trial can't wedge the gate.

Run (host):  python3 measure_gate.py --port 8266 --slots 2 --ttl 900
"""
from __future__ import annotations
import argparse, json, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

SLOTS = 2
TTL = 900.0            # seconds a slot may be held before it is force-reclaimed
ACQUIRE_WAIT = 1800.0  # max seconds a client blocks in /acquire before fail-open

_lock = threading.Lock()
_held: dict[str, float] = {}   # sid -> expiry timestamp
_sem = threading.Semaphore(SLOTS)


def _reap() -> int:
    """Force-release slots whose lease expired (holder died). Returns count reaped."""
    now = time.time()
    reaped = 0
    with _lock:
        for sid in [s for s, exp in _held.items() if exp <= now]:
            _held.pop(sid, None)
            _sem.release()
            reaped += 1
    return reaped


def _acquire(sid: str) -> bool:
    _reap()
    if _sem.acquire(timeout=ACQUIRE_WAIT):
        with _lock:
            _held[sid] = time.time() + TTL
        return True
    return False


def _release(sid: str) -> bool:
    with _lock:
        if sid in _held:
            _held.pop(sid, None)
            _sem.release()
            return True
    return False


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # quiet
        pass

    def _send(self, code: int, obj: dict) -> None:
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        try:
            self.wfile.write(body)
        except Exception:
            pass  # client gone; lease TTL will reclaim if it was an acquire

    def _sid(self) -> str:
        q = parse_qs(urlparse(self.path).query)
        return (q.get("sid") or ["anon"])[0]

    def do_GET(self):
        p = urlparse(self.path).path
        if p in ("/health", "/"):
            with _lock:
                self._send(200, {"slots": SLOTS, "in_use": len(_held),
                                 "held": list(_held.keys())})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        p = urlparse(self.path).path
        sid = self._sid()
        if p == "/acquire":
            self._send(200, {"acquired": _acquire(sid), "sid": sid})
        elif p == "/release":
            self._send(200, {"released": _release(sid), "sid": sid})
        else:
            self._send(404, {"error": "not found"})


def _reaper_loop():
    while True:
        time.sleep(10)
        _reap()


def main() -> None:
    global SLOTS, TTL, ACQUIRE_WAIT, _sem
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8266)
    ap.add_argument("--slots", type=int, default=2)
    ap.add_argument("--ttl", type=float, default=900.0)
    ap.add_argument("--acquire-wait", type=float, default=1800.0)
    a = ap.parse_args()
    SLOTS, TTL, ACQUIRE_WAIT = a.slots, a.ttl, a.acquire_wait
    _sem = threading.Semaphore(SLOTS)
    threading.Thread(target=_reaper_loop, daemon=True).start()
    srv = ThreadingHTTPServer(("0.0.0.0", a.port), Handler)
    print(f"[measure_gate] up on :{a.port} slots={SLOTS} ttl={TTL}s", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()

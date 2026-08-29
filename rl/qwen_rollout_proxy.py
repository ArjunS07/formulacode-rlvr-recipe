#!/usr/bin/env python3
"""Proxy between the qwen-code CLI and skyrl's vLLM engine that records the per-turn
token_ids+logprobs GRPO needs (qwen-code discards them). Trials pass OPENAI_BASE_URL=
.../trial/<session_id>/v1 so calls are keyed by session_id; each call is forwarded
non-streaming with logprobs+return_token_ids, recorded, then re-streamed to qwen as SSE.
GET /trial/<sid>/rollout returns the assembled per-turn arrays for the QwenCodeRL hook.

Run: /home/arjun/skyrl-formulacode/.venv/bin/python rl/qwen_rollout_proxy.py \
        --port 30022 --upstream http://127.0.0.1:30021/v1  [--verbose]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import time
import uuid
from collections import defaultdict
from typing import Any

from aiohttp import ClientSession, ClientTimeout, web

# ── in-memory per-session rollout store ─────────────────────────────────────
# session_id -> list of per-turn dicts {prompt_token_ids, completion_token_ids, logprobs}
_ROLLOUTS: dict[str, list[dict[str, Any]]] = defaultdict(list)
_LAST_WRITE: dict[str, float] = {}          # session_id -> last append time, for TTL eviction
_UPSTREAM: str = "http://127.0.0.1:30021/v1"
_SESSION: ClientSession | None = None       # shared upstream client (created on startup)
_VERBOSE = False

_UPSTREAM_TIMEOUT = 1800   # per-turn upstream ceiling (s); well under the 2400s agent cap
_SESSION_TTL = 3600        # evict a session's turns if unfetched this long (trial can't outlive it)
_SWEEP_INTERVAL = 300


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] [proxy] {msg}", flush=True)


def _extract_capture(resp: dict) -> dict[str, Any] | None:
    """Pull (prompt_token_ids, completion_token_ids, logprobs) from a non-streaming vLLM response.
    Returns None if shapes are missing, the completion/logprobs lengths disagree, or the completion
    is empty — recording a mismatched turn would silently corrupt or mask the whole trajectory."""
    try:
        choice = resp["choices"][0]
        prompt_token_ids = resp.get("prompt_token_ids")
        completion_token_ids = choice.get("token_ids")
        lp = (choice.get("logprobs") or {}).get("content") or []
        logprobs = [t.get("logprob") for t in lp]
        if prompt_token_ids is None or not completion_token_ids:
            return None
        if len(completion_token_ids) != len(logprobs):
            _log(f"WARN token_ids/logprobs length mismatch ({len(completion_token_ids)} vs "
                 f"{len(logprobs)}) — dropping turn")
            return None
        return {
            "prompt_token_ids": prompt_token_ids,
            "completion_token_ids": completion_token_ids,
            "logprobs": logprobs,
        }
    except (KeyError, IndexError, TypeError):
        return None


def _to_sse_chunks(resp: dict) -> list[str]:
    """Re-emit a completed ChatCompletionResponse as OpenAI streaming SSE chunks (role, content,
    tool_calls as one complete delta each, then a finish chunk and [DONE])."""
    cid = resp.get("id", f"chatcmpl-{uuid.uuid4().hex}")
    created = resp.get("created", int(time.time()))
    model = resp.get("model", "")
    choice = resp["choices"][0]
    msg = choice.get("message", {})
    base = {"id": cid, "object": "chat.completion.chunk", "created": created, "model": model}

    def chunk(delta: dict, finish: Any = None) -> str:
        c = {**base, "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
        return f"data: {json.dumps(c)}\n\n"

    out = [chunk({"role": "assistant"})]
    if msg.get("content"):
        out.append(chunk({"content": msg["content"]}))
    for i, tc in enumerate(msg.get("tool_calls") or []):
        out.append(chunk({"tool_calls": [{
            "index": i,
            "id": tc.get("id"),
            "type": "function",
            "function": {
                "name": tc["function"]["name"],
                "arguments": tc["function"].get("arguments", ""),
            },
        }]}))
    out.append(chunk({}, finish=choice.get("finish_reason") or "stop"))
    out.append("data: [DONE]\n\n")
    return out


async def _forward_nonstream(body: dict) -> dict:
    """Make a non-streaming upstream call with token/logprob capture enabled (shared session)."""
    up_body = dict(body)
    up_body["stream"] = False
    up_body.pop("stream_options", None)
    up_body["logprobs"] = True
    up_body.setdefault("top_logprobs", 1)
    up_body["return_token_ids"] = True
    assert _SESSION is not None, "upstream session not initialized"
    async with _SESSION.post(f"{_UPSTREAM}/chat/completions", json=up_body,
                             headers={"Content-Type": "application/json"}) as r:
        return await r.json()


async def handle_chat(request: web.Request) -> web.StreamResponse:
    sid = request.match_info.get("sid", "nosession")
    body = await request.json()
    wants_stream = bool(body.get("stream"))
    n_msgs = len(body.get("messages") or [])
    n_tools = len(body.get("tools") or [])
    if _VERBOSE:
        _log(f"sid={sid[:8]} chat/completions stream={wants_stream} n_messages={n_msgs} n_tools={n_tools}")

    resp = await _forward_nonstream(body)

    # surface upstream errors (or a malformed 200 with no choices) to qwen instead of crashing
    if "choices" not in resp:
        _log(f"sid={sid[:8]} UPSTREAM ERROR/malformed: {str(resp.get('error') or resp)[:160]}")
        code = (resp.get("error") or {}).get("code", 500)
        status = int(code) if str(code).isdigit() else 500
        return web.json_response(resp if "error" in resp else {"error": {"message": "no choices"}},
                                 status=status)

    cap = _extract_capture(resp)  # appended only after successful delivery (retry-safe)

    # strip fields qwen doesn't expect before handing back
    clean = dict(resp)
    clean.pop("prompt_token_ids", None)
    clean.pop("prompt_logprobs", None)
    for ch in clean.get("choices", []):
        ch.pop("token_ids", None)

    def _record() -> None:
        if cap is None:
            _log(f"sid={sid[:8]} WARN: could not capture token_ids/logprobs this turn")
            return
        _ROLLOUTS[sid].append(cap)
        _LAST_WRITE[sid] = time.time()
        if _VERBOSE:
            _log(f"sid={sid[:8]} CAPTURED turn#{len(_ROLLOUTS[sid])}: "
                 f"prompt={len(cap['prompt_token_ids'])} completion={len(cap['completion_token_ids'])}")

    if not wants_stream:
        resp_out = web.json_response(clean)
        _record()
        return resp_out

    sse = web.StreamResponse(status=200, headers={
        "Content-Type": "text/event-stream", "Cache-Control": "no-cache",
        "Connection": "keep-alive"})
    await sse.prepare(request)
    for c in _to_sse_chunks(clean):
        await sse.write(c.encode())
    await sse.write_eof()
    _record()   # record only after the turn is fully delivered, so a mid-stream retry can't double-count
    return sse


async def handle_models(request: web.Request) -> web.Response:
    # qwen-code may query /models on startup; return a minimal list.
    async with ClientSession(timeout=ClientTimeout(total=30)) as s:
        try:
            async with s.get(f"{_UPSTREAM}/models") as r:
                return web.json_response(await r.json())
        except Exception:
            return web.json_response({"object": "list", "data": [
                {"id": "qwen3p5-9b-local", "object": "model", "owned_by": "local"}]})


async def handle_rollout(request: web.Request) -> web.Response:
    sid = request.match_info.get("sid", "nosession")
    pop = request.query.get("pop") in ("1", "true", "yes")
    turns = _ROLLOUTS.pop(sid, []) if pop else _ROLLOUTS.get(sid, [])
    if pop and _VERBOSE:
        _log(f"sid={sid[:8]} rollout fetched+popped ({len(turns)} turns)")
    return web.json_response({
        "session_id": sid,
        "n_turns": len(turns),
        "turns": turns,
    })


async def handle_health(request: web.Request) -> web.Response:
    return web.json_response({"status": "ok", "sessions": len(_ROLLOUTS)})


async def _sweeper(app: web.Application) -> None:
    """Evict turns from sessions whose trial died before fetching (no ?pop), bounding memory."""
    try:
        while True:
            await asyncio.sleep(_SWEEP_INTERVAL)
            now = time.time()
            stale = [s for s, t in _LAST_WRITE.items() if now - t > _SESSION_TTL]
            for s in stale:
                _ROLLOUTS.pop(s, None)
                _LAST_WRITE.pop(s, None)
            if stale:
                _log(f"swept {len(stale)} stale sessions ({len(_ROLLOUTS)} live)")
    except asyncio.CancelledError:
        pass


async def _on_startup(app: web.Application) -> None:
    global _SESSION
    _SESSION = ClientSession(timeout=ClientTimeout(total=_UPSTREAM_TIMEOUT, sock_connect=30))
    app["sweeper"] = asyncio.create_task(_sweeper(app))


async def _on_cleanup(app: web.Application) -> None:
    app["sweeper"].cancel()
    if _SESSION is not None:
        await _SESSION.close()


def build_app() -> web.Application:
    app = web.Application(client_max_size=1024 ** 3)
    app.router.add_post("/trial/{sid}/v1/chat/completions", handle_chat)
    app.router.add_post("/trial/{sid}/v1/completions", handle_chat)  # tolerate /completions
    app.router.add_get("/trial/{sid}/v1/models", handle_models)
    app.router.add_get("/trial/{sid}/rollout", handle_rollout)
    app.router.add_get("/v1/models", handle_models)
    app.router.add_get("/health", handle_health)
    app.on_startup.append(_on_startup)
    app.on_cleanup.append(_on_cleanup)
    return app


def main() -> None:
    global _UPSTREAM, _VERBOSE
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=30022)
    ap.add_argument("--upstream", default="http://127.0.0.1:30021/v1")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    _UPSTREAM = args.upstream.rstrip("/")
    _VERBOSE = args.verbose
    _log(f"recording proxy on {args.host}:{args.port} -> upstream {_UPSTREAM} (verbose={_VERBOSE})")
    web.run_app(build_app(), host=args.host, port=args.port, print=None)


if __name__ == "__main__":
    main()

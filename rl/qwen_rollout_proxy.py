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
import json
import time
import uuid
from collections import defaultdict
from typing import Any

from aiohttp import ClientSession, ClientTimeout, web

# ── in-memory per-session rollout store ─────────────────────────────────────
# session_id -> list of per-turn dicts {prompt_token_ids, completion_token_ids, logprobs}
_ROLLOUTS: dict[str, list[dict[str, Any]]] = defaultdict(list)
_UPSTREAM: str = "http://127.0.0.1:30021/v1"
_VERBOSE = False


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] [proxy] {msg}", flush=True)


def _extract_capture(resp: dict) -> dict[str, Any] | None:
    """Pull (prompt_token_ids, completion_token_ids, logprobs) from a non-streaming
    vLLM ChatCompletionResponse. Returns None if the shapes are missing."""
    try:
        choice = resp["choices"][0]
        prompt_token_ids = resp.get("prompt_token_ids")
        completion_token_ids = choice.get("token_ids")
        lp = (choice.get("logprobs") or {}).get("content") or []
        logprobs = [t.get("logprob") for t in lp]
        if prompt_token_ids is None or completion_token_ids is None:
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
    """Make a non-streaming upstream call with token/logprob capture enabled."""
    up_body = dict(body)
    up_body["stream"] = False
    up_body.pop("stream_options", None)
    up_body["logprobs"] = True
    up_body.setdefault("top_logprobs", 1)
    up_body["return_token_ids"] = True
    timeout = ClientTimeout(total=None)
    async with ClientSession(timeout=timeout) as s:
        async with s.post(f"{_UPSTREAM}/chat/completions", json=up_body,
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

    if "error" in resp and "choices" not in resp:
        # surface upstream errors (e.g. context length) to qwen unchanged
        _log(f"sid={sid[:8]} UPSTREAM ERROR: {str(resp.get('error'))[:160]}")
        return web.json_response(resp, status=int(
            (resp.get("error") or {}).get("code", 500) if str(
                (resp.get("error") or {}).get("code", "")).isdigit() else 500))

    cap = _extract_capture(resp)
    if cap is not None:
        _ROLLOUTS[sid].append(cap)
        if _VERBOSE:
            _log(f"sid={sid[:8]} CAPTURED turn#{len(_ROLLOUTS[sid])}: "
                 f"prompt={len(cap['prompt_token_ids'])} completion={len(cap['completion_token_ids'])} "
                 f"logprobs={len(cap['logprobs'])}")
    else:
        _log(f"sid={sid[:8]} WARN: could not capture token_ids/logprobs this turn")

    # strip fields qwen doesn't expect before handing back
    clean = dict(resp)
    clean.pop("prompt_token_ids", None)
    clean.pop("prompt_logprobs", None)
    for ch in clean.get("choices", []):
        ch.pop("token_ids", None)

    if not wants_stream:
        return web.json_response(clean)

    sse = web.StreamResponse(status=200, headers={
        "Content-Type": "text/event-stream", "Cache-Control": "no-cache",
        "Connection": "keep-alive"})
    await sse.prepare(request)
    for c in _to_sse_chunks(clean):
        await sse.write(c.encode())
    await sse.write_eof()
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


def build_app() -> web.Application:
    app = web.Application(client_max_size=1024 ** 3)
    app.router.add_post("/trial/{sid}/v1/chat/completions", handle_chat)
    app.router.add_post("/trial/{sid}/v1/completions", handle_chat)  # tolerate /completions
    app.router.add_get("/trial/{sid}/v1/models", handle_models)
    app.router.add_get("/trial/{sid}/rollout", handle_rollout)
    app.router.add_get("/v1/models", handle_models)
    app.router.add_get("/health", handle_health)
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

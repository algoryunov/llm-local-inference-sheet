"""Deterministic mock streaming server for harness tests.

Everything it produces is SYNTHETIC. Runs against it are labelled
``ResultState.SYNTHETIC`` and are excluded from empirical reports.

Implemented on raw asyncio streams (not an HTTP framework), so tests can control
exactly how bytes hit the wire: arbitrary fragmentation, including splits inside
UTF-8 code points and CRLF pairs, as well as keep-alive comments, missing usage,
mid-stream errors and hangs.

Behaviour is set per request through a ``mock`` object in the JSON body
(fields of :class:`MockBehavior`). Server-wide defaults come from the constructor.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import random
from dataclasses import asdict, dataclass, field, fields
from typing import Any

WORDS = ["alpha", " beta", " gamma", " δέλτα", " 🙂", " ü", " 数据", " omega", "\n", " x"]


@dataclass
class MockBehavior:
    ttft_s: float = 0.01
    token_interval_s: float = 0.002
    tokens_per_chunk: int = 1
    # If > 0, stop "naturally" (finish_reason=stop) after this many tokens.
    eos_after: int = 0
    include_usage: bool = True
    include_role_chunk: bool = True
    include_timings: bool = False  # llama.cpp-style `timings`
    fragment: str = "none"  # none | bytes1 | random
    fragment_seed: int = 0
    line_ending: str = "\n"  # "\n" | "\r\n" | "\r"
    keepalive_comments: bool = False
    http_status: int = 200
    error_after_tokens: int = -1  # emit an SSE error object after N tokens, then close
    hang_after_tokens: int = -1  # stop sending (without closing) after N tokens
    drop_connection_after_tokens: int = -1
    reasoning_tokens: int = 0  # emit N reasoning tokens before content
    ignore_eos_supported: bool = True
    words: list[str] = field(default_factory=lambda: list(WORDS))

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None, base: MockBehavior) -> MockBehavior:
        merged = asdict(base)
        if d:
            known = {f.name for f in fields(cls)}
            merged.update({k: v for k, v in d.items() if k in known})
        return cls(**merged)


class MockServer:
    def __init__(self, host: str = "127.0.0.1", port: int = 0, defaults: MockBehavior | None = None):
        self.host = host
        self.port = port
        self.defaults = defaults or MockBehavior()
        self._server: asyncio.base_events.Server | None = None
        self.requests_seen: list[dict[str, Any]] = []

    async def start(self) -> None:
        self._server = await asyncio.start_server(self._handle, self.host, self.port)
        sock = self._server.sockets[0]
        self.port = sock.getsockname()[1]

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    async def stop(self) -> None:
        if self._server:
            self._server.close()
            with contextlib.suppress(Exception):
                await asyncio.wait_for(self._server.wait_closed(), 2)

    async def serve_forever(self) -> None:
        if self._server is None:
            await self.start()
        assert self._server
        async with self._server:
            await self._server.serve_forever()

    # ------------------------------------------------------------------ HTTP
    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            while True:
                req = await _read_request(reader)
                if req is None:
                    break
                method, path, headers, body = req
                keep = await self._route(method, path, body, writer)
                if not keep or headers.get("connection", "").lower() == "close":
                    break
        except (ConnectionError, asyncio.IncompleteReadError):
            pass
        finally:
            with contextlib.suppress(Exception):
                writer.close()

    async def _route(self, method: str, path: str, body: bytes, w: asyncio.StreamWriter) -> bool:
        if method == "GET" and path in ("/health", "/v1/health"):
            await _send_json(w, 200, {"status": "ok", "synthetic": True})
            return True
        if method == "GET" and path == "/v1/models":
            await _send_json(w, 200, {"object": "list", "data": [{"id": "mock-model", "synthetic": True}]})
            return True
        if method == "POST" and path in ("/v1/chat/completions", "/api/chat"):
            payload = json.loads(body or b"{}")
            self.requests_seen.append(payload)
            beh = MockBehavior.from_dict(payload.get("mock"), self.defaults)
            if beh.http_status != 200:
                await _send_json(w, beh.http_status, {"error": {"message": "mock failure", "code": beh.http_status}})
                return True
            if path == "/api/chat":
                return await self._stream_ollama(payload, beh, w)
            return await self._stream_openai(payload, beh, w)
        await _send_json(w, 404, {"error": "not found"})
        return True

    # ------------------------------------------------------------------ streams
    def _plan_tokens(self, payload: dict[str, Any], beh: MockBehavior) -> tuple[int, str]:
        max_tokens = int(payload.get("max_tokens") or payload.get("max_completion_tokens")
                         or (payload.get("options") or {}).get("num_predict") or 16)
        ignore_eos = bool(payload.get("ignore_eos")) and beh.ignore_eos_supported
        if beh.eos_after > 0 and not ignore_eos and beh.eos_after < max_tokens:
            return beh.eos_after, "stop"
        return max_tokens, "length"

    async def _stream_openai(self, payload: dict[str, Any], beh: MockBehavior, w: asyncio.StreamWriter) -> bool:
        n_tokens, finish = self._plan_tokens(payload, beh)
        stream = payload.get("stream", False)
        prompt_tokens = _approx_prompt_tokens(payload)
        if not stream:
            text = "".join(beh.words[i % len(beh.words)] for i in range(n_tokens))
            await _send_json(w, 200, {
                "id": "mock", "object": "chat.completion", "synthetic": True,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": finish}],
                "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": n_tokens},
            })
            return True
        headers = (
            "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n"
            "Cache-Control: no-cache\r\nConnection: close\r\n\r\n"
        )
        w.write(headers.encode())
        await w.drain()
        rng = random.Random(beh.fragment_seed)
        le = beh.line_ending

        async def emit(obj: dict[str, Any] | str) -> None:
            data = obj if isinstance(obj, str) else json.dumps(obj, ensure_ascii=False)
            raw = f"data: {data}{le}{le}".encode()
            if beh.keepalive_comments:
                raw = f": keepalive{le}".encode() + raw
            await _write_fragmented(w, raw, beh.fragment, rng)

        await asyncio.sleep(beh.ttft_s)
        base = {"id": "mock", "object": "chat.completion.chunk", "model": "mock-model"}
        if beh.include_role_chunk:
            await emit({**base, "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}}]})
        produced = 0
        total = beh.reasoning_tokens + n_tokens
        while produced < total:
            if produced == beh.hang_after_tokens:
                await asyncio.sleep(3600)
            if produced == beh.drop_connection_after_tokens:
                w.transport.abort()
                return False
            if produced == beh.error_after_tokens:
                await emit({"error": {"message": "mock mid-stream error", "type": "server_error"}})
                return False
            k = min(beh.tokens_per_chunk, total - produced)
            parts_r: list[str] = []
            parts_c: list[str] = []
            for j in range(k):
                idx = produced + j
                word = beh.words[idx % len(beh.words)]
                (parts_r if idx < beh.reasoning_tokens else parts_c).append(word)
            delta: dict[str, Any] = {}
            if parts_r:
                delta["reasoning_content"] = "".join(parts_r)
            if parts_c:
                delta["content"] = "".join(parts_c)
            await emit({**base, "choices": [{"index": 0, "delta": delta, "finish_reason": None}]})
            produced += k
            await asyncio.sleep(beh.token_interval_s * k)
        final: dict[str, Any] = {**base, "choices": [{"index": 0, "delta": {}, "finish_reason": finish}]}
        if beh.include_timings:
            final["timings"] = {
                "prompt_n": prompt_tokens, "prompt_ms": beh.ttft_s * 1000,
                "predicted_n": total, "predicted_ms": beh.token_interval_s * total * 1000,
                "cache_n": 0,
            }
        await emit(final)
        if beh.include_usage and (payload.get("stream_options") or {}).get("include_usage"):
            await emit({**base, "choices": [],
                        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": total,
                                  "total_tokens": prompt_tokens + total}})
        await emit("[DONE]")
        return False

    async def _stream_ollama(self, payload: dict[str, Any], beh: MockBehavior, w: asyncio.StreamWriter) -> bool:
        n_tokens, finish = self._plan_tokens(payload, beh)
        w.write(b"HTTP/1.1 200 OK\r\nContent-Type: application/x-ndjson\r\nConnection: close\r\n\r\n")
        await w.drain()
        rng = random.Random(beh.fragment_seed)
        await asyncio.sleep(beh.ttft_s)
        for i in range(n_tokens):
            line = json.dumps({"model": "mock", "message": {"role": "assistant",
                               "content": beh.words[i % len(beh.words)]}, "done": False}, ensure_ascii=False)
            await _write_fragmented(w, (line + "\n").encode(), beh.fragment, rng)
            await asyncio.sleep(beh.token_interval_s)
        p = _approx_prompt_tokens(payload)
        done = {"model": "mock", "message": {"role": "assistant", "content": ""}, "done": True,
                "done_reason": finish, "prompt_eval_count": p, "prompt_eval_duration": int(beh.ttft_s * 1e9),
                "eval_count": n_tokens, "eval_duration": int(beh.token_interval_s * n_tokens * 1e9),
                "load_duration": 0, "total_duration": 0}
        await _write_fragmented(w, (json.dumps(done) + "\n").encode(), beh.fragment, rng)
        return False


def _approx_prompt_tokens(payload: dict[str, Any]) -> int:
    text = " ".join(str(m.get("content", "")) for m in payload.get("messages", []))
    return max(1, len(text.split()))


async def _write_fragmented(w: asyncio.StreamWriter, raw: bytes, mode: str, rng: random.Random) -> None:
    if mode == "none":
        w.write(raw)
        await w.drain()
        return
    i = 0
    while i < len(raw):
        step = 1 if mode == "bytes1" else rng.randint(1, 7)
        w.write(raw[i : i + step])
        await w.drain()
        i += step
        await asyncio.sleep(0)


async def _read_request(r: asyncio.StreamReader) -> tuple[str, str, dict[str, str], bytes] | None:
    try:
        head = await r.readuntil(b"\r\n\r\n")
    except (asyncio.IncompleteReadError, ConnectionError):
        return None
    lines = head.decode("latin-1").split("\r\n")
    method, path, _ = lines[0].split(" ", 2)
    headers = {}
    for ln in lines[1:]:
        if ":" in ln:
            k, v = ln.split(":", 1)
            headers[k.strip().lower()] = v.strip()
    n = int(headers.get("content-length", "0") or 0)
    body = await r.readexactly(n) if n else b""
    return method, path, headers, body


async def _send_json(w: asyncio.StreamWriter, status: int, obj: Any) -> None:
    body = json.dumps(obj).encode()
    reason = {200: "OK", 404: "Not Found", 500: "Internal Server Error", 503: "Service Unavailable"}.get(status, "X")
    w.write(f"HTTP/1.1 {status} {reason}\r\nContent-Type: application/json\r\n"
            f"Content-Length: {len(body)}\r\n\r\n".encode() + body)
    await w.drain()


def run(host: str, port: int) -> None:
    async def main() -> None:
        srv = MockServer(host, port)
        await srv.start()
        print(f"SYNTHETIC mock server listening on {srv.base_url}", flush=True)
        await srv.serve_forever()

    asyncio.run(main())

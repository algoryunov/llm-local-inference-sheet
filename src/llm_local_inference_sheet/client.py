"""Async HTTP benchmark client (closed-loop load generation).

Load model: *closed loop*. ``C`` workers each keep exactly one request in flight.
When a request completes, the worker immediately dispatches the next request
from a shared FIFO queue. There is no think time and no arrival process. Offered
load adapts to service rate, which makes this a capacity/latency-at-concurrency
measurement, not a latency-under-given-arrival-rate measurement.

Warm-up requests run as a separate phase at the same concurrency, and all of them
complete before the measured phase begins, so warm-up and measured requests
never overlap.

Timing: ``time.perf_counter_ns()`` (monotonic). ``dispatch`` is taken immediately
before the request is handed to httpx. Payload arrival time is when the
network read containing the payload's final byte returned to the client.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import httpx

from .streaming import NDJSONParser, SSEParser, normalize_payload

log = logging.getLogger(__name__)


@dataclass
class RequestSpec:
    request_id: str
    phase: str  # "warmup" | "measured" | "probe"
    item_id: str
    path: str
    protocol: str  # "openai" | "ollama"
    payload: dict[str, Any]
    target_output_tokens: int | None = None
    expected_input_tokens: int | None = None
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class RequestResult:
    record: dict[str, Any]
    events: list[list[Any]]


class Runner:
    def __init__(
        self,
        base_url: str,
        *,
        request_timeout_s: float = 600,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.request_timeout_s = request_timeout_s
        self.headers = {"Content-Type": "application/json", **(headers or {})}
        self._inflight = 0

    def _client(self, concurrency: int) -> httpx.AsyncClient:
        limits = httpx.Limits(max_connections=concurrency + 2, max_keepalive_connections=concurrency + 2)
        return httpx.AsyncClient(
            base_url=self.base_url,
            limits=limits,
            timeout=httpx.Timeout(None, connect=10.0),
            headers=self.headers,
        )

    async def one(self, client: httpx.AsyncClient, spec: RequestSpec, worker: int = 0) -> RequestResult:
        rec: dict[str, Any] = {
            "request_id": spec.request_id,
            "phase": spec.phase,
            "item_id": spec.item_id,
            "worker": worker,
            "protocol": spec.protocol,
            "target_output_tokens": spec.target_output_tokens,
            "expected_input_tokens": spec.expected_input_tokens,
            "meta": spec.meta,
        }
        events: list[list[Any]] = []
        body = json.dumps(spec.payload, ensure_ascii=False).encode()
        terminal_seen = False
        stream_error: str | None = None
        self._inflight += 1
        rec["inflight_at_dispatch"] = self._inflight
        t0 = time.perf_counter_ns()
        rec["dispatch_abs_ns"] = t0
        rec["dispatch_wall"] = time.time()
        try:
            async with asyncio.timeout(self.request_timeout_s):
                req = client.build_request("POST", spec.path, content=body)
                resp = await client.send(req, stream=True)
                try:
                    rec["headers_ns"] = time.perf_counter_ns() - t0
                    rec["http_status"] = resp.status_code
                    if resp.status_code != 200:
                        err = (await resp.aread())[:2000].decode("utf-8", "replace")
                        rec["status"] = "http_error"
                        rec["error"] = f"HTTP {resp.status_code}: {err}"
                    else:
                        parser: SSEParser | NDJSONParser = (
                            SSEParser() if spec.protocol == "openai" else NDJSONParser()
                        )
                        async for chunk in resp.aiter_raw():
                            t = time.perf_counter_ns() - t0
                            for item in parser.feed(chunk):
                                data = item if isinstance(item, str) else item.data
                                events.append([t, data])
                        t = time.perf_counter_ns() - t0
                        for item in parser.close():
                            data = item if isinstance(item, str) else item.data
                            events.append([t, data])
                finally:
                    await resp.aclose()
            rec["end_ns"] = time.perf_counter_ns() - t0
            if rec.get("status") is None:
                for _, payload in events:
                    ch = normalize_payload(spec.protocol, payload)
                    if ch is None:
                        continue
                    if ch.error and stream_error is None:
                        stream_error = ch.error
                    if ch.done:
                        terminal_seen = True
                if stream_error:
                    rec["status"] = "error"
                    rec["error"] = stream_error
                elif not terminal_seen:
                    rec["status"] = "error"
                    rec["error"] = "stream ended without terminal event ([DONE] / done:true)"
                else:
                    rec["status"] = "ok"
        except TimeoutError:
            rec["end_ns"] = time.perf_counter_ns() - t0
            rec["status"] = "timeout"
            rec["error"] = f"request exceeded {self.request_timeout_s}s"
        except asyncio.CancelledError:
            rec["end_ns"] = time.perf_counter_ns() - t0
            rec["status"] = "cancelled"
            rec["error"] = "cancelled (phase deadline or interrupt)"
            self._inflight -= 1
            raise _Cancelled(RequestResult(rec, events)) from None
        except httpx.HTTPError as e:
            rec["end_ns"] = time.perf_counter_ns() - t0
            rec["status"] = "error"
            rec["error"] = f"{type(e).__name__}: {e}"
        self._inflight -= 1
        return RequestResult(rec, events)

    async def run_phase(
        self,
        specs: Sequence[RequestSpec],
        concurrency: int,
        *,
        deadline_s: float | None = None,
        on_result: Callable[[RequestResult], None] | None = None,
    ) -> list[RequestResult]:
        queue = deque(specs)
        results: list[RequestResult] = []

        async def worker(i: int, client: httpx.AsyncClient) -> None:
            while queue:
                spec = queue.popleft()
                try:
                    res = await self.one(client, spec, worker=i)
                except _Cancelled as c:
                    results.append(c.result)
                    if on_result:
                        on_result(c.result)
                    raise asyncio.CancelledError from None
                results.append(res)
                if on_result:
                    on_result(res)

        async with self._client(concurrency) as client:
            tasks = [asyncio.create_task(worker(i, client)) for i in range(concurrency)]
            try:
                if deadline_s is None:
                    await asyncio.gather(*tasks)
                else:
                    done, pending = await asyncio.wait(tasks, timeout=deadline_s)
                    for t in pending:
                        t.cancel()
                    await asyncio.gather(*pending, return_exceptions=True)
                    for t in done:
                        if t.exception():
                            raise t.exception()  # type: ignore[misc]
            finally:
                for t in tasks:
                    if not t.done():
                        t.cancel()
        for spec in queue:  # never dispatched because of the deadline
            results.append(RequestResult(
                {"request_id": spec.request_id, "phase": spec.phase, "item_id": spec.item_id,
                 "protocol": spec.protocol, "status": "not_dispatched",
                 "error": "phase deadline reached before dispatch",
                 "target_output_tokens": spec.target_output_tokens, "meta": spec.meta},
                [],
            ))
        return results


class _Cancelled(Exception):
    def __init__(self, result: RequestResult) -> None:
        super().__init__("cancelled")
        self.result = result

"""Metric definitions. Everything here is recomputable from stored raw events.

Timestamps are nanoseconds from a monotonic clock (``time.perf_counter_ns``),
stored relative to the request's dispatch instant. See docs/methodology.md for
the precise definitions. In short:

* ``ttft_s``: dispatch → arrival of the first stream payload containing non-empty
  generated text (content *or* reasoning). Role-only, metadata-only and usage-only
  payloads are excluded. This is *client-observed first-content latency*, not
  server-side first-token generation time.
* ``e2e_s``: dispatch → stream end (``[DONE]`` sentinel / ``done:true`` / connection
  close, whichever the client observed last).
* ``tpot_s``: (t_last_text − t_first_text) / (output_tokens − 1). Labelled
  ``token_level`` only when the number of text-bearing payloads equals the
  server-reported output token count, meaning one token per payload. Otherwise it is
  labelled ``chunk_derived``. Null when output_tokens < 2 or unknown.
* Inter-token latency is reported only for ``token_level`` streams. Otherwise the
  same gaps are reported as *inter-chunk* gaps.
"""

from __future__ import annotations

import itertools
import math
import random
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from typing import Any

from .streaming import NormalizedChunk, normalize_payload

NS = 1_000_000_000


@dataclass
class RequestMetrics:
    status: str
    ttft_s: float | None = None
    ttft_content_s: float | None = None
    headers_s: float | None = None
    e2e_s: float | None = None
    first_text_ns: int | None = None
    last_text_ns: int | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    output_tokens_method: str | None = None
    cached_tokens: int | None = None
    text_events: int = 0
    total_events: int = 0
    token_timing: str | None = None  # token_level | chunk_derived | None
    tpot_s: float | None = None
    decode_tps: float | None = None
    gap_mean_s: float | None = None
    gap_p50_s: float | None = None
    gap_p95_s: float | None = None
    gap_max_s: float | None = None
    gap_kind: str | None = None  # itl | inter_chunk
    finish_reason: str | None = None
    early_stop: bool | None = None
    truncated: bool | None = None
    reasoning_chars: int = 0
    content_chars: int = 0
    prefill_proxy_tps: float | None = None
    native_prompt_tokens: int | None = None
    native_cached_tokens: int | None = None
    native_prefill_s: float | None = None
    native_prefill_tps: float | None = None
    native_decode_tokens: int | None = None
    native_decode_s: float | None = None
    native_decode_tps: float | None = None
    native_source: str | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _pct(sorted_vals: Sequence[float], q: float) -> float:
    """Linear-interpolated percentile (same as numpy 'linear' / Excel PERCENTILE.INC)."""
    if not sorted_vals:
        raise ValueError("empty")
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])
    pos = (len(sorted_vals) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return float(sorted_vals[lo])
    return float(sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (pos - lo))


def percentile(values: Iterable[float], q: float) -> float | None:
    vals = sorted(v for v in values if v is not None)
    return _pct(vals, q) if vals else None


def compute_request_metrics(
    record: dict[str, Any],
    events: Sequence[Sequence[Any]],
    *,
    fallback_output_tokens: int | None = None,
    fallback_method: str | None = None,
) -> RequestMetrics:
    """Compute metrics for one request from its record and raw event list.

    ``events`` is a list of ``[t_rel_ns, payload_str]``. ``record`` holds dispatch
    metadata (status, protocol, headers_ns/end_ns relative to dispatch, target tokens).
    """
    protocol = record.get("protocol", "openai")
    m = RequestMetrics(status=record.get("status", "unknown"), error=record.get("error"))
    if record.get("headers_ns") is not None:
        m.headers_s = record["headers_ns"] / NS
    if record.get("end_ns") is not None and m.status == "ok":
        m.e2e_s = record["end_ns"] / NS

    usage: dict[str, Any] | None = None
    native: dict[str, Any] | None = None
    text_times: list[int] = []
    first_content_ns: int | None = None
    for t_ns, payload in events:
        m.total_events += 1
        ch: NormalizedChunk | None = normalize_payload(protocol, payload)
        if ch is None:
            continue
        if ch.error and m.error is None:
            m.error = ch.error
        if ch.has_generated_text:
            text_times.append(int(t_ns))
            m.reasoning_chars += len(ch.reasoning)
            m.content_chars += len(ch.content)
            if ch.content and first_content_ns is None:
                first_content_ns = int(t_ns)
        if ch.finish_reason:
            m.finish_reason = ch.finish_reason
        if ch.usage:
            usage = ch.usage
        if ch.native_timings:
            native = ch.native_timings

    m.text_events = len(text_times)
    if text_times:
        m.first_text_ns = text_times[0]
        m.last_text_ns = text_times[-1]
        m.ttft_s = text_times[0] / NS
    if first_content_ns is not None:
        m.ttft_content_s = first_content_ns / NS

    if usage:
        m.input_tokens = usage.get("prompt_tokens")
        if usage.get("completion_tokens") is not None:
            m.output_tokens = int(usage["completion_tokens"])
            m.output_tokens_method = "server_usage"
        details = usage.get("prompt_tokens_details") or {}
        if details.get("cached_tokens") is not None:
            m.cached_tokens = int(details["cached_tokens"])
    if m.output_tokens is None and fallback_output_tokens is not None:
        m.output_tokens = fallback_output_tokens
        m.output_tokens_method = fallback_method or "client_retokenized_text"

    if native:
        _apply_native(m, native)

    # Token/timing correspondence
    if m.output_tokens is not None and m.text_events > 0:
        m.token_timing = "token_level" if m.output_tokens == m.text_events else "chunk_derived"
    if (
        m.output_tokens is not None
        and m.output_tokens >= 2
        and m.first_text_ns is not None
        and m.last_text_ns is not None
        and m.last_text_ns > m.first_text_ns
    ):
        span = (m.last_text_ns - m.first_text_ns) / NS
        m.tpot_s = span / (m.output_tokens - 1)
        m.decode_tps = (m.output_tokens - 1) / span
    if len(text_times) >= 2:
        gaps = sorted((b - a) / NS for a, b in itertools.pairwise(text_times))
        m.gap_kind = "itl" if m.token_timing == "token_level" else "inter_chunk"
        m.gap_mean_s = sum(gaps) / len(gaps)
        m.gap_p50_s = _pct(gaps, 0.5)
        m.gap_p95_s = _pct(gaps, 0.95)
        m.gap_max_s = gaps[-1]

    target = record.get("target_output_tokens")
    if m.finish_reason is not None:
        m.truncated = m.finish_reason == "length"
        if target is not None and m.output_tokens is not None:
            m.early_stop = m.finish_reason != "length" and m.output_tokens < target
    if m.ttft_s and m.input_tokens:
        m.prefill_proxy_tps = m.input_tokens / m.ttft_s
    return m


def _apply_native(m: RequestMetrics, native: dict[str, Any]) -> None:
    src = native.get("source")
    m.native_source = src
    if src == "llama.cpp":
        # llama-server timings: prompt_n/prompt_ms/predicted_n/predicted_ms/cache_n
        pn, pms = native.get("prompt_n"), native.get("prompt_ms")
        dn, dms = native.get("predicted_n"), native.get("predicted_ms")
        m.native_prompt_tokens = pn
        m.native_cached_tokens = native.get("cache_n")
        if pn is not None and pms:
            m.native_prefill_s = pms / 1000
            m.native_prefill_tps = pn / (pms / 1000) if pms > 0 else None
        if dn is not None and dms:
            m.native_decode_tokens = dn
            m.native_decode_s = dms / 1000
            m.native_decode_tps = dn / (dms / 1000) if dms > 0 else None
    elif src == "ollama":
        pn, pd = native.get("prompt_eval_count"), native.get("prompt_eval_duration")
        dn, dd = native.get("eval_count"), native.get("eval_duration")
        m.native_prompt_tokens = pn
        if pn is not None and pd:
            m.native_prefill_s = pd / NS
            m.native_prefill_tps = pn / (pd / NS)
        if dn is not None and dd:
            m.native_decode_tokens = dn
            m.native_decode_s = dd / NS
            m.native_decode_tps = dn / (dd / NS)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

LATENCY_FIELDS = ("ttft_s", "e2e_s", "tpot_s", "decode_tps", "gap_p95_s", "prefill_proxy_tps",
                  "native_prefill_tps", "native_decode_tps")


def bootstrap_ci(
    values: Sequence[float], stat: str = "median", n_boot: int = 1000, seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float] | None:
    """Percentile bootstrap CI (deterministic seed). Returns None for n < 10."""
    vals = [v for v in values if v is not None]
    if len(vals) < 10:
        return None
    rng = random.Random(seed)
    n = len(vals)
    stats = []
    for _ in range(n_boot):
        s = sorted(vals[rng.randrange(n)] for _ in range(n))
        stats.append(_pct(s, 0.5) if stat == "median" else sum(s) / n)
    stats.sort()
    return _pct(stats, alpha / 2), _pct(stats, 1 - alpha / 2)


def summarize_field(
    values: Sequence[float | None], *, min_n_p95: int, min_n_p99: int
) -> dict[str, Any]:
    vals = sorted(v for v in values if v is not None)
    n = len(vals)
    if n == 0:
        return {"n": 0}
    mean = sum(vals) / n
    std = math.sqrt(sum((v - mean) ** 2 for v in vals) / (n - 1)) if n > 1 else 0.0
    out: dict[str, Any] = {
        "n": n,
        "mean": mean,
        "std": std,
        "min": vals[0],
        "p50": _pct(vals, 0.5),
        "p90": _pct(vals, 0.9),
        "max": vals[-1],
        "p95": _pct(vals, 0.95),
        "p95_reportable": n >= min_n_p95,
        "p99": _pct(vals, 0.99) if n >= min_n_p99 else None,
    }
    ci = bootstrap_ci(vals, "median")
    out["p50_ci95"] = list(ci) if ci else None
    return out


def aggregate_cell(
    records: Sequence[dict[str, Any]],
    metrics: Sequence[RequestMetrics],
    *,
    min_n_p95: int = 200,
    min_n_p99: int = 1000,
) -> dict[str, Any]:
    """Aggregate measured-phase requests of one cell.

    Aggregate throughput window: from the earliest measured dispatch to the latest
    measured stream end (absolute monotonic ns stored in the record as
    ``dispatch_abs_ns``). This includes the closed-loop ramp-down tail, and
    ``mean_inflight`` makes that visible.
    """
    pairs = [(r, m) for r, m in zip(records, metrics, strict=True) if r.get("phase") == "measured"]
    statuses: dict[str, int] = {}
    for _, m in pairs:
        statuses[m.status] = statuses.get(m.status, 0) + 1
    ok = [(r, m) for r, m in pairs if m.status == "ok"]
    out: dict[str, Any] = {
        "n_measured": len(pairs),
        "n_ok": len(ok),
        "status_counts": statuses,
    }
    if not ok:
        return out
    dispatched = [r for r, _ in pairs if r.get("dispatch_abs_ns") is not None]
    starts = [r["dispatch_abs_ns"] for r in dispatched]
    ends = [r["dispatch_abs_ns"] + (r.get("end_ns") or 0) for r in dispatched]
    window_s = (max(ends) - min(starts)) / NS
    total_out = sum(m.output_tokens or 0 for _, m in ok)
    busy = sum(m.e2e_s or 0 for _, m in ok)
    out.update(
        {
            "window_s": window_s,
            "total_output_tokens": total_out,
            "aggregate_output_tps": total_out / window_s if window_s > 0 else None,
            "successful_rps": len(ok) / window_s if window_s > 0 else None,
            "mean_inflight": busy / window_s if window_s > 0 else None,
            "early_stops": sum(1 for _, m in ok if m.early_stop),
            "truncated": sum(1 for _, m in ok if m.truncated),
            "output_tokens_methods": sorted({m.output_tokens_method or "unknown" for _, m in ok}),
            "token_timing": sorted({m.token_timing or "unknown" for _, m in ok}),
            "input_tokens": summarize_field([m.input_tokens for _, m in ok], min_n_p95=10**9, min_n_p99=10**9),
            "output_tokens": summarize_field([m.output_tokens for _, m in ok], min_n_p95=10**9, min_n_p99=10**9),
            "cached_tokens_max": max(
                [c for c in (m.cached_tokens if m.cached_tokens is not None else m.native_cached_tokens for _, m in ok) if c is not None],
                default=None,
            ),
            "reasoning_chars_total": sum(m.reasoning_chars for _, m in ok),
        }
    )
    for f in LATENCY_FIELDS:
        out[f] = summarize_field([getattr(m, f) for _, m in ok], min_n_p95=min_n_p95, min_n_p99=min_n_p99)
    # Full-length subset: requests that produced the target token count, reported separately.
    full = [m for r, m in ok if r.get("target_output_tokens") and m.output_tokens == r["target_output_tokens"]]
    out["n_full_length"] = len(full)
    if full and len(full) != len(ok):
        out["full_length_only"] = {
            f: summarize_field([getattr(m, f) for m in full], min_n_p95=min_n_p95, min_n_p99=min_n_p99)
            for f in ("ttft_s", "e2e_s", "tpot_s")
        }
    return out

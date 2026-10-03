"""Harness tests against the deterministic SYNTHETIC mock server (no models, CPU only)."""

import asyncio

import pytest

from llm_local_inference_sheet.client import RequestSpec, Runner
from llm_local_inference_sheet.evaluation import response_text
from llm_local_inference_sheet.metrics import compute_request_metrics
from llm_local_inference_sheet.mockserver import WORDS, MockBehavior, MockServer


@pytest.fixture
async def server():
    s = MockServer()
    await s.start()
    yield s
    await s.stop()


def spec(i=0, max_tokens=8, mock=None, protocol="openai", ignore_eos=False, phase="measured"):
    payload = {"model": "m", "messages": [{"role": "user", "content": "hello there"}], "max_tokens": max_tokens,
               "stream": True, "stream_options": {"include_usage": True}}
    path = "/v1/chat/completions"
    if protocol == "ollama":
        payload = {"model": "m", "messages": payload["messages"], "stream": True, "options": {"num_predict": max_tokens}}
        path = "/api/chat"
    if mock:
        payload["mock"] = mock
    if ignore_eos:
        payload["ignore_eos"] = True
    return RequestSpec(request_id=f"r{i}", phase=phase, item_id=str(i), path=path, protocol=protocol,
                       payload=payload, target_output_tokens=max_tokens)


def expected_text(n):
    return "".join(WORDS[i % len(WORDS)] for i in range(n))


async def one(server, s, timeout=10):
    r = Runner(server.base_url, request_timeout_s=timeout)
    return (await r.run_phase([s], 1))[0]


@pytest.mark.parametrize("fragment", ["none", "bytes1", "random"])
@pytest.mark.parametrize("line_ending", ["\n", "\r\n"])
async def test_fragmented_unicode_stream_reassembles(server, fragment, line_ending):
    res = await one(server, spec(max_tokens=12, mock={"fragment": fragment, "line_ending": line_ending,
                                                      "keepalive_comments": True, "fragment_seed": 3}))
    assert res.record["status"] == "ok"
    content, _ = response_text("openai", res.events)
    assert content == expected_text(12)
    m = compute_request_metrics(res.record, res.events)
    assert m.output_tokens == 12 and m.token_timing == "token_level"
    assert m.ttft_s is not None and m.ttft_s >= 0.009


async def test_multiple_tokens_per_chunk(server):
    res = await one(server, spec(max_tokens=9, mock={"tokens_per_chunk": 3}))
    m = compute_request_metrics(res.record, res.events)
    assert m.text_events == 3 and m.output_tokens == 9 and m.token_timing == "chunk_derived"


async def test_missing_usage(server):
    res = await one(server, spec(max_tokens=5, mock={"include_usage": False}))
    m = compute_request_metrics(res.record, res.events)
    assert res.record["status"] == "ok" and m.output_tokens is None and m.tpot_s is None


async def test_early_eos_and_ignore_eos(server):
    res = await one(server, spec(max_tokens=10, mock={"eos_after": 4}))
    m = compute_request_metrics(res.record, res.events)
    assert m.output_tokens == 4 and m.early_stop and m.finish_reason == "stop"
    res2 = await one(server, spec(max_tokens=10, mock={"eos_after": 4}, ignore_eos=True))
    m2 = compute_request_metrics(res2.record, res2.events)
    assert m2.output_tokens == 10 and m2.truncated and not m2.early_stop


async def test_http_error(server):
    res = await one(server, spec(mock={"http_status": 500}))
    assert res.record["status"] == "http_error" and "500" in res.record["error"]


async def test_mid_stream_error(server):
    res = await one(server, spec(max_tokens=10, mock={"error_after_tokens": 3}))
    assert res.record["status"] == "error" and "mid-stream" in res.record["error"]


async def test_dropped_connection(server):
    res = await one(server, spec(max_tokens=10, mock={"drop_connection_after_tokens": 2}))
    assert res.record["status"] == "error"


async def test_timeout(server):
    res = await one(server, spec(max_tokens=10, mock={"hang_after_tokens": 2}), timeout=0.5)
    assert res.record["status"] == "timeout"
    assert res.record["end_ns"] >= 0.5e9


async def test_phase_deadline_cancels_and_marks_undispatched(server):
    r = Runner(server.base_url, request_timeout_s=30)
    specs = [spec(i, max_tokens=5, mock={"hang_after_tokens": 1}) for i in range(4)]
    results = await r.run_phase(specs, 2, deadline_s=0.5)
    statuses = sorted(x.record["status"] for x in results)
    assert statuses == ["cancelled", "cancelled", "not_dispatched", "not_dispatched"]


async def test_reasoning_then_content(server):
    res = await one(server, spec(max_tokens=3, mock={"reasoning_tokens": 2}))
    content, reasoning = response_text("openai", res.events)
    assert reasoning == expected_text(2) and content == WORDS[2] + WORDS[3] + WORDS[4]
    m = compute_request_metrics(res.record, res.events)
    assert m.ttft_content_s > m.ttft_s


async def test_ollama_ndjson(server):
    res = await one(server, spec(max_tokens=6, protocol="ollama", mock={"fragment": "random"}))
    assert res.record["status"] == "ok"
    m = compute_request_metrics(res.record, res.events)
    assert m.output_tokens == 6 and m.native_source == "ollama" and m.native_decode_tps is not None


async def test_closed_loop_never_exceeds_concurrency(server):
    r = Runner(server.base_url, request_timeout_s=30)
    specs = [spec(i, max_tokens=4, mock={"token_interval_s": 0.01}) for i in range(12)]
    results = await r.run_phase(specs, 3)
    assert len(results) == 12 and all(x.record["status"] == "ok" for x in results)
    assert max(x.record["inflight_at_dispatch"] for x in results) <= 3
    assert {x.record["worker"] for x in results} == {0, 1, 2}


async def test_monotonic_event_times(server):
    res = await one(server, spec(max_tokens=20))
    ts = [t for t, _ in res.events]
    assert ts == sorted(ts) and res.record["headers_ns"] <= ts[0] <= res.record["end_ns"]


def test_mock_behavior_merge():
    b = MockBehavior.from_dict({"tokens_per_chunk": 2, "unknown": 1}, MockBehavior())
    assert b.tokens_per_chunk == 2


async def test_cancellation_propagates(server):
    r = Runner(server.base_url, request_timeout_s=30)
    task = asyncio.create_task(r.run_phase([spec(0, mock={"hang_after_tokens": 1})], 1))
    await asyncio.sleep(0.3)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

import json

import pytest

from llm_local_inference_sheet.metrics import (
    NS,
    aggregate_cell,
    compute_request_metrics,
    percentile,
    summarize_field,
)

MS = 1_000_000


def ev(t_ms, content=None, reasoning=None, role=None, finish=None, usage=None, timings=None):
    delta = {}
    if role:
        delta["role"] = role
    if content is not None:
        delta["content"] = content
    if reasoning is not None:
        delta["reasoning_content"] = reasoning
    obj = {"choices": [{"delta": delta, "finish_reason": finish}] if (delta or finish) else []}
    if usage:
        obj["usage"] = usage
    if timings:
        obj["timings"] = timings
    return [int(t_ms * MS), json.dumps(obj)]


def rec(**kw):
    base = {"status": "ok", "protocol": "openai", "headers_ns": 5 * MS, "end_ns": 1000 * MS,
            "phase": "measured", "dispatch_abs_ns": 0, "target_output_tokens": 4}
    base.update(kw)
    return base


def test_ttft_excludes_role_and_usage_events():
    events = [ev(50, role="assistant", content=""), ev(100, content="a"), ev(150, content="b"),
              ev(200, content="c"), ev(250, content="d"), ev(260, finish="length"),
              ev(270, usage={"prompt_tokens": 10, "completion_tokens": 4}), [280 * MS, "[DONE]"]]
    m = compute_request_metrics(rec(), events)
    assert m.ttft_s == pytest.approx(0.100)
    assert m.output_tokens == 4 and m.output_tokens_method == "server_usage"
    assert m.token_timing == "token_level"
    assert m.tpot_s == pytest.approx((0.250 - 0.100) / 3)
    assert m.decode_tps == pytest.approx(3 / 0.150)
    assert m.gap_kind == "itl"
    assert m.e2e_s == pytest.approx(1.0)
    assert m.truncated and not m.early_stop
    assert m.prefill_proxy_tps == pytest.approx(10 / 0.1)


def test_multiple_tokens_per_chunk_is_chunk_derived():
    events = [ev(100, content="ab"), ev(200, content="cd"), ev(210, finish="length"),
              ev(211, usage={"prompt_tokens": 3, "completion_tokens": 4})]
    m = compute_request_metrics(rec(), events)
    assert m.text_events == 2
    assert m.token_timing == "chunk_derived"
    assert m.gap_kind == "inter_chunk"
    # Still the conventional estimate, but explicitly labelled as chunk-derived.
    assert m.tpot_s == pytest.approx(0.1 / 3)


def test_missing_usage_gives_null_token_metrics():
    events = [ev(100, content="a"), ev(200, content="b"), ev(210, finish="stop")]
    m = compute_request_metrics(rec(), events)
    assert m.output_tokens is None and m.tpot_s is None and m.decode_tps is None and m.token_timing is None
    assert m.ttft_s == pytest.approx(0.1)
    # Inter-chunk gaps are still available, never relabelled as ITL.
    assert m.gap_kind == "inter_chunk"


def test_fallback_token_count_is_labelled():
    events = [ev(100, content="a"), ev(200, content="b")]
    m = compute_request_metrics(rec(), events, fallback_output_tokens=2, fallback_method="client_retokenized_text")
    assert m.output_tokens_method == "client_retokenized_text"


def test_early_eos_detected():
    events = [ev(100, content="a"), ev(150, content="b"), ev(160, finish="stop"),
              ev(161, usage={"prompt_tokens": 3, "completion_tokens": 2})]
    m = compute_request_metrics(rec(target_output_tokens=256), events)
    assert m.early_stop is True and m.truncated is False


def test_single_token_output_has_no_tpot():
    events = [ev(100, content="a"), ev(110, finish="length"), ev(111, usage={"prompt_tokens": 3, "completion_tokens": 1})]
    m = compute_request_metrics(rec(target_output_tokens=1), events)
    assert m.tpot_s is None and m.ttft_s == pytest.approx(0.1)


def test_reasoning_counts_for_ttft_but_not_content_ttft():
    events = [ev(100, reasoning="hmm"), ev(300, content="x"), ev(310, usage={"prompt_tokens": 1, "completion_tokens": 2})]
    m = compute_request_metrics(rec(), events)
    assert m.ttft_s == pytest.approx(0.1) and m.ttft_content_s == pytest.approx(0.3)
    assert m.reasoning_chars == 3


def test_error_status_has_no_e2e():
    m = compute_request_metrics(rec(status="timeout"), [ev(100, content="a")])
    assert m.e2e_s is None and m.status == "timeout"


def test_native_llama_and_ollama_timings():
    t = {"prompt_n": 100, "prompt_ms": 50.0, "predicted_n": 20, "predicted_ms": 400.0, "cache_n": 0}
    m = compute_request_metrics(rec(), [ev(100, content="a"), ev(500, finish="length", timings=t)])
    assert m.native_prefill_tps == pytest.approx(2000) and m.native_decode_tps == pytest.approx(50)
    o = [[100 * MS, json.dumps({"message": {"content": "a"}, "done": False})],
         [200 * MS, json.dumps({"done": True, "done_reason": "stop", "prompt_eval_count": 10,
                                "prompt_eval_duration": NS // 10, "eval_count": 5, "eval_duration": NS // 2})]]
    mo = compute_request_metrics(rec(protocol="ollama"), o)
    assert mo.native_prefill_tps == pytest.approx(100) and mo.native_decode_tps == pytest.approx(10)
    assert mo.output_tokens == 5


def test_percentile_linear():
    assert percentile([1, 2, 3, 4], 0.5) == 2.5
    assert percentile([10], 0.95) == 10
    assert percentile([], 0.5) is None
    assert percentile(range(1, 101), 0.95) == pytest.approx(95.05)


def test_summarize_field_reportability_flags():
    s = summarize_field([float(i) for i in range(30)], min_n_p95=200, min_n_p99=1000)
    assert s["n"] == 30 and s["p95_reportable"] is False and s["p99"] is None
    assert s["p50_ci95"] is not None and s["p50_ci95"][0] <= s["p50"] <= s["p50_ci95"][1]


def test_aggregate_window_and_throughput():
    records, metrics = [], []
    # two concurrent requests, 1 s each, overlapping fully; 10 tokens each
    for i in range(2):
        r = rec(dispatch_abs_ns=0, end_ns=NS, request_id=f"r{i}", target_output_tokens=10)
        evs = [ev(100, content="x")] + [ev(100 + 10 * k, content="x") for k in range(1, 10)] + \
              [ev(900, usage={"prompt_tokens": 5, "completion_tokens": 10})]
        records.append(r)
        metrics.append(compute_request_metrics(r, evs))
    records.append(rec(phase="warmup", dispatch_abs_ns=-5 * NS, end_ns=NS))
    metrics.append(compute_request_metrics(records[-1], []))
    agg = aggregate_cell(records, metrics)
    assert agg["n_measured"] == 2 and agg["n_ok"] == 2
    assert agg["window_s"] == pytest.approx(1.0)
    assert agg["aggregate_output_tps"] == pytest.approx(20.0)
    assert agg["mean_inflight"] == pytest.approx(2.0)
    assert agg["n_full_length"] == 2

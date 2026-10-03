import json

import pytest

from llm_local_inference_sheet.streaming import (
    NDJSONParser,
    SSEParser,
    normalize_ollama_native,
    normalize_openai_chunk,
    normalize_payload,
)


def _chunk(content=None, reasoning=None, role=None, finish=None):
    delta = {}
    if role:
        delta["role"] = role
    if content is not None:
        delta["content"] = content
    if reasoning is not None:
        delta["reasoning_content"] = reasoning
    return {"choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}


STREAM = (
    ": keepalive\n\n"
    f"data: {json.dumps(_chunk(role='assistant', content=''))}\n\n"
    f"data: {json.dumps(_chunk(content='Γειά'), ensure_ascii=False)}\n\n"
    f"data: {json.dumps(_chunk(content=' 🙂 世界'), ensure_ascii=False)}\n\n"
    f"data: {json.dumps(_chunk(finish='stop'))}\n\n"
    'data: {"choices":[],"usage":{"prompt_tokens":5,"completion_tokens":3}}\n\n'
    "data: [DONE]\n\n"
).encode()


def _parse_all(raw: bytes, splits: list[int]) -> list[str]:
    p = SSEParser()
    out = []
    prev = 0
    for s in [*splits, len(raw)]:
        out += [e.data for e in p.feed(raw[prev:s])]
        prev = s
    out += [e.data for e in p.close()]
    return out


def test_sse_whole_stream():
    datas = _parse_all(STREAM, [])
    assert len(datas) == 6
    assert datas[-1] == "[DONE]"


@pytest.mark.parametrize("line_ending", ["\n", "\r\n", "\r"])
def test_sse_every_split_point_and_line_endings(line_ending):
    raw = STREAM.decode().replace("\n", line_ending).encode()
    expected = _parse_all(raw, [])
    assert len(expected) == 6
    # Split at every single byte boundary, including inside multi-byte UTF-8 and CRLF pairs.
    for i in range(1, len(raw)):
        assert _parse_all(raw, [i]) == expected, f"split at {i}"


def test_sse_one_byte_at_a_time_preserves_unicode():
    raw = STREAM
    p = SSEParser()
    datas = []
    for i in range(len(raw)):
        datas += [e.data for e in p.feed(raw[i : i + 1])]
    text = "".join(normalize_payload("openai", d).content for d in datas if d != "[DONE]")
    assert text == "Γειά 🙂 世界"


def test_sse_multiline_data_and_fields():
    p = SSEParser()
    evs = p.feed(b"event: message\nid: 7\ndata: line1\ndata: line2\n\n")
    assert len(evs) == 1
    assert evs[0].data == "line1\nline2" and evs[0].event == "message" and evs[0].id == "7"


def test_sse_final_event_without_blank_line_is_flushed():
    p = SSEParser()
    assert p.feed(b"data: [DONE]") == []
    assert [e.data for e in p.close()] == ["[DONE]"]


def test_ndjson_fragmented():
    lines = [json.dumps({"message": {"content": "é"}, "done": False}, ensure_ascii=False),
             json.dumps({"done": True, "eval_count": 1})]
    raw = ("\n".join(lines) + "\n").encode()
    for i in range(1, len(raw)):
        p = NDJSONParser()
        got = p.feed(raw[:i]) + p.feed(raw[i:]) + p.close()
        assert got == lines


def test_normalize_role_only_and_usage_only_are_not_text():
    assert not normalize_openai_chunk(_chunk(role="assistant", content="")).has_generated_text
    u = normalize_openai_chunk({"choices": [], "usage": {"prompt_tokens": 1, "completion_tokens": 2}})
    assert not u.has_generated_text and u.usage["completion_tokens"] == 2


def test_normalize_reasoning_spellings():
    a = normalize_openai_chunk({"choices": [{"delta": {"reasoning_content": "think"}}]})
    b = normalize_openai_chunk({"choices": [{"delta": {"reasoning": "think"}}]})
    assert a.reasoning == b.reasoning == "think"
    assert a.content == ""


def test_normalize_error_and_done_and_garbage():
    assert normalize_openai_chunk({"error": {"message": "boom"}}).error == "boom"
    assert normalize_payload("openai", "[DONE]").done
    assert normalize_payload("openai", "{not json").error


def test_normalize_llama_timings():
    c = normalize_openai_chunk({"choices": [{"delta": {}, "finish_reason": "length"}],
                                "timings": {"prompt_n": 10, "prompt_ms": 20.0}})
    assert c.native_timings["source"] == "llama.cpp"
    assert c.finish_reason == "length"


def test_normalize_ollama_chat_and_generate():
    c = normalize_ollama_native({"message": {"role": "assistant", "content": "hi", "thinking": "hm"}, "done": False})
    assert c.content == "hi" and c.reasoning == "hm" and not c.done
    g = normalize_ollama_native({"response": "yo", "done": False})
    assert g.content == "yo"
    d = normalize_ollama_native({"done": True, "done_reason": "length", "prompt_eval_count": 7, "eval_count": 3,
                                 "prompt_eval_duration": 1_000_000, "eval_duration": 2_000_000})
    assert d.done and d.finish_reason == "length" and d.usage == {"prompt_tokens": 7, "completion_tokens": 3}
    assert d.native_timings["source"] == "ollama"

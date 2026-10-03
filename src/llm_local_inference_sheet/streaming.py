"""Incremental stream parsing and chunk normalization.

Two wire formats are supported:

* Server-Sent Events (OpenAI-compatible ``/v1/chat/completions`` with ``stream: true``),
  used by llama-server and mlx_lm.server.
* Newline-delimited JSON (Ollama's native ``/api/chat``), which is the only Ollama
  endpoint that returns server-side timings (prompt_eval_duration, eval_duration).

The parsers are byte-oriented and incremental. Network chunks can split anywhere,
including inside a multi-byte UTF-8 sequence, inside a ``\\r\\n`` pair, or inside
a JSON payload. They never assume one network read equals one event, and
normalization never assumes one event equals one token.
"""

from __future__ import annotations

import codecs
import json
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "NDJSONParser",
    "NormalizedChunk",
    "SSEEvent",
    "SSEParser",
    "normalize_ollama_native",
    "normalize_openai_chunk",
    "normalize_payload",
]


@dataclass(frozen=True)
class SSEEvent:
    data: str
    event: str | None = None
    id: str | None = None


class SSEParser:
    """WHATWG-style SSE parser (the subset relevant to LLM streaming).

    - Accepts ``\\n``, ``\\r\\n`` and ``\\r`` line terminators, even when split across feeds.
    - Joins multi-line ``data:`` fields with ``\\n``.
    - Ignores comment lines (``:keepalive``) and unknown fields.
    - Decodes UTF-8 incrementally, so a code point split across reads is preserved.
    """

    def __init__(self) -> None:
        self._decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        self._buf = ""
        self._pending_cr = False
        self._data: list[str] = []
        self._event: str | None = None
        self._id: str | None = None

    def feed(self, chunk: bytes) -> list[SSEEvent]:
        text = self._decoder.decode(chunk)
        return self._consume(text)

    def close(self) -> list[SSEEvent]:
        """Flush at end of stream. A final event without a trailing blank line is emitted."""
        text = self._decoder.decode(b"", final=True)
        events = self._consume(text)
        if self._buf:
            events.extend(self._line(self._buf))
            self._buf = ""
        if self._data:
            events.append(self._dispatch())
        return events

    def _consume(self, text: str) -> list[SSEEvent]:
        events: list[SSEEvent] = []
        if not text:
            return events
        if self._pending_cr:
            # Previous feed ended in '\r'; a leading '\n' completes the CRLF pair.
            self._pending_cr = False
            if text.startswith("\n"):
                text = text[1:]
        buf = self._buf + text
        start = 0
        i = 0
        n = len(buf)
        while i < n:
            c = buf[i]
            if c == "\n" or c == "\r":
                line = buf[start:i]
                if c == "\r":
                    if i + 1 < n:
                        if buf[i + 1] == "\n":
                            i += 1
                    else:
                        self._pending_cr = True
                events.extend(self._line(line))
                start = i + 1
            i += 1
        self._buf = buf[start:]
        return events

    def _line(self, line: str) -> list[SSEEvent]:
        if line == "":
            return [self._dispatch()] if (self._data or self._event is not None) else []
        if line.startswith(":"):
            return []
        name, sep, value = line.partition(":")
        if sep and value.startswith(" "):
            value = value[1:]
        if name == "data":
            self._data.append(value)
        elif name == "event":
            self._event = value
        elif name == "id":
            self._id = value
        return []

    def _dispatch(self) -> SSEEvent:
        ev = SSEEvent(data="\n".join(self._data), event=self._event, id=self._id)
        self._data = []
        self._event = None
        return ev


class NDJSONParser:
    """Incremental newline-delimited JSON line splitter (Ollama native API)."""

    def __init__(self) -> None:
        self._decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        self._buf = ""

    def feed(self, chunk: bytes) -> list[str]:
        self._buf += self._decoder.decode(chunk)
        *lines, self._buf = self._buf.split("\n")
        return [ln.strip() for ln in lines if ln.strip()]

    def close(self) -> list[str]:
        self._buf += self._decoder.decode(b"", final=True)
        rest, self._buf = self._buf.strip(), ""
        return [rest] if rest else []


@dataclass
class NormalizedChunk:
    """Backend-independent view of one stream payload.

    ``content`` and ``reasoning`` are *text*, which may contain zero, one or many
    tokens. Metrics must never count chunks as tokens.
    """

    content: str = ""
    reasoning: str = ""
    role: str | None = None
    finish_reason: str | None = None
    usage: dict[str, Any] | None = None
    native_timings: dict[str, Any] | None = None
    done: bool = False
    error: str | None = None
    tool_calls: bool = False
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def has_generated_text(self) -> bool:
        return bool(self.content) or bool(self.reasoning)


def normalize_openai_chunk(obj: dict[str, Any]) -> NormalizedChunk:
    """Normalize an OpenAI-style ``chat.completion.chunk`` (or text-completion chunk).

    Handles the reasoning field spellings: ``reasoning_content`` (llama.cpp, older vLLM)
    and ``reasoning`` (mlx-lm, newer vLLM, Ollama OpenAI-compat). Also handles
    llama-server's ``timings`` object, usage-only chunks with empty ``choices``,
    and error objects.
    """
    out = NormalizedChunk()
    if obj.get("error"):
        err = obj["error"]
        out.error = err.get("message", json.dumps(err)) if isinstance(err, dict) else str(err)
        return out
    choices = obj.get("choices") or []
    for ch in choices:
        delta = ch.get("delta")
        if delta is None:
            # text completion chunk or non-delta message
            delta = ch.get("message") or {}
            if "text" in ch and isinstance(ch["text"], str):
                out.content += ch["text"]
        if isinstance(delta, dict):
            if delta.get("role"):
                out.role = delta["role"]
            c = delta.get("content")
            if isinstance(c, str):
                out.content += c
            r = delta.get("reasoning_content")
            if r is None:
                r = delta.get("reasoning")
            if isinstance(r, str):
                out.reasoning += r
            if delta.get("tool_calls"):
                out.tool_calls = True
        if ch.get("finish_reason"):
            out.finish_reason = ch["finish_reason"]
    if obj.get("usage"):
        out.usage = obj["usage"]
    if obj.get("timings"):
        out.native_timings = {"source": "llama.cpp", **obj["timings"]}
    return out


_OLLAMA_TIMING_KEYS = (
    "total_duration",
    "load_duration",
    "prompt_eval_count",
    "prompt_eval_duration",
    "eval_count",
    "eval_duration",
)


def normalize_ollama_native(obj: dict[str, Any]) -> NormalizedChunk:
    """Normalize one Ollama ``/api/chat`` NDJSON object."""
    out = NormalizedChunk()
    if obj.get("error"):
        out.error = str(obj["error"])
        return out
    msg = obj.get("message") or {}
    if msg.get("role"):
        out.role = msg["role"]
    if isinstance(msg.get("content"), str):
        out.content = msg["content"]
    if isinstance(msg.get("thinking"), str):
        out.reasoning = msg["thinking"]
    if msg.get("tool_calls"):
        out.tool_calls = True
    # /api/generate objects carry text at the top level
    if isinstance(obj.get("response"), str):
        out.content += obj["response"]
    if isinstance(obj.get("thinking"), str):
        out.reasoning += obj["thinking"]
    if obj.get("done"):
        out.done = True
        out.finish_reason = obj.get("done_reason") or "stop"
        timings = {k: obj[k] for k in _OLLAMA_TIMING_KEYS if k in obj}
        if timings:
            out.native_timings = {"source": "ollama", **timings}
        if "prompt_eval_count" in obj or "eval_count" in obj:
            out.usage = {
                "prompt_tokens": obj.get("prompt_eval_count"),
                "completion_tokens": obj.get("eval_count"),
            }
    return out


def normalize_payload(protocol: str, payload: str) -> NormalizedChunk | None:
    """Parse and normalize one raw payload string as stored in the event log.

    Returns ``None`` for payloads that carry no information (e.g. unparsable
    keep-alives), a chunk with ``done=True`` for the OpenAI ``[DONE]`` sentinel.
    """
    if protocol == "openai":
        if payload.strip() == "[DONE]":
            return NormalizedChunk(done=True)
        try:
            obj = json.loads(payload)
        except json.JSONDecodeError:
            return NormalizedChunk(error=f"unparsable SSE payload: {payload[:200]!r}")
        if not isinstance(obj, dict):
            return None
        return normalize_openai_chunk(obj)
    if protocol == "ollama":
        try:
            obj = json.loads(payload)
        except json.JSONDecodeError:
            return NormalizedChunk(error=f"unparsable NDJSON line: {payload[:200]!r}")
        return normalize_ollama_native(obj) if isinstance(obj, dict) else None
    raise ValueError(f"unknown protocol {protocol!r}")

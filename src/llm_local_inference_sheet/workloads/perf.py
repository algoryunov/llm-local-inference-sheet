"""Performance workloads: prompts that hit an exact *rendered* input-token target.

``input_tokens`` counts everything the model prefills: chat-template tokens (role
markers, the empty ``<think></think>`` block Qwen3 inserts when thinking is disabled,
etc.) plus the user text. It is measured with each model's pinned tokenizer and
pinned chat template. The template overhead is recorded separately.

Every prompt starts with a unique request tag, so no two prompts share more than
the template's fixed prefix (a handful of tokens). That keeps the baseline free
of unintended prefix-cache reuse. Warm-up prompts are distinct from measured prompts.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..config import PROJECT_ROOT, ModelEntry
from ..models import download_tokenizer

# A neutral vocabulary; prompts are filler for timing and carry no task semantics.
_VOCAB = ["system", "memory", "latency", "cache", "model", "token", "server", "queue", "batch", "kernel", "device", "thread", "signal", "river", "mountain", "window", "garden", "music", "paper", "light", "stone", "cloud", "field", "market", "station", "bridge", "north", "south", "early", "late", "quiet", "bright", "simple", "careful", "rapid", "gentle", "steady", "local", "remote", "measure", "observe", "record", "compare", "report", "adjust", "verify", "improve", "describe", "explain"]

INSTRUCTION = "\n\nContinue the text above with a long, detailed passage."


@dataclass
class PerfItem:
    item_id: str
    phase: str
    messages: list[dict[str, str]]
    rendered_prompt: str
    input_tokens: int
    template_overhead_tokens: int
    target_output_tokens: int

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


class TemplateRenderer:
    """Renders prompts with the model's pinned HF tokenizer + chat template."""

    def __init__(self, model: ModelEntry, thinking: str = "disabled") -> None:
        from transformers import AutoTokenizer

        self.model = model
        self.dir = download_tokenizer(model)
        self.tok = AutoTokenizer.from_pretrained(str(self.dir))
        r = model.reasoning
        if thinking == "disabled":
            self.kwargs = dict(r.disable_chat_template_kwargs or {})
        else:
            self.kwargs = dict(r.enable_chat_template_kwargs or {})
        tpl = self.tok.chat_template if isinstance(self.tok.chat_template, str) else None
        self.template_sha256 = hashlib.sha256(tpl.encode()).hexdigest() if tpl else None
        self.date_dependent = bool(tpl and "strftime_now" in tpl)

    def render(self, messages: list[dict[str, str]]) -> str:
        return str(self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **self.kwargs))

    def count(self, messages: list[dict[str, str]]) -> int:
        ids: Any = self.tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, **self.kwargs)
        if not isinstance(ids, list):  # transformers >= 5 returns a BatchEncoding (UserDict)
            ids = ids["input_ids"]
        return len(ids)



def _filler(rng: random.Random, n_words: int) -> str:
    return " ".join(rng.choice(_VOCAB) for _ in range(n_words))


def build_prompt(r: TemplateRenderer, target: int, tag: str, seed: int) -> tuple[list[dict[str, str]], int, int]:
    """Return (messages, actual_tokens, template_overhead). Aims for actual == target exactly."""
    rng = random.Random(seed)
    head = f"[{tag}] "
    base_msgs = [{"role": "user", "content": head + INSTRUCTION.strip()}]
    overhead = r.count([{"role": "user", "content": ""}])
    fixed = r.count(base_msgs)
    if fixed > target:
        raise ValueError(f"target {target} smaller than the fixed prompt ({fixed} tokens)")
    words = _filler(rng, max(16, 2 * (target - fixed) + 64))
    ids = r.tok(words, add_special_tokens=False)["input_ids"]
    n = target - fixed
    best: tuple[int, list[dict[str, str]], int] | None = None
    for _ in range(30):
        n = max(0, min(n, len(ids)))
        body = str(r.tok.decode(ids[:n])).strip()
        msgs = [{"role": "user", "content": head + body + INSTRUCTION}]
        got = r.count(msgs)
        if best is None or abs(got - target) < abs(best[0] - target):
            best = (got, msgs, overhead)
        if got == target:
            break
        n += target - got
    assert best is not None
    return best[1], best[0], best[2]


def build_perf_workload(model: ModelEntry, *, input_tokens: int, output_tokens: int, n_warmup: int,
                        n_measured: int, thinking: str = "disabled", seed: int = 1234,
                        cache_dir: Path | None = None) -> dict[str, Any]:
    cache_dir = cache_dir or PROJECT_ROOT / ".cache" / "workloads" / model.key
    r = TemplateRenderer(model, thinking)
    spec = {"model": model.key, "tokenizer_repo": model.tokenizer_repo, "tokenizer_revision": model.tokenizer_revision,
            "input_tokens": input_tokens, "output_tokens": output_tokens, "n_warmup": n_warmup,
            "n_measured": n_measured, "thinking": thinking, "seed": seed, "generator": "perf-v1",
            "template_sha256": r.template_sha256}
    if r.date_dependent:
        # e.g. SmolLM3 injects "Today Date: ..." via strftime_now; token counts can change by date.
        import datetime as _dt

        spec["render_date"] = _dt.date.today().isoformat()
    spec_hash = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:16]
    path = cache_dir / f"perf-{input_tokens}-{output_tokens}-{spec_hash}.json"
    if path.exists():
        return json.loads(path.read_text())
    items: list[dict[str, Any]] = []
    for phase, count in (("warmup", n_warmup), ("measured", n_measured)):
        for i in range(count):
            tag = f"{phase[0]}{i:05d}-{spec_hash[:6]}"
            msgs, actual, overhead = build_prompt(r, input_tokens, tag, seed + (0 if phase == "measured" else 10**6) + i)
            items.append(PerfItem(tag, phase, msgs, r.render(msgs), actual, overhead, output_tokens).to_dict())
    content_hash = hashlib.sha256(json.dumps([(it["messages"], it["target_output_tokens"]) for it in items],
                                             sort_keys=True).encode()).hexdigest()
    wl = {
        "spec": spec,
        "spec_hash": spec_hash,
        "workload_sha256": content_hash,
        "template_sha256": r.template_sha256,
        "template_matches_registry": (r.template_sha256 == model.chat_template_sha256) if model.chat_template_sha256 else None,
        "chat_template_kwargs": r.kwargs,
        "template_date_dependent": r.date_dependent,
        "exact_hits": sum(1 for it in items if it["input_tokens"] == input_tokens),
        "items": items,
    }
    cache_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(wl, ensure_ascii=False))
    return wl

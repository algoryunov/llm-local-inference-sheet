#!/usr/bin/env python3
"""Build the GitHub Pages site (docs/site/) from the git-tracked part of results-public/.

    python scripts/build_site.py

Writes:
  - docs/site/technical.html  the technical report (scripts/site_technical_template.html + charts drawn here as SVG)
  - docs/site/proofreading.html  the proofreading report (scripts/site_proofreading_template.html)
  - docs/site/dashboard.html  a copy of results-public/dashboard/index.html
  - docs/site/index.html      a landing page linking both

Every number on the pages is read from results-public/: results.csv and manifest.json for the technical page,
and the per-item scores.jsonl (a raw file, in the release bundle) for the proofreading page. The tone-rewrite
example input comes from the synthetic items in .cache/datasets/proofread-v1 (scripts/prepare_proofread.py).
"""

from __future__ import annotations

import csv
import json
import re
import shutil
import statistics
from functools import partial
from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results-public"
SITE = ROOT / "docs" / "site"

MODEL_NAMES = {"qwen3-1.7b": "Qwen3-1.7B", "qwen3-4b": "Qwen3-4B", "qwen3-8b": "Qwen3-8B",
               "qwen3-14b": "Qwen3-14B", "smollm3-3b": "SmolLM3-3B"}
BACKEND_NAMES = {"llama-cpp": "llama.cpp", "mlx-lm": "MLX-LM", "ollama": "Ollama"}


def load_rows() -> list[dict[str, str]]:
    rows = list(csv.DictReader((RESULTS / "report" / "results.csv").open()))
    return [r for r in rows if r["state"] == "measured"]


def ctx_label(tokens: int) -> str:
    return f"{tokens // 1024}k" if tokens >= 1024 else str(tokens)


# ---------------------------------------------------------------- chart: GPU memory breakdown

def memory_rows(rows: list[dict[str, str]]) -> tuple[list[dict], int]:
    out, limit = [], 0
    for r in rows:
        if r["backend"] != "llama-cpp" or r["experiment"] not in ("mac-capacity", "mac-kvquant-q8"):
            continue
        man = json.loads((ROOT / r["run_dir"] / "manifest.json").read_text())
        bd = (man.get("final_log_facts") or {}).get("gpu_memory_breakdown_mib") or {}
        if not bd:
            continue
        limit = bd["device_total"]
        q8 = r["experiment"] == "mac-kvquant-q8"
        if q8 and int(r["input_tokens"]) != 16384:
            continue  # the q8_0 row is shown once, at the context where it matters
        out.append({
            "model": f"{MODEL_NAMES[r['model']]}, {r['quant']}", "params": float(r["params_b"]),
            "tokens": int(r["input_tokens"]), "q8": q8,
            "ctx": f"{ctx_label(int(r['input_tokens']))}, q8_0 KV" if q8 else f"{ctx_label(int(r['input_tokens']))} context",
            "weights": bd["model"], "kv": bd["context_kv"], "compute": bd["compute"],
        })
    out.sort(key=lambda d: (d["params"], d["tokens"], d["q8"]))
    return out, limit


def memory_chart(data: list[dict], limit_mib: int) -> str:
    big = max(data, key=lambda d: (d["params"], d["tokens"], not d["q8"]))
    big_total = (big["weights"] + big["kv"] + big["compute"]) / 1024
    title = (f"Context grows the KV cache, not the weights: {big['model'].split(',')[0]} at "
             f"{ctx_label(big['tokens'])} uses {big_total:.1f} of {limit_mib / 1024:.1f} GiB")
    left, width, max_mib, y0 = 150, 540, 14 * 1024, 104

    def x(mib: float) -> float:
        return left + mib / max_mib * width

    parts, yy, prev = [], y0, None
    for d in data:
        if d["model"] != prev:
            parts.append(f'<text class="t-group" x="0" y="{yy + 12}">{escape(d["model"])}</text>')
            yy += 22
            prev = d["model"]
        total = d["weights"] + d["kv"] + d["compute"]
        name = f'{d["model"]}, {d["ctx"]}'
        parts.append(f'<text class="t-lab" x="{left - 8}" y="{yy + 12}" text-anchor="end">{escape(d["ctx"])}</text>')
        for cls, start, size, what in (("weights", 0, d["weights"], "weights"),
                                       ("kv", d["weights"], d["kv"], "KV cache"),
                                       ("compute", d["weights"] + d["kv"], d["compute"], "compute buffers")):
            parts.append(f'<rect class="{cls}" x="{x(start):.1f}" y="{yy}" width="{x(size) - x(0):.1f}" height="16">'
                         f'<title>{escape(name)}: {what} {size:,} MiB</title></rect>')
        parts.append(f'<text class="t-val" x="{x(total) + 6:.1f}" y="{yy + 12}">{total / 1024:.1f} GiB</text>')
        yy += 24
    y_end = yy
    grid = "".join(
        f'<line class="grid" x1="{x(t * 1024):.1f}" x2="{x(t * 1024):.1f}" y1="{y0 - 4}" y2="{y_end}"/>'
        f'<text class="t-axis" x="{x(t * 1024):.1f}" y="{y_end + 18}" text-anchor="middle">{t} GiB</text>'
        for t in (0, 4, 8, 12))
    lim = (f'<line class="ref" x1="{x(limit_mib):.1f}" x2="{x(limit_mib):.1f}" y1="{y0 - 8}" y2="{y_end}"/>'
           f'<text class="t-lab" x="{x(limit_mib):.1f}" y="{y0 - 12}" text-anchor="end">'
           f'Metal GPU limit: {limit_mib / 1024:.1f} GiB</text>')
    legend = ('<rect class="weights" x="0" y="64" width="12" height="12"/><text class="t-lab" x="18" y="74">Weights</text>'
              '<rect class="kv" x="84" y="64" width="12" height="12"/><text class="t-lab" x="102" y="74">KV cache</text>'
              '<rect class="compute" x="174" y="64" width="12" height="12"/>'
              '<text class="t-lab" x="192" y="74">Compute buffers</text>')
    height = y_end + 30
    return (f'<svg viewBox="0 0 760 {height}" role="img" aria-label="{escape(title)}">'
            f'<text class="t-title" x="0" y="22">{escape(title)}</text>'
            f'<text class="t-sub" x="0" y="44">llama.cpp b11246 GPU memory at exit, one request. '
            f'A q8_0 KV cache nearly halves the cache.</text>'
            f'{legend}{grid}{"".join(parts)}{lim}</svg>')


# ---------------------------------------------------------------- chart: client vs server timer

def timer_rows(rows: list[dict[str, str]]) -> list[dict]:
    out = []
    for r in rows:
        if r["concurrency"] != "1" or not r["native_prefill_tps_p50"] or not r["prefill_proxy_tps_p50"]:
            continue
        native, proxy = float(r["native_prefill_tps_p50"]), float(r["prefill_proxy_tps_p50"])
        if r["experiment"] in ("quality", "proofread"):
            what = "task suite" if r["experiment"] == "quality" else "proofreading"
            long_prompt = False
        else:
            tokens = int(r["input_tokens"])
            what = f"{ctx_label(tokens)} in / {r['output_tokens']} out"
            what += ", q8_0 KV" if r["experiment"] == "mac-kvquant-q8" else ""
            what += ", rerun" if r["experiment"] == "mac-smoke" else ""
            long_prompt = tokens >= 1024
        out.append({"cell": f"{MODEL_NAMES[r['model']]} · {what} · {BACKEND_NAMES[r['backend']]}",
                    "long": long_prompt, "gap": (native - proxy) / native * 100})
    out.sort(key=lambda d: (not d["long"], -d["gap"]))
    return out


def timer_chart(data: list[dict]) -> str:
    max_long = max(d["gap"] for d in data if d["long"])
    title = f"On long prompts, client and server prefill timing agree within {max_long:.1f}%"
    left, width, max_pct, y0 = 330, 380, 8.0, 96

    def x(pct: float) -> float:
        return left + pct / max_pct * width

    group_names = {True: "Performance runs, prompts of 1k–16k tokens", False: "Short prompts and task suites"}
    top_gap = {g: max(d["gap"] for d in data if d["long"] == g) for g in (True, False)}
    parts, yy, prev = [], y0, None
    for d in data:
        if d["long"] != prev:
            parts.append(f'<text class="t-group" x="0" y="{yy + 14}">{group_names[d["long"]]}</text>')
            yy += 24
            prev = d["long"]
        dot = "dot-accent" if d["long"] else "dot-muted"
        parts.append(f'<text class="t-lab" x="{left - 10}" y="{yy + 4}" text-anchor="end">{escape(d["cell"])}</text>'
                     f'<line class="stem" x1="{x(0):.1f}" x2="{x(d["gap"]):.1f}" y1="{yy}" y2="{yy}"/>'
                     f'<circle class="{dot}" cx="{x(d["gap"]):.1f}" cy="{yy}" r="4">'
                     f'<title>{escape(d["cell"])}: {d["gap"]:.2f}% gap</title></circle>')
        if d["gap"] == top_gap[d["long"]]:
            parts.append(f'<text class="t-val" x="{x(d["gap"]) + 8:.1f}" y="{yy + 4}">{d["gap"]:.2f}%</text>')
        yy += 17
    y_end = yy
    grid = "".join(
        f'<line class="grid" x1="{x(t):.1f}" x2="{x(t):.1f}" y1="{y0 - 4}" y2="{y_end}"/>'
        f'<text class="t-axis" x="{x(t):.1f}" y="{y_end + 18}" text-anchor="middle">{t}%</text>'
        for t in (0, 2, 4, 6, 8))
    ref = (f'<line class="ref" x1="{x(1):.1f}" x2="{x(1):.1f}" y1="{y0 - 10}" y2="{y_end}"/>'
           f'<text class="t-lab" x="{x(1) + 4:.1f}" y="{y0 - 12}">1% gap</text>')
    height = y_end + 30
    return (f'<svg viewBox="0 0 760 {height}" role="img" aria-label="{escape(title)}">'
            f'<text class="t-title" x="0" y="22">{escape(title)}</text>'
            f'<text class="t-sub" x="0" y="44">Gap between the prefill rate derived from client time to first token '
            f'and the server timer, one request per cell</text>{grid}{ref}{"".join(parts)}</svg>')


# ---------------------------------------------------------------- proofreading page

PROOF_CONFIGS = {  # label -> cell id in results-public/proofread
    "Qwen3-8B Q4_K_M, llama.cpp": "llama-cpp__qwen3-8b-q4__proofread-v1-test__c1",
    "Qwen3-8B MLX 4-bit, MLX-LM": "mlx-lm__qwen3-8b-mlx4__proofread-v1-test__c1",
    "Qwen3-4B MLX 4-bit, MLX-LM": "mlx-lm__qwen3-4b-mlx4__proofread-v1-test__c1",
    "Qwen3-4B Q4_K_M, llama.cpp": "llama-cpp__qwen3-4b-q4__proofread-v1-test__c1",
}
L8, M8, M4, L4 = PROOF_CONFIGS  # labels in the order above
COMPARISONS = [("8B − 4B, llama.cpp", L8, L4), ("8B − 4B, MLX-LM", M8, M4), ("4B: llama.cpp − MLX-LM", L4, M4)]
N_BOOT = 3000


def load_scores(cell: str) -> dict[str, dict]:
    """Per-item scores of a cell; a later run (bench --fill-missing) supplies the items an earlier one skipped."""
    out: dict[str, dict] = {}
    files = sorted((RESULTS / "proofread" / "cells" / cell).glob("*/scores.jsonl"))
    if not files:
        raise SystemExit(f"no scores.jsonl for {cell}: extract the release bundle or run scripts/export_public.py")
    for path in files:
        for line in path.open():
            s = json.loads(line)
            out[s["item_id"]] = s
    return out


def f05(tp: int, fp: int, fn: int) -> float:
    p = tp / (tp + fp) if tp + fp else 1.0
    r = tp / (tp + fn) if tp + fn else 1.0
    return 1.25 * p * r / (0.25 * p + r) if p + r else 0.0


def split_items(scores: dict[str, dict]) -> tuple[list[dict], list[dict], list[dict]]:
    gec = [s for s in scores.values() if s["task"] == "gec" and not s["control"]]
    ctl = [s for s in scores.values() if s["task"] == "gec" and s["control"]]
    style = [s for s in scores.values() if s["task"] == "style"]
    return gec, ctl, style


def proof_metrics(scores: dict[str, dict]) -> dict:
    gec, ctl, style = split_items(scores)
    tp, fp, fn = (sum(s[k] for s in gec) for k in ("tp", "fp", "fn"))
    changed = [s for s in ctl if not s["unchanged"]]
    outcome = {"match": 0, "partial": 0, "untouched": 0, "wrong": 0}
    for s in gec:
        if s["fp"] == 0 and s["fn"] == 0:
            outcome["match"] += 1
        elif s["hyp_edits"] == 0:
            outcome["untouched"] += 1
        elif s["tp"] == 0:
            outcome["wrong"] += 1
        else:
            outcome["partial"] += 1
    return {
        "f": f05(tp, fp, fn), "p": tp / (tp + fp), "r": tp / (tp + fn),
        "keep": 1 - len(changed) / len(ctl), "tone": sum(s["pass"] for s in style) / len(style),
        "n_gec": len(gec), "outcome": outcome, "changed": changed,
        "verbatim": sum(s["facts_preserved_verbatim"] for s in style) / len(style), "style": style,
    }


def paired_bootstrap(a: dict[str, dict], b: dict[str, dict], seed: int) -> dict[str, tuple[float, float, float]]:
    """Point difference a − b and 95% interval for F0.5, keep rate and tone pass, resampling shared items."""
    import random

    rng = random.Random(seed)
    out = {}
    for metric in ("f", "keep", "tone"):
        if metric == "f":
            ids = sorted(k for k, s in a.items() if s["task"] == "gec" and not s["control"] and k in b)

            def score(side: dict[str, dict], sample: list[str]) -> float:
                return f05(*(sum(side[k][f] for k in sample) for f in ("tp", "fp", "fn")))
        elif metric == "keep":
            ids = sorted(k for k, s in a.items() if s["task"] == "gec" and s["control"] and k in b)

            def score(side: dict[str, dict], sample: list[str]) -> float:
                return sum(side[k]["unchanged"] for k in sample) / len(sample)
        else:
            ids = sorted(k for k, s in a.items() if s["task"] == "style" and k in b)

            def score(side: dict[str, dict], sample: list[str]) -> float:
                return sum(side[k]["pass"] for k in sample) / len(sample)
        point = score(a, ids) - score(b, ids)
        diffs = []
        for _ in range(N_BOOT):
            sample = [ids[rng.randrange(len(ids))] for _ in ids]
            diffs.append(score(a, sample) - score(b, sample))
        diffs.sort()
        out[metric] = (point, diffs[int(0.025 * N_BOOT)], diffs[int(0.975 * N_BOOT) - 1])
    return out


def diff_chart(diffs: list[tuple[str, dict]]) -> str:
    title = "8B corrects grammar better on both runtimes; the other gaps are small or noise"
    panels = [("f", "Grammar F0.5"), ("keep", "Correct text untouched"), ("tone", "Tone rewrites pass")]
    left, pw, gap, y0, row_h = 190, 170, 20, 112, 36
    parts = []
    for p_i, (metric, name) in enumerate(panels):
        x0 = left + p_i * (pw + gap)
        span = max(max(abs(d[metric][1]), abs(d[metric][2])) for _, d in diffs) * 1.15

        def x(v: float, x0: float = x0, span: float = span) -> float:
            return x0 + pw / 2 + v / span * (pw / 2)

        y_end = y0 + len(diffs) * row_h - 12
        parts.append(f'<text class="t-head" x="{x0 + pw / 2:.1f}" y="{y0 - 22}" text-anchor="middle">{name}</text>'
                     f'<line class="zero" x1="{x(0):.1f}" x2="{x(0):.1f}" y1="{y0 - 12}" y2="{y_end}"/>'
                     f'<text class="t-axis" x="{x(0):.1f}" y="{y_end + 16}" text-anchor="middle">0</text>')
        for i, (label, d) in enumerate(diffs):
            point, lo, hi = d[metric]
            cy = y0 + i * row_h
            real = lo > 0 or hi < 0
            cls = "accent" if real else "muted"
            parts.append(f'<line class="whisker-{cls}" x1="{x(lo):.1f}" x2="{x(hi):.1f}" y1="{cy}" y2="{cy}"/>'
                         f'<circle class="dot-{cls}" cx="{x(point):.1f}" cy="{cy}" r="4.5">'
                         f'<title>{escape(label)}, {name}: {point:+.3f} ({lo:+.3f} to {hi:+.3f})</title></circle>'
                         f'<text class="t-val" x="{x(point):.1f}" y="{cy - 9}" text-anchor="middle">{point:+.3f}</text>')
    for i, (label, _) in enumerate(diffs):
        parts.append(f'<text class="t-lab" x="0" y="{y0 + i * row_h + 4}">{escape(label)}</text>')
    height = y0 + len(diffs) * row_h + 22
    return (f'<svg viewBox="0 0 760 {height}" role="img" aria-label="{escape(title)}">'
            f'<text class="t-title" x="0" y="22">{escape(title)}</text>'
            f'<text class="t-sub" x="0" y="44">Difference in each score with its 95% interval; blue = interval excludes zero</text>'
            f'{"".join(parts)}</svg>')


def outcome_chart(metrics: dict[str, dict]) -> str:
    def share(label: str, key: str) -> float:
        return metrics[label]["outcome"][key] / metrics[label]["n_gec"] * 100

    title = (f"Qwen3-8B matches a human correction on {share(L8, 'match'):.0f}% of sentences, "
             f"Qwen3-4B on {share(L4, 'match'):.0f}% (llama.cpp)")
    segs = [("match", "Matches a human correction"), ("partial", "Partly fixed"),
            ("untouched", "Errors left untouched"), ("wrong", "Only wrong edits")]
    left, width, y0 = 190, 540, 104
    legend, lx = [], 0
    for key, name in segs:
        legend.append(f'<rect class="seg-{key}" x="{lx}" y="62" width="12" height="12"/>'
                      f'<text class="t-lab" x="{lx + 18}" y="72">{name}</text>')
        lx += 18 + len(name) * 6.4 + 18
    parts = []
    for i, label in enumerate(PROOF_CONFIGS):
        cy = y0 + i * 34
        parts.append(f'<text class="t-lab" x="{left - 10}" y="{cy + 15}" text-anchor="end">{escape(label)}</text>')
        x = float(left)
        for key, name in segs:
            pct = share(label, key)
            w = pct / 100 * width
            parts.append(f'<rect class="seg-{key}" x="{x:.1f}" y="{cy}" width="{w:.1f}" height="22">'
                         f'<title>{escape(label)}: {name.lower()} {pct:.1f}% '
                         f'({metrics[label]["outcome"][key]} of {metrics[label]["n_gec"]})</title></rect>')
            if w > 34:
                cls = "t-val-in" if key in ("match", "wrong") else "t-val"
                parts.append(f'<text class="{cls}" x="{x + w / 2:.1f}" y="{cy + 15}" text-anchor="middle">{pct:.0f}%</text>')
            x += w
    height = y0 + len(PROOF_CONFIGS) * 34 + 6
    return (f'<svg viewBox="0 0 760 {height}" role="img" aria-label="{escape(title)}">'
            f'<text class="t-title" x="0" y="22">{escape(title)}</text>'
            f'<text class="t-sub" x="0" y="44">Outcome per learner sentence, compared at edit level with the closest '
            f'of its 4 human corrections</text>{"".join(legend)}{"".join(parts)}</svg>')


# Curated examples: (group, item id, expected (4B, 8B) outcome, note). The build fails if an outcome no longer
# holds, so the groups never drift from the data. Outcomes: match / partial / untouched / wrong (grammar),
# kept / changed (already-correct text), pass / fail (tone).
EXAMPLES = [
    ("Both fixed it", "proofread-v1-test-gec-en-0026", ("match", "match"), ""),
    ("Both fixed it", "proofread-v1-test-gec-en-0035", ("match", "match"), ""),
    ("Both fixed it", "proofread-v1-test-gec-en-0025", ("match", "match"), ""),
    ("Only Qwen3-8B fixed it", "proofread-v1-test-gec-en-0056", ("untouched", "match"), ""),
    ("Only Qwen3-8B fixed it", "proofread-v1-test-gec-en-0059", ("wrong", "match"), ""),
    ("Only Qwen3-8B fixed it", "proofread-v1-test-gec-en-0104", ("wrong", "match"), ""),
    ("Only Qwen3-8B fixed it", "proofread-v1-test-gec-en-0333", ("untouched", "match"), ""),
    ("Only Qwen3-4B fixed it", "proofread-v1-test-gec-en-0416", ("match", "wrong"),
     "8B's fix reads fine too, but no human wrote it, so the score counts it as wrong."),
    ("Only Qwen3-4B fixed it", "proofread-v1-test-gec-en-0085", ("match", "partial"),
     "8B also changed \"have\" to \"had\", which fits \"thought\", but no human made that change."),
    ("Neither fixed it", "proofread-v1-test-gec-en-0094", ("wrong", "wrong"), ""),
    ("Neither fixed it", "proofread-v1-test-gec-en-0009", ("untouched", "untouched"), ""),
    ("Neither fixed it", "proofread-v1-test-gec-en-0146", ("wrong", "wrong"),
     "Both wrote \"various fields\", which reads fine; the humans wrote \"a variety of fields\"."),
    ("Already-correct text", "proofread-v1-test-gec-en-ctrl-003", ("changed", "kept"), ""),
    ("Already-correct text", "proofread-v1-test-gec-en-ctrl-011", ("kept", "changed"),
     "8B fixed a comma splice that the human correction left in; the score counts it as over-correction."),
    ("Already-correct text", "proofread-v1-test-gec-en-ctrl-015", ("changed", "changed"),
     "Counted as over-correction, but \"media\" is a real fix: the human correction kept the error."),
    ("Tone rewrite, formal to casual", "proofread-v1-test-style-en-casual-006", ("fail", "pass"), ""),
    ("Tone rewrite, casual to formal", "proofread-v1-test-style-en-formal-000", ("pass", "pass"), ""),
]
TONE_MARKS = {"in Berlin": "bad", "4 PM": "mid"}  # spots to point at in the tone example
VERDICTS = {
    "match": ("ok", "✓ matches a human correction"), "partial": ("mid", "~ partly fixed"),
    "untouched": ("bad", "✗ error left in place"), "wrong": ("bad", "✗ only wrong edits"),
    "kept": ("ok", "✓ left as it was"), "changed": ("bad", "✗ changed correct text"),
    "pass": ("ok", "✓ facts kept, tone changed"), "fail": ("bad", "✗ lost a fact"),
}
TOKEN = re.compile(r"\w+(?:'\w+)?|[^\w\s]")


def detokenize(text: str) -> str:
    """JFLEG text is space-tokenized ("do n't", "school ."); show it as normal prose."""
    text = re.sub(r" n't\b", "n't", text)
    text = re.sub(r" '(s|re|ve|ll|d|m)\b", r"'\1", text)
    text = re.sub(r" ([.,!?;:%)])", r"\1", text)
    return re.sub(r"\( ", "(", text).strip()


def diff_html(original: str, output: str) -> str:
    """Output text with added/changed words in <ins> and removed words in <del>, compared word by word."""
    import difflib

    src = [m.group() for m in TOKEN.finditer(original)]
    out = list(TOKEN.finditer(output))
    ops = difflib.SequenceMatcher(a=src, b=[m.group() for m in out], autojunk=False).get_opcodes()
    marks_before: dict[int, str] = {}  # output token index -> struck-through words shown before it
    ins = set()
    for tag, i1, i2, j1, j2 in ops:
        if tag in ("delete", "replace"):
            marks_before[j1] = marks_before.get(j1, "") + f'<del>{escape(" ".join(src[i1:i2]))}</del> '
        if tag in ("insert", "replace"):
            ins.update(range(j1, j2))
    html, pos = [], 0
    for j, m in enumerate(out):
        html.append(escape(output[pos:m.start()]))
        html.append(marks_before.get(j, ""))
        word = escape(m.group())
        html.append(f"<ins>{word}</ins>" if j in ins else word)
        pos = m.end()
    html.append(escape(output[pos:]))
    html.append(marks_before.get(len(out), "").rstrip())
    return "".join(html)


def closest_ref(refs: list[str], scored: tuple[dict, ...]) -> str:
    """The human correction nearest to a model output that matched one, else nearest to any output."""
    import difflib

    outs = [s["output"] for s in scored if outcome_of(s) == "match"] or [s["output"] for s in scored]
    words = [[m.group() for m in TOKEN.finditer(o)] for o in outs]
    cands = [detokenize(r) for r in refs]
    return max(cands, key=lambda r: max(difflib.SequenceMatcher(
        a=[m.group() for m in TOKEN.finditer(r)], b=w, autojunk=False).ratio() for w in words))


def tone_html(text: str) -> str:
    html = "<br>".join(escape(line.strip()) for line in text.strip().splitlines() if line.strip())
    for spot, cls in TONE_MARKS.items():
        html = html.replace(escape(spot), f'<mark class="{cls}">{escape(spot)}</mark>')
    return html


def outcome_of(s: dict) -> str:
    if s["task"] == "style":
        return "pass" if s["pass"] else "fail"
    if s["control"]:
        return "kept" if s["unchanged"] else "changed"
    if s["fp"] == 0 and s["fn"] == 0:
        return "match"
    if s["hyp_edits"] == 0:
        return "untouched"
    return "wrong" if s["tp"] == 0 else "partial"


def example_rows(scores: dict[str, dict[str, dict]], items: dict[str, dict]) -> str:
    rows, group = [], None
    for name, item_id, expected, note in EXAMPLES:
        s4, s8 = scores[L4][item_id], scores[L8][item_id]
        got = (outcome_of(s4), outcome_of(s8))
        if got != expected:
            raise SystemExit(f"example {item_id} is now {got}, not {expected}: update EXAMPLES in build_site.py")
        if name != group:
            rows.append(f'<tr class="group"><td colspan="3">{escape(name)}</td></tr>')
            group = name
        item = items[item_id]
        if s4["task"] == "style":
            original, ref = item["messages"][-1]["content"], ""
            render = tone_html
        else:
            original = detokenize(item["expected"]["source"])
            ref = "" if s4["control"] else (
                f'<span class="ref">Human: {escape(closest_ref(item["expected"]["refs"], (s4, s8)))}</span>')
            render = partial(diff_html, original)
        cells = []
        for s in (s4, s8):
            cls, label = VERDICTS[outcome_of(s)]
            cells.append(f'<td>{render(s["output"])}<span class="verdict {cls}">{label}</span></td>')
        note_html = f'<span class="note">{escape(note)}</span>' if note else ""
        rows.append(f'<tr><td>{escape(original)}{ref}{note_html}</td>{"".join(cells)}</tr>')
    return "\n".join(rows)


def proofreading_page(rows: list[dict[str, str]], data_date: str) -> str:
    scores = {label: load_scores(cell) for label, cell in PROOF_CONFIGS.items()}
    metrics = {label: proof_metrics(s) for label, s in scores.items()}
    latency = {r["cell_id"]: r for r in rows if r["experiment"] == "proofread"}
    diffs = [(name, paired_bootstrap(scores[a], scores[b], seed=20261001 + i))
             for i, (name, a, b) in enumerate(COMPARISONS)]
    best_f = max(m["f"] for m in metrics.values())
    table = []
    for label, cell in PROOF_CONFIGS.items():
        m, lat = metrics[label], latency[cell]
        f_cell = f"<strong>{m['f']:.3f}</strong>" if m["f"] == best_f else f"{m['f']:.3f}"
        table.append(f'<tr><td>{escape(label)}</td><td class="n">{f_cell}</td><td class="n">{m["p"]:.3f}</td>'
                     f'<td class="n">{m["r"]:.3f}</td><td class="n">{m["keep"]:.0%}</td><td class="n">{m["tone"]:.0%}</td>'
                     f'<td class="n">{float(lat["e2e_p50_s"]):.2f} / {float(lat["e2e_p95_s"]):.2f}</td></tr>')

    # tone-rewrite example and failure counts
    items = {}
    for line in (ROOT / ".cache" / "datasets" / "proofread-v1" / "test.jsonl").open():
        it = json.loads(line)
        items[it["id"]] = it
    drops = [s for s in metrics[L4]["style"] if not s["checks"].get("facts_preserved", True)
             and "Berlin" in s["output"] and "Berlin office" not in s["output"]]
    if not drops:
        raise SystemExit("tone example not found: update the proofreading template")
    slang = sum(1 for s in metrics[M8]["style"] if not s["checks"].get("no_slang", True) and "awesome" in s["output"].lower())
    changed_all = [s for m in metrics.values() for s in m["changed"]]
    wrong = [m["outcome"]["wrong"] / m["n_gec"] * 100 for m in metrics.values()]
    ctrl = [len(m["changed"]) for m in metrics.values()]
    tone = [m["tone"] * 100 for m in metrics.values()]

    def untouched(label: str) -> str:
        return f"{metrics[label]['outcome']['untouched'] / metrics[label]['n_gec'] * 100:.1f}"

    lo, hi = diffs[0][1]["f"][1], diffs[0][1]["f"][2]
    values = {
        "DATA_DATE": data_date, "N_BOOT": f"{N_BOOT:,}",
        "F_8B_LLAMA": f"{metrics[L8]['f']:.3f}", "F_4B_LLAMA": f"{metrics[L4]['f']:.3f}",
        "D_F_LLAMA": f"{diffs[0][1]['f'][0]:+.3f}, 95% interval {lo:+.3f} to {hi:+.3f}",
        "SEC_8B_LLAMA": f"{float(latency[PROOF_CONFIGS[L8]]['e2e_p50_s']):.2f}",
        "SEC_4B_LLAMA": f"{float(latency[PROOF_CONFIGS[L4]]['e2e_p50_s']):.2f}",
        "RESULTS_ROWS": "\n".join(table),
        "DIFF_CHART": diff_chart(diffs), "OUTCOME_CHART": outcome_chart(metrics),
        "WRONG_MIN": f"{min(wrong):.0f}", "WRONG_MAX": f"{max(wrong):.0f}",
        "UNTOUCHED_4B_LLAMA": untouched(L4), "UNTOUCHED_8B_LLAMA": untouched(L8),
        "R_8B_LLAMA": f"{metrics[L8]['r']:.2f}", "R_4B_LLAMA": f"{metrics[L4]['r']:.2f}",
        "CTRL_CHANGED_MIN": str(min(ctrl)), "CTRL_CHANGED_MAX": str(max(ctrl)),
        "CTRL_ONE_EDIT": f"{sum(1 for s in changed_all if s['hyp_edits'] == 1) / len(changed_all) * 100:.0f}",
        "TONE_MIN": f"{min(tone):.0f}", "TONE_MAX": f"{max(tone):.0f}",
        "EXAMPLE_ROWS": example_rows(scores, items),
        "PROMPT_GEC": escape(items[EXAMPLES[0][1]]["messages"][0]["content"]),
        "PROMPT_CASUAL": escape(items["proofread-v1-test-style-en-casual-006"]["messages"][0]["content"]),
        "PROMPT_FORMAL": escape(items["proofread-v1-test-style-en-formal-000"]["messages"][0]["content"]),
        "TONE_4B_FACTS": str(len(drops)), "TONE_8B_MLX_SLANG": str(slang),
        "TONE_8B_REFORMAT": f"{(1 - metrics[L8]['verbatim']) * 100:.0f}",
        "TONE_4B_REFORMAT": f"{(1 - metrics[L4]['verbatim']) * 100:.0f}",
    }
    page = (ROOT / "scripts" / "site_proofreading_template.html").read_text()
    for key, value in values.items():
        page = page.replace("{{" + key + "}}", value)
    if "{{" in page:
        raise SystemExit("unfilled placeholder in the proofreading page")
    for name, d in diffs:
        print(f"  {name}: " + ", ".join(f"{k} {v[0]:+.3f} ({v[1]:+.3f} to {v[2]:+.3f})" for k, v in d.items()))
    return page


# ---------------------------------------------------------------- pages

INDEX ="""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>LLM Local Inference Sheet</title>
<style>
:root { --bg: #f4f5f8; --ink: #14171c; --ink-2: #454c57; --accent: #2a78d6; --line: #dfe3e9; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
  --bg: #101216; --ink: #f1f3f6; --ink-2: #c2c8d0; --accent: #3987e5; --line: #2a2f38; color-scheme: dark; } }
:root[data-theme="dark"] { --bg: #101216; --ink: #f1f3f6; --ink-2: #c2c8d0; --accent: #3987e5; --line: #2a2f38;
  color-scheme: dark; }
body { margin: 0; background: var(--bg); color: var(--ink); font: 17px/1.6 "Segoe UI", system-ui, sans-serif; }
.wrap { max-width: 720px; margin: 0 auto; padding: 40px 16px; }
h1 { font-size: 30px; margin: 0 0 8px; } p { color: var(--ink-2); } a { color: var(--accent); }
li { margin: 10px 0; } li span { color: var(--ink-2); }
</style></head><body><div class="wrap">
<h1>LLM Local Inference Sheet</h1>
<p>Which local LLM and runtime to run on a MacBook, measured on an Apple M3 Pro (18 GB).</p>
<ul>
<li><a href="dashboard.html">Results dashboard</a> <span>— speed, quality, capacity and compatibility</span></li>
<li><a href="proofreading.html">Proofreading on a Mac</a> <span>— Qwen3-8B vs Qwen3-4B for grammar correction and tone rewrites</span></li>
<li><a href="technical.html">Technical Kitchen</a> <span>— how the numbers are measured and how far to trust them</span></li>
<li><a href="https://github.com/algoryunov/llm-local-inference-sheet">Repository</a> <span>— code, data and findings</span></li>
</ul>
</div></body></html>
"""


def main() -> None:
    rows = load_rows()
    mem, limit = memory_rows(rows)
    timer = timer_rows(rows)
    avail = [float(r["available_at_start_gib"]) for r in rows if r["available_at_start_gib"]]
    pressured = sum(1 for r in rows if r["peak_pressure"] and int(r["peak_pressure"]) >= 2)
    run_dates = sorted(r["run_id"][:8] for r in rows)
    values = {
        "DATA_DATE": f"{run_dates[-1][:4]}-{run_dates[-1][4:6]}-{run_dates[-1][6:8]}",
        "MAX_LONG_GAP": f"{max(d['gap'] for d in timer if d['long']):.1f}",
        "N_RUNS": str(len(rows)), "N_PRESSURED": str(pressured),
        "AVAIL_MIN": f"{min(avail):.1f}", "AVAIL_MAX": f"{max(avail):.1f}", "AVAIL_MED": f"{statistics.median(avail):.1f}",
        "N_TIMER": str(len(timer)),
        "MEMORY_CHART": memory_chart(mem, limit), "TIMER_CHART": timer_chart(timer),
    }
    page = (ROOT / "scripts" / "site_technical_template.html").read_text()
    for key, value in values.items():
        page = page.replace("{{" + key + "}}", value)
    if "{{" in page:
        raise SystemExit("unfilled placeholder in the technical page")
    SITE.mkdir(parents=True, exist_ok=True)
    (SITE / "technical.html").write_text(page)
    (SITE / "proofreading.html").write_text(proofreading_page(rows, values["DATA_DATE"]))
    shutil.copyfile(RESULTS / "dashboard" / "index.html", SITE / "dashboard.html")
    (SITE / "index.html").write_text(INDEX)
    nojekyll = ROOT / "docs" / ".nojekyll"  # serve files as they are; the Markdown docs are read on GitHub itself
    nojekyll.touch()
    print(f"wrote {SITE}/technical.html, proofreading.html, dashboard.html, index.html")


if __name__ == "__main__":
    main()

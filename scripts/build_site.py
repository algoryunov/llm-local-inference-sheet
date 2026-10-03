#!/usr/bin/env python3
"""Build the GitHub Pages site (docs/site/) from the git-tracked part of results-public/.

    python scripts/build_site.py

Writes:
  - docs/site/technical.html  the technical report (scripts/site_technical_template.html + charts drawn here as SVG)
  - docs/site/dashboard.html  a copy of results-public/dashboard/index.html
  - docs/site/index.html      a landing page linking both

Every number on the pages is read from results-public/report/results.csv or the per-run manifest.json files,
so the site needs no raw files and rebuilds with the rest of the export.
"""

from __future__ import annotations

import csv
import json
import shutil
import statistics
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


# ---------------------------------------------------------------- pages

INDEX = """<!doctype html>
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
    shutil.copyfile(RESULTS / "dashboard" / "index.html", SITE / "dashboard.html")
    (SITE / "index.html").write_text(INDEX)
    nojekyll = ROOT / "docs" / ".nojekyll"  # serve files as they are; the Markdown docs are read on GitHub itself
    nojekyll.touch()
    print(f"wrote {SITE}/technical.html, dashboard.html, index.html")


if __name__ == "__main__":
    main()

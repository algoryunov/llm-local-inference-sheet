#!/usr/bin/env python3
"""Build a self-contained results dashboard (results-public/dashboard/index.html) from saved observations.

Inputs (all regenerated, never typed by hand):
  - results-public/report/results.csv  (from `llm-sheet report --runs results-public/... --out results-public/report`)
  - results-public/investigations/llama-cpp-batching/batched-bench-fine-qwen3-4b-q4.txt  (raw llama-batched-bench output)
  - docs/compatibility.md             (from scripts/gen_compat.py)
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIELDS = [
    "experiment", "cell_id", "backend", "artifact", "model", "params_b", "quant", "workload", "input_tokens",
    "output_tokens", "concurrency", "state", "reason", "n_ok", "ttft_p50_s", "tpot_p50_ms", "e2e_p50_s",
    "decode_tps_p50", "agg_output_tps", "native_prefill_tps_p50", "prefill_proxy_tps_p50", "gpu_mem_delta_gib",
    "mlx_peak_gib", "llama_gpu_self_mib", "llama_kv_mib", "mem_contaminated", "quality_composite",
    "extraction_field_acc", "routing_macro_f1", "cv_field_acc", "parse_strict",
    "early_stops", "status_counts", "gec_en_f05", "gec_en_precision", "gec_en_recall", "gec_en_keep", "style_en_pass",
]
NUMERIC = {"params_b", "input_tokens", "output_tokens", "concurrency", "n_ok", "ttft_p50_s", "tpot_p50_ms",
           "e2e_p50_s", "decode_tps_p50", "agg_output_tps", "native_prefill_tps_p50", "prefill_proxy_tps_p50",
           "gpu_mem_delta_gib", "mlx_peak_gib", "llama_gpu_self_mib", "llama_kv_mib", "quality_composite",
           "extraction_field_acc", "routing_macro_f1", "cv_field_acc", "parse_strict",
           "early_stops", "gec_en_f05", "gec_en_precision", "gec_en_recall", "gec_en_keep", "style_en_pass"}


def latest_rows() -> list[dict]:
    rows = list(csv.DictReader((ROOT / "results-public/report/results.csv").open()))
    best: dict[tuple[str, str], dict] = {}
    for r in rows:
        k = (r["experiment"], r["cell_id"])
        if k not in best or r["run_id"] > best[k]["run_id"]:
            best[k] = r
    out = []
    for r in best.values():
        d = {}
        for f in FIELDS:
            v = r.get(f, "")
            if f in NUMERIC:
                d[f] = float(v) if v not in ("", None) else None
            elif f == "mem_contaminated":
                d[f] = v == "True"
            else:
                d[f] = v
        out.append(d)
    return out


def batched_bench() -> list[dict]:
    p = ROOT / "results-public/investigations/llama-cpp-batching/batched-bench-fine-qwen3-4b-q4.txt"
    pts = []
    for line in p.read_text().splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) == 10 and cells[0].isdigit():
            pts.append({"b": int(cells[2]), "tps": float(cells[7])})
    return pts


def compat() -> list[dict]:
    lines = (ROOT / "docs/compatibility.md").read_text().splitlines()
    head = next(i for i, ln in enumerate(lines) if ln.startswith("| artifact"))
    cols = [c.strip() for c in lines[head].strip("|").split("|")]
    rows = []
    for ln in lines[head + 2:]:
        if not ln.startswith("|"):
            break
        cells = [re.sub(r"[*`]", "", c).strip() for c in ln.strip("|").split("|")]
        rows.append(dict(zip(cols, cells, strict=False)))
    return rows


def main() -> None:
    data = {"rows": latest_rows(), "bench": batched_bench(), "compat": compat()}
    tpl = (ROOT / "scripts/dashboard_template.html").read_text()
    out = ROOT / "results-public/dashboard/index.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(tpl.replace("__DATA__", json.dumps(data, separators=(",", ":")).replace("</", "<\\/")))
    print(f"wrote {out} ({out.stat().st_size // 1024} KiB, {len(data['rows'])} cells)")


if __name__ == "__main__":
    main()

"""Report generation from saved observations only (never runs inference).

Every metric is recomputed from ``requests.jsonl`` + ``events.jsonl.gz`` (and quality
scores from the raw response text), so a report regenerates identically from the
stored observations. Cached ``summary.json`` files are not trusted.
Synthetic (mock) runs are excluded unless explicitly requested, and even then they
are labelled SYNTHETIC in every table.
"""

from __future__ import annotations

import csv
import html
import json
from pathlib import Path
from typing import Any

from .bench import UNSCORED_STATUSES, load_app_items, merged_run_data, score_run, summarize_run
from .config import PROJECT_ROOT, Workload
from .evaluation import aggregate_scores
from .manifests import load_jsonl
from .telemetry import memory_contamination


def collect_runs(roots: list[Path], include_synthetic: bool = False) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for root in roots:
        for man_path in sorted(root.rglob("manifest.json")):
            run = man_path.parent
            if run in seen or "report" in run.parts:
                continue
            seen.add(run)
            man = json.loads(man_path.read_text())
            if not man.get("finished"):
                continue
            if man.get("fill_of"):
                continue  # merged into the run it completes
            synthetic = bool(man.get("synthetic")) or man.get("state") == "synthetic"
            if synthetic and not include_synthetic:
                continue
            rows.append(_row(run, man, synthetic))
    return rows


def _p(d: dict[str, Any] | None, key: str) -> Any:
    return (d or {}).get(key)


def _row(run: Path, man: dict[str, Any], synthetic: bool) -> dict[str, Any]:
    w = man.get("workload") or {}
    host = man.get("host") or {}
    mac = host.get("mac") or {}
    hw = man.get("hardware_label") or (f"{mac.get('chip')} {round((mac.get('memory_bytes') or 0) / 2**30)}GB" if mac else None)
    row: dict[str, Any] = {
        "run_dir": _rel(run), "experiment": man.get("experiment"), "cell_id": man.get("cell_id"),
        "run_id": run.name, "state": "synthetic" if synthetic else man.get("state"), "reason": man.get("reason"),
        "synthetic": synthetic, "backend": man.get("backend"), "artifact": man.get("artifact"),
        "model": man.get("model"), "family": man.get("family"), "params_b": man.get("params_total_b"),
        "quant": (man.get("quantization") or {}).get("method"), "weight_dtype": man.get("weight_dtype"),
        "platform": man.get("platform"), "hardware": hw, "kind": man.get("kind"),
        "workload": w.get("name"), "input_tokens": w.get("input_tokens"), "output_tokens": w.get("output_tokens"),
        "concurrency": man.get("concurrency"), "repeat": man.get("repeat"),
        "backend_version": (man.get("backend_availability") or {}).get("version"),
        "power_source": mac.get("power_source"), "low_power_mode": mac.get("low_power_mode"),
        "pressure_at_start": (man.get("system_baseline") or {}).get("vm_pressure_level"),
        "available_at_start_gib": _gib((man.get("system_baseline") or {}).get("sys_mem_available")),
        "load_first_gen_s": _p(man.get("readiness"), "first_generation_ready_s"),
        "load_health_s": _p(man.get("readiness"), "health_ready_s"),
    }
    mc = memory_contamination(man.get("system_baseline"), man.get("telemetry_summary"))
    row["mem_contaminated"] = mc["contaminated"]
    row["mem_contamination_reasons"] = "; ".join(mc["reasons"])
    tv = man.get("template_verification") or {}
    row["template_identical"] = tv.get("rendered_prompt_identical")
    tel = (man.get("telemetry_summary") or {})
    row.update({
        "peak_footprint_gib": _gib(tel.get("proc_max_lifetime_phys_footprint") or tel.get("peak_proc_tree_phys_footprint")),
        "peak_rss_gib": _gib(tel.get("peak_proc_tree_rss")),
        "peak_agx_inuse_gib": _gib(tel.get("peak_agx_in_use_system_memory")),
        "peak_gpu_util_pct": tel.get("peak_gpu_device_util_pct"),
        "peak_pressure": tel.get("peak_vm_pressure_level"),
        "swap_out_delta_mib": (tel.get("swap_out_bytes_delta") or 0) / 2**20 if tel.get("swap_out_bytes_delta") is not None else None,
        "peak_nvidia_mem_mib": tel.get("peak_nvidia_memory_used_mib"),
    })
    bd = (man.get("final_log_facts") or {}).get("gpu_memory_breakdown_mib") or {}
    if not bd and str(man.get("backend", "")).startswith("llama-cpp"):
        bd = _llama_breakdown_from_log(run)
    row["llama_gpu_self_mib"] = bd.get("self")
    row["llama_kv_mib"] = bd.get("context_kv")
    base_agx = (man.get("system_baseline") or {}).get("agx_in_use_system_memory")
    peak_agx = tel.get("peak_agx_in_use_system_memory")
    # System-wide GPU-accessible memory in use (all processes) minus the pre-launch baseline.
    row["gpu_mem_delta_gib"] = _gib(peak_agx - base_agx) if peak_agx is not None and base_agx is not None else None
    if row["gpu_mem_delta_gib"] is None and tel.get("peak_nvidia_memory_used_mib") is not None:
        row["gpu_mem_delta_gib"] = round(tel["peak_nvidia_memory_used_mib"] / 1024, 3)
    mlx_mem = load_jsonl(run / "mlx_memory.jsonl")
    peaks = [r["mlx_peak"] for r in mlx_mem if "mlx_peak" in r]
    row["mlx_peak_gib"] = _gib(max(peaks)) if peaks else None
    records, events, n_filled = merged_run_data(run)
    row["filled_items"] = n_filled
    if records:
        s = summarize_run(records, events, None, None)
        perf = s["performance"]
        row.update({
            "n_measured": perf.get("n_measured"), "n_ok": perf.get("n_ok"),
            "n_unanswered": sum(v for k, v in (perf.get("status_counts") or {}).items() if k in UNSCORED_STATUSES),
            "status_counts": json.dumps(perf.get("status_counts")),
            "early_stops": perf.get("early_stops"), "n_full_length": perf.get("n_full_length"),
            "ttft_p50_s": _p(perf.get("ttft_s"), "p50"), "ttft_p95_s": _p(perf.get("ttft_s"), "p95"),
            "ttft_p95_reportable": _p(perf.get("ttft_s"), "p95_reportable"),
            "ttft_p50_ci95": json.dumps(_p(perf.get("ttft_s"), "p50_ci95")),
            "tpot_p50_ms": _ms(_p(perf.get("tpot_s"), "p50")), "tpot_p95_ms": _ms(_p(perf.get("tpot_s"), "p95")),
            "e2e_p50_s": _p(perf.get("e2e_s"), "p50"), "e2e_p95_s": _p(perf.get("e2e_s"), "p95"),
            "decode_tps_p50": _p(perf.get("decode_tps"), "p50"),
            "agg_output_tps": perf.get("aggregate_output_tps"), "successful_rps": perf.get("successful_rps"),
            "mean_inflight": perf.get("mean_inflight"),
            "native_prefill_tps_p50": _p(perf.get("native_prefill_tps"), "p50"),
            "native_decode_tps_p50": _p(perf.get("native_decode_tps"), "p50"),
            "prefill_proxy_tps_p50": _p(perf.get("prefill_proxy_tps"), "p50"),
            "gap_p95_ms_p50": _ms(_p(perf.get("gap_p95_s"), "p50")),
            "token_timing": ",".join(perf.get("token_timing") or []),
            "out_tokens_method": ",".join(perf.get("output_tokens_methods") or []),
            "input_tokens_actual_p50": _p(perf.get("input_tokens"), "p50"),
            "output_tokens_actual_p50": _p(perf.get("output_tokens"), "p50"),
            "cached_tokens_max": perf.get("cached_tokens_max"),
            "prompt_tokens_match": (s["prompt_token_check"] or {}).get("all_equal"),
            "prompt_tokens_diff": f"{s['prompt_token_check'].get('min_diff')}..{s['prompt_token_check'].get('max_diff')}"
            if s["prompt_token_check"].get("n") else None,
            "reasoning_chars_total": perf.get("reasoning_chars_total"),
        })
        if w.get("kind") == "application":
            try:
                _, test = load_app_items(Workload.model_validate(w))
                q = aggregate_scores(score_run(records, events, test))
                if "gec/en" in q or "style/en" in q:
                    row.update({
                        "quality_composite": q["composite"]["primary_mean"],
                        "gec_en_f05": _p(q.get("gec/en"), "f0_5"), "gec_en_precision": _p(q.get("gec/en"), "precision"),
                        "gec_en_recall": _p(q.get("gec/en"), "recall"), "gec_en_keep": _p(q.get("gec/en"), "keep_rate"),
                        "style_en_pass": _p(q.get("style/en"), "pass_rate"),
                    })
                    for key, v in q.items():  # any further languages present (only in private reports)
                        task, _, lang = key.partition("/")
                        if lang and lang != "en" and task in ("gec", "style"):
                            row[f"{task}_{lang}_primary"] = v.get("primary")
                    return row
                row.update({
                    "quality_composite": q["composite"]["primary_mean"],
                    "extraction_field_acc": _p(q.get("extraction/all"), "field_accuracy"),
                    "extraction_exact": _p(q.get("extraction/all"), "exact_all"),
                    "extraction_null_acc": _p(q.get("extraction/all"), "null_handling_accuracy"),
                    "extraction_schema_valid": _p(q.get("extraction/all"), "schema_valid"),
                    "routing_macro_f1": _p(q.get("routing/all"), "macro_f1"),
                    "routing_unnecessary_calls": _p(q.get("routing/all"), "unnecessary_tool_calls"),
                    "cv_field_acc": _p(q.get("cv_report/all"), "field_accuracy"),
                    "cv_exact": _p(q.get("cv_report/all"), "exact_all"),
                    "parse_strict": _mean([_p(q.get(f"{t}/all"), "parse_strict") for t in ("extraction", "routing", "cv_report")]),
                    "quality_en": _mean([_p(q.get(f"{t}/en"), "primary") for t in ("extraction", "routing", "cv_report")]),
                })
                for lang in sorted({k.split("/")[1] for k in q if "/" in k} - {"en", "all"}):
                    row[f"quality_{lang}"] = _mean([_p(q.get(f"{t}/{lang}"), "primary") for t in ("extraction", "routing", "cv_report")])
            except Exception as e:
                row["quality_error"] = repr(e)
    return row


def _llama_breakdown_from_log(run: Path) -> dict[str, Any]:
    import gzip
    import tempfile

    from .backends.llama_cpp import LlamaCppBackend

    gz = run / "server.log.gz"
    if not gz.exists():
        return {}
    with tempfile.NamedTemporaryFile("wb", suffix=".log", delete=True) as tmp:
        tmp.write(gzip.decompress(gz.read_bytes()))
        tmp.flush()
        return LlamaCppBackend().final_log_facts(Path(tmp.name)).get("gpu_memory_breakdown_mib") or {}


def _rel(p: Path) -> str:
    """Repo-relative path for published tables (never an absolute local path)."""
    try:
        return str(p.resolve().relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return p.name


def _mean(vals: list[Any]) -> float | None:
    v = [x for x in vals if x is not None]
    return sum(v) / len(v) if v else None


def _gib(v: Any) -> float | None:
    return round(v / 2**30, 3) if isinstance(v, int | float) else None


def _ms(v: Any) -> float | None:
    return v * 1000 if isinstance(v, int | float) else None


def latest_per_cell(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    best: dict[tuple[str, str], dict[str, Any]] = {}
    for r in rows:
        k = (r["experiment"], r["cell_id"])
        if k not in best or r["run_id"] > best[k]["run_id"]:
            best[k] = r
    return sorted(best.values(), key=lambda r: (r["experiment"] or "", r["cell_id"] or ""))


def _fmt(v: Any, nd: int = 2) -> str:
    if v is None or v == "":
        return "–"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


PERF_COLS = [
    ("experiment", "experiment"), ("backend", "backend"), ("artifact", "artifact"), ("quant", "quant"),
    ("hardware", "hardware"), ("workload", "workload"), ("concurrency", "C"), ("state", "state"),
    ("n_ok", "n ok"), ("ttft_p50_s", "TTFT p50 s"), ("ttft_p95_s", "TTFT p95 s"), ("tpot_p50_ms", "TPOT p50 ms"),
    ("e2e_p50_s", "E2E p50 s"), ("decode_tps_p50", "decode tok/s/req p50"), ("agg_output_tps", "agg out tok/s"),
    ("native_prefill_tps_p50", "native prefill tok/s"), ("gpu_mem_delta_gib", "GPU mem Δ GiB"),
    ("peak_rss_gib", "peak RSS GiB"), ("mlx_peak_gib", "MLX peak GiB"), ("llama_gpu_self_mib", "llama.cpp GPU self MiB"),
    ("load_first_gen_s", "ready s"), ("early_stops", "early stops"), ("token_timing", "timing basis"),
    ("mem_contaminated", "memory-pressure flag"),
]
QUALITY_COLS = [
    ("backend", "backend"), ("artifact", "artifact"), ("quant", "quant"), ("state", "state"), ("n_ok", "n"),
    ("quality_composite", "composite"), ("extraction_field_acc", "extract field acc"),
    ("extraction_null_acc", "null handling"), ("routing_macro_f1", "routing macro-F1"),
    ("routing_unnecessary_calls", "unneeded tool calls"), ("cv_field_acc", "CV field acc"),
    ("parse_strict", "strict JSON"), ("e2e_p50_s", "E2E p50 s"),
    ("gpu_mem_delta_gib", "GPU mem Δ GiB"), ("mem_contaminated", "memory-pressure flag"),
]


def markdown_tables(rows: list[dict[str, Any]]) -> str:
    latest = latest_per_cell(rows)
    perf = [r for r in latest if r.get("kind") == "performance"]
    qual = [r for r in latest if r.get("kind") == "quality"]
    out = ["# Results tables", "",
           "Generated from saved observations by `llm-sheet report`. Latest run per cell. "
           "p95 is shown for every run but is only *reportable* at n ≥ 200 (see methodology). "
           "Rows marked SYNTHETIC come from the mock server and are not model performance. "
           "`memory-pressure flag = yes` means pressure rose above normal or >256 MiB was swapped during "
           "the run: those timings may include paging and are not headline-quality.", ""]

    def table(rs: list[dict[str, Any]], cols: list[tuple[str, str]]) -> None:
        out.append("| " + " | ".join(h for _, h in cols) + " |")
        out.append("|" + "|".join("---" for _ in cols) + "|")
        for r in rs:
            cells = []
            for k, _ in cols:
                v = r.get(k)
                if k == "ttft_p95_s" and v is not None and not r.get("ttft_p95_reportable"):
                    cells.append(f"({_fmt(v)})")
                elif k == "state" and r.get("synthetic"):
                    cells.append("SYNTHETIC")
                else:
                    cells.append(_fmt(v))
            out.append("| " + " | ".join(cells) + " |")
        out.append("")

    if perf:
        out += ["## Performance", "", "TTFT p95 in parentheses: n < 200, exploratory only.", ""]
        table(perf, PERF_COLS)
    if qual:
        out += ["## Application quality", ""]
        table(qual, QUALITY_COLS)
    return "\n".join(out)


def build_report(roots: list[Path], out_dir: Path | None = None, include_synthetic: bool = False) -> Path:
    rows = collect_runs(roots, include_synthetic)
    out_dir = out_dir or roots[0] / "report"
    out_dir.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with (out_dir / "results.csv").open("w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=keys)
        wr.writeheader()
        for r in rows:
            wr.writerow(r)
    (out_dir / "results.md").write_text(markdown_tables(rows))
    (out_dir / "index.html").write_text(render_html(rows))
    return out_dir / "index.html"


# ---------------------------------------------------------------------------
def render_html(rows: list[dict[str, Any]]) -> str:
    data = json.dumps(latest_per_cell(rows), default=str).replace("</", "<\\/")
    return _HTML.replace("__DATA__", data).replace("__N_RUNS__", str(len(rows))).replace(
        "__PERF_COLS__", json.dumps(PERF_COLS)).replace("__QUAL_COLS__", json.dumps(QUALITY_COLS)).replace(
        "__ESC__", html.escape(""))


_HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>LLM Inference Sheet</title>
<style>
:root{color-scheme:light;--surface-0:#f4f3f0;--surface-1:#fcfcfb;--border:#dddcd6;--grid:#ebeae6;
--text-primary:#0b0b0b;--text-secondary:#52514e;--text-muted:#7b7a75;
--s1:#2a78d6;--s2:#eb6834;--s3:#1baf7a;--s4:#eda100;--s5:#e87ba4;--s6:#008300;--s7:#4a3aa7;--s8:#e34948;
--warn-bg:#fff4d6;--warn-fg:#6b4a00}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){color-scheme:dark;--surface-0:#121211;--surface-1:#1a1a19;
--border:#383835;--grid:#2a2a28;--text-primary:#fff;--text-secondary:#c3c2b7;--text-muted:#8f8e86;
--s1:#3987e5;--s2:#d95926;--s3:#199e70;--s4:#c98500;--s5:#d55181;--s6:#008300;--s7:#9085e9;--s8:#e66767;
--warn-bg:#3a2f12;--warn-fg:#f3d58a}}
:root[data-theme="dark"]{color-scheme:dark;--surface-0:#121211;--surface-1:#1a1a19;--border:#383835;--grid:#2a2a28;
--text-primary:#fff;--text-secondary:#c3c2b7;--text-muted:#8f8e86;--s1:#3987e5;--s2:#d95926;--s3:#199e70;--s4:#c98500;
--s5:#d55181;--s6:#008300;--s7:#9085e9;--s8:#e66767;--warn-bg:#3a2f12;--warn-fg:#f3d58a}
*{box-sizing:border-box}
body{margin:0;background:var(--surface-0);color:var(--text-primary);font:14px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",Inter,sans-serif}
main{max-width:1280px;margin:0 auto;padding:24px 16px 64px}
h1{font-size:22px;margin:0 0 4px}h2{font-size:17px;margin:32px 0 8px}
.sub{color:var(--text-secondary);margin:0 0 16px}
.note{background:var(--warn-bg);color:var(--warn-fg);border-radius:8px;padding:10px 12px;margin:12px 0}
.filters{display:flex;flex-wrap:wrap;gap:8px 16px;align-items:end;padding:12px;background:var(--surface-1);border:1px solid var(--border);border-radius:10px;position:sticky;top:0;z-index:5}
.filters label{display:flex;flex-direction:column;font-size:12px;color:var(--text-secondary);gap:2px}
select{font:inherit;padding:4px 6px;border-radius:6px;border:1px solid var(--border);background:var(--surface-1);color:var(--text-primary);min-width:130px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,560px),1fr));gap:16px}
.card{background:var(--surface-1);border:1px solid var(--border);border-radius:10px;padding:14px;min-width:0}
.card h3{margin:0 0 2px;font-size:14px}.card p{margin:0 0 8px;color:var(--text-muted);font-size:12px}
.legend{display:flex;flex-wrap:wrap;gap:4px 14px;font-size:12px;color:var(--text-secondary);margin-bottom:6px}
.legend i{display:inline-block;width:10px;height:10px;border-radius:3px;margin-right:5px;vertical-align:-1px}
svg{display:block;width:100%;height:auto;overflow:visible}
svg text{fill:var(--text-muted);font-size:11px}
.tablewrap{overflow-x:auto;background:var(--surface-1);border:1px solid var(--border);border-radius:10px}
table{border-collapse:collapse;width:100%;font-size:12.5px;font-variant-numeric:tabular-nums}
th,td{padding:6px 8px;border-bottom:1px solid var(--grid);text-align:right;white-space:nowrap}
th{position:sticky;top:0;background:var(--surface-1);color:var(--text-secondary);font-weight:600;cursor:pointer}
td:nth-child(-n+4),th:nth-child(-n+4){text-align:left}
tr.state-bad td{color:var(--text-muted)}
.muted{color:var(--text-muted)}
#tip{position:fixed;pointer-events:none;background:var(--surface-1);color:var(--text-primary);border:1px solid var(--border);
border-radius:8px;padding:8px 10px;font-size:12px;box-shadow:0 4px 16px rgba(0,0,0,.18);display:none;z-index:20;max-width:320px}
.empty{color:var(--text-muted);padding:28px;text-align:center}
</style></head><body><main>
<h1>LLM Local Inference Sheet</h1>
<p class="sub">Regenerated from __N_RUNS__ saved runs (latest run per cell shown). Every value is recomputed from raw stream events.</p>
<div class="note">Read with the methodology: TTFT is client-observed first-content latency; TPOT is <em>chunk-derived</em> unless the timing basis says <code>token_level</code>. p95 is exploratory below n=200. GGUF Q4_K_M and MLX 4-bit are different quantization schemes.</div>
<div class="filters" id="filters"></div>
<h2>Performance</h2>
<div class="grid">
<div class="card"><h3>Time to first content vs concurrency</h3><p>p50 seconds, closed loop. Lower is better.</p><div class="legend" id="lg-ttft"></div><div id="ch-ttft"></div></div>
<div class="card"><h3>Aggregate output throughput vs concurrency</h3><p>output tokens / measurement window. Higher is better.</p><div class="legend" id="lg-tps"></div><div id="ch-tps"></div></div>
<div class="card"><h3>GPU memory vs context per request</h3><p>GiB, concurrency 1. System-wide AGX in-use memory minus pre-launch baseline</p><div class="legend" id="lg-mem"></div><div id="ch-mem"></div></div>
<div class="card"><h3>Quality vs latency</h3><p>application suite composite vs end-to-end p50 seconds</p><div class="legend" id="lg-q"></div><div id="ch-q"></div></div>
</div>
<h2>Performance table</h2><div class="tablewrap"><table id="t-perf"></table></div>
<h2>Quality table</h2><div class="tablewrap"><table id="t-qual"></table></div>
<div id="tip"></div>
</main>
<script>
const DATA = __DATA__;
const PERF_COLS = __PERF_COLS__, QUAL_COLS = __QUAL_COLS__;
const BACKEND_SLOT = {"llama-cpp":1,"mlx-lm":2,"ollama":3,"mock":8};
const col = b => `var(--s${BACKEND_SLOT[b]||8})`;
const FILTERS = [["model","model"],["backend","backend"],["hardware","hardware"],["quant","quantization"],["workload","workload / context"],["experiment","experiment"]];
const state = {};
const fbox = document.getElementById('filters');
for (const [k,label] of FILTERS){
  const vals=[...new Set(DATA.map(r=>r[k]).filter(v=>v!=null))].sort();
  const l=document.createElement('label'); l.textContent=label;
  const s=document.createElement('select'); s.innerHTML='<option value="">all</option>'+vals.map(v=>`<option>${esc(v)}</option>`).join('');
  s.onchange=()=>{state[k]=s.value;render()}; l.appendChild(s); fbox.appendChild(l);
}
function esc(v){return String(v).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]))}
function rows(){return DATA.filter(r=>FILTERS.every(([k])=>!state[k]||String(r[k])===state[k]))}
const tip=document.getElementById('tip');
function showTip(e,h){tip.innerHTML=h;tip.style.display='block';const x=Math.min(e.clientX+14,innerWidth-340);tip.style.left=x+'px';tip.style.top=(e.clientY+14)+'px'}
function hideTip(){tip.style.display='none'}
function fmt(v,nd=2){if(v==null||v==='')return '–';if(typeof v==='boolean')return v?'yes':'no';if(typeof v==='number')return Number.isInteger(v)?String(v):v.toFixed(nd);return String(v)}

function lineChart(el, lg, rs, xk, yk, {xlog=true, ylabel=''}={}) {
  const ok = rs.filter(r=>r[xk]!=null&&r[yk]!=null&&(r.state==='measured'||r.state==='synthetic'));
  const box=document.getElementById(el), leg=document.getElementById(lg);
  if(!ok.length){box.innerHTML='<div class="empty">No measured data for this selection</div>';leg.innerHTML='';return}
  const key = r=>`${r.backend} · ${r.artifact}${r.workload?' · '+r.workload:''}`;
  const series={}; ok.forEach(r=>{(series[key(r)] ||= {b:r.backend,pts:[]}).pts.push(r)});
  const names=Object.keys(series).sort();
  const W=560,H=260,m={l:48,r:12,t:10,b:30};
  const xs=[...new Set(ok.map(r=>r[xk]))].sort((a,b)=>a-b);
  const xmin=Math.min(...xs),xmax=Math.max(...xs), ymax=Math.max(...ok.map(r=>r[yk]))*1.1||1;
  const X=v=> xs.length===1? (m.l+(W-m.l-m.r)/2) : m.l+(W-m.l-m.r)*(xlog?(Math.log2(v)-Math.log2(xmin))/(Math.log2(xmax)-Math.log2(xmin)||1):(v-xmin)/(xmax-xmin||1));
  const Y=v=>H-m.b-(H-m.t-m.b)*v/ymax;
  let s=`<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(yk)} by ${esc(xk)}">`;
  for(let i=0;i<=4;i++){const v=ymax*i/4;s+=`<line x1="${m.l}" x2="${W-m.r}" y1="${Y(v)}" y2="${Y(v)}" stroke="var(--grid)"/><text x="${m.l-6}" y="${Y(v)+4}" text-anchor="end">${fmt(v, v<10?2:0)}</text>`}
  xs.forEach(v=>{s+=`<text x="${X(v)}" y="${H-m.b+16}" text-anchor="middle">${v}</text>`});
  s+=`<text x="${W-m.r}" y="${H-4}" text-anchor="end">${esc(xk)}</text>`;
  names.forEach((n,i)=>{const sr=series[n]; const pts=sr.pts.sort((a,b)=>a[xk]-b[xk]);
    const dash = i>=3? 'stroke-dasharray="5 3"':'';
    s+=`<polyline fill="none" stroke="${col(sr.b)}" stroke-width="2" ${dash} points="${pts.map(p=>X(p[xk])+','+Y(p[yk])).join(' ')}"/>`;
    pts.forEach(p=>{s+=`<circle cx="${X(p[xk])}" cy="${Y(p[yk])}" r="4.5" fill="${col(sr.b)}" stroke="var(--surface-1)" stroke-width="2"/>`+
      `<circle class="hit" data-n="${esc(n)}" data-x="${p[xk]}" data-y="${p[yk]}" data-extra="${esc('n='+fmt(p.n_ok)+(p.synthetic?' · SYNTHETIC':''))}" cx="${X(p[xk])}" cy="${Y(p[yk])}" r="12" fill="transparent"/>`})});
  s+='</svg>'; box.innerHTML=s;
  leg.innerHTML=names.map((n,i)=>`<span><i style="background:${col(series[n].b)}"></i>${esc(n)}${i>=3?' (dashed)':''}</span>`).join('');
  box.querySelectorAll('.hit').forEach(c=>{c.onmousemove=e=>showTip(e,`<b>${c.dataset.n}</b><br>${xk} = ${c.dataset.x}<br>${yk} = ${fmt(+c.dataset.y,3)}<br><span class="muted">${c.dataset.extra}</span>`);c.onmouseleave=hideTip});
}

function scatter(el, lg, rs, xk, yk){
  const ok=rs.filter(r=>r[xk]!=null&&r[yk]!=null);
  const box=document.getElementById(el), leg=document.getElementById(lg);
  if(!ok.length){box.innerHTML='<div class="empty">No quality runs for this selection</div>';leg.innerHTML='';return}
  const W=560,H=260,m={l:48,r:12,t:10,b:30};
  const xmax=Math.max(...ok.map(r=>r[xk]))*1.15||1;
  const X=v=>m.l+(W-m.l-m.r)*v/xmax, Y=v=>H-m.b-(H-m.t-m.b)*v;
  let s=`<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="quality vs latency">`;
  for(let i=0;i<=4;i++){const v=i/4;s+=`<line x1="${m.l}" x2="${W-m.r}" y1="${Y(v)}" y2="${Y(v)}" stroke="var(--grid)"/><text x="${m.l-6}" y="${Y(v)+4}" text-anchor="end">${v.toFixed(2)}</text>`}
  for(let i=0;i<=4;i++){const v=xmax*i/4;s+=`<text x="${X(v)}" y="${H-m.b+16}" text-anchor="middle">${v.toFixed(1)}</text>`}
  s+=`<text x="${W-m.r}" y="${H-4}" text-anchor="end">${esc(xk)}</text>`;
  ok.forEach(r=>{s+=`<circle cx="${X(r[xk])}" cy="${Y(r[yk])}" r="6" fill="${col(r.backend)}" stroke="var(--surface-1)" stroke-width="2"/>`+
    `<text x="${X(r[xk])+9}" y="${Y(r[yk])+4}" style="fill:var(--text-secondary)">${esc(r.artifact)}</text>`+
    `<circle class="hit" data-t="${esc(r.backend+' · '+r.artifact)}" data-x="${r[xk]}" data-y="${r[yk]}" cx="${X(r[xk])}" cy="${Y(r[yk])}" r="13" fill="transparent"/>`});
  s+='</svg>'; box.innerHTML=s;
  const bs=[...new Set(ok.map(r=>r.backend))].sort();
  leg.innerHTML=bs.map(b=>`<span><i style="background:${col(b)}"></i>${esc(b)}</span>`).join('');
  box.querySelectorAll('.hit').forEach(c=>{c.onmousemove=e=>showTip(e,`<b>${c.dataset.t}</b><br>composite = ${fmt(+c.dataset.y,3)}<br>E2E p50 = ${fmt(+c.dataset.x,2)} s`);c.onmouseleave=hideTip});
}

function table(id, rs, cols){
  const t=document.getElementById(id);
  if(!rs.length){t.innerHTML='<tr><td class="empty">No rows for this selection</td></tr>';return}
  t.innerHTML='<thead><tr>'+cols.map(([k,h])=>`<th data-k="${k}">${esc(h)}</th>`).join('')+'</tr></thead><tbody>'+
   rs.map(r=>`<tr class="${['measured','synthetic'].includes(r.state)?'':'state-bad'}" title="${esc(r.reason||'')}">`+cols.map(([k])=>{
     let v=r[k]; if(k==='state'&&r.synthetic) v='SYNTHETIC';
     if(k==='ttft_p95_s'&&v!=null&&!r.ttft_p95_reportable) return `<td class="muted">(${fmt(v)})</td>`;
     return `<td>${esc(fmt(v))}</td>`}).join('')+'</tr>').join('')+'</tbody>';
  t.querySelectorAll('th').forEach(th=>th.onclick=()=>{const k=th.dataset.k;const asc=th.dataset.asc!=='1';
    rs.sort((a,b)=>(a[k]??-1e18)>(b[k]??-1e18)?(asc?1:-1):-(asc?1:-1)); table(id,rs,cols);
    const nth=t.querySelector(`th[data-k="${k}"]`); if(nth) nth.dataset.asc=asc?'1':'0'});
}

function render(){
  const rs=rows();
  const perf=rs.filter(r=>r.kind==='performance'), qual=rs.filter(r=>r.kind==='quality');
  lineChart('ch-ttft','lg-ttft',perf,'concurrency','ttft_p50_s');
  lineChart('ch-tps','lg-tps',perf,'concurrency','agg_output_tps');
  const mem=perf.map(r=>({...r,ctx:(r.input_tokens||0)+(r.output_tokens||0),mem:r.gpu_mem_delta_gib}));
  lineChart('ch-mem','lg-mem',mem.filter(r=>r.concurrency===1||!r.concurrency).map(r=>({...r,workload:null})),'ctx','mem',{xlog:true});
  scatter('ch-q','lg-q',qual,'e2e_p50_s','quality_composite');
  table('t-perf',perf,PERF_COLS); table('t-qual',qual,QUAL_COLS);
}
render();
</script></body></html>
"""

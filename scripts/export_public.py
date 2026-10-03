#!/usr/bin/env python3
"""Export local results (results/, git-ignored) into the publishable results-public/.

    python scripts/export_public.py            # rebuilds results-public/ from results/

What the export does, per finished run:
  * application-quality runs: keeps only items whose language is in PUBLIC_LANGS; requests, raw events and
    scores of all other items are dropped, and summaries are recomputed from what remains;
  * performance runs: generated text in raw events is replaced by a same-length placeholder ("x"). The prompts
    are synthetic filler, the text carries no information, and lengths/timings/token counts are unchanged,
    so every published metric recomputes exactly;
  * absolute local paths are rewritten (<repo>, ~, /Users/<user>);
  * server-log lines containing characters outside the public scripts are dropped.
It then scans every exported file and fails if a character from a blocked Unicode block remains.

Git tracks only the compact layer of results-public/ (manifests, configs, summaries, report, dashboard);
the raw files (RAW_FILES, git-ignored) stay local. The full tree is packed into BUNDLE for a GitHub Release:
download it and extract it over the repo root to recompute every metric from raw events.
"""

from __future__ import annotations

import gzip
import json
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from llm_local_inference_sheet.bench import score_run, summarize_run  # noqa: E402
from llm_local_inference_sheet.config import Workload  # noqa: E402
from llm_local_inference_sheet.evaluation import aggregate_scores  # noqa: E402
from llm_local_inference_sheet.manifests import load_jsonl  # noqa: E402

SRC = ROOT / "results"
DST = ROOT / "results-public"
PUBLIC_LANGS = {"en"}
SKIP_DIRS = {"_scratch", "report", "dashboard"}
# Raw per-run files: git-ignored inside results-public/ and published only in BUNDLE.
RAW_FILES = ("requests.jsonl", "events.jsonl.gz", "telemetry.jsonl", "mlx_memory.jsonl", "server.log.gz", "scores.jsonl")
BUNDLE = ROOT / "dist" / "results-public-full.tar.gz"
BLOCKED = re.compile("[\u0400-\u052F\u2DE0-\u2DFF\uA640-\uA69F]")  # Unicode blocks that must not be published
HOME = str(Path.home())


def scrub(text: str) -> str:
    text = text.replace(str(ROOT), "<repo>").replace(HOME, "~")
    return re.sub(r"/Users/[^/\s\"']+", "/Users/<user>", text)


def item_langs() -> dict[str, str]:
    """id -> language for every dataset item available locally (private copies include all languages)."""
    out: dict[str, str] = {}
    for base in (ROOT / "datasets", ROOT / ".cache" / "datasets", ROOT / "private"):
        for path in base.rglob("*.jsonl") if base.exists() else []:
            for line in path.open():
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(d, dict) and "id" in d and "lang" in d:
                    out[d["id"]] = d["lang"]
    return out


def public_items(workload: dict, langs: dict[str, str]) -> list[dict]:
    """Test items (with expected answers) for scoring, from the public dataset location only."""
    from llm_local_inference_sheet.bench import load_app_items

    _, test = load_app_items(Workload.model_validate(workload))
    return [it for it in test if it.get("lang") in PUBLIC_LANGS]


def redact_events(events: list[list], protocol: str) -> list[list]:
    out = []
    for t, payload in events:
        try:
            obj = json.loads(payload)
        except (json.JSONDecodeError, TypeError):
            out.append([t, payload])
            continue
        if protocol == "openai" and isinstance(obj, dict):
            for ch in obj.get("choices") or []:
                d = ch.get("delta") or ch.get("message") or {}
                for k in ("content", "reasoning_content", "reasoning"):
                    if isinstance(d.get(k), str):
                        d[k] = "x" * len(d[k])
                if isinstance(ch.get("text"), str):
                    ch["text"] = "x" * len(ch["text"])
        elif protocol == "ollama" and isinstance(obj, dict):
            for k in ("response", "thinking"):
                if isinstance(obj.get(k), str):
                    obj[k] = "x" * len(obj[k])
            msg = obj.get("message") or {}
            for k in ("content", "thinking"):
                if isinstance(msg.get(k), str):
                    msg[k] = "x" * len(msg[k])
        out.append([t, json.dumps(obj, ensure_ascii=False, separators=(",", ":"))])
    return out


def write_jsonl(path: Path, rows: list[dict], gz: bool = False) -> None:
    opener = gzip.open if gz else open
    with opener(path, "wt", encoding="utf-8") as f:  # type: ignore[operator]
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def export_run(run: Path, dst: Path, langs: dict[str, str]) -> dict:
    man = json.loads(run.read_text() if run.is_file() else (run / "manifest.json").read_text())
    dst.mkdir(parents=True, exist_ok=True)
    stats = {"run": str(run.relative_to(SRC)), "dropped_requests": 0, "redacted": False}
    kind = (man.get("workload") or {}).get("kind")
    records = load_jsonl(run / "requests.jsonl")
    events = {e["request_id"]: e["events"] for e in load_jsonl(run / "events.jsonl.gz")}
    if records:
        if kind == "application":
            keep = [r for r in records if langs.get(r.get("item_id"), "en") in PUBLIC_LANGS]
            stats["dropped_requests"] = len(records) - len(keep)
            records = keep
            events = {r["request_id"]: events.get(r["request_id"], []) for r in records}
        else:
            events = {rid: redact_events(ev, next((r["protocol"] for r in records if r["request_id"] == rid), "openai"))
                      for rid, ev in events.items()}
            stats["redacted"] = True
        # mlx_lm.server echoes the absolute model path in every chunk: scrub payloads too.
        events = {rid: [[t, scrub(pl) if isinstance(pl, str) else pl] for t, pl in ev] for rid, ev in events.items()}
        write_jsonl(dst / "requests.jsonl", records)
        write_jsonl(dst / "events.jsonl.gz", [{"request_id": k, "events": v} for k, v in events.items()], gz=True)
        summary = summarize_run(records, events, None, None)
        summary.pop("per_request", None)  # recomputable from events; keeps the git-tracked summary small
        old = json.loads((run / "summary.json").read_text()) if (run / "summary.json").exists() else {}
        summary["telemetry"] = old.get("telemetry")
        if kind == "application":
            items = public_items(man["workload"], langs)
            scores = score_run(records, events, items)
            write_jsonl(dst / "scores.jsonl", scores)
            summary["quality"] = aggregate_scores(scores)
            summary["public_items"] = len(items)
        (dst / "summary.json").write_text(scrub(json.dumps(summary, indent=2, ensure_ascii=False)))
    man["public_export"] = {
        "languages": sorted(PUBLIC_LANGS),
        "dropped_requests_other_languages": stats["dropped_requests"],
        "generated_text_redacted": stats["redacted"],
        "note": ("Exported by scripts/export_public.py. Performance-run generated text is replaced by same-length "
                 "placeholders; timings, token counts and lengths are unchanged."),
    }
    (dst / "manifest.json").write_text(scrub(json.dumps(man, indent=2, ensure_ascii=False)))
    for name in ("effective_config.json", "workload.json", "telemetry.jsonl", "mlx_memory.jsonl",
                 "embedded_chat_template.jinja", "template_diff.txt", "Modelfile"):
        p = run / name
        if p.exists():
            (dst / name).write_text(scrub(p.read_text(errors="replace")))
    log = run / "server.log.gz"
    if log.exists():
        lines = gzip.decompress(log.read_bytes()).decode("utf-8", "replace").splitlines()
        kept = [scrub(ln) for ln in lines if not BLOCKED.search(ln)]
        with gzip.open(dst / "server.log.gz", "wt", encoding="utf-8") as f:
            f.write("\n".join(kept) + "\n")
        stats["log_lines_dropped"] = len(lines) - len(kept)
    return stats


def verify(root: Path) -> list[str]:
    bad = []
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        data = gzip.decompress(p.read_bytes()) if p.suffix == ".gz" else p.read_bytes()
        text = data.decode("utf-8", "replace")
        why = [w for w, hit in (("blocked characters", BLOCKED.search(text)), ("home path", HOME in text)) if hit]
        if why:
            bad.append(f"{p.relative_to(root)} ({', '.join(why)})")
    return bad


def main() -> None:
    langs = item_langs()
    if DST.exists():
        keep_readme = (DST / "README.md").read_text() if (DST / "README.md").exists() else None
        shutil.rmtree(DST)
        DST.mkdir()
        if keep_readme:
            (DST / "README.md").write_text(keep_readme)
    n_runs, dropped = 0, 0
    for exp in sorted(p for p in SRC.iterdir() if p.is_dir() and p.name not in SKIP_DIRS):
        if exp.name == "investigations":
            for f in exp.rglob("*"):
                if f.is_file():
                    out = DST / f.relative_to(SRC)
                    out.parent.mkdir(parents=True, exist_ok=True)
                    out.write_text(scrub(f.read_text(errors="replace")))
            continue
        if (exp / "experiment.json").exists():
            (DST / exp.name).mkdir(parents=True, exist_ok=True)
            ej = json.loads((exp / "experiment.json").read_text())
            ej.get("config", {}).pop("description", None)  # free text from older configs; settings + identity stay
            (DST / exp.name / "experiment.json").write_text(scrub(json.dumps(ej, indent=2, ensure_ascii=False)))
        for man in sorted(exp.rglob("manifest.json")):
            m = json.loads(man.read_text())
            if not m.get("finished") or m.get("synthetic"):
                continue
            st = export_run(man.parent, DST / man.parent.relative_to(SRC), langs)
            n_runs += 1
            dropped += st["dropped_requests"]
    bad = verify(DST)
    print(f"exported {n_runs} runs to {DST}; dropped {dropped} non-public requests")
    if bad:
        print("BLOCKED CONTENT REMAINS in:", *bad[:20], sep="\n  ")
        sys.exit(1)
    print("verification passed: no blocked characters and no home-directory paths in results-public/")
    # Regenerate the public report, compatibility matrix and dashboard from the exported data only.
    import subprocess

    exps = [str(p) for p in sorted(DST.iterdir()) if p.is_dir() and (p / "cells").exists()]
    subprocess.run([sys.executable, "-m", "llm_local_inference_sheet.cli", "report", "--runs", *exps,
                    "--out", str(DST / "report")], check=True, cwd=ROOT)
    subprocess.run([sys.executable, str(ROOT / "scripts" / "gen_compat.py")], check=True, cwd=ROOT)
    subprocess.run([sys.executable, str(ROOT / "scripts" / "build_dashboard.py")], check=True, cwd=ROOT)
    bad = verify(DST)
    if bad:
        print("BLOCKED CONTENT IN GENERATED REPORTS:", *bad[:20], sep="\n  ")
        sys.exit(1)
    write_bundle()


def write_bundle() -> None:
    import hashlib
    import tarfile

    BUNDLE.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(BUNDLE, "w:gz") as tar:
        tar.add(DST, arcname=DST.name)
    digest = hashlib.sha256(BUNDLE.read_bytes()).hexdigest()
    (BUNDLE.parent / (BUNDLE.name + ".sha256")).write_text(f"{digest}  {BUNDLE.name}\n")
    size_mib = BUNDLE.stat().st_size / 2**20
    print(f"wrote {BUNDLE} ({size_mib:.1f} MiB, sha256 {digest[:12]}…); attach it to a GitHub Release")


if __name__ == "__main__":
    main()

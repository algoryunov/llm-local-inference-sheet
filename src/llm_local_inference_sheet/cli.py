"""``llm-sheet`` command line interface."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

from .config import PROJECT_ROOT, load_experiment, load_registry

RESULTS = Path(os.environ.get("LLM_SHEET_RESULTS") or PROJECT_ROOT / "results")


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        d = {"ts": round(record.created, 3), "level": record.levelname, "logger": record.name, "msg": record.getMessage()}
        if record.exc_info:
            d["exc"] = self.formatException(record.exc_info)
        return json.dumps(d, ensure_ascii=False)


def _setup_logging(json_logs: bool, verbose: bool) -> None:
    h = logging.StreamHandler(sys.stderr)
    h.setFormatter(_JsonFormatter() if json_logs else logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO, handlers=[h], force=True)
    for noisy in ("httpx", "httpcore", "urllib3", "filelock", "huggingface_hub"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def _load_dotenv() -> None:
    env = PROJECT_ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                if v.strip() and k.strip() not in os.environ:
                    os.environ[k.strip()] = v.strip()


def _gib(n: int | None) -> str:
    return "?" if n is None else f"{n / 2**30:.2f} GiB"


# ---------------------------------------------------------------- doctor
def cmd_doctor(args: argparse.Namespace) -> int:
    from .backends import BACKEND_NAMES, get_backend
    from .manifests import host_info
    from .telemetry import system_snapshot

    info = host_info()
    snap = system_snapshot()
    print("== Host")
    mac = info.get("mac")
    if mac:
        print(f"  {mac['chip']} ({mac['model']}), {_gib(mac['memory_bytes'])} unified memory, "
              f"CPU {mac['cpu_cores_logical']} ({mac['cpu_cores_performance']}P+{mac['cpu_cores_efficiency']}E), "
              f"GPU cores {mac['gpu_cores']}, macOS {mac['macos_version']} ({mac['macos_build']})")
        print(f"  power: {mac['power_source']}, low power mode: {mac['low_power_mode']}, {mac['battery_line']}")
        if mac["shell_translated_rosetta"]:
            print("  WARNING: this shell runs under Rosetta (x86_64). Backends are launched with `arch -arm64`.")
        print(f"  memory pressure level: {snap.get('vm_pressure_level')} (1=normal, 2=warn, 4=critical); "
              f"available {_gib(snap['sys_mem_available'])}; swap used {_gib(snap['swap_used'])}")
        if (snap.get("vm_pressure_level") or 1) > 1:
            print("  WARNING: system is already under memory pressure. Close other apps before measuring.")
    if info.get("nvidia"):
        print(f"  NVIDIA: {info['nvidia']}")
    print(f"  python {info['python']} ({info['machine']}), disk free {_gib(info['disk_free_bytes'])}, git {info['git']}")
    print(f"  HF_TOKEN: {'set' if os.environ.get('HF_TOKEN') else 'not set (gated models unavailable)'}")
    print("== Backends")
    for name in BACKEND_NAMES:
        b = get_backend(name)
        av = b.availability()
        arch = f" [{av.binary_archs}]" if av.binary_archs else ""
        print(f"  {name:16s} {'OK ' if av.ok else '-- '} {av.version or ''}{arch}  {'' if av.ok else av.detail}")
    return 0


# ---------------------------------------------------------------- models
def cmd_models(args: argparse.Namespace) -> int:
    from .models import artifact_path, download_artifact, is_downloaded, verify_artifact

    reg = load_registry()
    if args.models_cmd == "list":
        print(f"{'artifact':22s} {'model':34s} {'format':11s} {'quant':16s} {'size':>10s} {'platforms':12s} {'access':7s} local")
        for m in reg.models:
            for a in m.artifacts:
                print(f"{a.key:22s} {m.hf_id:34s} {a.format:11s} {a.quantization.method:16s} "
                      f"{_gib(a.total_bytes):>10s} {','.join(a.platforms):12s} {m.access:7s} "
                      f"{'yes' if is_downloaded(a) else '-'}")
        return 0
    if args.models_cmd == "show":
        m, a = reg.artifact(args.key)
        print(json.dumps({"model": m.model_dump(exclude={"artifacts"}), "artifact": a.model_dump(),
                          "local_path": str(artifact_path(a))}, indent=2))
        return 0
    if args.models_cmd == "download":
        for key in args.keys:
            m, a = reg.artifact(key)
            if m.access == "gated" and not os.environ.get("HF_TOKEN"):
                print(f"{key}: gated ({m.hf_id}); set HF_TOKEN", file=sys.stderr)
                continue
            free = shutil.disk_usage(PROJECT_ROOT).free
            if a.total_bytes and a.total_bytes > free - 5 * 2**30:
                print(f"{key}: needs {_gib(a.total_bytes)}, only {_gib(free)} free (keeping 5 GiB headroom)", file=sys.stderr)
                return 2
            print(f"downloading {key} ({_gib(a.total_bytes)}) @ {a.revision[:10]} ...", flush=True)
            p = download_artifact(a)
            v = verify_artifact(a)
            print(f"  {p}  sha256 {'OK' if v['all_ok'] else 'MISMATCH'}")
            if not v["all_ok"]:
                return 1
        return 0
    if args.models_cmd == "verify":
        rc = 0
        for key in args.keys:
            _, a = reg.artifact(key)
            v = verify_artifact(a, force=args.force)
            print(key, "OK" if v["all_ok"] else "FAIL", json.dumps(v["files"]))
            rc |= 0 if v["all_ok"] else 1
        return rc
    return 1


# ---------------------------------------------------------------- plan / bench / evaluate
def cmd_plan(args: argparse.Namespace) -> int:
    from .bench import plan

    exp = load_experiment(Path(args.config))
    p = plan(exp, load_registry(), Path(args.results))
    if args.json:
        print(json.dumps(p, indent=2))
        return 0
    print(f"experiment {p['experiment']} (identity {p['identity']}): {p['n_cells']} cells, {p['n_runnable']} runnable")
    print(f"{'cell':70s} {'state':12s} {'req':>5s} {'ctx':>6s} {'KV est':>8s}  reason")
    for r in p["cells"]:
        kv = f"{r['kv_cache_estimate_mib_f16']} MiB" if r["kv_cache_estimate_mib_f16"] is not None else "?"
        print(f"{r['cell']:70s} {r['state']:12s} {r['requests']:5d} {r['ctx_per_request']:6d} {kv:>8s}  {r['reason']}")
    print(f"pending requests: {p['total_requests_runnable']}")
    print("storage:")
    for r in p["storage"]["rows"]:
        print(f"  {r['artifact']:22s} {_gib(r['size_bytes'])} downloaded={r['downloaded']} "
              f"ollama_copy={r['ollama_blob_copy']} additional={_gib(r['additional_bytes'])}")
    print(f"  additional total {_gib(p['storage']['additional_bytes_total'])}; disk free {_gib(p['disk_free_bytes'])}")
    print(p["note"])
    return 0


def cmd_bench(args: argparse.Namespace) -> int:
    from .bench import CellRunner

    exp = load_experiment(Path(args.config))
    runner = CellRunner(exp, load_registry(), Path(args.results), allow_download=args.download,
                        retry_failed=args.retry_failed, force=args.force,
                        fill_missing=getattr(args, "fill_missing", False))
    out = runner.run_all(only=args.only)
    for s in out:
        print(f"{s['cell']:75s} {s['state']}{' (skipped: already done)' if s.get('skipped') else ''}")
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    if args.runs:
        return _rescore(Path(args.runs))
    exp = load_experiment(Path(args.config))
    if exp.kind != "quality":
        print("evaluate expects a quality experiment config", file=sys.stderr)
        return 2
    return cmd_bench(args)


def _rescore(root: Path) -> int:
    """Re-score saved application runs from raw events (no inference)."""
    from .bench import load_app_items, merged_run_data, score_run
    from .config import Workload
    from .evaluation import aggregate_scores

    n = 0
    for man_path in sorted(root.rglob("manifest.json")):
        man = json.loads(man_path.read_text())
        if (man.get("workload") or {}).get("kind") != "application" or not man.get("finished") or man.get("fill_of"):
            continue
        run = man_path.parent
        records, events, _ = merged_run_data(run)
        if not records:
            continue
        _, test = load_app_items(Workload.model_validate(man["workload"]))
        scores = score_run(records, events, test)
        with (run / "scores.jsonl").open("w") as f:
            for s in scores:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        summ_path = run / "summary.json"
        summ = json.loads(summ_path.read_text()) if summ_path.exists() else {}
        summ["quality"] = aggregate_scores(scores)
        summ_path.write_text(json.dumps(summ, indent=2, ensure_ascii=False))
        n += 1
        print(f"rescored {run} composite={summ['quality']['composite']['primary_mean']:.3f}")
    print(f"{n} runs rescored")
    return 0


# ---------------------------------------------------------------- serve
def cmd_serve(args: argparse.Namespace) -> int:
    from .backends import get_backend
    from .backends.base import free_port
    from .models import artifact_path, is_downloaded

    reg = load_registry()
    model, art = reg.artifact(args.model)
    b = get_backend(args.backend)
    ok, why = b.supports(model, art)
    if not ok:
        print(why, file=sys.stderr)
        return 2
    if not is_downloaded(art):
        print(f"artifact not downloaded: run `llm-sheet models download {art.key}`", file=sys.stderr)
        return 2
    port = args.port or free_port()
    settings: dict[str, Any] = {"_cache_policy": "no_reuse"}
    spec = b.launch_spec(model, art, artifact_path(art), settings, concurrency=args.concurrency,
                         ctx_per_request=args.ctx, port=port)
    log_path = RESULTS / "_scratch" / f"serve-{args.backend}-{art.key}-{int(time.time())}.log"
    print("launch:", " ".join(spec.argv))
    h = b.start(spec, log_path)

    async def _ready() -> None:
        deadline = time.monotonic() + 900
        await b.wait_health(h, deadline)
        await b.prepare(h, model, art, artifact_path(art))
        t = await b.wait_first_generation(h, deadline, b.probe_payload(b.request_model_name(model, art, artifact_path(art))))
        print(f"ready in {t:.1f}s at {h.base_url}{b.chat_path} (log: {log_path}); Ctrl+C to stop")

    try:
        asyncio.run(_ready())
        h.proc.wait()
    except KeyboardInterrupt:
        pass
    finally:
        print("stopping:", b.stop(h))
    return 0


# ---------------------------------------------------------------- report
def cmd_report(args: argparse.Namespace) -> int:
    from .reports import build_report

    out = build_report([Path(p) for p in args.runs], Path(args.out) if args.out else None,
                       include_synthetic=args.include_synthetic)
    print(f"report written: {out}")
    return 0


def cmd_workloads(args: argparse.Namespace) -> int:
    from .workloads.perf import build_perf_workload

    reg = load_registry()
    m = reg.model(args.model)
    wl = build_perf_workload(m, input_tokens=args.input_tokens, output_tokens=args.output_tokens,
                             n_warmup=args.warmup, n_measured=args.measured)
    print(json.dumps({k: v for k, v in wl.items() if k != "items"}, indent=2))
    return 0


def cmd_mock_server(args: argparse.Namespace) -> int:
    from .mockserver import run

    run(args.host, args.port)
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="llm-sheet", description="Reproducible local LLM serving benchmarks")
    ap.add_argument("--json-logs", action="store_true", help="structured JSON logs on stderr")
    ap.add_argument("-v", "--verbose", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("doctor", help="detect hardware, OS, power, memory pressure and backend availability")

    m = sub.add_parser("models", help="model registry")
    msub = m.add_subparsers(dest="models_cmd", required=True)
    msub.add_parser("list")
    s = msub.add_parser("show")
    s.add_argument("key")
    d = msub.add_parser("download")
    d.add_argument("keys", nargs="+")
    v = msub.add_parser("verify")
    v.add_argument("keys", nargs="+")
    v.add_argument("--force", action="store_true", help="re-hash even if cached")

    p = sub.add_parser("plan", help="validate an experiment and list cells, downloads, storage, requests (no inference)")
    p.add_argument("--config", required=True)
    p.add_argument("--results", default=str(RESULTS))
    p.add_argument("--json", action="store_true")

    for name, helptext in (("bench", "run a performance experiment"), ("evaluate", "run/rescore a quality experiment")):
        b = sub.add_parser(name, help=helptext)
        b.add_argument("--config", required=name == "bench")
        b.add_argument("--results", default=str(RESULTS))
        b.add_argument("--download", action="store_true", help="download missing pinned artifacts")
        b.add_argument("--retry-failed", action="store_true")
        b.add_argument("--force", action="store_true", help="re-run completed cells (keeps previous runs)")
        b.add_argument("--only", help="regex filter on cell ids")
        b.add_argument("--fill-missing", action="store_true",
                       help="run only items a previous run skipped (deadline) and merge them into its results")
        if name == "evaluate":
            b.add_argument("--runs", help="rescore saved runs under this directory (no inference)")

    sv = sub.add_parser("serve", help="launch one backend server in the foreground")
    sv.add_argument("--backend", required=True)
    sv.add_argument("--model", required=True, help="artifact key, e.g. qwen3-8b-q4")
    sv.add_argument("--port", type=int)
    sv.add_argument("--concurrency", type=int, default=1)
    sv.add_argument("--ctx", type=int, default=4096, help="context per request")

    r = sub.add_parser("report", help="regenerate tables/charts from saved observations")
    r.add_argument("--runs", nargs="+", required=True, help="result directories (experiments or parents)")
    r.add_argument("--out", help="output directory (default: <first runs dir>/report)")
    r.add_argument("--include-synthetic", action="store_true", help="include mock/synthetic runs (clearly labelled)")

    w = sub.add_parser("workloads", help="build/inspect a performance workload for a model")
    w.add_argument("--model", required=True)
    w.add_argument("--input-tokens", type=int, default=1024)
    w.add_argument("--output-tokens", type=int, default=256)
    w.add_argument("--warmup", type=int, default=5)
    w.add_argument("--measured", type=int, default=30)

    ms = sub.add_parser("mock-server", help="run the deterministic SYNTHETIC mock server")
    ms.add_argument("--host", default="127.0.0.1")
    ms.add_argument("--port", type=int, default=18080)
    return ap


def main(argv: list[str] | None = None) -> int:
    _load_dotenv()
    args = build_parser().parse_args(argv)
    _setup_logging(args.json_logs, args.verbose)
    handlers = {
        "doctor": cmd_doctor, "models": cmd_models, "plan": cmd_plan, "bench": cmd_bench,
        "evaluate": cmd_evaluate, "serve": cmd_serve, "report": cmd_report, "workloads": cmd_workloads,
        "mock-server": cmd_mock_server,
    }
    return handlers[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())

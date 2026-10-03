"""Experiment planning and execution.

A *cell* is one (backend, artifact, workload, concurrency, repeat) combination.
Each cell gets a fresh server process, so model memory is released between cells
and load time is observed per cell. Cells run strictly sequentially.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import platform
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .backends import Backend, BackendError, ServerHandle, get_backend
from .backends.base import detect_oom, free_port, tail_log
from .client import RequestSpec, Runner
from .config import (
    PROJECT_ROOT,
    Artifact,
    ExperimentConfig,
    ModelEntry,
    Registry,
    ResultState,
    Workload,
    dataset_dir,
    stable_hash,
)
from .evaluation import aggregate_scores, response_text, score_item
from .manifests import RunDir, find_completed_run, host_info, now_run_id
from .metrics import aggregate_cell, compute_request_metrics
from .models import artifact_path, download_artifact, is_downloaded, storage_plan, verify_artifact
from .telemetry import TelemetrySampler, memory_contamination, summarize_telemetry, system_snapshot

log = logging.getLogger(__name__)

MIN_FREE_DISK_BYTES = 5 * 2**30


@dataclass
class Cell:
    backend: str
    artifact_key: str
    model: ModelEntry
    artifact: Artifact
    workload: Workload
    concurrency: int
    repeat: int
    pre_state: str | None = None  # None = runnable; else a ResultState value
    pre_reason: str = ""
    identity: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def cell_id(self) -> str:
        r = f"-r{self.repeat}" if self.repeat else ""
        return f"{self.backend}__{self.artifact_key}__{self.workload.name}__c{self.concurrency}{r}"


def host_platform() -> str:
    if platform.system() == "Darwin":
        return "mac"
    from shutil import which

    return "nvidia" if which("nvidia-smi") else "cpu"


def ctx_required(w: Workload) -> int:
    if w.kind == "synthetic_tokens":
        assert w.input_tokens and w.output_tokens
        return w.input_tokens + w.output_tokens
    return 4096  # application suite: longest item (CV logs) ~1.3k tokens + max_tokens; validated per run


def expand_cells(exp: ExperimentConfig, reg: Registry, *, check_availability: bool = True) -> list[Cell]:
    cells: list[Cell] = []
    avail_cache: dict[str, Any] = {}
    here = host_platform()
    has_token = bool(os.environ.get("HF_TOKEN"))
    for bname in exp.backends:
        backend = get_backend(bname)
        for akey in exp.artifacts:
            model, art = reg.artifact(akey)
            for w in exp.workloads:
                for c in w.concurrency:
                    for rep in range(exp.repeats):
                        cell = Cell(bname, akey, model, art, w, c, rep)
                        ok, why = backend.supports(model, art)
                        compat = art.runtimes.get(bname)
                        if ok and compat is not None and compat.status == "unsupported":
                            ok, why = False, f"registry: {compat.evidence}"
                        if not ok:
                            cell.pre_state, cell.pre_reason = ResultState.UNSUPPORTED, why
                        elif model.access == "gated" and not has_token:
                            cell.pre_state, cell.pre_reason = ResultState.GATED, f"{model.hf_id} is gated; set HF_TOKEN with granted access"
                        elif backend.platform not in ("any", here) and not getattr(backend, "synthetic", False):
                            cell.pre_state, cell.pre_reason = ResultState.UNTESTED, f"needs {backend.platform} host (this host: {here})"
                        elif w.kind == "application" and not (
                                dataset_dir(w.dataset or "") / f"{w.split or 'test'}.jsonl").exists():
                            hint = ("run scripts/prepare_proofread.py" if "proofread" in (w.dataset or "")
                                    else "run scripts/generate_app_dataset.py")
                            cell.pre_state, cell.pre_reason = ResultState.UNTESTED, f"dataset {w.dataset} not prepared: {hint}"
                        elif ctx_required(w) > model.context.native_max:
                            cell.pre_state, cell.pre_reason = ResultState.UNSUPPORTED, "workload exceeds native context"
                        elif check_availability:
                            if bname not in avail_cache:
                                avail_cache[bname] = backend.availability()
                            av = avail_cache[bname]
                            if not av.ok:
                                cell.pre_state, cell.pre_reason = ResultState.UNTESTED, f"backend unavailable: {av.detail}"
                        if getattr(backend, "synthetic", False):
                            cell.extra["synthetic"] = True
                        cell.identity = stable_hash({
                            "cell": cell.cell_id, "experiment": exp.name, "artifact_revision": art.revision,
                            "settings": exp.backend_settings.get(bname, {}), "sampling": exp.sampling.model_dump(),
                            "requests": exp.requests.model_dump(), "thinking": exp.thinking,
                            "length": exp.length_enforcement, "cache": exp.cache_policy,
                            "workload": w.model_dump(), "harness": "v1",
                        })
                        cells.append(cell)
    return cells


def plan(exp: ExperimentConfig, reg: Registry, results_root: Path) -> dict[str, Any]:
    cells = expand_cells(exp, reg)
    runnable = [c for c in cells if c.pre_state is None]
    per_cell = exp.requests.warmup + exp.requests.measured
    rows = []
    for c in cells:
        done = find_completed_run(results_root / exp.name / "cells" / c.cell_id, c.identity)
        n_req = per_cell if c.workload.kind == "synthetic_tokens" else _app_count(c.workload) + exp.requests.warmup
        kv = c.model.arch.kv_bytes_per_token() * ctx_required(c.workload) * c.concurrency if c.model.arch else None
        rows.append({
            "cell": c.cell_id, "state": c.pre_state or ("done" if done else "pending"),
            "reason": c.pre_reason, "requests": n_req, "ctx_per_request": ctx_required(c.workload),
            "kv_cache_estimate_mib_f16": round(kv / 2**20) if kv else None,
            "weights_bytes": c.artifact.total_bytes,
        })
    arts = sorted({c.artifact_key for c in runnable})
    storage = storage_plan(reg, arts, exp.backends)
    import shutil

    return {
        "experiment": exp.name, "identity": exp.identity(), "cells": rows,
        "n_cells": len(cells), "n_runnable": len(runnable),
        "total_requests_runnable": sum(r["requests"] for r in rows if r["state"] == "pending"),
        "storage": storage, "disk_free_bytes": shutil.disk_usage(PROJECT_ROOT).free,
        "note": "KV estimates = 2*layers*kv_heads*head_dim*2B*tokens*concurrency: a hypothesis, not a measurement.",
    }


def _app_count(w: Workload) -> int:
    path = dataset_dir(w.dataset or "") / f"{w.split or 'test'}.jsonl"
    return sum(1 for _ in path.open()) if path.exists() else 0


def load_app_items(w: Workload) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    base = dataset_dir(w.dataset or "")
    test = [json.loads(line) for line in (base / f"{w.split or 'test'}.jsonl").open()]
    dev = [json.loads(line) for line in (base / "dev.jsonl").open()]
    return dev, test


# ---------------------------------------------------------------------------
class CellRunner:
    def __init__(self, exp: ExperimentConfig, reg: Registry, results_root: Path, *,
                 allow_download: bool = False, retry_failed: bool = False, force: bool = False,
                 fill_missing: bool = False) -> None:
        self.exp = exp
        self.reg = reg
        self.root = results_root / exp.name
        self.allow_download = allow_download
        self.retry_failed = retry_failed
        self.force = force
        self.fill_missing = fill_missing
        self._only_items: set[str] | None = None

    def run_all(self, only: str | None = None) -> list[dict[str, Any]]:
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "experiment.json").write_text(json.dumps(
            {"config": self.exp.model_dump(mode="json"), "identity": self.exp.identity()}, indent=2))
        summaries = []
        for cell in expand_cells(self.exp, self.reg):
            if only and not re.search(only, cell.cell_id):
                continue
            summaries.append(self.run_cell(cell))
        return summaries

    def _fill_target(self, cell: Cell, cell_dir: Path) -> tuple[Path, set[str]] | None:
        """Latest finished original run of this cell (same artifact revision + workload) with skipped items."""
        if not cell_dir.exists():
            return None
        for run in sorted(cell_dir.iterdir(), reverse=True):
            m = run / "manifest.json"
            if not m.exists():
                continue
            man = json.loads(m.read_text())
            if man.get("fill_of") or not man.get("finished") or man.get("state") not in (ResultState.MEASURED, ResultState.SYNTHETIC):
                continue
            if man.get("artifact_revision") != cell.artifact.revision or man.get("workload") != cell.workload.model_dump():
                continue
            records, _, _ = merged_run_data(run)
            missing = {r["item_id"] for r in records if r.get("phase") == "measured" and r.get("status") in UNSCORED_STATUSES}
            return (run, missing) if missing else None
        return None

    def run_cell(self, cell: Cell) -> dict[str, Any]:
        cell_dir = self.root / "cells" / cell.cell_id
        if self.fill_missing:
            target = self._fill_target(cell, cell_dir)
            if target is None:
                return {"cell": cell.cell_id, "state": "nothing to fill", "skipped": True}
            self._only_items = target[1]
            log.info("%s: filling %d skipped items of run %s", cell.cell_id, len(target[1]), target[0].name)
            res = self._run_new(cell, cell_dir, fill_of=target[0].name)
            self._only_items = None
            return res
        prev = find_completed_run(cell_dir, cell.identity)
        if prev and not self.force:
            man = json.loads((prev / "manifest.json").read_text())
            if not (self.retry_failed and man["state"] in (ResultState.FAILED, ResultState.TIMEOUT)):
                log.info("skip %s (completed run %s, state=%s)", cell.cell_id, prev.name, man["state"])
                return {"cell": cell.cell_id, "state": man["state"], "skipped": True, "run": str(prev)}
        return self._run_new(cell, cell_dir)

    def _run_new(self, cell: Cell, cell_dir: Path, fill_of: str | None = None) -> dict[str, Any]:
        run = RunDir(cell_dir / now_run_id())
        manifest: dict[str, Any] = {
            "experiment": self.exp.name, "experiment_identity": self.exp.identity(),
            "cell_id": cell.cell_id, "cell_identity": cell.identity, "backend": cell.backend,
            "artifact": cell.artifact_key, "model": cell.model.key, "model_hf_id": cell.model.hf_id,
            "family": cell.model.family, "params_total_b": cell.model.params_total_b,
            "artifact_repo": cell.artifact.repo, "artifact_revision": cell.artifact.revision,
            "artifact_format": cell.artifact.format, "quantization": cell.artifact.quantization.model_dump(),
            "weight_dtype": cell.artifact.weight_dtype, "workload": cell.workload.model_dump(),
            "concurrency": cell.concurrency, "repeat": cell.repeat, "platform": self.exp.platform,
            "hardware_label": self.exp.hardware_label, "kind": self.exp.kind,
            "synthetic": bool(cell.extra.get("synthetic")), "started_wall": time.time(), "finished": False,
            "state": "running", "sampling": self.exp.sampling.model_dump(), "thinking": self.exp.thinking,
            "cache_policy": self.exp.cache_policy,
        }
        if fill_of:
            manifest["fill_of"] = fill_of
            manifest["fill_items"] = len(self._only_items or ())
        run.write_json("manifest.json", manifest)
        if cell.pre_state:
            manifest.update({"state": cell.pre_state, "reason": cell.pre_reason, "finished": True})
            run.write_json("manifest.json", manifest)
            log.info("%s: %s (%s)", cell.cell_id, cell.pre_state, cell.pre_reason)
            return {"cell": cell.cell_id, "state": cell.pre_state}
        manifest["host"] = host_info()
        import shutil

        free = shutil.disk_usage(self.root).free
        if free < MIN_FREE_DISK_BYTES:
            manifest.update({"state": ResultState.FAILED, "finished": True,
                             "reason": f"disk guard: only {free / 2**30:.1f} GiB free (< {MIN_FREE_DISK_BYTES / 2**30:.0f} GiB); "
                                       "swap growth could fill the disk mid-run"})
            run.write_json("manifest.json", manifest)
            return {"cell": cell.cell_id, "state": ResultState.FAILED}
        try:
            result = asyncio.run(self._execute(cell, run, manifest))
        except KeyboardInterrupt:
            manifest.update({"state": ResultState.FAILED, "reason": "interrupted", "finished": False})
            run.write_json("manifest.json", manifest)
            raise
        manifest.update(result)
        manifest["memory_contamination"] = memory_contamination(manifest.get("system_baseline"),
                                                                manifest.get("telemetry_summary"))
        manifest["finished"] = True
        manifest["finished_wall"] = time.time()
        if manifest.get("synthetic") and manifest["state"] == ResultState.MEASURED:
            manifest["state"] = ResultState.SYNTHETIC
        run.write_json("manifest.json", manifest)
        log.info("%s -> %s", cell.cell_id, manifest["state"])
        return {"cell": cell.cell_id, "state": manifest["state"], "run": str(run.path)}

    # ------------------------------------------------------------------
    async def _execute(self, cell: Cell, run: RunDir, manifest: dict[str, Any]) -> dict[str, Any]:
        backend = get_backend(cell.backend)
        av = backend.availability()
        manifest["backend_availability"] = av.__dict__
        synthetic = bool(cell.extra.get("synthetic"))
        # ---- artifact
        a_path = artifact_path(cell.artifact)
        if not synthetic:
            if not is_downloaded(cell.artifact):
                if not self.allow_download:
                    return {"state": ResultState.UNTESTED, "reason": f"artifact not downloaded ({a_path}); use --download"}
                download_artifact(cell.artifact)
            ver = verify_artifact(cell.artifact)
            manifest["artifact_verification"] = ver
            if not ver["all_ok"]:
                return {"state": ResultState.FAILED, "reason": "artifact sha256 mismatch"}
        # ---- workload
        items_w, items_m, wl_meta = self._workload(cell)
        run.write_json("workload.json", wl_meta)
        manifest["workload_sha256"] = wl_meta.get("workload_sha256")
        # ---- launch
        port = self.exp.ports.get(cell.backend) or free_port()
        settings = {**self.exp.backend_settings.get(cell.backend, {}), "_cache_policy": self.exp.cache_policy,
                    "_run_dir": str(run.path)}
        if cell.backend == "mlx-lm":
            settings["_mem_log"] = str(run.path / "mlx_memory.jsonl")
        ctx_req = wl_meta["ctx_required_per_request"]
        spec = backend.launch_spec(cell.model, cell.artifact, a_path, settings, concurrency=cell.concurrency,
                                   ctx_per_request=ctx_req, port=port)
        if "_mem_log" in settings:
            spec.effective["_mem_log"] = settings["_mem_log"]
        model_name = backend.request_model_name(cell.model, cell.artifact, a_path)
        ignore_eos = (self.exp.length_enforcement == "ignore_eos_if_supported" and backend.supports_ignore_eos
                      and cell.workload.kind == "synthetic_tokens")
        effective = {
            "launch_argv": spec.argv, "launch_env": spec.env, "settings": spec.effective,
            "scheduling": backend.scheduling, "request_path": backend.chat_path, "protocol": backend.protocol,
            "model_name": model_name,
            "length_enforcement": "ignore_eos" if ignore_eos else "none (natural EOS; early stops recorded)",
            "cache_policy": self.exp.cache_policy, "thinking": self.exp.thinking,
            "chat_template_kwargs": wl_meta.get("chat_template_kwargs"),
            "load_generation": f"closed-loop, concurrency={cell.concurrency}, no think time",
        }
        run.write_json("effective_config.json", effective)
        baseline = system_snapshot()
        manifest["system_baseline"] = baseline
        log_path = run.path / "server.log"
        h: ServerHandle | None = None
        sampler: TelemetrySampler | None = None
        try:
            h = backend.start(spec, log_path)
            sampler = TelemetrySampler(h.proc.pid, out_path=run.path / "telemetry.jsonl")
            sampler.start()
            deadline = time.monotonic() + self.exp.readiness_timeout_s
            readiness: dict[str, Any] = {}
            readiness["health_ready_s"] = await backend.wait_health(h, deadline)
            sampler.mark("health_ready")
            prep = await backend.prepare(h, cell.model, cell.artifact, a_path)
            if prep:
                readiness["prepare"] = prep
                readiness["prepare_done_s"] = (time.perf_counter_ns() - h.t_launch_ns) / 1e9
            readiness["first_generation_ready_s"] = await backend.wait_first_generation(
                h, deadline, backend.probe_payload(model_name))
            sampler.mark("ready")
            manifest["readiness"] = readiness
            manifest["readiness_note"] = ("launch->health and launch->first 1-token generation; OS file-cache state "
                                          "not controlled (weights may be warm in page cache)")
            meta = await backend.metadata(h)
            manifest["backend_metadata"] = _trim_meta(meta)
            if "embedded_chat_template" in meta:
                (run.path / "embedded_chat_template.jinja").write_text(meta["embedded_chat_template"] or "")
            verification = await self._verify_template(backend, h, cell, items_m, wl_meta)
            manifest["template_verification"] = verification
            # ---- load
            runner = Runner(h.base_url, request_timeout_s=self.exp.request_timeout_s)
            specs_w = [self._spec(backend, cell, it, "warmup", model_name, ignore_eos, wl_meta) for it in items_w]
            specs_m = [self._spec(backend, cell, it, "measured", model_name, ignore_eos, wl_meta) for it in items_m]
            t_start = time.perf_counter_ns()
            sampler.mark("warmup_start")
            res_w = await runner.run_phase(specs_w, cell.concurrency, deadline_s=self.exp.max_cell_duration_s)
            sampler.mark("measured_start")
            remaining = self.exp.max_cell_duration_s - (time.perf_counter_ns() - t_start) / 1e9
            res_m = await runner.run_phase(specs_m, cell.concurrency, deadline_s=max(1.0, remaining))
            sampler.mark("measured_end")
            meta_after = await backend.metadata(h)
            manifest["backend_metadata_after"] = _trim_meta(meta_after)
        except BackendError as e:
            return {"state": e.state, "reason": str(e)[:4000], **self._finish(backend, h, sampler, run, baseline)}
        except Exception as e:
            log.exception("cell failed")
            tail = tail_log(log_path)
            state = ResultState.OUT_OF_MEMORY if detect_oom(tail) else ResultState.FAILED
            return {"state": state, "reason": f"{type(e).__name__}: {e}"[:4000],
                    **self._finish(backend, h, sampler, run, baseline)}
        fin = self._finish(backend, h, sampler, run, baseline)
        # ---- persist
        all_res = res_w + res_m
        records = [r.record for r in all_res]
        run.append_jsonl("requests.jsonl", records)
        run.write_events([(r.record["request_id"], r.events) for r in all_res])
        summary = summarize_run(records, {r.record["request_id"]: r.events for r in all_res}, self.exp, cell.workload)
        summary["telemetry"] = fin["telemetry_summary"]
        if cell.workload.kind == "application":
            scores = score_run(records, {r.record["request_id"]: r.events for r in all_res}, items_m)
            run.append_jsonl("scores.jsonl", scores)
            summary["quality"] = aggregate_scores(scores)
        run.write_json("summary.json", summary)
        perf = summary["performance"]
        if perf["n_ok"] == 0:
            tail = tail_log(log_path)
            state = ResultState.OUT_OF_MEMORY if detect_oom(tail) else ResultState.FAILED
            return {"state": state, "reason": "no successful measured requests", **fin}
        unscored = {k: v for k, v in perf["status_counts"].items() if k in UNSCORED_STATUSES}
        if unscored:
            fin["incomplete"] = {**unscored, "reason": f"max_cell_duration_s={self.exp.max_cell_duration_s} reached; "
                                 "run `bench --fill-missing` to complete the skipped items"}
            log.warning("%s incomplete: %s", cell.cell_id, unscored)
        if perf["status_counts"].get("timeout") and perf["n_ok"] < perf["n_measured"]:
            fin["partial_timeouts"] = perf["status_counts"]["timeout"]
        return {"state": ResultState.MEASURED, **fin}

    # ------------------------------------------------------------------
    def _workload(self, cell: Cell) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
        w = cell.workload
        if w.kind == "synthetic_tokens":
            if cell.extra.get("synthetic"):
                items = [{"item_id": f"{ph}{i}", "phase": ph,
                          "messages": [{"role": "user", "content": f"synthetic prompt {ph} {i}"}],
                          "rendered_prompt": f"synthetic prompt {ph} {i}", "input_tokens": None,
                          "target_output_tokens": w.output_tokens}
                         for ph, n in (("warmup", self.exp.requests.warmup), ("measured", self.exp.requests.measured))
                         for i in range(n)]
                meta = {"synthetic": True, "workload_sha256": stable_hash(items, 64),
                        "ctx_required_per_request": ctx_required(w)}
            else:
                from .workloads.perf import build_perf_workload

                wl = build_perf_workload(cell.model, input_tokens=w.input_tokens or 0, output_tokens=w.output_tokens or 0,
                                         n_warmup=self.exp.requests.warmup, n_measured=self.exp.requests.measured,
                                         thinking=self.exp.thinking)
                items = wl["items"]
                meta = {k: v for k, v in wl.items() if k != "items"}
                meta["items"] = [{k: it[k] for k in ("item_id", "phase", "input_tokens", "template_overhead_tokens")}
                                 for it in items]
                meta["ctx_required_per_request"] = ctx_required(w)
            warm = [i for i in items if i["phase"] == "warmup"]
            meas = [i for i in items if i["phase"] == "measured"]
            return warm, meas, meta
        dev, test = load_app_items(w)
        if self._only_items is not None:
            test = [t for t in test if t["id"] in self._only_items]
        warm = dev[: self.exp.requests.warmup]
        meta = {"dataset": w.dataset, "split": w.split, "n_items": len(test),
                "workload_sha256": stable_hash([t["id"] for t in test], 64),
                "ctx_required_per_request": 4096, "chat_template_kwargs": None}
        if not cell.extra.get("synthetic"):
            from .workloads.perf import TemplateRenderer

            r = TemplateRenderer(cell.model, self.exp.thinking)
            meta["chat_template_kwargs"] = r.kwargs
            meta["template_sha256"] = r.template_sha256
            counts = []
            for it in warm + test:
                it["rendered_prompt"] = r.render(it["messages"])
                it["input_tokens"] = r.count(it["messages"])
                counts.append(it["input_tokens"])
            max_tok = w.max_tokens or 1024
            meta["ctx_required_per_request"] = max(counts) + max_tok
            meta["input_tokens_max"] = max(counts)
        return warm, test, meta

    def _spec(self, backend: Backend, cell: Cell, item: dict[str, Any], phase: str, model_name: str,
              ignore_eos: bool, wl_meta: dict[str, Any]) -> RequestSpec:
        w = cell.workload
        max_tokens = w.output_tokens if w.kind == "synthetic_tokens" else (w.max_tokens or 1024)
        kwargs = wl_meta.get("chat_template_kwargs")  # thinking on/off kwargs from the pinned template
        payload = backend.build_payload(
            model_name=model_name, messages=item["messages"], max_tokens=max_tokens or 256,
            sampling=self.exp.sampling.model_dump(), chat_template_kwargs=kwargs, ignore_eos=ignore_eos,
            no_cache_reuse=self.exp.cache_policy == "no_reuse", rendered_prompt=item.get("rendered_prompt"),
            num_ctx=None, think=(self.exp.thinking != "disabled") if backend.name == "ollama" else None,
            mock=self.exp.backend_settings.get("mock", {}).get("behavior"),
        )
        iid = str(item.get("item_id") or item.get("id"))
        return RequestSpec(
            request_id=f"{phase}-{iid}", phase=phase, item_id=iid, path=backend.chat_path,
            protocol=backend.protocol, payload=payload,
            target_output_tokens=w.output_tokens if w.kind == "synthetic_tokens" else None,
            expected_input_tokens=item.get("input_tokens"),
        )

    async def _verify_template(self, backend: Backend, h: ServerHandle, cell: Cell, items: list[dict[str, Any]],
                               wl_meta: dict[str, Any]) -> dict[str, Any]:
        """Server-side rendering check where the backend exposes it (llama-server /apply-template)."""
        out: dict[str, Any] = {"method": None}
        if cell.extra.get("synthetic") or not items:
            return out
        if hasattr(backend, "apply_template"):
            it = items[0]
            try:
                r = await backend.apply_template(h, it["messages"], wl_meta.get("chat_template_kwargs"))
                out.update({
                    "method": "llama-server /apply-template + /tokenize",
                    "server_prompt_tokens": r["n_tokens"], "expected_prompt_tokens": it.get("input_tokens"),
                    "rendered_prompt_identical": r["prompt"] == it.get("rendered_prompt"),
                })
                if r["prompt"] != it.get("rendered_prompt"):
                    (Path(h.log_path.parent) / "template_diff.txt").write_text(
                        "=== server ===\n" + r["prompt"] + "\n=== pinned HF template ===\n" + str(it.get("rendered_prompt")))
            except Exception as e:
                out["error"] = repr(e)
        else:
            out["method"] = "post-hoc: usage.prompt_tokens vs expected (see summary.prompt_token_check)"
        return out

    def _finish(self, backend: Backend, h: ServerHandle | None, sampler: TelemetrySampler | None, run: RunDir,
                baseline: dict[str, Any]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        if h is not None:
            out["shutdown"] = backend.stop(h)
            out["final_log_facts"] = backend.final_log_facts(h.log_path)
        if sampler is not None:
            sampler.stop()
            out["telemetry_summary"] = summarize_telemetry(sampler.samples, baseline)
        else:
            out["telemetry_summary"] = {}
        # Wait briefly for unified memory to be returned before the next cell.
        time.sleep(3)
        out["system_after_shutdown"] = system_snapshot()
        run.gzip_file(run.path / "server.log", "server.log.gz")
        return out


def _trim_meta(meta: dict[str, Any]) -> dict[str, Any]:
    m = dict(meta)
    m.pop("embedded_chat_template", None)
    return m


# ---------------------------------------------------------------------------
def summarize_run(records: list[dict[str, Any]], events: dict[str, list[list[Any]]], exp: ExperimentConfig | None,
                  workload: Workload | None) -> dict[str, Any]:
    """Recompute everything from records + raw events (also used by `report`)."""
    metrics = [compute_request_metrics(r, events.get(r["request_id"], [])) for r in records]
    min95 = exp.min_n_p95 if exp else 200
    min99 = exp.min_n_p99 if exp else 1000
    perf = aggregate_cell(records, metrics, min_n_p95=min95, min_n_p99=min99)
    # Prompt-token check: server-reported vs pinned-template expectation
    diffs = [m.input_tokens - r["expected_input_tokens"] for r, m in zip(records, metrics, strict=True)
             if r.get("phase") == "measured" and m.input_tokens is not None and r.get("expected_input_tokens") is not None]
    check = {"n": len(diffs)}
    if diffs:
        check.update({"all_equal": all(d == 0 for d in diffs), "min_diff": min(diffs), "max_diff": max(diffs)})
    return {"performance": perf, "prompt_token_check": check,
            "per_request": [dict(request_id=r["request_id"], phase=r.get("phase"), **m.to_dict())
                            for r, m in zip(records, metrics, strict=True)]}


UNSCORED_STATUSES = {"not_dispatched", "cancelled"}  # never answered because of a deadline: missing coverage, not a model failure


def merged_run_data(run: Path) -> tuple[list[dict[str, Any]], dict[str, list[list[Any]]], int]:
    """Records + events of ``run`` merged with any --fill-missing runs that completed its skipped items.

    Original records whose items were later answered by a fill run are replaced by the fill run's records.
    Returns (records, events, n_filled).
    """
    from .manifests import load_jsonl

    records = load_jsonl(run / "requests.jsonl")
    events = {e["request_id"]: e["events"] for e in load_jsonl(run / "events.jsonl.gz")}
    filled: list[dict[str, Any]] = []
    for other in sorted(run.parent.iterdir()):
        m = other / "manifest.json"
        if other == run or not m.exists():
            continue
        man = json.loads(m.read_text())
        if man.get("fill_of") == run.name and man.get("finished"):
            for r in load_jsonl(other / "requests.jsonl"):
                if r.get("phase") == "measured" and r.get("status") not in UNSCORED_STATUSES:
                    r = {**r, "request_id": f"fill:{other.name}:{r['request_id']}", "filled_from": other.name}
                    filled.append(r)
            for e in load_jsonl(other / "events.jsonl.gz"):
                events[f"fill:{other.name}:{e['request_id']}"] = e["events"]
    done = {r["item_id"] for r in filled}
    merged = [r for r in records if not (r.get("status") in UNSCORED_STATUSES and r.get("item_id") in done)] + filled
    return merged, events, len(filled)


def score_run(records: list[dict[str, Any]], events: dict[str, list[list[Any]]], items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id = {it["id"]: it for it in items}
    scores = []
    for r in records:
        if r.get("phase") != "measured" or r["item_id"] not in by_id or r.get("status") in UNSCORED_STATUSES:
            continue
        content, reasoning = response_text(r["protocol"], events.get(r["request_id"], []))
        s = score_item(by_id[r["item_id"]], content if r.get("status") == "ok" else "")
        s.update({"request_id": r["request_id"], "status": r.get("status"), "content": content,
                  "reasoning_chars": len(reasoning)})
        scores.append(s)
    return scores

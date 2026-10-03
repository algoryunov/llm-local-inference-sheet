"""Host detection, run manifests and on-disk result layout.

Layout::

    results/<experiment>/
      experiment.json                    # effective experiment config + identity
      cells/<cell_id>/<run_id>/
        manifest.json                    # host, backend, artifact, versions, state
        effective_config.json            # launch command, request template, settings
        workload.json                    # workload hash + per-item token counts
        requests.jsonl                   # one record per request (no raw events)
        events.jsonl.gz                  # raw stream payloads with timestamps
        telemetry.jsonl                  # sampler output
        server.log.gz                    # backend stdout/stderr
        summary.json                     # recomputable aggregate
"""

from __future__ import annotations

import contextlib
import datetime as dt
import gzip
import json
import os
import platform
import shutil
import socket
import subprocess
from pathlib import Path
from typing import Any

import psutil

from . import __version__
from .config import PROJECT_ROOT, ResultState


def _run(cmd: list[str], timeout: float = 5) -> str | None:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return out.stdout.strip() if out.returncode == 0 else None
    except Exception:
        return None


def _sysctl(name: str) -> str | None:
    return _run(["sysctl", "-n", name])


def git_state() -> dict[str, Any]:
    head = _run(["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"])
    dirty = _run(["git", "-C", str(PROJECT_ROOT), "status", "--porcelain"])
    return {"commit": head, "dirty": bool(dirty) if dirty is not None else None}


def mac_host_info() -> dict[str, Any]:
    batt = _run(["pmset", "-g", "batt"]) or ""
    power_source = None
    if "AC Power" in batt:
        power_source = "ac"
    elif "Battery Power" in batt:
        power_source = "battery"
    pm = _run(["pmset", "-g"]) or ""
    low_power = None
    power_mode = None
    for line in pm.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0] == "lowpowermode":
            low_power = parts[1] == "1"
        if len(parts) >= 2 and parts[0] == "powermode":
            power_mode = parts[1]  # 0 auto, 1 low, 2 high (on machines that expose it)
    gpu_cores = None
    sp = _run(["system_profiler", "SPDisplaysDataType"], timeout=15) or ""
    for line in sp.splitlines():
        if "Total Number of Cores" in line:
            with contextlib.suppress(ValueError):
                gpu_cores = int(line.split(":")[1])
    return {
        "chip": _sysctl("machdep.cpu.brand_string"),
        "model": _sysctl("hw.model"),
        "memory_bytes": int(_sysctl("hw.memsize") or 0) or None,
        "cpu_cores_logical": int(_sysctl("hw.ncpu") or 0) or None,
        "cpu_cores_performance": int(_sysctl("hw.perflevel0.physicalcpu") or 0) or None,
        "cpu_cores_efficiency": int(_sysctl("hw.perflevel1.physicalcpu") or 0) or None,
        "gpu_cores": gpu_cores,
        "macos_version": platform.mac_ver()[0],
        "macos_build": _run(["sw_vers", "-buildVersion"]),
        "power_source": power_source,
        "low_power_mode": low_power,
        "power_mode": power_mode,
        "battery_line": next((ln.strip() for ln in batt.splitlines() if "%" in ln), None),
        "shell_translated_rosetta": _sysctl("sysctl.proc_translated") == "1",
        "iogpu_wired_limit_mb": _sysctl("iogpu.wired_limit_mb"),
    }


def nvidia_host_info() -> dict[str, Any] | None:
    if not shutil.which("nvidia-smi"):
        return None
    q = _run(["nvidia-smi", "--query-gpu=name,uuid,memory.total,driver_version,power.limit,"
              "power.max_limit,pci.bus_id,compute_cap", "--format=csv,noheader"])
    cuda = None
    hdr = _run(["nvidia-smi"]) or ""
    for tok in hdr.split("|"):
        if "CUDA Version" in tok:
            cuda = tok.split("CUDA Version:")[1].strip().split()[0]
    return {"gpus": [ln.strip() for ln in (q or "").splitlines()], "driver_cuda_version": cuda,
            "nvcc": _run(["nvcc", "--version"])}


def host_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "hostname_hash": _short_hash(socket.gethostname()),
        "os": platform.system(),
        "os_release": platform.release(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "memory_total": psutil.virtual_memory().total,
        "disk_free_bytes": shutil.disk_usage(PROJECT_ROOT).free,
        "harness_version": __version__,
        "git": git_state(),
        "container": Path("/.dockerenv").exists() or bool(os.environ.get("VAST_CONTAINERLABEL")),
        "vast_instance": {k: os.environ[k] for k in ("VAST_CONTAINERLABEL", "CONTAINER_ID") if k in os.environ},
    }
    if platform.system() == "Darwin":
        info["mac"] = mac_host_info()
    nv = nvidia_host_info()
    if nv:
        info["nvidia"] = nv
    return info


def _short_hash(s: str) -> str:
    import hashlib

    return hashlib.sha256(s.encode()).hexdigest()[:10]


def now_run_id() -> str:
    return dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")


class RunDir:
    def __init__(self, path: Path) -> None:
        self.path = path
        path.mkdir(parents=True, exist_ok=True)

    def write_json(self, name: str, obj: Any) -> None:
        tmp = self.path / f"{name}.tmp"
        tmp.write_text(json.dumps(obj, indent=2, default=str, ensure_ascii=False))
        tmp.replace(self.path / name)

    def append_jsonl(self, name: str, rows: list[dict[str, Any]]) -> None:
        with (self.path / name).open("a") as f:
            for r in rows:
                f.write(json.dumps(r, default=str, ensure_ascii=False) + "\n")

    def write_events(self, per_request: list[tuple[str, list[list[Any]]]]) -> None:
        with gzip.open(self.path / "events.jsonl.gz", "at", encoding="utf-8") as f:
            for rid, events in per_request:
                f.write(json.dumps({"request_id": rid, "events": events}, ensure_ascii=False) + "\n")

    def gzip_file(self, src: Path, name: str) -> None:
        if src.exists():
            with src.open("rb") as fi, gzip.open(self.path / name, "wb") as fo:
                shutil.copyfileobj(fi, fo)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def find_completed_run(cell_dir: Path, identity: str) -> Path | None:
    """Return the latest run of this cell whose identity matches and which finished.

    Any terminal state counts as finished (measured, OOM, unsupported, ...), because
    re-running an OOM cell with an identical config would produce the same observation.
    Pass --retry-failed to re-run failed/timeout cells.
    """
    if not cell_dir.exists():
        return None
    for run in sorted(cell_dir.iterdir(), reverse=True):
        m = run / "manifest.json"
        if not m.exists():
            continue
        try:
            man = json.loads(m.read_text())
        except json.JSONDecodeError:
            continue
        if man.get("fill_of"):
            continue
        if man.get("cell_identity") == identity and man.get("state") and man.get("finished"):
            return run
    return None


TERMINAL_STATES = {s.value for s in ResultState}

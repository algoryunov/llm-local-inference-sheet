"""Host telemetry sampling (background thread).

Each metric carries its source. Unified memory is not VRAM, so overlapping
quantities (RSS, phys_footprint, AGX in-use memory) are reported side by side
and never added together.

macOS sources (all unprivileged):
  * ``proc_pid_rusage(RUSAGE_INFO_V4)``: per-process ``phys_footprint`` (what Activity
    Monitor calls "Memory"; includes dirty anonymous memory and GPU allocations attributed
    to the process, excludes clean file-backed pages such as mmap'd weights), plus
    ``lifetime_max_phys_footprint`` maintained by the kernel, so peaks between samples are not missed.
  * psutil RSS (resident pages, *including* clean mmap'd weight pages).
  * ``ioreg`` AGXAccelerator PerformanceStatistics: *system-wide* GPU "In use system memory"
    and "Device Utilization %". Not per process.
  * ``kern.memorystatus_vm_pressure_level`` (1 normal, 2 warning, 4 critical) and swap counters.
NVIDIA: ``nvidia-smi --query-gpu`` and ``--query-compute-apps`` polling.
"""

from __future__ import annotations

import contextlib
import ctypes
import json
import platform
import re
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

import psutil

IS_MAC = platform.system() == "Darwin"


# ---------------------------------------------------------------------------
# macOS proc_pid_rusage
# ---------------------------------------------------------------------------
class _RusageInfoV4(ctypes.Structure):
    _fields_ = [("ri_uuid", ctypes.c_uint8 * 16)] + [
        (name, ctypes.c_uint64)
        for name in (
            "ri_user_time", "ri_system_time", "ri_pkg_idle_wkups", "ri_interrupt_wkups",
            "ri_pageins", "ri_wired_size", "ri_resident_size", "ri_phys_footprint",
            "ri_proc_start_abstime", "ri_proc_exit_abstime", "ri_child_user_time",
            "ri_child_system_time", "ri_child_pkg_idle_wkups", "ri_child_interrupt_wkups",
            "ri_child_pageins", "ri_child_elapsed_abstime", "ri_diskio_bytesread",
            "ri_diskio_byteswritten", "ri_cpu_time_qos_default", "ri_cpu_time_qos_maintenance",
            "ri_cpu_time_qos_background", "ri_cpu_time_qos_utility", "ri_cpu_time_qos_legacy",
            "ri_cpu_time_qos_user_initiated", "ri_cpu_time_qos_user_interactive",
            "ri_billed_system_time", "ri_serviced_system_time", "ri_logical_writes",
            "ri_lifetime_max_phys_footprint", "ri_instructions", "ri_cycles",
            "ri_billed_energy", "ri_serviced_energy", "ri_interval_max_phys_footprint",
            "ri_runnable_time",
        )
    ]


_libproc = None
if IS_MAC:
    with contextlib.suppress(OSError):
        _libproc = ctypes.CDLL("/usr/lib/libproc.dylib")


def mac_rusage(pid: int) -> dict[str, int] | None:
    if _libproc is None:
        return None
    info = _RusageInfoV4()
    rc = _libproc.proc_pid_rusage(ctypes.c_int(pid), ctypes.c_int(4), ctypes.byref(info))
    if rc != 0:
        return None
    return {
        "phys_footprint": int(info.ri_phys_footprint),
        "lifetime_max_phys_footprint": int(info.ri_lifetime_max_phys_footprint),
        "resident_size": int(info.ri_resident_size),
        "wired_size": int(info.ri_wired_size),
    }


def mac_pressure_level() -> int | None:
    try:
        out = subprocess.run(["sysctl", "-n", "kern.memorystatus_vm_pressure_level"],
                             capture_output=True, text=True, timeout=2)
        return int(out.stdout.strip())
    except Exception:
        return None


_AGX_RE = {
    "gpu_device_util_pct": re.compile(r'"Device Utilization %"=(\d+)'),
    "gpu_renderer_util_pct": re.compile(r'"Renderer Utilization %"=(\d+)'),
    "agx_in_use_system_memory": re.compile(r'"In use system memory"=(\d+)'),
    "agx_alloc_system_memory": re.compile(r'"Alloc system memory"=(\d+)'),
}


def mac_agx_stats() -> dict[str, int]:
    try:
        out = subprocess.run(["ioreg", "-r", "-d", "1", "-w", "0", "-c", "AGXAccelerator"],
                             capture_output=True, text=True, timeout=3).stdout
    except Exception:
        return {}
    res = {}
    for k, rx in _AGX_RE.items():
        m = rx.search(out)
        if m:
            res[k] = int(m.group(1))
    return res


# ---------------------------------------------------------------------------
# NVIDIA
# ---------------------------------------------------------------------------
_NV_FIELDS = [
    "index", "memory.used", "memory.total", "utilization.gpu", "utilization.memory",
    "power.draw", "power.limit", "temperature.gpu", "clocks.sm", "clocks.mem",
    "clocks_throttle_reasons.active",
]


def nvidia_sample() -> list[dict[str, Any]] | None:
    if not shutil.which("nvidia-smi"):
        return None
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={','.join(_NV_FIELDS)}", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        ).stdout
    except Exception:
        return None
    gpus = []
    for line in out.strip().splitlines():
        vals = [v.strip() for v in line.split(",")]
        row: dict[str, Any] = {}
        for k, v in zip(_NV_FIELDS, vals, strict=False):
            try:
                row[k] = float(v) if k != "clocks_throttle_reasons.active" else v
            except ValueError:
                row[k] = None  # "[N/A]" etc.: unavailable, not zero
        gpus.append(row)
    return gpus


def nvidia_compute_apps() -> list[dict[str, Any]]:
    if not shutil.which("nvidia-smi"):
        return []
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        ).stdout
    except Exception:
        return []
    apps = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 2 and parts[0].isdigit():
            with contextlib.suppress(ValueError):
                apps.append({"pid": int(parts[0]), "used_memory_mib": float(parts[1])})
    return apps


# ---------------------------------------------------------------------------
# Sampler
# ---------------------------------------------------------------------------
def _tree(pid: int) -> list[psutil.Process]:
    try:
        p = psutil.Process(pid)
        return [p, *p.children(recursive=True)]
    except psutil.Error:
        return []


def system_snapshot() -> dict[str, Any]:
    vm = psutil.virtual_memory()
    sw = psutil.swap_memory()
    snap: dict[str, Any] = {
        "sys_mem_total": vm.total,
        "sys_mem_available": vm.available,
        "sys_mem_used": vm.used,
        "swap_used": sw.used,
        "swap_sin": sw.sin,
        "swap_sout": sw.sout,
    }
    if IS_MAC:
        snap["vm_pressure_level"] = mac_pressure_level()
        snap.update(mac_agx_stats())
    return snap


class TelemetrySampler:
    """Samples a process tree plus system state every ``interval_s`` until stopped."""

    def __init__(self, pid: int | None, interval_s: float = 0.5, out_path: Path | None = None) -> None:
        self.pid = pid
        self.interval_s = interval_s
        self.samples: list[dict[str, Any]] = []
        self.out_path = out_path
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._nvidia = shutil.which("nvidia-smi") is not None
        self.marks: list[dict[str, Any]] = []
        self._procs: dict[int, psutil.Process] = {}  # persistent objects so cpu_percent has a baseline

    def mark(self, label: str) -> None:
        self.marks.append({"t_ns": time.perf_counter_ns(), "label": label})

    def start(self) -> None:
        self._thread = threading.Thread(target=self._loop, name="telemetry", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=10)
        if self.out_path:
            with self.out_path.open("w") as f:
                for s in self.samples:
                    f.write(json.dumps(s) + "\n")
                for m in self.marks:
                    f.write(json.dumps({"mark": m}) + "\n")

    def sample_once(self) -> dict[str, Any]:
        s: dict[str, Any] = {"t_ns": time.perf_counter_ns(), "wall": time.time()}
        s.update(system_snapshot())
        if self.pid is not None:
            procs = []
            for p in _tree(self.pid):
                procs.append(self._procs.setdefault(p.pid, p))
            rss = 0
            cpu = 0.0
            for p in procs:
                with contextlib.suppress(psutil.Error):
                    rss += p.memory_info().rss
                    cpu += p.cpu_percent(None)
            s["proc_tree_rss"] = rss if procs else None
            s["proc_tree_cpu_pct"] = cpu if procs else None
            s["proc_tree_pids"] = [p.pid for p in procs]
            if IS_MAC:
                fp = 0
                lifetime_max = 0
                ok = False
                for p in procs:
                    r = mac_rusage(p.pid)
                    if r:
                        ok = True
                        fp += r["phys_footprint"]
                        lifetime_max = max(lifetime_max, r["lifetime_max_phys_footprint"])
                s["proc_tree_phys_footprint"] = fp if ok else None
                # max over processes (not summed): the kernel tracks a per-process lifetime peak
                s["proc_max_lifetime_phys_footprint"] = lifetime_max if ok else None
        if self._nvidia:
            s["nvidia"] = nvidia_sample()
            s["nvidia_apps"] = nvidia_compute_apps()
        return s

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self.samples.append(self.sample_once())
            except Exception as e:
                self.samples.append({"t_ns": time.perf_counter_ns(), "error": repr(e)})
            self._stop.wait(self.interval_s)


def summarize_telemetry(samples: list[dict[str, Any]], baseline: dict[str, Any] | None = None) -> dict[str, Any]:
    """Peaks and deltas. Missing sources give null, never zero."""

    def peak(key: str) -> Any:
        vals = [s[key] for s in samples if s.get(key) is not None]
        return max(vals) if vals else None

    out: dict[str, Any] = {
        "n_samples": len(samples),
        "peak_proc_tree_rss": peak("proc_tree_rss"),
        "peak_proc_tree_phys_footprint": peak("proc_tree_phys_footprint"),
        "proc_max_lifetime_phys_footprint": peak("proc_max_lifetime_phys_footprint"),
        "peak_agx_in_use_system_memory": peak("agx_in_use_system_memory"),
        "peak_gpu_device_util_pct": peak("gpu_device_util_pct"),
        "peak_vm_pressure_level": peak("vm_pressure_level"),
        "min_sys_mem_available": min((s["sys_mem_available"] for s in samples if s.get("sys_mem_available")), default=None),
        "peak_swap_used": peak("swap_used"),
        "peak_proc_tree_cpu_pct": peak("proc_tree_cpu_pct"),
    }
    sw = [s for s in samples if s.get("swap_sout") is not None]
    if sw:
        first = baseline if baseline and baseline.get("swap_sout") is not None else sw[0]
        out["swap_used_delta"] = sw[-1]["swap_used"] - first["swap_used"]
        out["swap_out_bytes_delta"] = sw[-1]["swap_sout"] - first["swap_sout"]
        out["swap_in_bytes_delta"] = sw[-1]["swap_sin"] - first["swap_sin"]
    nv = [g for s in samples for g in (s.get("nvidia") or [])]
    if nv:
        def nvpeak(k: str) -> Any:
            vals = [g[k] for g in nv if isinstance(g.get(k), float)]
            return max(vals) if vals else None
        out.update({
            "peak_nvidia_memory_used_mib": nvpeak("memory.used"),
            "nvidia_memory_total_mib": nvpeak("memory.total"),
            "peak_nvidia_util_pct": nvpeak("utilization.gpu"),
            "peak_nvidia_power_w": nvpeak("power.draw"),
            "nvidia_power_limit_w": nvpeak("power.limit"),
            "peak_nvidia_temp_c": nvpeak("temperature.gpu"),
            "max_nvidia_sm_clock_mhz": nvpeak("clocks.sm"),
            "throttle_reasons_seen": sorted({str(g.get("clocks_throttle_reasons.active")) for g in nv}),
        })
    return out


SWAP_THRESHOLD_BYTES = 256 * 2**20


def memory_contamination(baseline: dict[str, Any] | None, tel: dict[str, Any] | None) -> dict[str, Any]:
    """Flag runs whose timings may include paging effects rather than inference alone.

    Criteria (any): memory pressure above "normal" at start or at any sample, or more than
    256 MiB swapped in/out during the run. Derived from recorded data only; missing telemetry
    yields ``contaminated=None`` (unknown), not False.
    """
    baseline = baseline or {}
    tel = tel or {}
    reasons = []
    known = False
    bp = baseline.get("vm_pressure_level")
    pp = tel.get("peak_vm_pressure_level")
    if bp is not None:
        known = True
        if bp > 1:
            reasons.append(f"pressure level {bp} at start")
    if pp is not None:
        known = True
        if pp > 1:
            reasons.append(f"peak pressure level {pp}")
    for k, label in (("swap_in_bytes_delta", "swapped in"), ("swap_out_bytes_delta", "swapped out")):
        v = tel.get(k)
        if v is not None:
            known = True
            if v > SWAP_THRESHOLD_BYTES:
                reasons.append(f"{v / 2**30:.2f} GiB {label}")
    return {"contaminated": bool(reasons) if known else None, "reasons": reasons}

"""Backend adapter interface.

Adapters own *process control*: availability, launch configuration, readiness,
metadata and shutdown. They also describe the request payload quirks of their
server. They never measure anything. The HTTP benchmark client is separate.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import platform
import re
import shutil
import signal
import subprocess
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

from ..config import PROJECT_ROOT, Artifact, ModelEntry

TOOLS_DIR = Path(os.environ.get("LLM_SHEET_TOOLS_DIR") or PROJECT_ROOT / ".tools")

# Errors that retrying cannot fix: fail fast instead of burning the readiness deadline.
UNSUPPORTED_PATTERNS = [
    r"unknown model architecture: '?[\w.-]+'?",
    r"unsupported model architecture",
    r"model architecture .* is not supported",
]

OOM_PATTERNS = [
    r"out of memory",
    r"failed to allocate",
    r"insufficient memory",
    r"kIOGPUCommandBufferCallbackErrorOutOfMemory",
    r"CUDA error: out of memory",
    r"torch\.OutOfMemoryError",
    r"CUDA out of memory",
    r"ggml_metal.*error.*buffer",
    r"not enough memory",
    r"memory layout cannot be allocated",
    r"No available memory for the cache blocks",
    r"model requires more system memory",
]


@dataclass
class Availability:
    ok: bool
    version: str | None = None
    detail: str = ""
    binary: str | None = None
    binary_archs: str | None = None


@dataclass
class LaunchSpec:
    argv: list[str]
    env: dict[str, str]
    port: int
    effective: dict[str, Any]  # settings as configured, for the manifest
    cwd: str | None = None


@dataclass
class ServerHandle:
    proc: subprocess.Popen[bytes]
    spec: LaunchSpec
    log_path: Path
    t_launch_ns: int
    base_url: str
    readiness: dict[str, Any] = field(default_factory=dict)


class BackendError(RuntimeError):
    def __init__(self, msg: str, state: str = "failed") -> None:
        super().__init__(msg)
        self.state = state


class Backend(ABC):
    name: str = "base"
    platform: str = "any"  # mac | nvidia | any
    protocol: str = "openai"
    chat_path: str = "/v1/chat/completions"
    health_path: str | None = "/health"
    artifact_formats: tuple[str, ...] = ()
    supports_ignore_eos: bool = False
    supports_chat_template_kwargs: bool = True
    # Human-readable description of how concurrency is served (reported verbatim).
    scheduling: str = ""

    # ------------------------------------------------------------ discovery
    @abstractmethod
    def availability(self) -> Availability: ...

    def supports(self, model: ModelEntry, artifact: Artifact) -> tuple[bool, str]:
        if artifact.format not in self.artifact_formats:
            return False, f"{self.name} does not load {artifact.format} artifacts"
        return True, ""

    # ------------------------------------------------------------ launch
    @abstractmethod
    def launch_spec(
        self,
        model: ModelEntry,
        artifact: Artifact,
        artifact_path: Path,
        settings: dict[str, Any],
        *,
        concurrency: int,
        ctx_per_request: int,
        port: int,
    ) -> LaunchSpec: ...

    def request_model_name(self, model: ModelEntry, artifact: Artifact, artifact_path: Path) -> str:
        return artifact.key

    def start(self, spec: LaunchSpec, log_path: Path) -> ServerHandle:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        logf = log_path.open("wb")
        env = {**os.environ, **spec.env}
        t0 = time.perf_counter_ns()
        proc = subprocess.Popen(
            spec.argv, stdout=logf, stderr=subprocess.STDOUT, env=env, cwd=spec.cwd,
            start_new_session=True,
        )
        return ServerHandle(proc=proc, spec=spec, log_path=log_path, t_launch_ns=t0,
                            base_url=f"http://127.0.0.1:{spec.port}")

    def probe_payload(self, model_name: str) -> dict[str, Any]:
        return {"model": model_name, "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 1, "temperature": 0, "stream": False}

    async def prepare(self, h: ServerHandle, model: ModelEntry, artifact: Artifact,
                      artifact_path: Path) -> dict[str, Any]:
        """Hook between health and first generation (e.g. Ollama model import)."""
        return {}

    async def wait_health(self, h: ServerHandle, deadline: float) -> float | None:
        if not self.health_path:
            return None
        async with httpx.AsyncClient(base_url=h.base_url, timeout=httpx.Timeout(10, connect=2)) as c:
            while True:
                self._check_alive(h)
                if time.monotonic() > deadline:
                    raise BackendError("health endpoint not ready before deadline", "timeout")
                with contextlib.suppress(httpx.HTTPError):
                    r = await c.get(self.health_path)
                    if r.status_code == 200:
                        return (time.perf_counter_ns() - h.t_launch_ns) / 1e9
                await asyncio.sleep(0.25)

    async def wait_first_generation(self, h: ServerHandle, deadline: float, probe: dict[str, Any]) -> float:
        """Readiness means one completed 1-token generation, not just a healthy endpoint.

        Some servers report healthy before the model has loaded (mlx_lm.server 0.31.3).
        """
        last_err = None
        async with httpx.AsyncClient(base_url=h.base_url, timeout=httpx.Timeout(None, connect=2)) as c:
            while True:
                self._check_alive(h)
                if time.monotonic() > deadline:
                    raise BackendError(f"first generation not ready before deadline: {last_err}", "timeout")
                try:
                    r = await c.post(self.chat_path, json=probe, timeout=max(5.0, deadline - time.monotonic()))
                    if r.status_code == 200:
                        return (time.perf_counter_ns() - h.t_launch_ns) / 1e9
                    last_err = f"HTTP {r.status_code}: {r.text[:300]}"
                    log_tail = tail_log(h.log_path)
                    if detect_oom(r.text) or detect_oom(log_tail):
                        raise BackendError(f"OOM during first generation: {r.text[:500]}", "out_of_memory")
                    why = detect_unsupported(r.text + "\n" + log_tail)
                    if why:
                        raise BackendError(f"{self.name} cannot load this model: {why}", "unsupported")
                except httpx.HTTPError as e:
                    last_err = repr(e)
                await asyncio.sleep(0.5)

    def _check_alive(self, h: ServerHandle) -> None:
        rc = h.proc.poll()
        if rc is not None:
            tail = tail_log(h.log_path)
            state = "out_of_memory" if detect_oom(tail) else "failed"
            raise BackendError(f"server exited with code {rc} during startup. Log tail:\n{tail[-2000:]}", state)

    async def metadata(self, h: ServerHandle) -> dict[str, Any]:
        return {}

    def final_log_facts(self, log_path: Path) -> dict[str, Any]:
        """Facts only logged at shutdown (e.g. llama.cpp's memory breakdown)."""
        return {}

    def stop(self, h: ServerHandle, grace_s: float = 30) -> dict[str, Any]:
        """Terminate the process group, then wait for the process tree to exit."""
        info: dict[str, Any] = {}
        if h.proc.poll() is None:
            try:
                pgid = os.getpgid(h.proc.pid)
                os.killpg(pgid, signal.SIGTERM)
            except ProcessLookupError:
                pgid = None
            try:
                h.proc.wait(timeout=grace_s)
            except subprocess.TimeoutExpired:
                if pgid is not None:
                    with contextlib.suppress(ProcessLookupError):
                        os.killpg(pgid, signal.SIGKILL)
                h.proc.wait(timeout=10)
                info["killed"] = True
        info["exit_code"] = h.proc.returncode
        return info

    # ------------------------------------------------------------ requests
    def build_payload(
        self,
        *,
        model_name: str,
        messages: list[dict[str, Any]],
        max_tokens: int,
        sampling: dict[str, Any],
        chat_template_kwargs: dict[str, Any] | None,
        ignore_eos: bool,
        no_cache_reuse: bool,
        stream: bool = True,
        **_extra: Any,  # backend-specific extras (rendered_prompt, num_ctx, think, mock) ignored here
    ) -> dict[str, Any]:
        p: dict[str, Any] = {
            "model": model_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "stream": stream,
            "temperature": sampling.get("temperature", 0.0),
            "top_p": sampling.get("top_p", 1.0),
        }
        if stream:
            p["stream_options"] = {"include_usage": True}
        if sampling.get("seed") is not None:
            p["seed"] = sampling["seed"]
        if chat_template_kwargs and self.supports_chat_template_kwargs:
            p["chat_template_kwargs"] = chat_template_kwargs
        if ignore_eos and self.supports_ignore_eos:
            p["ignore_eos"] = True
        return p


# ---------------------------------------------------------------- helpers
def tail_log(path: Path, n_bytes: int = 20000) -> str:
    if not path.exists():
        return ""
    with path.open("rb") as f:
        f.seek(0, 2)
        size = f.tell()
        f.seek(max(0, size - n_bytes))
        return f.read().decode("utf-8", "replace")


def detect_unsupported(text: str) -> str | None:
    for p in UNSUPPORTED_PATTERNS:
        m = re.search(p, text, re.IGNORECASE)
        if m:
            return m.group(0)
    return None


def detect_oom(text: str) -> bool:
    return any(re.search(p, text, re.IGNORECASE) for p in OOM_PATTERNS)


def binary_archs(path: str) -> str | None:
    if platform.system() != "Darwin" or not shutil.which("lipo"):
        return None
    try:
        return subprocess.run(["lipo", "-archs", path], capture_output=True, text=True, timeout=5).stdout.strip()
    except Exception:
        return None


def native_prefix() -> list[str]:
    """On Apple Silicon, force arm64 even when the parent shell runs under Rosetta."""
    if platform.system() == "Darwin" and shutil.which("arch"):
        try:
            translated = subprocess.run(["sysctl", "-n", "sysctl.proc_translated"],
                                        capture_output=True, text=True).stdout.strip()
        except Exception:
            translated = ""
        brand = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout
        if translated == "1" or "Apple" in brand:
            return ["arch", "-arm64"]
    return []


def free_port() -> int:
    import socket

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])

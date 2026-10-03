"""llama.cpp ``llama-server`` adapter (Metal on macOS, CUDA on Linux)."""

from __future__ import annotations

import hashlib
import os
import platform
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

import httpx

from ..config import Artifact, ModelEntry
from .base import (
    TOOLS_DIR,
    Availability,
    Backend,
    LaunchSpec,
    ServerHandle,
    binary_archs,
    native_prefix,
)

PINNED_BUILD = "b11246"


def default_binary() -> str | None:
    env = os.environ.get("LLAMA_SERVER_BIN")
    if env:
        return env
    asset = "macos-arm64" if platform.system() == "Darwin" else "ubuntu-cuda-12.8-x64"
    p = TOOLS_DIR / "llama.cpp" / f"{PINNED_BUILD}-{asset}" / "llama-server"
    if p.exists():
        return str(p)
    return shutil.which("llama-server")


DEFAULTS: dict[str, Any] = {
    "n_gpu_layers": 999,  # all layers on GPU (Metal/CUDA)
    "flash_attn": "on",
    "cache_type_k": "f16",
    "cache_type_v": "f16",
    "batch_size": 2048,
    "ubatch_size": 512,
    "parallel": "match_concurrency",
    "ctx_per_slot": None,  # None: derived from the workload (input + output + margin)
    "ctx_margin": 64,
    "kv_unified": False,
    "cont_batching": True,
    "fit": "off",  # b11246 defaults to --fit on, which silently adjusts unset args
    "cache_ram_mib": None,  # None: 0 under cache_policy=no_reuse, else server default (8192)
    "mmap": True,
    "threads": None,
    "extra_args": [],
}


class LlamaCppBackend(Backend):
    name = "llama-cpp"
    platform = "any"
    artifact_formats = ("gguf",)
    supports_ignore_eos = True
    scheduling = (
        "llama-server: N parallel slots (-np), continuous batching across slots. Each slot has "
        "its own KV region when --no-kv-unified (total ctx = N x per-slot ctx)."
    )

    def __init__(self, variant: str = "metal") -> None:
        self.variant = variant
        if variant == "cuda":
            self.name = "llama-cpp-cuda"
            self.platform = "nvidia"
        else:
            self.platform = "mac"

    def availability(self) -> Availability:
        if self.variant == "cuda" and platform.system() == "Darwin":
            return Availability(False, detail="CUDA build requires a Linux/NVIDIA host")
        b = default_binary()
        if not b:
            return Availability(False, detail="llama-server not found; run scripts/install_llama_cpp.sh")
        try:
            out = subprocess.run([*native_prefix(), b, "--version"], capture_output=True, text=True, timeout=20)
            text = out.stdout + out.stderr
        except Exception as e:
            return Availability(False, detail=repr(e), binary=b)
        m = re.search(r"version: (\S+) \(build (\d+), commit (\w+)\)", text)
        version = f"b{m.group(2)} ({m.group(3)})" if m else text.strip()[:80]
        return Availability(True, version=version, binary=b, binary_archs=binary_archs(b), detail=text.strip()[-200:])

    def launch_spec(self, model: ModelEntry, artifact: Artifact, artifact_path: Path, settings: dict[str, Any],
                    *, concurrency: int, ctx_per_request: int, port: int) -> LaunchSpec:
        s = {**DEFAULTS, **settings}
        n_par = concurrency if s["parallel"] == "match_concurrency" else int(s["parallel"])
        per_slot = s["ctx_per_slot"] or _round_up(ctx_per_request + s["ctx_margin"], 256)
        total_ctx = per_slot * n_par  # same total either way; kv_unified only changes sharing
        b = default_binary()
        if not b:
            raise RuntimeError("llama-server binary not found")
        argv = [
            *(native_prefix() if self.variant == "metal" else []),
            b, "-m", str(artifact_path), "-a", artifact.key,
            "--host", "127.0.0.1", "--port", str(port),
            "-ngl", str(s["n_gpu_layers"]),
            "-fa", str(s["flash_attn"]),
            "-c", str(total_ctx), "-np", str(n_par),
            "-b", str(s["batch_size"]), "-ub", str(s["ubatch_size"]),
            "-ctk", s["cache_type_k"], "-ctv", s["cache_type_v"],
            "--fit", s["fit"],
            "--kv-unified" if s["kv_unified"] else "--no-kv-unified",
            "--cont-batching" if s["cont_batching"] else "--no-cont-batching",
            "--jinja", "--reasoning-format", "deepseek",
            "--no-webui", "--metrics", "--slots", "--log-timestamps", "-lv", "4",
        ]
        cache_ram = s["cache_ram_mib"]
        if cache_ram is None and settings.get("_cache_policy", "no_reuse") == "no_reuse":
            cache_ram = 0
        if cache_ram is not None:
            argv += ["--cache-ram", str(cache_ram)]
        if not s["mmap"]:
            argv.append("--no-mmap")
        if s["threads"]:
            argv += ["-t", str(s["threads"])]
        argv += list(s["extra_args"])
        effective = {k: v for k, v in s.items() if not k.startswith("_")}
        effective.update({
            "parallel_slots": n_par,
            "ctx_total": total_ctx,
            "ctx_per_slot": total_ctx // n_par,
            "ctx_required_per_request": ctx_per_request,
            "cache_ram_mib": cache_ram,
            "prompt_cache_per_request": "cache_prompt=false" if settings.get("_cache_policy", "no_reuse") == "no_reuse" else "server default",
        })
        return LaunchSpec(argv=argv, env={}, port=port, effective=effective)

    def build_payload(self, **kw: Any) -> dict[str, Any]:
        no_cache = kw.get("no_cache_reuse", False)
        p = super().build_payload(**kw)
        if no_cache:
            # Per-request: do not reuse the slot's previous KV for a matching prefix.
            p["cache_prompt"] = False
        return p

    async def metadata(self, h: ServerHandle) -> dict[str, Any]:
        meta: dict[str, Any] = {}
        async with httpx.AsyncClient(base_url=h.base_url, timeout=10) as c:
            try:
                props = (await c.get("/props")).json()
                tpl = props.get("chat_template") or ""
                meta["props"] = {k: v for k, v in props.items() if k not in ("chat_template",)}
                meta["embedded_chat_template_sha256"] = hashlib.sha256(tpl.encode()).hexdigest() if tpl else None
                meta["embedded_chat_template"] = tpl
            except Exception as e:
                meta["props_error"] = repr(e)
            try:
                meta["slots"] = (await c.get("/slots")).json()
            except Exception as e:
                meta["slots_error"] = repr(e)
        meta["log_facts"] = parse_log_facts(h.log_path.read_text(errors="replace") if h.log_path.exists() else "")
        return meta

    def final_log_facts(self, log_path: Path) -> dict[str, Any]:
        text = log_path.read_text(errors="replace") if log_path.exists() else ""
        m = re.findall(r"- (MTL0|CUDA0) \((.+?)\)\s*\|\s*(\d+) = (\d+) \+ \((\d+) =\s*(\d+) \+\s*(\d+) \+\s*(\d+)\)", text)
        if not m:
            return {}
        dev, name, total, free, self_, model, ctx, compute = m[-1]
        return {"gpu_memory_breakdown_mib": {
            "device": f"{dev} ({name})", "device_total": int(total), "free": int(free), "self": int(self_),
            "model": int(model), "context_kv": int(ctx), "compute": int(compute),
            "source": "llama.cpp common_memory_breakdown_print at exit"}}

    async def apply_template(self, h: ServerHandle, messages: list[dict[str, Any]],
                             chat_template_kwargs: dict[str, Any] | None) -> dict[str, Any]:
        """Render the prompt server-side (llama-server /apply-template) and tokenize it."""
        async with httpx.AsyncClient(base_url=h.base_url, timeout=30) as c:
            body: dict[str, Any] = {"messages": messages}
            if chat_template_kwargs:
                body["chat_template_kwargs"] = chat_template_kwargs
            r = await c.post("/apply-template", json=body)
            prompt = r.json().get("prompt", "")
            t = await c.post("/tokenize", json={"content": prompt, "add_special": True, "parse_special": True})
            return {"prompt": prompt, "n_tokens": len(t.json().get("tokens", []))}


_LOG_FACTS = {
    # b11246 at -lv 4; patterns are best-effort and absent facts are simply omitted
    "gpu_device": r"- (?:MTL0|CUDA0)\s*: (.+)",
    "metal_features": r"ggml_metal_init: (use \w+(?: \w+)?\s*= \w+)",
    "file_type": r"print_info: file type\s*=\s*(.+)",
    "n_ctx_train": r"n_ctx_train\s*=\s*(\d+)",
    "offloaded_layers": r"offloaded (\d+/\d+) layers to GPU",
    "gpu_model_buffer": r"(?:MTL0|CUDA0)_Mapped model buffer size\s*=\s*([\d.]+ MiB)",
    "cpu_model_buffer": r"CPU_Mapped model buffer size\s*=\s*([\d.]+ MiB)",
    "kv_buffer": r"(?:MTL0|CUDA0) KV buffer size\s*=\s*([\d.]+ MiB)",
    "gpu_compute_buffer": r"(?:MTL0|CUDA0) compute buffer size\s*=\s*([\d.]+ MiB)",
    "n_seq_max": r"llama_context: n_seq_max\s*=\s*(\d+)",
    "n_ctx": r"llama_context: n_ctx\s*=\s*(\d+)",
    "n_ctx_seq": r"llama_context: n_ctx_seq\s*=\s*(\d+)",
    "n_batch": r"llama_context: n_batch\s*=\s*(\d+)",
    "n_ubatch": r"llama_context: n_ubatch\s*=\s*(\d+)",
    "flash_attn": r"llama_context: flash_attn\s*=\s*(\w+)",
    "kv_unified": r"llama_context: kv_unified\s*=\s*(\w+)",
    "n_slots": r"load_model: initializing, n_slots = (\d+)",
    "n_ctx_slot": r"n_ctx_slot = (\d+), kv_unified",
    "n_threads": r"threadpool init, n_threads = (\d+)",
    "warnings": r" W (?:srv|load|cmn)\s+\S+: (.{0,160})",
}


def parse_log_facts(log: str) -> dict[str, Any]:
    facts: dict[str, Any] = {}
    for k, rx in _LOG_FACTS.items():
        found = re.findall(rx, log)
        if found:
            facts[k] = found[-1] if len(set(found)) == 1 else found[-4:]
    return facts


def _round_up(x: int, m: int) -> int:
    return ((x + m - 1) // m) * m

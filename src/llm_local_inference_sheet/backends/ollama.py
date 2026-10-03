"""Ollama adapter.

Ollama is a model manager and server, not an independent kernel implementation.
Its runner is built on ggml/llama.cpp code (either the vendored llama.cpp runner or
Ollama's own Go "ollama engine" on top of ggml, depending on version and model). The adapter
records which runner the server log reports and never labels results as a separate
inference engine.

We run a private ``ollama serve`` instance (own port, own OLLAMA_MODELS directory),
so the user's desktop Ollama and its models are untouched.

Two request modes:
  * ``generate_raw`` (default for runtime comparisons): ``/api/generate`` with ``raw: true``
    and a prompt rendered client-side from the model's *pinned* HF chat template. The prompt
    is then byte-identical to what the harness intended, and ``prompt_eval_count`` verifies it.
  * ``chat``: ``/api/chat`` with Ollama's own template handling and ``think`` control.
    This is the user-facing path. Template detection for imported GGUFs is version-dependent.
"""

from __future__ import annotations

import asyncio
import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

import httpx

from ..config import Artifact, ModelEntry
from ..models import models_dir
from .base import Availability, Backend, BackendError, LaunchSpec, ServerHandle, binary_archs

DEFAULTS: dict[str, Any] = {
    "api": "generate_raw",
    "flash_attention": True,
    "kv_cache_type": "f16",
    "num_parallel": "match_concurrency",
    "num_gpu": None,  # None: Ollama decides layer placement (recorded from /api/ps + log)
    "debug_log": True,
    "extra_env": {},
}


class OllamaBackend(Backend):
    name = "ollama"
    platform = "mac"
    protocol = "ollama"
    health_path = "/api/version"
    artifact_formats = ("gguf",)
    supports_ignore_eos = False
    supports_chat_template_kwargs = False
    scheduling = (
        "ollama serve: OLLAMA_NUM_PARALLEL slots in one runner process; requests beyond that "
        "queue in Ollama's scheduler (OLLAMA_MAX_QUEUE). Context per slot = num_ctx."
    )

    def __init__(self) -> None:
        self._api = "generate_raw"

    @property
    def chat_path(self) -> str:  # type: ignore[override]
        return "/api/generate" if self._api == "generate_raw" else "/api/chat"

    def availability(self) -> Availability:
        b = shutil.which("ollama")
        if not b:
            return Availability(False, detail="ollama not installed")
        out = subprocess.run([b, "--version"], capture_output=True, text=True, timeout=10)
        text = out.stdout + out.stderr
        m = re.search(r"(?:client )?version is (\S+)", text)
        return Availability(True, version=m.group(1) if m else text.strip()[:60], binary=b,
                            binary_archs=binary_archs(b), detail=text.strip())

    def ollama_model_name(self, artifact: Artifact) -> str:
        return f"llmsheet-{artifact.key}".lower().replace(".", "-")

    def launch_spec(self, model: ModelEntry, artifact: Artifact, artifact_path: Path, settings: dict[str, Any],
                    *, concurrency: int, ctx_per_request: int, port: int) -> LaunchSpec:
        s = {**DEFAULTS, **settings}
        self._api = s["api"]
        n_par = concurrency if s["num_parallel"] == "match_concurrency" else int(s["num_parallel"])
        num_ctx = _round_up(ctx_per_request + 64, 256)
        env = {
            "OLLAMA_HOST": f"127.0.0.1:{port}",
            "OLLAMA_MODELS": str(models_dir() / "ollama"),
            "OLLAMA_NUM_PARALLEL": str(n_par),
            "OLLAMA_MAX_LOADED_MODELS": "1",
            "OLLAMA_KEEP_ALIVE": "-1",
            "OLLAMA_FLASH_ATTENTION": "1" if s["flash_attention"] else "0",
            "OLLAMA_KV_CACHE_TYPE": s["kv_cache_type"],
            "OLLAMA_NOPRUNE": "1",
            **({"OLLAMA_DEBUG": "1"} if s["debug_log"] else {}),
            **{k: str(v) for k, v in s["extra_env"].items()},
        }
        b = shutil.which("ollama")
        if not b:
            raise RuntimeError("ollama binary not found")
        effective = {k: v for k, v in s.items() if not k.startswith("_")}
        effective.update({
            "num_parallel": n_par,
            "num_ctx_per_request": num_ctx,
            "ctx_total_expected": num_ctx * n_par,
            "ctx_required_per_request": ctx_per_request,
            "env": env,
            "cache_control": "none available; reuse detected via prompt_eval_count < prompt tokens",
            "ollama_model": self.ollama_model_name(artifact),
        })
        return LaunchSpec(argv=[b, "serve"], env=env, port=port, effective=effective)

    def request_model_name(self, model: ModelEntry, artifact: Artifact, artifact_path: Path) -> str:
        return self.ollama_model_name(artifact)

    async def prepare(self, h: ServerHandle, model: ModelEntry, artifact: Artifact, artifact_path: Path) -> dict[str, Any]:
        """Import the exact local GGUF via a Modelfile, then verify the blob digest."""
        name = self.ollama_model_name(artifact)
        num_ctx = h.spec.effective["num_ctx_per_request"]
        mf_dir = h.log_path.parent
        mf = mf_dir / "Modelfile"
        lines = [f"FROM {artifact_path}", f"PARAMETER num_ctx {num_ctx}"]
        if h.spec.effective.get("num_gpu") is not None:
            lines.append(f"PARAMETER num_gpu {h.spec.effective['num_gpu']}")
        mf.write_text("\n".join(lines) + "\n")
        proc = await asyncio.create_subprocess_exec(
            "ollama", "create", name, "-f", str(mf),
            env={**_os_env(), "OLLAMA_HOST": h.spec.env["OLLAMA_HOST"]},
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
        )
        out, _ = await proc.communicate()
        if proc.returncode != 0:
            raise BackendError(f"ollama create failed: {out.decode(errors='replace')[-1000:]}")
        info: dict[str, Any] = {"modelfile": mf.read_text(), "create_output_tail": out.decode(errors="replace")[-500:]}
        # Blob digest == file sha256 means Ollama serves the byte-identical GGUF.
        manifest = _find_manifest(models_dir() / "ollama", name)
        if manifest:
            layers = json.loads(manifest.read_text()).get("layers", [])
            model_layer = next((la for la in layers if la.get("mediaType", "").endswith(".model")), None)
            if model_layer:
                digest = model_layer["digest"].removeprefix("sha256:")
                expected = artifact.files[0].sha256 if artifact.files else None
                info["blob_digest"] = digest
                info["blob_matches_artifact_sha256"] = (digest == expected) if expected else None
        return info

    def build_payload(self, *, model_name: str, messages: list[dict[str, Any]], max_tokens: int,
                      sampling: dict[str, Any], chat_template_kwargs: dict[str, Any] | None,
                      ignore_eos: bool, no_cache_reuse: bool, stream: bool = True,
                      rendered_prompt: str | None = None, think: bool | None = None,
                      num_ctx: int | None = None, **_extra: Any) -> dict[str, Any]:
        options: dict[str, Any] = {
            "num_predict": max_tokens,
            "temperature": sampling.get("temperature", 0.0),
            "top_p": sampling.get("top_p", 1.0),
        }
        if sampling.get("seed") is not None:
            options["seed"] = sampling["seed"]
        if num_ctx:
            options["num_ctx"] = num_ctx
        if self._api == "generate_raw":
            if rendered_prompt is None:
                raise ValueError("generate_raw mode needs a client-rendered prompt")
            return {"model": model_name, "prompt": rendered_prompt, "raw": True, "stream": stream,
                    "options": options, "keep_alive": -1}
        p: dict[str, Any] = {"model": model_name, "messages": messages, "stream": stream,
                             "options": options, "keep_alive": -1}
        if think is not None:
            p["think"] = think
        return p

    def probe_payload(self, model_name: str) -> dict[str, Any]:
        opts = {"num_predict": 1, "temperature": 0}
        if self._api == "generate_raw":
            return {"model": model_name, "prompt": "Hello", "raw": True, "stream": False, "options": opts}
        return {"model": model_name, "messages": [{"role": "user", "content": "Hello"}], "stream": False,
                "options": opts}

    async def metadata(self, h: ServerHandle) -> dict[str, Any]:
        meta: dict[str, Any] = {}
        name = h.spec.effective.get("ollama_model")
        async with httpx.AsyncClient(base_url=h.base_url, timeout=15) as c:
            for key, fn in (
                ("version", lambda: c.get("/api/version")),
                ("ps", lambda: c.get("/api/ps")),
                ("show", lambda: c.post("/api/show", json={"model": name})),
            ):
                try:
                    r = await fn()
                    j = r.json()
                    if key == "show":
                        j = {k: v for k, v in j.items() if k not in ("tensors", "license")}
                        mi = j.get("model_info") or {}
                        j["model_info"] = {k: v for k, v in mi.items() if not isinstance(v, list)}
                    meta[key] = j
                except Exception as e:
                    meta[f"{key}_error"] = repr(e)
        log = h.log_path.read_text(errors="replace") if h.log_path.exists() else ""
        meta["log_facts"] = parse_ollama_log(log)
        return meta


def parse_ollama_log(log: str) -> dict[str, Any]:
    facts: dict[str, Any] = {}
    pats = {
        "runner_cmd": r"(starting llama server|starting runner).{0,400}",
        "ollama_engine": r"(--ollama-engine)",
        "offloaded": r"offloaded (\d+/\d+) layers to GPU",
        "flash_attn": r"flash_attn\s*=\s*(\w+)",
        "n_ctx": r"n_ctx\s*=\s*(\d+)",
        "n_seq_max": r"n_seq_max\s*=\s*(\d+)",
        "kv_unified": r"kv_unified\s*=\s*(\w+)",
        "model_buffer": r"Metal\S* model buffer size\s*=\s*([\d.]+ MiB)",
        "kv_buffer": r"Metal\S*KV buffer size\s*=\s*([\d.]+ MiB)",
        "gpu_name": r"GPU name:\s+(.+)",
        "inference_compute": r"inference compute.{0,200}",
        "load_duration": r"llama runner started in ([\d.]+ \w+)",
    }
    for k, rx in pats.items():
        found = re.findall(rx, log)
        if found:
            facts[k] = found[-1] if isinstance(found[-1], str) else found[-1][0]
    return facts


def _find_manifest(root: Path, name: str) -> Path | None:
    base = root / "manifests"
    if not base.exists():
        return None
    for p in base.rglob("*"):
        if p.is_file() and p.parent.name == name and p.name == "latest":
            return p
    return None


def _os_env() -> dict[str, str]:
    import os

    return dict(os.environ)


def _round_up(x: int, m: int) -> int:
    return ((x + m - 1) // m) * m

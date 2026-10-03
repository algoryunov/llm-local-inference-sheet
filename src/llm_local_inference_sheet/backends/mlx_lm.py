"""MLX-LM ``mlx_lm.server`` adapter (Apple Silicon, Metal via MLX).

mlx-lm is installed in its own venv (``.venvs/mlx``) with pinned versions, so its
transformers/mlx dependency set never mixes with the harness environment.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

from ..config import PROJECT_ROOT, Artifact, ModelEntry
from .base import Availability, Backend, LaunchSpec, ServerHandle, binary_archs, native_prefix

MLX_VENV = Path(os.environ.get("LLM_SHEET_MLX_VENV") or PROJECT_ROOT / ".venvs" / "mlx")
WRAPPER = Path(__file__).resolve().parents[1] / "servers" / "mlx_server_wrapper.py"

DEFAULTS: dict[str, Any] = {
    "decode_concurrency": 32,  # mlx-lm 0.31.3 default
    "prompt_concurrency": 8,  # mlx-lm 0.31.3 default
    "prefill_step_size": 2048,  # mlx-lm 0.31.3 default
    "prompt_cache_size": None,  # None: 0 under cache_policy=no_reuse, else server default (10)
    "extra_args": [],
}


class MlxLmBackend(Backend):
    name = "mlx-lm"
    platform = "mac"
    artifact_formats = ("mlx",)
    supports_ignore_eos = False  # no ignore_eos request field in mlx_lm.server 0.31.3
    scheduling = (
        "mlx_lm.server: continuous batching via BatchGenerator when the request is batchable "
        "(no seed set); up to --decode-concurrency sequences decode together and up to "
        "--prompt-concurrency prompts prefill together."
    )

    @property
    def python(self) -> Path:
        return MLX_VENV / "bin" / "python"

    def availability(self) -> Availability:
        if not self.python.exists():
            return Availability(False, detail=f"MLX venv missing at {MLX_VENV}; run scripts/install_mlx.sh")
        try:
            out = subprocess.run(
                [*native_prefix(), str(self.python), "-c",
                 "import importlib.metadata as m, json, platform; "
                 "print(json.dumps({p: m.version(p) for p in ('mlx','mlx-lm','transformers')} | "
                 "{'machine': platform.machine()}))"],
                capture_output=True, text=True, timeout=60,
            )
            info = json.loads(out.stdout.strip().splitlines()[-1])
        except Exception as e:
            return Availability(False, detail=repr(e))
        return Availability(True, version=f"mlx-lm {info['mlx-lm']} / mlx {info['mlx']}",
                            binary=str(self.python), binary_archs=binary_archs(str(self.python.resolve())),
                            detail=json.dumps(info))

    def launch_spec(self, model: ModelEntry, artifact: Artifact, artifact_path: Path, settings: dict[str, Any],
                    *, concurrency: int, ctx_per_request: int, port: int) -> LaunchSpec:
        s = {**DEFAULTS, **settings}
        cache_size = s["prompt_cache_size"]
        if cache_size is None and settings.get("_cache_policy", "no_reuse") == "no_reuse":
            cache_size = 0
        mem_log = settings.get("_mem_log")
        argv = [*native_prefix(), str(self.python), str(WRAPPER)]
        if mem_log:
            argv += ["--mem-log", str(mem_log)]
        argv += [
            "--", "--model", str(artifact_path), "--host", "127.0.0.1", "--port", str(port),
            "--decode-concurrency", str(s["decode_concurrency"]),
            "--prompt-concurrency", str(s["prompt_concurrency"]),
            "--prefill-step-size", str(s["prefill_step_size"]),
            "--log-level", "INFO",
        ]
        if cache_size is not None:
            argv += ["--prompt-cache-size", str(cache_size)]
        argv += list(s["extra_args"])
        effective = {k: v for k, v in s.items() if not k.startswith("_")}
        effective.update({
            "prompt_cache_size": cache_size,
            "ctx_required_per_request": ctx_per_request,
            "context_allocation": "dynamic (KV cache grows per sequence; no fixed context window setting)",
            "kv_cache_dtype": "model dtype (no KV quantization flag set)",
        })
        return LaunchSpec(argv=argv, env={"PYTHONUNBUFFERED": "1"}, port=port, effective=effective)

    def request_model_name(self, model: ModelEntry, artifact: Artifact, artifact_path: Path) -> str:
        # mlx_lm.server treats `model` as a path/repo to (re)load. Pass the loaded path.
        return str(artifact_path)

    def build_payload(self, **kw: Any) -> dict[str, Any]:
        sampling = dict(kw.get("sampling") or {})
        if sampling.get("seed") is not None:
            # A seed makes requests non-batchable in mlx_lm.server 0.31.3 (_is_batchable).
            # It is dropped here and the drop is recorded, so concurrency is never silently serialized.
            sampling.pop("seed")
            kw["sampling"] = sampling
        return super().build_payload(**kw)

    async def metadata(self, h: ServerHandle) -> dict[str, Any]:
        meta: dict[str, Any] = {"note": "seed is never sent to mlx_lm.server (it disables batching)"}
        mem_log = h.spec.effective.get("_mem_log")
        if mem_log:
            meta["mlx_mem_log"] = str(mem_log)
        return meta

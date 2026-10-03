"""Mock backend: runs the deterministic SYNTHETIC mock server as a subprocess.

Used for end-to-end harness tests. Results are always labelled ``synthetic``.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from ..config import Artifact, ModelEntry
from .base import Availability, Backend, LaunchSpec


class MockBackend(Backend):
    name = "mock"
    platform = "any"
    artifact_formats = ("gguf", "mlx", "safetensors")
    supports_ignore_eos = True
    synthetic = True
    scheduling = "SYNTHETIC mock server: asyncio, unlimited concurrency, deterministic timings."

    def availability(self) -> Availability:
        return Availability(True, version="synthetic", detail="deterministic mock streaming server")

    def launch_spec(self, model: ModelEntry, artifact: Artifact, artifact_path: Path, settings: dict[str, Any],
                    *, concurrency: int, ctx_per_request: int, port: int) -> LaunchSpec:
        argv = [sys.executable, "-m", "llm_local_inference_sheet.cli", "mock-server", "--port", str(port)]
        return LaunchSpec(argv=argv, env={}, port=port,
                          effective={"synthetic": True, **{k: v for k, v in settings.items() if not k.startswith("_")}})

    def build_payload(self, **kw: Any) -> dict[str, Any]:
        mock = kw.pop("mock", None)
        p = super().build_payload(**kw)
        if mock:
            p["mock"] = mock
        return p

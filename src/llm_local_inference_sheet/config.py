"""Typed configuration: model registry and experiment definitions."""

from __future__ import annotations

import hashlib
import json
import os
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


def _project_root() -> Path:
    """Repo checkout (editable install) or $LLM_SHEET_ROOT (container installs into a venv)."""
    env = os.environ.get("LLM_SHEET_ROOT")
    if env:
        return Path(env)
    here = Path(__file__).resolve().parents[2]
    if (here / "configs" / "models.yaml").exists():
        return here
    return Path.cwd()


PROJECT_ROOT = _project_root()


def private_mode() -> bool:
    """LLM_SHEET_PRIVATE=1 makes datasets resolve to private/<dataset> (git-ignored) when it exists."""
    return os.environ.get("LLM_SHEET_PRIVATE") == "1"


def dataset_dir(name: str) -> Path:
    if private_mode() and (PROJECT_ROOT / "private" / name).exists():
        return PROJECT_ROOT / "private" / name
    return PROJECT_ROOT / name


class ResultState(StrEnum):
    MEASURED = "measured"
    UNTESTED = "untested"
    UNSUPPORTED = "unsupported"
    GATED = "gated"
    OUT_OF_MEMORY = "out_of_memory"
    TIMEOUT = "timeout"
    FAILED = "failed"
    SYNTHETIC = "synthetic"  # mock-server/fixture data. Never an empirical result.


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

UNKNOWN = "unknown"


class ArtifactFile(_Strict):
    name: str
    size_bytes: int | None = None
    sha256: str | None = None  # LFS sha256 from the HF API at the pinned revision


class Quantization(_Strict):
    method: str  # e.g. "Q4_K_M", "mlx-affine-4bit", "none"
    scheme: str | None = None
    group_size: int | None = None
    bits: int | None = None
    notes: str | None = None


class Provenance(_Strict):
    derived_from: str | None = None
    derived_from_revision: str | None = None
    verified: bool = False
    evidence: str | None = None


class RuntimeCompat(_Strict):
    status: Literal["verified", "expected", "attempt", "unsupported", "untested", "unknown"]
    evidence: str | None = None
    checked: str | None = None


class Artifact(_Strict):
    key: str
    format: Literal["gguf", "mlx", "safetensors"]
    repo: str
    revision: str
    files: list[ArtifactFile] = Field(default_factory=list)
    quantization: Quantization
    weight_dtype: str
    compute_dtype: str = "runtime default"
    kv_cache_dtype: str = "runtime default"
    platforms: list[Literal["mac", "nvidia"]]
    provenance: Provenance = Field(default_factory=Provenance)
    runtimes: dict[str, RuntimeCompat] = Field(default_factory=dict)
    notes: str | None = None

    @property
    def total_bytes(self) -> int | None:
        sizes = [f.size_bytes for f in self.files]
        return sum(s for s in sizes if s) if sizes and all(sizes) else None


class ReasoningControls(_Strict):
    supports_thinking: bool
    default_mode: Literal["thinking", "non_thinking", "none"]
    # How to disable/enable, per mechanism. Recorded rather than inferred.
    disable_chat_template_kwargs: dict[str, Any] | None = None
    enable_chat_template_kwargs: dict[str, Any] | None = None
    disable_system_flag: str | None = None
    ollama_think: bool | None = None
    notes: str | None = None


class ContextInfo(_Strict):
    native_max: int
    extended_max: int | None = None
    extension_method: str | None = None


class ArchInfo(_Strict):
    """From the pinned config.json. Used only for *estimates* (e.g. KV-cache bytes/token)."""

    layers: int
    kv_heads: int
    head_dim: int
    hidden_size: int
    num_experts: int | None = None

    def kv_bytes_per_token(self, bytes_per_elem: float = 2.0) -> float:
        # K and V, per layer, per KV head. Ignores allocator padding and runtime metadata.
        return 2 * self.layers * self.kv_heads * self.head_dim * bytes_per_elem


class ModelEntry(_Strict):
    key: str
    hf_id: str
    family: str
    tier: Literal["comparison", "capacity", "optional"]
    params_total_b: float
    params_active_b: float | None = None
    source_revision: str
    tokenizer_repo: str
    tokenizer_revision: str
    chat_template_sha256: str | None = None
    license: str
    access: Literal["open", "gated"]
    reasoning: ReasoningControls
    context: ContextInfo
    arch: ArchInfo | None = None
    artifacts: list[Artifact] = Field(default_factory=list)
    notes: str | None = None


class Registry(_Strict):
    version: int
    models: list[ModelEntry]

    @model_validator(mode="after")
    def _unique(self) -> Registry:
        keys = [m.key for m in self.models] + [a.key for m in self.models for a in m.artifacts]
        dup = {k for k in keys if keys.count(k) > 1}
        if dup:
            raise ValueError(f"duplicate registry keys: {sorted(dup)}")
        return self

    def model(self, key: str) -> ModelEntry:
        for m in self.models:
            if m.key == key or m.hf_id == key:
                return m
        raise KeyError(f"unknown model {key!r}")

    def artifact(self, key: str) -> tuple[ModelEntry, Artifact]:
        for m in self.models:
            for a in m.artifacts:
                if a.key == key:
                    return m, a
        raise KeyError(f"unknown artifact {key!r}; see `llm-sheet models list`")


def load_registry(path: Path | None = None) -> Registry:
    path = path or PROJECT_ROOT / "configs" / "models.yaml"
    return Registry.model_validate(yaml.safe_load(path.read_text()))


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------


class Workload(_Strict):
    name: str
    kind: Literal["synthetic_tokens", "application"] = "synthetic_tokens"
    input_tokens: int | None = None  # total rendered prompt tokens INCLUDING template overhead
    output_tokens: int | None = None  # target (max) output tokens
    concurrency: list[int] = Field(default_factory=lambda: [1])
    dataset: str | None = None  # application suite path (relative to project root)
    split: Literal["dev", "test"] | None = None
    max_tokens: int | None = None  # application runs: generous cap, natural stop

    @model_validator(mode="after")
    def _check(self) -> Workload:
        if self.kind == "synthetic_tokens" and (not self.input_tokens or not self.output_tokens):
            raise ValueError(f"workload {self.name}: synthetic_tokens needs input_tokens and output_tokens")
        if self.kind == "application" and not self.dataset:
            raise ValueError(f"workload {self.name}: application needs dataset")
        if any(c < 1 for c in self.concurrency):
            raise ValueError("concurrency must be >= 1")
        return self


class RequestCounts(_Strict):
    warmup: int = 5
    measured: int = 30

    @field_validator("warmup")
    @classmethod
    def _min_warmup(cls, v: int) -> int:
        if v < 5:
            raise ValueError("project methodology requires at least 5 warm-up requests")
        return v


class Sampling(_Strict):
    temperature: float = 0.0
    top_p: float = 1.0
    seed: int | None = None  # NOTE: mlx_lm.server disables batching when seed is set


class ExperimentConfig(_Strict):
    name: str
    description: str = ""
    kind: Literal["performance", "quality"]
    profile: Literal["smoke", "comparison", "extended", "capacity", "quality"]
    platform: Literal["mac", "nvidia"]
    hardware_label: str | None = None
    backends: list[str]
    artifacts: list[str]
    workloads: list[Workload]
    requests: RequestCounts = Field(default_factory=RequestCounts)
    repeats: int = 1
    sampling: Sampling = Field(default_factory=Sampling)
    thinking: Literal["disabled", "enabled"] = "disabled"
    length_enforcement: Literal["ignore_eos_if_supported", "natural"] = "ignore_eos_if_supported"
    cache_policy: Literal["no_reuse", "prefix_reuse"] = "no_reuse"
    request_timeout_s: float = 600
    readiness_timeout_s: float = 900
    max_cell_duration_s: float = 3600
    min_n_p95: int = 200
    min_n_p99: int = 1000
    backend_settings: dict[str, dict[str, Any]] = Field(default_factory=dict)
    # Skip (artifact, backend) pairs; e.g. mlx artifacts only run on mlx-lm.
    ports: dict[str, int] = Field(default_factory=dict)

    def identity(self) -> str:
        return stable_hash(self.model_dump(mode="json", exclude={"description"}))


def load_experiment(path: Path) -> ExperimentConfig:
    return ExperimentConfig.model_validate(yaml.safe_load(Path(path).read_text()))


def stable_hash(obj: Any, n: int = 16) -> str:
    blob = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:n]

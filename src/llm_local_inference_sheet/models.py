"""Artifact resolution: pinned download, local layout, sha256 verification, storage estimates."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from .config import PROJECT_ROOT, Artifact, ModelEntry, Registry


def models_dir() -> Path:
    return Path(os.environ.get("LLM_SHEET_MODELS_DIR") or PROJECT_ROOT / "models")


def artifact_dir(a: Artifact) -> Path:
    # <models>/<repo>/<revision>/, so two revisions never overwrite each other.
    return models_dir() / a.repo.replace("/", "__") / a.revision


def artifact_path(a: Artifact) -> Path:
    """Path passed to the runtime: the file for single-file GGUF, the directory otherwise."""
    d = artifact_dir(a)
    if a.format == "gguf":
        return d / a.files[0].name
    return d


def is_downloaded(a: Artifact) -> bool:
    d = artifact_dir(a)
    if not d.exists():
        return False
    return all((d / f.name).exists() and (f.size_bytes is None or (d / f.name).stat().st_size == f.size_bytes)
               for f in a.files) and bool(a.files)


def sha256_file(path: Path, chunk: int = 8 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def verify_artifact(a: Artifact, *, force: bool = False) -> dict[str, Any]:
    """Verify every pinned file's sha256. Results are cached in a sidecar keyed by (size, mtime)."""
    d = artifact_dir(a)
    sidecar = d / ".llm-sheet-verified.json"
    cache: dict[str, Any] = json.loads(sidecar.read_text()) if sidecar.exists() and not force else {}
    results = {}
    for f in a.files:
        p = d / f.name
        if not p.exists():
            results[f.name] = {"ok": False, "reason": "missing"}
            continue
        st = p.stat()
        key = f"{st.st_size}:{int(st.st_mtime)}"
        cached = cache.get(f.name)
        if cached and cached.get("key") == key:
            digest = cached["sha256"]
        else:
            digest = sha256_file(p)
            cache[f.name] = {"key": key, "sha256": digest}
        results[f.name] = {"ok": f.sha256 is None or digest == f.sha256, "sha256": digest, "expected": f.sha256}
    if d.exists():
        sidecar.write_text(json.dumps(cache, indent=2))
    return {"all_ok": all(r["ok"] for r in results.values()) and bool(results), "files": results}


def download_artifact(a: Artifact, token: str | None = None) -> Path:
    """Download exactly the pinned revision's files (plus small config/tokenizer files for MLX/HF dirs)."""
    from huggingface_hub import hf_hub_download, snapshot_download

    d = artifact_dir(a)
    d.mkdir(parents=True, exist_ok=True)
    token = token or os.environ.get("HF_TOKEN") or None
    if a.format == "gguf":
        for f in a.files:
            hf_hub_download(a.repo, f.name, revision=a.revision, local_dir=d, token=token)
    else:
        snapshot_download(a.repo, revision=a.revision, local_dir=d, token=token,
                          allow_patterns=["*.json", "*.safetensors", "*.jinja", "*.txt", "*.model", "*.tiktoken", "*.py"])
    return artifact_path(a)


def download_tokenizer(m: ModelEntry, token: str | None = None) -> Path:
    """Fetch only tokenizer + chat-template files at the pinned tokenizer revision."""
    from huggingface_hub import snapshot_download

    d = models_dir() / "tokenizers" / m.tokenizer_repo.replace("/", "__") / m.tokenizer_revision
    token = token or os.environ.get("HF_TOKEN") or None
    snapshot_download(m.tokenizer_repo, revision=m.tokenizer_revision, local_dir=d, token=token,
                      allow_patterns=["tokenizer*", "chat_template.jinja", "special_tokens_map.json",
                                      "vocab.json", "merges.txt", "config.json", "generation_config.json"])
    return d


def storage_plan(registry: Registry, artifact_keys: list[str], backends: list[str]) -> dict[str, Any]:
    """Estimate *additional* disk needed. Ollama import copies the GGUF into its blob store."""
    rows = []
    total_new = 0
    for key in artifact_keys:
        _, a = registry.artifact(key)
        size = a.total_bytes
        present = is_downloaded(a)
        extra_copies = 1 if ("ollama" in backends and a.format == "gguf") else 0
        need = (0 if present else (size or 0)) + (size or 0) * extra_copies
        total_new += need
        rows.append({"artifact": key, "size_bytes": size, "downloaded": present,
                     "ollama_blob_copy": bool(extra_copies), "additional_bytes": need,
                     "size_known": size is not None})
    return {"rows": rows, "additional_bytes_total": total_new}

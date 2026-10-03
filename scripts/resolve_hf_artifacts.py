#!/usr/bin/env python3
"""Resolve pinned revisions, file sizes/LFS sha256 and chat-template hashes from the HF API.

Used to author/refresh configs/models.yaml. Prints JSON; does not modify the registry.

    python scripts/resolve_hf_artifacts.py Qwen/Qwen3-8B Qwen/Qwen3-8B-GGUF:Qwen3-8B-Q4_K_M.gguf

A repo spec may carry ``:file1,file2`` to restrict listed files. ``@rev`` pins a revision.
Gated repos return ``{"error": ...}`` unless HF_TOKEN grants access.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import urllib.request

API = "https://huggingface.co/api/models/{repo}/revision/{rev}?blobs=true"
RAW = "https://huggingface.co/{repo}/resolve/{rev}/{path}"


def _get(url: str) -> bytes:
    req = urllib.request.Request(url)
    tok = os.environ.get("HF_TOKEN")
    if tok:
        req.add_header("Authorization", f"Bearer {tok}")
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


def chat_template_info(repo: str, rev: str, siblings: set[str]) -> dict:
    """Return sha256 of the effective HF chat template (chat_template.jinja wins over tokenizer_config)."""
    try:
        if "chat_template.jinja" in siblings:
            tpl = _get(RAW.format(repo=repo, rev=rev, path="chat_template.jinja")).decode()
            src = "chat_template.jinja"
        elif "tokenizer_config.json" in siblings:
            cfg = json.loads(_get(RAW.format(repo=repo, rev=rev, path="tokenizer_config.json")))
            tpl = cfg.get("chat_template")
            if isinstance(tpl, list):  # named templates
                tpl = next((t["template"] for t in tpl if t.get("name") == "default"), None)
            src = "tokenizer_config.json"
        else:
            return {"chat_template_sha256": None, "source": None}
        if not tpl:
            return {"chat_template_sha256": None, "source": src}
        return {
            "chat_template_sha256": hashlib.sha256(tpl.encode()).hexdigest(),
            "source": src,
            "mentions_enable_thinking": "enable_thinking" in tpl,
            "mentions_no_think": "/no_think" in tpl,
        }
    except Exception as e:
        return {"chat_template_sha256": None, "error": str(e)}


def resolve(spec: str) -> dict:
    files_filter: list[str] | None = None
    if ":" in spec:
        spec, f = spec.split(":", 1)
        files_filter = f.split(",")
    rev = "main"
    if "@" in spec:
        spec, rev = spec.split("@", 1)
    try:
        d = json.loads(_get(API.format(repo=spec, rev=rev)))
    except Exception as e:
        return {"repo": spec, "error": str(e)}
    siblings = {s["rfilename"] for s in d.get("siblings", [])}
    files = []
    for s in d.get("siblings", []):
        name = s["rfilename"]
        if files_filter is not None and name not in files_filter:
            continue
        if files_filter is None and not name.endswith((".gguf", ".safetensors")):
            continue
        files.append({
            "name": name,
            "size_bytes": s.get("size"),
            "sha256": (s.get("lfs") or {}).get("sha256"),
        })
    card = d.get("cardData") or {}
    return {
        "repo": spec,
        "revision": d.get("sha"),
        "last_modified": d.get("lastModified"),
        "gated": d.get("gated"),
        "license": card.get("license"),
        "base_model": card.get("base_model"),
        "files": files,
        "total_bytes": sum(f["size_bytes"] or 0 for f in files),
        **chat_template_info(spec, d.get("sha") or rev, siblings),
    }


if __name__ == "__main__":
    print(json.dumps([resolve(s) for s in sys.argv[1:]], indent=2))

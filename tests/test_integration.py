"""Opt-in smoke tests against REAL installed backends. Never run in CI.

    LLM_SHEET_INTEGRATION=1 pytest -m integration -k llama

They require downloaded artifacts and write to a temp results dir.
"""

import os

import pytest

from llm_local_inference_sheet.backends import get_backend
from llm_local_inference_sheet.bench import CellRunner
from llm_local_inference_sheet.config import ExperimentConfig, load_registry
from llm_local_inference_sheet.models import is_downloaded

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(os.environ.get("LLM_SHEET_INTEGRATION") != "1", reason="set LLM_SHEET_INTEGRATION=1"),
]

CASES = [("llama-cpp", "qwen3-4b-q4"), ("mlx-lm", "qwen3-4b-mlx4"), ("ollama", "qwen3-4b-q4")]


@pytest.mark.parametrize(("backend", "artifact"), CASES)
def test_backend_smoke(tmp_path, backend, artifact):
    if not get_backend(backend).availability().ok:
        pytest.skip(f"{backend} not installed")
    _, a = load_registry().artifact(artifact)
    if not is_downloaded(a):
        pytest.skip(f"{artifact} not downloaded")
    exp = ExperimentConfig.model_validate({
        "name": f"integration-{backend}", "kind": "performance", "profile": "smoke",
        "platform": get_backend(backend).platform if get_backend(backend).platform != "any" else "mac",
        "backends": [backend], "artifacts": [artifact],
        "workloads": [{"name": "tiny", "input_tokens": 128, "output_tokens": 16, "concurrency": [1]}],
        "requests": {"warmup": 5, "measured": 3},
    })
    out = CellRunner(exp, load_registry(), tmp_path).run_all()
    assert out[0]["state"] == "measured", out

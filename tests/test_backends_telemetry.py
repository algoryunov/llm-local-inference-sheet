from pathlib import Path

from llm_local_inference_sheet.backends.llama_cpp import LlamaCppBackend, parse_log_facts
from llm_local_inference_sheet.bench import expand_cells
from llm_local_inference_sheet.config import ExperimentConfig, RuntimeCompat, load_registry
from llm_local_inference_sheet.telemetry import memory_contamination

FIX = Path(__file__).parent / "fixtures" / "llama_server_b11246_excerpt.log"


def test_llama_log_facts_from_real_excerpt():
    f = parse_log_facts(FIX.read_text())
    assert f["offloaded_layers"] == "37/37"
    assert f["kv_unified"] == "false" and f["flash_attn"] == "enabled"
    assert f["kv_buffer"] == "216.00 MiB"
    bd = LlamaCppBackend().final_log_facts(FIX)["gpu_memory_breakdown_mib"]
    assert bd["self"] == bd["model"] + bd["context_kv"] + bd["compute"]
    assert bd["device_total"] == 13639


def test_kv_estimate_matches_llama_kv_buffer():
    m = load_registry().model("qwen3-8b")
    # 1536-token context -> llama.cpp reported 216.00 MiB KV buffer (f16)
    assert m.arch.kv_bytes_per_token() * 1536 / 2**20 == 216.0


def test_memory_contamination_flags():
    assert memory_contamination({"vm_pressure_level": 1}, {"peak_vm_pressure_level": 1, "swap_in_bytes_delta": 0,
                                                           "swap_out_bytes_delta": 0})["contaminated"] is False
    c = memory_contamination({"vm_pressure_level": 2}, {"swap_in_bytes_delta": 2**30})
    assert c["contaminated"] and len(c["reasons"]) == 2
    assert memory_contamination({}, {})["contaminated"] is None


def test_registry_unsupported_blocks_cell():
    reg = load_registry()
    _, art = reg.artifact("smollm3-3b-q4")
    art.runtimes["ollama"] = RuntimeCompat(status="unsupported", evidence="unknown model architecture")
    exp = ExperimentConfig.model_validate({
        "name": "t", "kind": "performance", "profile": "smoke", "platform": "mac", "backends": ["ollama"],
        "artifacts": ["smollm3-3b-q4"],
        "workloads": [{"name": "w", "input_tokens": 64, "output_tokens": 8}]})
    cell = expand_cells(exp, reg, check_availability=False)[0]
    assert cell.pre_state == "unsupported" and "registry" in cell.pre_reason


def test_detect_unsupported_architecture():
    from llm_local_inference_sheet.backends.base import detect_unsupported

    log = "llama_model_load: error loading model: error loading model architecture: unknown model architecture: 'smollm3'"
    assert detect_unsupported(log) == "unknown model architecture: 'smollm3'"
    assert detect_unsupported("HTTP 503 loading") is None

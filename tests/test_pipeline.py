"""End-to-end: mock backend -> run dirs -> report regeneration. Everything here is SYNTHETIC."""

import json
from pathlib import Path

import pytest

from llm_local_inference_sheet.bench import CellRunner, expand_cells, plan
from llm_local_inference_sheet.config import ExperimentConfig, load_experiment, load_registry
from llm_local_inference_sheet.reports import build_report, collect_runs

ROOT = Path(__file__).resolve().parents[1]


def mock_experiment(**over):
    cfg = {
        "name": "synthetic-pipeline-test", "kind": "performance", "profile": "smoke", "platform": "mac",
        "backends": ["mock"], "artifacts": ["qwen3-8b-q4"],
        "workloads": [{"name": "tiny", "input_tokens": 64, "output_tokens": 8, "concurrency": [1, 2]}],
        "requests": {"warmup": 5, "measured": 6},
        "backend_settings": {"mock": {"behavior": {"ttft_s": 0.005, "token_interval_s": 0.001}}},
    }
    cfg.update(over)
    return ExperimentConfig.model_validate(cfg)


def test_all_repo_configs_validate():
    reg = load_registry()
    paths = sorted((ROOT / "configs" / "experiments").glob("*.yaml"))
    assert paths
    for p in paths:
        exp = load_experiment(p)
        for a in exp.artifacts:
            reg.artifact(a)


def test_warmup_minimum_enforced():
    with pytest.raises(ValueError):
        mock_experiment(requests={"warmup": 2, "measured": 5})


def test_expand_states():
    reg = load_registry()
    exp = mock_experiment(backends=["llama-cpp", "mlx-lm", "ollama"], artifacts=["qwen3-8b-mlx4", "llama3.1-8b-q4"])
    cells = {c.cell_id: c for c in expand_cells(exp, reg, check_availability=False)}
    assert cells["llama-cpp__qwen3-8b-mlx4__tiny__c1"].pre_state == "unsupported"
    assert cells["ollama__qwen3-8b-mlx4__tiny__c1"].pre_state == "unsupported"


def test_end_to_end_and_report_regeneration(tmp_path, monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    reg = load_registry()
    exp = mock_experiment()
    p = plan(exp, reg, tmp_path)
    assert p["n_runnable"] == 2
    out = CellRunner(exp, reg, tmp_path).run_all()
    assert [o["state"] for o in out] == ["synthetic", "synthetic"]
    # resume: identical config is skipped, previous observations retained
    out2 = CellRunner(exp, reg, tmp_path).run_all()
    assert all(o.get("skipped") for o in out2)
    run_dirs = sorted(tmp_path.rglob("manifest.json"))
    assert len(run_dirs) == 2
    run = run_dirs[0].parent
    for f in ("effective_config.json", "workload.json", "requests.jsonl", "events.jsonl.gz", "telemetry.jsonl",
              "summary.json", "server.log.gz"):
        assert (run / f).exists(), f
    summ = json.loads((run / "summary.json").read_text())
    assert summ["performance"]["n_ok"] == 6

    # Synthetic runs are excluded from empirical reports by default ...
    assert collect_runs([tmp_path]) == []
    # ... and when included, labelled, and regeneration is deterministic.
    r1 = build_report([tmp_path], tmp_path / "rep1", include_synthetic=True)
    r2 = build_report([tmp_path], tmp_path / "rep2", include_synthetic=True)
    assert (tmp_path / "rep1" / "results.md").read_text() == (tmp_path / "rep2" / "results.md").read_text()
    assert "SYNTHETIC" in (tmp_path / "rep1" / "results.md").read_text()
    assert r1.read_text() == r2.read_text()
    csv_text = (tmp_path / "rep1" / "results.csv").read_text()
    assert "synthetic" in csv_text


def test_pre_state_cells_are_recorded(tmp_path, monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    exp = mock_experiment(name="synthetic-gated", backends=["mock"], artifacts=["llama3.1-8b-q4"],
                          workloads=[{"name": "tiny", "input_tokens": 64, "output_tokens": 8, "concurrency": [1]}])
    out = CellRunner(exp, load_registry(), tmp_path).run_all()
    assert out[0]["state"] == "gated"


def test_deadline_skips_are_flagged_and_fill_missing_completes_them(tmp_path, monkeypatch):
    """A cell that hits max_cell_duration_s must not score skipped items as failures; --fill-missing completes them."""
    monkeypatch.delenv("HF_TOKEN", raising=False)
    from llm_local_inference_sheet.bench import merged_run_data

    reg = load_registry()
    cfg = {
        "name": "synthetic-fill", "kind": "quality", "profile": "quality", "platform": "mac",
        "backends": ["mock"], "artifacts": ["qwen3-4b-q4"],
        "workloads": [{"name": "app", "kind": "application", "dataset": "datasets/app-v1", "split": "test",
                       "max_tokens": 8, "concurrency": [1]}],
        "requests": {"warmup": 5, "measured": 60}, "length_enforcement": "natural",
        "max_cell_duration_s": 2.0,
        "backend_settings": {"mock": {"behavior": {"ttft_s": 0.02, "token_interval_s": 0.01}}},
    }
    exp = ExperimentConfig.model_validate(cfg)
    CellRunner(exp, reg, tmp_path).run_all()
    run = next(tmp_path.rglob("manifest.json")).parent
    man = json.loads((run / "manifest.json").read_text())
    assert man["incomplete"]["not_dispatched"] > 0
    scores = [json.loads(x) for x in (run / "scores.jsonl").open()]
    assert len(scores) < 60 and all(s["status"] == "ok" for s in scores)

    exp2 = ExperimentConfig.model_validate({**cfg, "max_cell_duration_s": 600})
    out = CellRunner(exp2, reg, tmp_path, fill_missing=True).run_all()
    assert out[0]["state"] == "synthetic"
    records, _, n_filled = merged_run_data(run)
    measured = [r for r in records if r["phase"] == "measured"]
    assert n_filled > 0 and len({r["item_id"] for r in measured}) == 60
    assert not any(r["status"] in ("not_dispatched", "cancelled") for r in measured)
    rows = collect_runs([tmp_path], include_synthetic=True)
    assert len(rows) == 1 and rows[0]["filled_items"] == n_filled and rows[0]["n_unanswered"] == 0


def test_missing_dataset_is_untested_with_hint(tmp_path):
    exp = ExperimentConfig.model_validate({
        "name": "synthetic-nodata", "kind": "quality", "profile": "quality", "platform": "mac",
        "backends": ["mock"], "artifacts": ["qwen3-4b-q4"],
        "workloads": [{"name": "p", "kind": "application", "dataset": ".cache/datasets/does-not-exist-proofread",
                       "split": "test", "concurrency": [1]}]})
    cell = expand_cells(exp, load_registry(), check_availability=False)[0]
    assert cell.pre_state == "untested" and "prepare_proofread.py" in cell.pre_reason

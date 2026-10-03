import json
from pathlib import Path

from llm_local_inference_sheet.evaluation import (
    aggregate_scores,
    macro_f1,
    parse_json_output,
    score_item,
)

DATA = Path(__file__).resolve().parents[1] / "datasets" / "app-v1"


def items(task, lang="en", split="test"):
    return [json.loads(line) for line in (DATA / f"{split}.jsonl").open()
            if json.loads(line)["task"] == task and json.loads(line)["lang"] == lang]


def test_parse_levels():
    assert parse_json_output('{"a": 1}')[1:] == (True, True)
    obj, strict, lenient = parse_json_output('Sure!\n```json\n{"a": "}"}\n```')
    assert obj == {"a": "}"} and not strict and lenient
    assert parse_json_output("no json here")[0] is None


def test_extraction_gold_scores_perfectly_and_nulls_matter():
    it = items("extraction")[0]
    s = score_item(it, json.dumps(it["expected"], ensure_ascii=False))
    assert s["schema_valid"] and s["exact_all"] and s["field_accuracy"] == 1.0 and s["parse_strict"]
    wrong = dict(it["expected"])
    wrong["quantity"] = 999
    s2 = score_item(it, json.dumps(wrong))
    assert not s2["exact_all"] and s2["fields"]["quantity"] is False


def test_extraction_numeric_string_is_semantically_right_but_schema_invalid():
    it = next(i for i in items("extraction") if i["expected"]["quantity"] is not None)
    pred = dict(it["expected"])
    pred["quantity"] = str(pred["quantity"])
    s = score_item(it, json.dumps(pred, ensure_ascii=False))
    assert s["fields"]["quantity"] and not s["schema_valid"]


def test_routing_metrics():
    rows = items("routing")
    scores = [score_item(it, json.dumps(it["expected"])) for it in rows]
    agg = aggregate_scores(scores)
    assert agg["routing/all"]["accuracy"] == 1.0 and agg["routing/all"]["macro_f1"] == 1.0
    none_item = next(i for i in rows if i["expected"]["action"] == "none")
    s = score_item(none_item, '{"action": "create_ticket"}')
    assert s["unnecessary_tool_call"] and not s["correct"]


def test_macro_f1_hand_computed():
    g = ["a", "a", "b", "b"]
    p = ["a", "b", "b", "b"]
    # a: P=1 R=.5 F=.667 ; b: P=.667 R=1 F=.8 -> mean .7333
    assert abs(macro_f1(g, p, ["a", "b"]) - 0.7333) < 1e-3


def test_cv_gold_and_hallucinated_refs():
    it = items("cv_report")[0]
    pred = dict(it["expected"], summary="A person entered a restricted zone.")
    s = score_item(it, json.dumps(pred))
    assert s["exact_all"] and s["schema_valid"] and s["hallucinated_refs"] == 0
    pred["event_refs"] = [*pred["event_refs"], "e99"]
    s2 = score_item(it, json.dumps(pred))
    assert s2["hallucinated_refs"] == 1 and not s2["fields"]["event_refs"]


def test_dataset_splits_disjoint_and_versioned():
    dev = {json.loads(x)["id"] for x in (DATA / "dev.jsonl").open()}
    test = {json.loads(x)["id"] for x in (DATA / "test.jsonl").open()}
    assert dev.isdisjoint(test) and len(test) == 60 and len(dev) == 18
    man = json.loads((DATA / "manifest.json").read_text())
    assert man["synthetic"] is True and man["languages"] == ["en"]

"""Deterministic scoring for the application suite (datasets/app-v1).

No model judge (paid or local) is used. Free-text CV summaries are exported for
human rating against docs/rubrics/cv-summary.md and are not auto-scored.

Parsing is reported at two levels:
  * strict: the whole response is one JSON object (``json.loads`` on the stripped text);
  * lenient: the first balanced ``{...}`` block after removing Markdown code fences.
Field scores use the lenient parse, so formatting noise and semantic errors are counted separately.
Schema validity is checked separately from semantic correctness.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from collections.abc import Iterable
from typing import Any

from .streaming import normalize_payload

EXTRACTION_KEYS = ("customer_name", "order_id", "product", "quantity", "delivery_date", "priority")
ACTIONS = ["create_ticket", "check_order_status", "cancel_order", "update_address", "escalate_to_human", "none"]
CV_FIELDS = ("incident_type", "start_time", "end_time", "camera_ids", "event_refs", "person_count",
             "vehicle_involved", "uncertain", "missing_evidence")
CV_SET_FIELDS = {"camera_ids", "event_refs", "missing_evidence"}
_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


# ---------------------------------------------------------------- parsing
def parse_json_output(text: str) -> tuple[Any, bool, bool]:
    """Return (obj_or_None, strict_ok, lenient_ok)."""
    s = text.strip()
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return obj, True, True
    except json.JSONDecodeError:
        pass
    s2 = re.sub(r"```(?:json)?", "", s)
    start = s2.find("{")
    while start != -1:
        depth = 0
        in_str = False
        esc = False
        for i in range(start, len(s2)):
            c = s2[i]
            if in_str:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    in_str = False
                continue
            if c == '"':
                in_str = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(s2[start : i + 1])
                        if isinstance(obj, dict):
                            return obj, False, True
                    except json.JSONDecodeError:
                        pass
                    break
        start = s2.find("{", start + 1)
    return None, False, False


def _norm(v: Any) -> Any:
    if isinstance(v, str):
        return " ".join(v.strip().casefold().split())
    return v


def _num_equal(pred: Any, gold: Any) -> bool:
    if gold is None:
        return pred is None
    if isinstance(pred, bool):
        return False
    if isinstance(pred, int | float) and float(pred) == float(gold):
        return True
    if isinstance(pred, str) and pred.strip().isdigit():
        return int(pred.strip()) == gold
    return False


# ---------------------------------------------------------------- task scorers
def score_extraction(obj: dict[str, Any] | None, gold: dict[str, Any], tags: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {"schema_valid": False, "fields": {}, "exact_all": False}
    if obj is None:
        out["fields"] = {k: False for k in EXTRACTION_KEYS}
        out["field_accuracy"] = 0.0
        out["null_handling"] = _null_handling({}, gold, tags)
        return out
    schema = set(obj) == set(EXTRACTION_KEYS)
    types_ok = (
        all(isinstance(obj.get(k), str | type(None)) for k in ("customer_name", "order_id", "product"))
        and (obj.get("quantity") is None or (isinstance(obj.get("quantity"), int) and not isinstance(obj.get("quantity"), bool)))
        and (obj.get("delivery_date") is None or (isinstance(obj.get("delivery_date"), str) and bool(_DATE.match(obj["delivery_date"]))))
        and obj.get("priority") in (None, "low", "normal", "high")
    )
    out["schema_valid"] = schema and types_ok
    for k in EXTRACTION_KEYS:
        p, g = obj.get(k), gold[k]
        ok = _num_equal(p, g) if k == "quantity" else _norm(p) == _norm(g)
        out["fields"][k] = ok
    out["field_accuracy"] = sum(out["fields"].values()) / len(EXTRACTION_KEYS)
    out["exact_all"] = all(out["fields"].values())
    out["null_handling"] = _null_handling(obj, gold, tags)
    return out


def _null_handling(obj: dict[str, Any], gold: dict[str, Any], tags: list[str]) -> dict[str, Any]:
    res: dict[str, Any] = {}
    for t in tags:
        kind, _, field = t.partition(":")
        if kind in ("absent", "ambiguous"):
            res[t] = obj.get(field, "MISSING_KEY") is None
    return res


def score_routing(obj: dict[str, Any] | None, gold: dict[str, Any]) -> dict[str, Any]:
    pred = obj.get("action") if isinstance(obj, dict) else None
    pred = pred if pred in ACTIONS else "invalid"
    g = gold["action"]
    return {
        "pred": pred,
        "gold": g,
        "correct": pred == g,
        "schema_valid": isinstance(obj, dict) and set(obj) == {"action"} and pred != "invalid",
        "unnecessary_tool_call": g == "none" and pred not in ("none", "invalid"),
        "missed_tool_call": g != "none" and pred == "none",
    }


def score_cv(obj: dict[str, Any] | None, gold: dict[str, Any], log_ids: set[str]) -> dict[str, Any]:
    out: dict[str, Any] = {"fields": {}, "schema_valid": False}
    if obj is None:
        out["fields"] = {k: False for k in CV_FIELDS}
        out["field_accuracy"] = 0.0
        out["exact_all"] = False
        out["hallucinated_refs"] = 0
        out["summary"] = None
        return out
    out["schema_valid"] = set(CV_FIELDS) | {"summary"} == set(obj)
    for k in CV_FIELDS:
        p, g = obj.get(k), gold[k]
        if k in CV_SET_FIELDS:
            ok = isinstance(p, list) and {_norm(x) for x in p} == {_norm(x) for x in g}
        elif k == "person_count":
            ok = _num_equal(p, g)
        elif isinstance(g, bool):
            ok = p is g
        else:
            ok = _norm(p) == _norm(g)
        out["fields"][k] = ok
    out["field_accuracy"] = sum(out["fields"].values()) / len(CV_FIELDS)
    out["exact_all"] = all(out["fields"].values())
    raw_refs = obj.get("event_refs")
    refs: list[Any] = raw_refs if isinstance(raw_refs, list) else []
    out["hallucinated_refs"] = len([r for r in refs if str(r) not in log_ids])
    s = obj.get("summary")
    out["summary"] = s if isinstance(s, str) else None
    return out


# ---------------------------------------------------------------- per request
def response_text(protocol: str, events: Iterable[list[Any]]) -> tuple[str, str]:
    """Reassemble (content, reasoning) from raw stream events."""
    content, reasoning = [], []
    for _, payload in events:
        ch = normalize_payload(protocol, payload)
        if ch is None:
            continue
        content.append(ch.content)
        reasoning.append(ch.reasoning)
    return "".join(content), "".join(reasoning)


def score_item(item: dict[str, Any], content: str) -> dict[str, Any]:
    if item["task"] in ("gec", "style"):
        from .proofread import score_gec, score_style

        res = score_gec(item, content) if item["task"] == "gec" else score_style(item, content)
        res.update({"item_id": item["id"], "task": item["task"], "lang": item["lang"],
                    "direction": item["expected"].get("direction")})
        return res
    obj, strict, lenient = parse_json_output(content)
    base = {"item_id": item["id"], "task": item["task"], "lang": item["lang"],
            "parse_strict": strict, "parse_lenient": lenient}
    if item["task"] == "extraction":
        base.update(score_extraction(obj, item["expected"], item.get("tags", [])))
    elif item["task"] == "routing":
        base.update(score_routing(obj, item["expected"]))
    elif item["task"] == "cv_report":
        log = json.loads(item["messages"][-1]["content"].split("\n", 1)[1])
        base.update(score_cv(obj, item["expected"], {e["id"] for e in log["events"]}))
    return base


# ---------------------------------------------------------------- aggregation
def macro_f1(golds: list[str], preds: list[str], labels: list[str]) -> float:
    f1s = []
    for lab in labels:
        tp = sum(1 for g, p in zip(golds, preds, strict=True) if g == lab and p == lab)
        fp = sum(1 for g, p in zip(golds, preds, strict=True) if g != lab and p == lab)
        fn = sum(1 for g, p in zip(golds, preds, strict=True) if g == lab and p != lab)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return sum(f1s) / len(f1s)


def aggregate_scores(scores: list[dict[str, Any]]) -> dict[str, Any]:
    if scores and all(s["task"] in ("gec", "style") for s in scores):
        from .proofread import aggregate_proofread

        return aggregate_proofread(scores)
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for s in scores:
        groups[(s["task"], s["lang"])].append(s)
        groups[(s["task"], "all")].append(s)
    out: dict[str, Any] = {}
    for (task, lang), rows in sorted(groups.items()):
        n = len(rows)
        agg: dict[str, Any] = {
            "n": n,
            "parse_strict": sum(r["parse_strict"] for r in rows) / n,
            "parse_lenient": sum(r["parse_lenient"] for r in rows) / n,
            "schema_valid": sum(bool(r.get("schema_valid")) for r in rows) / n,
        }
        if task == "extraction":
            agg["field_accuracy"] = sum(r["field_accuracy"] for r in rows) / n
            agg["exact_all"] = sum(r["exact_all"] for r in rows) / n
            for k in EXTRACTION_KEYS:
                agg[f"field:{k}"] = sum(r["fields"].get(k, False) for r in rows) / n
            nh = [v for r in rows for v in r["null_handling"].values()]
            agg["null_handling_accuracy"] = sum(nh) / len(nh) if nh else None
            agg["primary"] = agg["field_accuracy"]
        elif task == "routing":
            golds = [r["gold"] for r in rows]
            preds = [r["pred"] for r in rows]
            agg["accuracy"] = sum(r["correct"] for r in rows) / n
            agg["macro_f1"] = macro_f1(golds, preds, ACTIONS)
            agg["unnecessary_tool_calls"] = sum(r["unnecessary_tool_call"] for r in rows)
            agg["missed_tool_calls"] = sum(r["missed_tool_call"] for r in rows)
            agg["n_gold_none"] = sum(1 for g in golds if g == "none")
            agg["primary"] = agg["macro_f1"]
        elif task == "cv_report":
            agg["field_accuracy"] = sum(r["field_accuracy"] for r in rows) / n
            agg["exact_all"] = sum(r["exact_all"] for r in rows) / n
            for k in CV_FIELDS:
                agg[f"field:{k}"] = sum(r["fields"].get(k, False) for r in rows) / n
            agg["hallucinated_refs_total"] = sum(r["hallucinated_refs"] for r in rows)
            agg["summaries_present"] = sum(1 for r in rows if r.get("summary")) / n
            agg["primary"] = agg["field_accuracy"]
        out[f"{task}/{lang}"] = agg
    prim = [out[f"{t}/all"]["primary"] for t in ("extraction", "routing", "cv_report") if f"{t}/all" in out]
    out["composite"] = {"primary_mean": sum(prim) / len(prim) if prim else None, "tasks": len(prim),
                        "definition": "mean of extraction field accuracy, routing macro-F1, CV field accuracy"}
    return out

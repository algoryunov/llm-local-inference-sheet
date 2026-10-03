"""Scoring for the proofreading suite (proofread-v1). Deterministic; no model judge.

GEC (grammatical error correction) is scored at the *edit* level, as in BEA-2019 / CoNLL-2014:
the system's edits (source → output) are compared with each reference's edits (source → reference).
  * English: ERRANT edits (spaCy en_core_web_sm), matched on (start, end, correction). This is the standard GEC scorer.
  * Other languages: edits come from a token-level diff (difflib) with the same matching rule. That is reported
    as "token-diff F0.5" and is not comparable to ERRANT numbers.
Corpus precision/recall/F0.5 come from summed TP/FP/FN. With several references, each sentence uses
the reference that gives the highest sentence-level F0.5 (ties: more TP), the usual multi-reference choice.
Controls (source already correct) measure over-correction: keep rate = share of controls left with zero edits.

Style rewrites are checked with constraints: every fact preserved (equivalent forms accepted: 24h/12h times,
thousands separators, a leading article; the strict verbatim rate is reported too), text actually changed, output in the
same writing system as the source, and for "formal": no listed slang, no emoji, and (English) no contractions.
"""

from __future__ import annotations

import difflib
import re
import unicodedata
from functools import lru_cache
from typing import Any

_TOKEN = re.compile(r"\w+|[^\w\s]", re.UNICODE)
_PREAMBLE_EN = r"^(here is|here's|corrected( text)?|rewritten)"
_CONTRACTION = re.compile(r"\b\w+(n't|'re|'ll|'ve|'m|'d)\b|\b(it's|that's|there's|what's|let's)\b", re.IGNORECASE)
_EMOJI = re.compile("[\U0001F300-\U0001FAFF☀-➿]")


def clean_output(text: str, source: str, preamble_regex: str | None = None) -> tuple[str, bool]:
    """Strip whitespace and wrapping quotes the source didn't have; flag chatty preambles (not removed).

    Items may carry their own ``preamble_regex`` (e.g. for a non-English language); English patterns always apply.
    """
    t = text.strip()
    for q in ('"', "«", "“", "'"):
        close = {"«": "»", "“": "”"}.get(q, q)
        if len(t) > 1 and t.startswith(q) and t.endswith(close) and not source.strip().startswith(q):
            t = t[1:-1].strip()
    patterns = [_PREAMBLE_EN] + ([preamble_regex] if preamble_regex else [])
    return t, any(re.match(p, t, re.IGNORECASE) for p in patterns)


@lru_cache(maxsize=1)
def _errant() -> Any:
    import errant

    return errant.load("en")


def _edits_en(src: str, tgt: str) -> set[tuple[int, int, str]]:
    a = _errant()
    # tokenise=True: JFLEG sources/references are pre-tokenized ("school .") while models write normal text
    # ("school."). Whitespace splitting would turn every detokenized punctuation mark into a false edit.
    o, c = a.parse(src, tokenise=True), a.parse(tgt, tokenise=True)
    return {(e.o_start, e.o_end, e.c_str.lower()) for e in a.annotate(o, c) if e.type != "noop"}


def _edits_tokens(src: str, tgt: str) -> set[tuple[int, int, str]]:
    s, t = _TOKEN.findall(src), _TOKEN.findall(tgt)
    out = set()
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(a=s, b=t, autojunk=False).get_opcodes():
        if tag != "equal":
            out.add((i1, i2, " ".join(t[j1:j2]).lower()))
    return out


def _f(tp: int, fp: int, fn: int, beta: float = 0.5) -> float:
    p = tp / (tp + fp) if tp + fp else 1.0
    r = tp / (tp + fn) if tp + fn else 1.0
    b2 = beta * beta
    return (1 + b2) * p * r / (b2 * p + r) if p + r else 0.0


def score_gec(item: dict[str, Any], content: str) -> dict[str, Any]:
    src = item["expected"]["source"]
    out, preamble = clean_output(content, src, item["expected"].get("preamble_regex"))
    edits = _edits_en if item["lang"] == "en" else _edits_tokens
    hyp = edits(src, out) if out else edits(src, "")
    best: tuple[float, int, int, int, int] | None = None
    for ref in item["expected"]["refs"]:
        gold = edits(src, ref)
        tp = len(hyp & gold)
        fp = len(hyp - gold)
        fn = len(gold - hyp)
        cand = (_f(tp, fp, fn), tp, fp, fn, len(gold))
        if best is None or (cand[0], cand[1]) > (best[0], best[1]):
            best = cand
    assert best is not None
    norm = lambda x: " ".join(x.split())  # noqa: E731
    return {
        "metric": "errant" if item["lang"] == "en" else "token_diff",
        "tp": best[1], "fp": best[2], "fn": best[3], "gold_edits": best[4], "hyp_edits": len(hyp),
        "control": any(t.startswith("control") for t in item.get("tags", [])),
        "exact_ref": any(norm(out) == norm(r) for r in item["expected"]["refs"]),
        "unchanged": len(hyp) == 0,
        "preamble": preamble,
        "empty": not out,
        "output": out,
    }


def _script(text: str) -> str | None:
    """Dominant writing system of the letters in ``text`` (first word of the Unicode character name)."""
    counts: dict[str, int] = {}
    for ch in text:
        if ch.isalpha():
            sc = unicodedata.name(ch, "UNKNOWN").split(" ")[0]
            counts[sc] = counts.get(sc, 0) + 1
    return max(counts, key=lambda k: counts[k]) if counts else None


_TIME = re.compile(r"^(\d{1,2}):(\d{2})$")


def fact_variants(fact: str) -> list[str]:
    """Equivalent surface forms of a fact: 24h/12h times, thousands separators, a leading article."""
    f = fact.lower()
    out = [f]
    m = _TIME.match(f)
    if m:
        h, mm = int(m.group(1)), m.group(2)
        h12 = h % 12 or 12
        suffix = "pm" if h >= 12 else "am"
        out += [f"{h12}:{mm} {suffix}", f"{h12}:{mm}{suffix}", f"{h12}:{mm}"]
        if mm == "00":
            out += [f"{h12} {suffix}", f"{h12}{suffix}", f"{h12} o'clock"]
    elif f.isdigit() and len(f) >= 4:
        out += [f"{int(f):,}", f"{int(f):,}".replace(",", " "), f"{int(f):,}".replace(",", ".")]
    if f.startswith("the "):
        out.append(f[4:])
    return out


def _fact_present(fact: str, low: str) -> bool:
    return any(v in low for v in fact_variants(fact))


def score_style(item: dict[str, Any], content: str) -> dict[str, Any]:
    exp = item["expected"]
    out, preamble = clean_output(content, exp["source"], exp.get("preamble_regex"))
    low = f" {out.lower()} "
    facts_ok = [_fact_present(f, low.replace("\u202f", " ")) for f in exp["facts"]]
    facts_strict = [f.lower() in low for f in exp["facts"]]
    changed = " ".join(out.split()).lower() != " ".join(exp["source"].split()).lower()
    checks = {"facts_preserved": all(facts_ok), "changed": changed, "script_kept": out != "" and _script(out) == _script(exp["source"]),
              "no_preamble": not preamble}
    if exp["direction"] == "formal":
        checks["no_slang"] = not any(s.strip().lower() and f" {s.strip().lower()} " in re.sub(r"[^\w\s/]", " ", low)
                                     for s in exp["slang"])
        checks["no_emoji"] = not _EMOJI.search(out)
        if item["lang"] == "en":
            checks["no_contractions"] = not _CONTRACTION.search(out)
    return {"checks": checks, "pass": all(checks.values()), "facts_kept_ratio": sum(facts_ok) / max(1, len(facts_ok)),
            "facts_preserved_verbatim": all(facts_strict), "output": out}


def aggregate_proofread(scores: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    langs = sorted({s["lang"] for s in scores})
    for lang in langs:
        rows = [s for s in scores if s["task"] == "gec" and s["lang"] == lang]
        if not rows:
            continue
        err = [s for s in rows if not s["control"]]
        ctrl = [s for s in rows if s["control"]]
        tp, fp, fn = (sum(s[k] for s in err) for k in ("tp", "fp", "fn"))
        p = tp / (tp + fp) if tp + fp else None
        r = tp / (tp + fn) if tp + fn else None
        out[f"gec/{lang}"] = {
            "metric": err[0]["metric"] if err else None, "n": len(err), "n_controls": len(ctrl),
            "precision": p, "recall": r, "f0_5": _f(tp, fp, fn) if err else None,
            "exact_ref": sum(s["exact_ref"] for s in err) / len(err) if err else None,
            "keep_rate": sum(s["unchanged"] for s in ctrl) / len(ctrl) if ctrl else None,
            "control_edits_mean": sum(s["hyp_edits"] for s in ctrl) / len(ctrl) if ctrl else None,
            "preamble_rate": sum(s["preamble"] for s in rows) / len(rows),
            "primary": _f(tp, fp, fn) if err else None,
        }
    for lang in langs:
        rows = [s for s in scores if s["task"] == "style" and s["lang"] == lang]
        if not rows:
            continue
        agg: dict[str, Any] = {"n": len(rows), "pass_rate": sum(s["pass"] for s in rows) / len(rows),
                               "facts_verbatim_rate": sum(s.get("facts_preserved_verbatim", False) for s in rows) / len(rows)}
        for direction in ("formal", "casual"):
            sub = [s for s in rows if s.get("direction") == direction]
            if sub:
                agg[f"pass_{direction}"] = sum(s["pass"] for s in sub) / len(sub)
        keys = sorted({k for s in rows for k in s["checks"]})
        for k in keys:
            vals = [s["checks"][k] for s in rows if k in s["checks"]]
            agg[f"check:{k}"] = sum(vals) / len(vals)
        agg["primary"] = agg["pass_rate"]
        out[f"style/{lang}"] = agg
    prim = [v["primary"] for v in out.values() if v.get("primary") is not None]
    out["composite"] = {"primary_mean": sum(prim) / len(prim) if prim else None,
                        "definition": "mean over languages present of GEC F0.5 (ERRANT for English, token-diff otherwise) and style pass rate"}
    return out

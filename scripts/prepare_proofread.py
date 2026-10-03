#!/usr/bin/env python3
"""Prepare the proofreading evaluation set (proofread-v1).

    python scripts/prepare_proofread.py            # -> .cache/datasets/proofread-v1 (English only)

Third-party data is downloaded at pinned revisions and NOT committed (JFLEG is CC BY-NC-SA 4.0).
The synthetic style subset is generated deterministically here.

Tasks:
  gec    : correct grammar/spelling/punctuation; scored against human references (ERRANT F0.5)
           - JFLEG test (jhu-clsp/jfleg@8b9fb0e6b3), 748 sentences x 4 references
           - control: 150 already-correct sentences (JFLEG reference #1 as input) -> should stay unchanged
  style  : rewrite a casual message formally (or a formal one casually) while keeping its facts;
           synthetic, scored by deterministic constraints

Extra languages come from git-ignored private/langpacks/proofread_*.py; when present, a combined set is
written to private/.cache/datasets/proofread-v1 and the public output stays English-only.
"""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import random
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NAME = Path(".cache") / "datasets" / "proofread-v1"
VERSION = "proofread-v1"
HF = "https://huggingface.co/datasets/{repo}/resolve/{rev}/{path}"
JFLEG = ("jhu-clsp/jfleg", "8b9fb0e6b3")  # short prefix; resolved to the full sha via the HF API

GEC_SYSTEM_EN = ("You are a careful proofreader. Correct grammar, spelling and punctuation errors in the user's text and make "
                 "it read fluently. Keep the meaning and the author's wording; change as little as necessary. If the text is "
                 "already correct, return it unchanged. Reply with only the corrected text, without quotes or comments.")
STYLE_EN = {
    "people": ["Maria", "Tom", "Aisha", "Daniel", "Priya", "Lukas", "Chen", "Sofia"],
    "places": ["Room 4B", "the Berlin office", "Hall C", "the main lobby"],
    "dates": ["Monday", "Tuesday", "March 14", "June 3"],
    "casual": [
        "hey {p}! gonna be late for the {t} sync, like 15 min. can u start w/o me in {pl}? btw the budget is {n} euros now 🙏",
        "yo {p}, the client call got moved to {d} at {t}. we're still on for {pl}, right? also they want {n} licenses lol",
        "{p} quick one - can't make it {d}, kinda swamped. can we do {t} instead? the invoice is {n} usd btw",
        "hey {p}, thx for the deck!! looks awesome. tiny thing: slide 7 says {n} users but i think it's way more. chat at {t}?",
    ],
    "formal": [
        "Dear {p}, I would like to inform you that the quarterly review has been rescheduled to {d} at {t} in {pl}. "
        "Please note that the approved budget is {n} euros.",
        "Dear {p}, thank you for your report. Could we please discuss the figure of {n} units at {t} on {d}? "
        "The meeting will take place in {pl}.",
    ],
    "slang": ["gonna", "wanna", "kinda", "lol", "btw", "thx", "yo", "hey", " u ", "w/o", "awesome", "swamped"],
    "system_formal": ("Rewrite the user's message in a formal, polite business tone. Keep every fact exactly: names, dates, "
                      "times, numbers, amounts and places. Do not use contractions, slang or emoji. Reply with only the "
                      "rewritten message."),
    "system_casual": ("Rewrite the user's message in a relaxed, friendly, casual tone, as a short chat message to a colleague. "
                      "Keep every fact exactly: names, dates, times, numbers, amounts and places. Reply with only the "
                      "rewritten message."),
}


def fetch(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=120) as r:
        return r.read()


def resolve_rev(repo: str, rev: str) -> str:
    return json.loads(fetch(f"https://huggingface.co/api/datasets/{repo}/revision/{rev}"))["sha"]


def jfleg(split: str, rev: str) -> list[dict]:
    import pyarrow.parquet as pq

    raw = fetch(HF.format(repo=JFLEG[0], rev=rev, path=f"data/{split}-00000-of-00001.parquet"))
    return pq.read_table(io.BytesIO(raw)).to_pylist()


def gec_item(iid: str, lang: str, src: str, refs: list[str], tags: list[str], system: str = GEC_SYSTEM_EN,
             preamble_regex: str | None = None) -> dict:
    expected: dict = {"source": src, "refs": refs}
    if preamble_regex:
        expected["preamble_regex"] = preamble_regex
    return {"id": iid, "task": "gec", "lang": lang, "tags": tags,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": src}],
            "expected": expected}


def style_items(rng: random.Random, lang: str, S: dict, n_each: int, split: str) -> list[dict]:
    items = []
    for direction, templates in (("formal", S["casual"]), ("casual", S["formal"])):
        for i in range(n_each):
            p, pl, d = rng.choice(S["people"]), rng.choice(S["places"]), rng.choice(S["dates"])
            t = f"{rng.randint(9, 17)}:{rng.choice(['00', '15', '30', '45'])}"
            n = str(rng.choice([120, 350, 1200, 2500, 48, 75]))
            text = rng.choice(templates).format(p=p, pl=pl, d=d, t=t, n=n)
            facts = [f for f in (p, t, n, d, pl) if f in text]
            items.append({
                "id": f"{VERSION}-{split}-style-{lang}-{direction}-{i:03d}", "task": "style", "lang": lang,
                "tags": [f"to:{direction}"],
                "messages": [{"role": "system", "content": S[f"system_{direction}"]},
                             {"role": "user", "content": text}],
                "expected": {"source": text, "direction": direction, "facts": facts, "slang": S["slang"]},
            })
    return items


def load_private_packs() -> list:
    mods = []
    for path in sorted((ROOT / "private" / "langpacks").glob("proofread_*.py")):
        spec = importlib.util.spec_from_file_location(path.stem, path)
        assert spec and spec.loader
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        mods.append(mod)
    return mods


def build(packs: list) -> tuple[list[dict], list[dict], dict]:
    jrev = resolve_rev(*JFLEG)
    sources: dict = {"jfleg": {"repo": JFLEG[0], "revision": jrev, "license": "CC BY-NC-SA 4.0"},
                     "style": "synthetic (this script)"}
    rng = random.Random(20261001)
    test: list[dict] = []
    dev: list[dict] = []
    jt = jfleg("test", jrev)
    for i, r in enumerate(jt):
        test.append(gec_item(f"{VERSION}-test-gec-en-{i:04d}", "en", r["sentence"], r["corrections"], ["jfleg"]))
    for k, i in enumerate(rng.sample(range(len(jt)), 150)):
        ref = jt[i]["corrections"][0]
        test.append(gec_item(f"{VERSION}-test-gec-en-ctrl-{k:03d}", "en", ref, [ref], ["control:already_correct"]))
    for i, r in enumerate(jfleg("validation", jrev)[:8]):
        dev.append(gec_item(f"{VERSION}-dev-gec-en-{i:03d}", "en", r["sentence"], r["corrections"], ["jfleg"]))
    for mod in packs:  # same rng stream continues, so pack items are reproducible
        mod.extend(test, dev, rng, version=VERSION, fetch=fetch, hf=HF, resolve_rev=resolve_rev, gec_item=gec_item,
                   manifest_sources=sources)
    for lang, S in [("en", STYLE_EN)] + [(m.LANG, m.STYLE) for m in packs]:
        test += style_items(random.Random(f"style-test-{lang}"), lang, S, 20, "test")
        dev += style_items(random.Random(f"style-dev-{lang}"), lang, S, 2, "dev")
    return test, dev, sources


def write(out: Path, test: list[dict], dev: list[dict], sources: dict) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in (("test", test), ("dev", dev)):
        with (out / f"{name}.jsonl").open("w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    manifest = {"version": VERSION, "sources": sources,
                "languages": sorted({r["lang"] for r in test}),
                "counts": {"test": len(test), "dev": len(dev)},
                "sha256": {n: hashlib.sha256((out / f"{n}.jsonl").read_bytes()).hexdigest() for n in ("test", "dev")}}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-public", action="store_true", help="only (re)build the private combined set")
    args = ap.parse_args()
    if not args.skip_public:
        print(json.dumps(write(ROOT / NAME, *build([])), indent=2))
    packs = load_private_packs()
    if packs:
        m = write(ROOT / "private" / NAME, *build(packs))
        print(f"private dataset: {ROOT / 'private' / NAME} languages={m['languages']} counts={m['counts']}")


if __name__ == "__main__":
    main()

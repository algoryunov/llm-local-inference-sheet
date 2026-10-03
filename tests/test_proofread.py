import importlib.util

import pytest

from llm_local_inference_sheet.proofread import (
    aggregate_proofread,
    clean_output,
    score_gec,
    score_style,
)

HAS_ERRANT = importlib.util.find_spec("errant") is not None and importlib.util.find_spec("en_core_web_sm") is not None


def gec(lang, src, refs, control=False):
    return {"id": "x", "task": "gec", "lang": lang, "tags": ["control:already_correct"] if control else [],
            "expected": {"source": src, "refs": refs}}


def _row(item, out):
    s = score_gec(item, out)
    s.update(task="gec", lang=item["lang"])
    return s


def test_token_diff_oracle_copy_and_overcorrection():
    # non-English languages use the token-diff scorer; German keeps this test independent of ERRANT
    it = gec("de", "Ich habe nach Hause gegeht", ["Ich bin nach Hause gegangen"])
    assert _row(it, "Ich bin nach Hause gegangen")["fp"] == 0
    copy = _row(it, "Ich habe nach Hause gegeht")
    assert copy["fn"] >= 1 and copy["unchanged"]
    ctrl = gec("de", "Alles gut.", ["Alles gut."], control=True)
    over = _row(ctrl, "Alles prima.")
    agg = aggregate_proofread([_row(it, "Ich bin nach Hause gegangen"), over])
    assert agg["gec/de"]["f0_5"] == 1.0 and agg["gec/de"]["keep_rate"] == 0.0 and agg["gec/de"]["metric"] == "token_diff"


@pytest.mark.skipif(not HAS_ERRANT, reason="proofread extra not installed")
def test_en_errant_multi_reference_picks_best():
    it = gec("en", "He go to school .", ["He goes to school .", "He went to school ."])
    s = _row(it, "He went to school.")
    assert s["metric"] == "errant" and s["tp"] == 1 and s["fp"] == 0


def test_clean_output_quotes_and_preamble():
    assert clean_output('"Fixed text."', "Fixed txt.") == ("Fixed text.", False)
    assert clean_output("Here is the corrected text: x", "x")[1] is True


def test_style_constraints():
    item = {"lang": "en", "expected": {"source": "hey Tom, gonna be late, 15:30 ok? budget 350", "direction": "formal",
                                       "facts": ["Tom", "15:30", "350"], "slang": ["gonna", "hey"]}}
    good = score_style(item, "Dear Tom, I will be late. Would 15:30 suit you? The budget is 350.")
    bad = score_style(item, "Hey Tom, I'm gonna be late at 15:30.")
    assert good["pass"] and not bad["pass"] and not bad["checks"]["facts_preserved"]


def test_fact_variants_accept_equivalent_forms():
    from llm_local_inference_sheet.proofread import _fact_present

    assert _fact_present("14:45", "moved to 2:45 pm in the lobby")
    assert _fact_present("16:00", "chat at 4 pm on monday")
    assert _fact_present("1200", "they want 1,200 licenses")
    assert _fact_present("the Berlin office", "the meeting is in berlin office")
    assert not _fact_present("14:45", "moved to 3:45 pm")
    assert not _fact_present("1200", "they want 120 licenses")

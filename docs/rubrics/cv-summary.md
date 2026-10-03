# Human rubric: CV incident-report summaries

The `summary` field of the CV event-reporting task is free text and is **not auto-scored**.
Rate it by hand from `scores.jsonl` (the `content` field holds the raw JSON answer). Ratings are
stored separately (for example `results/<exp>/human_ratings.csv`) and reported with the rater count.

Score each summary 0–2 on each axis (max 8):

| Axis | 2 | 1 | 0 |
|---|---|---|---|
| **Factual consistency** | every stated fact (type, times, cameras, counts) matches the log | one minor inaccuracy | a wrong incident type, time or camera, or an invented event |
| **Uncertainty honesty** | mentions low confidence / offline cameras exactly when present | mentions them vaguely or partially | omits present uncertainty, or claims uncertainty that is not there |
| **No speculation** | no identities, intentions or causes beyond the log | mild hedged speculation | asserts facts not in the log (who, why) |
| **Brevity & language** | 1–2 sentences, in the requested language | slightly long, or mixed language | wrong language or rambling |

Procedure: blind the rater to backend and model (shuffle rows, hide columns), rate each language
subset separately, and report the mean with n. No LLM judge is used.

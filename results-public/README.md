# results-public/

The published, filtered copy of the local measurement runs. It is produced by

    python scripts/export_public.py

from the git-ignored `results/` directory that `llm-sheet bench` writes. The export:

- keeps only English items of the application-quality runs and recomputes their summaries;
- replaces generated text in performance runs with same-length placeholders (the prompts are synthetic filler).
  Timings, token counts and lengths are unchanged, so every published metric recomputes exactly. The export was
  checked to reproduce all performance metrics identically;
- rewrites absolute local paths (`<repo>`, `~`, `/Users/<user>`).

Git tracks the compact files of each run. The raw files, marked *(bundle)* below, are git-ignored and published as
a release asset, `results-public-full.tar.gz`, built by the same export (with a `.sha256` next to it). Extract it at
the repo root to restore them:

    curl -LO https://github.com/algoryunov/llm-local-inference-sheet/releases/latest/download/results-public-full.tar.gz
    tar -xzf results-public-full.tar.gz

Each run lives in `results-public/<experiment>/cells/<cell_id>/<run_id>/`:

| file | content |
|---|---|
| `manifest.json` | state, host (chip, memory, OS, power, Rosetta), backend version, artifact revision + sha256 verification, readiness, template verification, telemetry summary, memory-pressure flag, export notes |
| `effective_config.json` | exact launch argv/env, settings, scheduling description, request path, length enforcement, cache policy |
| `workload.json` | workload spec + hash, template hash, per-item token counts |
| `requests.jsonl` *(bundle)* | one record per request (phase, status, timings relative to dispatch) |
| `events.jsonl.gz` *(bundle)* | raw stream payloads with monotonic arrival times: the source of truth for every metric |
| `telemetry.jsonl` *(bundle)* | 0.5 s samples: process RSS/footprint, system memory, swap, pressure, AGX GPU stats |
| `server.log.gz` *(bundle)* | backend stdout/stderr |
| `summary.json` | aggregate metrics without the per-request list (convenience only; reports recompute from events) |
| `scores.jsonl` *(bundle)* | application-suite scores per item (quality runs) |

Regenerate the tables and dashboard from this directory (needs the bundle):

    llm-sheet report --runs results-public/<experiment> ... --out results-public/report
    python scripts/build_dashboard.py

Synthetic (mock-server) runs are never exported.

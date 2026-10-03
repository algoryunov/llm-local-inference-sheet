# LLM Local Inference Sheet

Reproducible LLM serving benchmarks for Apple Silicon: which model and runtime to use on a MacBook, how fast it is,
and how well it does real tasks, including grammar correction.

A small, typed Python harness that launches real inference servers, drives them with a
closed-loop async streaming client, stores every raw stream event with monotonic timestamps,
and regenerates all tables and charts from those observations. It answers three questions:

1. Which model and runtime are practical on my Mac (M3 Pro, 18 GB)?
2. How do model size, quantization, KV-cache settings and serving architecture affect speed and task quality?
3. Which configurations complete useful tasks (JSON extraction, routing, incident reports, proofreading) accurately
   within a memory and latency budget?

> **Status (2026-10-02).** Everything for the Mac is implemented and measured: harness, model registry, three
> runtimes, performance and capacity sweeps, an application-task suite and a grammar/proofreading suite.
> Measurements were taken **under memory pressure** (other apps held much of the 18 GB), so every run carries
> `memory-pressure flag = yes` and absolute speeds are pessimistic.

## What was actually tested

<!-- RESULTS:BEGIN (summary of results/report/results.md; all numbers regenerate from raw events) -->
Measured on an Apple M3 Pro (18 GB, macOS 26.6.2), n = 30 per performance cell and 60 items per application-quality cell.
**Every Mac run is flagged for memory pressure** (level 2, several GiB swapped in during runs), so absolute values
are pessimistic. Full tables, CIs and caveats: [docs/findings.md](docs/findings.md).

| Qwen3-8B, 1,024 in / 256 out | TTFT p50 C=1 | TPOT p50 C=1 | agg tok/s C=1 → C=4 |
|---|---|---|---|
| llama.cpp b11246, Q4_K_M | 3.74 s | 39.7 ms | 18.4 → 21.0 |
| MLX-LM 0.31.3, MLX 4-bit | 3.81 s | 37.6 ms | 18.8 → **35.5** |
| Ollama 0.9.0, same Q4_K_M file | 3.98 s | 41.9 ms | 17.2 → 18.3 |

| Application suite (60 items) | composite | E2E p50 | GPU mem Δ |
|---|---|---|---|
| Qwen3-4B MLX 4-bit, MLX-LM | **0.864** | 2.18 s | 3.4 GiB |
| Qwen3-4B Q4_K_M, llama.cpp | 0.858 | 1.85 s | 3.3 GiB |
| Qwen3-8B Q4_K_M / MLX 4-bit | 0.819 / 0.815 | 3.5–3.6 s | 5.8–6.1 GiB |
| SmolLM3-3B Q4_K_M / MLX | 0.775 / 0.739 | 1.3–1.7 s | 2.3–3.1 GiB |

Key findings:
- **Profiling investigation:** llama.cpp's Metal backend stops scaling at 4–8 concurrent requests with K-quant GGUFs.
  Aggregate throughput *drops* from C=3 to C=4 and jumps +63% at C=9. The cause is the batch-size thresholds in
  its matmul kernel selection (source + counter-experiment + end-to-end check, [§C](docs/findings.md#c-profiling-investigation-why-llamacpp-stops-scaling-at-4-concurrent-requests-on-metal)).
- Qwen3-8B buys no measurable quality over Qwen3-4B on this suite (paired bootstrap, no significant difference), at ~2× memory and latency.
- GGUF Q4_K_M vs MLX 4-bit: no significant quality difference. llama.cpp vs Ollama on the identical file: Δ 0.004.
- **Proofreading (JFLEG, ERRANT F0.5):** Qwen3-8B corrects grammar measurably better than 4B (0.718 vs 0.675 on llama.cpp,
  paired bootstrap), at 1.7× the latency. Both models edit about a third of already-correct sentences, and both handle
  formal/casual tone rewrites well ([§F](docs/findings.md#f-proofreading-and-style-rewriting-results-publicproofread)).
- Ollama 0.9.0 cannot load SmolLM3 (`unknown model architecture`), aborts some greedy generations with its
  repetition guard, and silently reuses shared-prefix KV (no control to disable it).
<!-- RESULTS:END -->

| Area | State |
|---|---|
| CPU-only harness tests (stream fragmentation, Unicode, metadata-only events, early EOS, missing usage, errors, timeouts, cancellation, metrics, aggregation, report regeneration, grammar scoring) | **working**: `pytest` passes (69 tests; 6 opt-in integration tests skipped) |
| Grammar / proofreading suite (JFLEG + ERRANT, over-correction controls, tone rewrites) | **measured**: Qwen3-4B and Qwen3-8B on llama.cpp and MLX-LM ([how to run](#grammar-and-proofreading-suite)) |
| llama.cpp Metal (b11246) | **measured**: Qwen3-8B/4B, SmolLM3-3B, concurrency sweeps (flagged: memory pressure) |
| MLX-LM 0.31.3 | **measured**: Qwen3-8B/4B, SmolLM3-3B (perf + quality) |
| Ollama 0.9.0 | **measured**: Qwen3-8B/4B; SmolLM3 **unsupported** by this Ollama version |
| Capacity (Qwen3-1.7B/8B/14B × 1k/4k/16k) and KV-cache q8_0 | **measured** (n=5, exploratory): 14B at 16k fits in 18 GB (11.3 of 13.6 GiB Metal limit) but has 3 min TTFT; q8_0 KV halves KV memory |
| Llama-3.1-8B-Instruct | **gated** (no HF_TOKEN with granted access) |

## Quick start (Mac)

```bash
uv venv --python /opt/homebrew/opt/python@3.12/bin/python3.12 .venv   # native arm64 Python
uv pip install --python .venv/bin/python -e ".[dev,workloads,proofread]"
scripts/install_llama_cpp.sh          # pinned official llama.cpp b11246, sha256-verified, into .tools/
scripts/install_mlx.sh                # isolated pinned mlx-lm venv in .venvs/mlx
source .venv/bin/activate

llm-sheet doctor                      # hardware, OS, power, memory pressure, Rosetta, backends
llm-sheet models list
llm-sheet models download qwen3-8b-q4 # pinned revision, sha256-verified
llm-sheet plan  --config configs/experiments/mac-smoke.yaml
llm-sheet serve --backend llama-cpp --model qwen3-8b-q4
llm-sheet bench --config configs/experiments/mac-smoke.yaml
llm-sheet evaluate --config configs/experiments/quality.yaml
llm-sheet report --runs results/mac-smoke          # local report; publish via scripts/export_public.py
pytest                                # CPU-only; LLM_SHEET_INTEGRATION=1 pytest -m integration for real backends
```

Experiment profiles in `configs/experiments/`: `mac-smoke`, `mac-comparison` (initial 4-model matrix),
`mac-extended` (decode/balanced/prefill sweeps, exploratory n=30), `mac-published` (n=200 × 3 repeats),
`quality`, `proofread`, `mac-capacity` (1.7B/8B/14B × context), `mac-kvquant-q8`, `mac-investigation-batching`.
`bench` resumes by configuration identity and never overwrites
earlier runs; `bench --fill-missing` completes items a run skipped because of its time limit.

## Grammar and proofreading suite

Measures a Grammarly-style use: fix grammar and spelling without rewriting correct text, and change a message's tone
without losing its facts. Results and method: [docs/findings.md §F](docs/findings.md#f-proofreading-and-style-rewriting-results-publicproofread).

```bash
uv pip install --python .venv/bin/python -e ".[proofread]"   # ERRANT + spaCy English model + pyarrow
python scripts/prepare_proofread.py      # downloads JFLEG at a pinned revision into .cache/ (not redistributed)
llm-sheet plan --config configs/experiments/proofread.yaml
llm-sheet evaluate --config configs/experiments/proofread.yaml --download
llm-sheet evaluate --runs results/proofread   # re-score saved outputs without running the models again
```

- **Grammar:** 748 JFLEG learner sentences with 4 human corrections each, scored with ERRANT edit-level precision,
  recall and F0.5 (best reference per sentence), the standard metric for grammatical error correction.
- **Over-correction:** 150 already-correct sentences; the keep rate is the share returned with no edits.
- **Tone rewrites:** 40 synthetic messages rewritten casual → formal or formal → casual, checked for preserved facts,
  slang, contractions and emoji.
- Until `prepare_proofread.py` has run, `plan` shows the proofreading cells as untested with a pointer to it.

## Local results vs published results

`llm-sheet bench` writes complete, unfiltered runs to `results/`, which is git-ignored. What gets published is
`results-public/`, produced by:

```bash
python scripts/export_public.py   # filter + scrub + recompute, then regenerate report, compatibility matrix, dashboard
```

The export keeps only public evaluation items, replaces synthetic filler text in performance runs with same-length
placeholders (metrics are unchanged; verified), rewrites local paths, and fails if blocked content remains.
Optional extra language packs and their datasets live in the git-ignored `private/` directory, and
`LLM_SHEET_PRIVATE=1` makes local runs and reports include them.

Git tracks only the compact part of `results-public/` (about 3 MB): manifests, launch configs, workloads,
summaries, the report and the dashboard. The raw per-run files (stream events, requests, telemetry, server logs,
per-item scores) are published as a release asset, which the export also builds (`dist/results-public-full.tar.gz`).
To recompute every metric from raw events:

```bash
curl -LO https://github.com/algoryunov/llm-local-inference-sheet/releases/latest/download/results-public-full.tar.gz
tar -xzf results-public-full.tar.gz        # fills in the raw files under results-public/
llm-sheet report --runs results-public/mac-comparison --out /tmp/report
```

## Selection guidance

<!-- GUIDANCE:BEGIN -->
For this 18 GB M3 Pro, with a budget of composite ≥ 0.80, E2E ≤ 3 s per task and ≤ 6 GiB GPU memory:

- **Model:** Qwen3-4B at ~4-bit. 8B fails the latency budget without a quality gain here, and SmolLM3-3B fails the quality bar.
- **Runtime, single user / interactive:** llama.cpp (`llama-server`): lowest TTFT, ≤ 3 concurrent requests.
- **Runtime, several concurrent requests:** MLX-LM. Its batching scales (2.5× aggregate throughput at C=4 and 2.9× at C=8 vs C=1, decode-heavy Qwen3-4B sweep), while
  llama.cpp's K-quant path plateaus at 4–8 (or use ≥ 9 slots if memory allows).
- **Ollama:** convenient, and it runs the same llama.cpp-derived kernels. Check that the installed version supports the model
  (0.9.0 does not know SmolLM3), and remember that it reuses prefix caches you cannot turn off.
- **Proofreading:** Qwen3-8B corrects grammar measurably better (F0.5 0.718 vs 0.675); use 4B when speed matters more
  and a person reviews each suggestion. Neither is safe for blind auto-correct: both edit about a third of correct sentences.
<!-- GUIDANCE:END -->

## How it works

```
llm-sheet plan/bench/evaluate
  ├─ config.py      pydantic-validated registry (configs/models.yaml) + experiments
  ├─ backends/      process control only: availability, launch argv, readiness, metadata, shutdown
  │    llama_cpp · mlx_lm (+ in-process allocator sampler) · ollama · mock
  ├─ workloads/     exact-token prompts from each model's pinned tokenizer + chat template
  ├─ client.py      closed-loop asyncio/httpx load generator, raw events with perf_counter_ns
  ├─ streaming.py   incremental SSE / NDJSON parser (byte-level, UTF-8-safe) + chunk normalization
  ├─ metrics.py     TTFT, TPOT (token-level vs chunk-derived), E2E, throughput, native timings, CIs
  ├─ telemetry.py   proc_pid_rusage footprint, RSS, AGX GPU stats, swap, pressure
  ├─ evaluation.py  deterministic scoring for extraction / routing / CV-report tasks
  ├─ proofread.py   ERRANT-based grammar scoring, over-correction keep rate, tone-rewrite checks
  └─ reports.py     CSV + Markdown + self-contained HTML, recomputed from raw events
```

Why a common harness instead of each runtime's benchmark tool: one definition of TTFT/TPOT across
runtimes, raw events kept for recomputation, identical prompts (verified per backend), and explicit
labelling when a runtime cannot provide token-level timing or length enforcement. Native timings
(llama.cpp `timings`, Ollama durations) are recorded next to client timings as a cross-check.

## Documentation

- [docs/architecture.md](docs/architecture.md): overview diagrams of the data flow, one benchmark cell and the code layout
- [docs/methodology.md](docs/methodology.md): metric definitions, load model, cache and template controls, LLM trade-offs for CV/TensorRT engineers
- [docs/macos.md](docs/macos.md): native Mac backends, the Rosetta trap, unified-memory metrics, Ollama notes
- [docs/compatibility.md](docs/compatibility.md): generated model × runtime matrix (verified / expected / unsupported / gated)
- [docs/findings.md](docs/findings.md): evidence-backed findings and the profiling investigation
- [docs/interview-talk-track.md](docs/interview-talk-track.md): short demo narrative
- [docs/implementation-plan.md](docs/implementation-plan.md): the plan written before implementation
- [results-public/README.md](results-public/README.md): published run layout and how it is exported

## Limitations

- Mac numbers in this repository were measured while other applications held most of the unified
  memory. They are useful for relative comparisons made under the same conditions, but absolute values
  are pessimistic. Re-run after `llm-sheet doctor` reports pressure level 1.
- GGUF Q4_K_M and MLX 4-bit (g128 for Qwen3-8B) are different quantizations. "llama.cpp vs MLX" on the
  Mac compares runtime *and* quantization; only llama.cpp vs Ollama uses byte-identical weights (blob
  digest checked).
- MLX-LM and Ollama cannot force a fixed output length. Early EOS is recorded, and full-length-only
  statistics are reported separately.
- Closed-loop load only. There are no open-loop arrival-rate tests yet.
- No thermal or power telemetry on the Mac (it would need `sudo powermetrics`).
- Proofreading results contain model outputs on JFLEG sentences (CC BY-NC-SA 4.0); the dataset itself is downloaded by
  `scripts/prepare_proofread.py`, not redistributed.
- Quality set: 60 synthetic held-out items (3 tasks × 20). It is small, and differences of a few
  points are within noise. CV summaries need human rating (rubric provided); they are not auto-scored.

## TODO

- Re-run headline Mac cells with memory pressure at level 1, and log thermal state per run.
- Proofreading: test a stricter minimal-edit prompt against the same 150 over-correction controls.
- Application suite v2 with harder extraction and routing items.
- Llama-3.1-8B-Instruct once an HF token with the licence accepted is available.

## License

[MIT](LICENSE) for the code and the synthetic datasets. Model weights are downloaded under their own licences;
JFLEG (CC BY-NC-SA 4.0) is downloaded by `scripts/prepare_proofread.py` and not redistributed.

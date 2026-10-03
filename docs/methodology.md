# Methodology

This page defines exactly what is measured, how, and what each number does *not* mean. It
also explains the LLM-serving trade-offs that differ from CNN/detector inference. Those are
the places where intuition from CUDA/TensorRT computer vision can mislead.

## 1. LLM serving in one page (for a CV/TensorRT engineer)

| CV inference intuition | LLM serving reality |
|---|---|
| One forward pass per request, fixed shape | **Two phases.** *Prefill* processes the whole prompt in parallel: compute-bound, like a big batched GEMM. *Decode* then emits one token per forward pass: memory-bandwidth-bound, because every step re-reads all weights plus the KV cache for a single new token. |
| Latency = one number | Users feel **TTFT** (prefill + queueing) and **per-token pace** (decode). A 2 s TTFT with fast streaming can feel better than 0.5 s TTFT with slow tokens. |
| Batch = throughput knob chosen offline | Requests arrive and finish at different times. **Continuous batching** adds and removes sequences between decode steps. Batching decode is nearly free while bandwidth-bound: the same weights are read once for N sequences. So aggregate tokens/s rises with concurrency while per-request speed falls a little. |
| Activation memory is transient | The **KV cache** is persistent per-sequence state: `2 × layers × kv_heads × head_dim × bytes × tokens`. For Qwen3-8B at FP16 that is 144 KiB/token, so 4k context costs 576 MiB *per concurrent sequence*. Context length × concurrency, not just model size, decides memory fit. |
| INT8/FP16 engines with calibrated scales | Weight-only 4-bit formats (GGUF k-quants, MLX affine groups) shrink the **bandwidth** cost of decode, which is why they speed up decode on bandwidth-bound devices. They are *different algorithms*: "4-bit" GGUF Q4_K_M and MLX 4-bit g128 are not the same weights, so quality has to be measured, not assumed. |
| Engine build = compile once per shape | Modern TRT-LLM runs through a PyTorch-based LLM API by default (no separate engine build). vLLM/TRT-LLM capture CUDA graphs and autotune at startup, which shows up in startup time. |
| Tokenizer is irrelevant | Different families tokenize differently: "1,024 tokens" is a different amount of text for Qwen3 and Llama 3. Chat templates add tokens (Qwen3: 12, SmolLM3: 68 with its metadata block). Reasoning models may emit hidden "thinking" tokens unless disabled. |
| Apple GPU = discrete GPU | Unified memory: the GPU uses system RAM. Metal caps the GPU working set (here 13,639 MiB of 18 GiB). Other apps compete for the same pool, and paging slows everything down. |

## 2. Load generation

- **Closed loop** (`client.py`): C workers each keep exactly one request in flight, with no think time.
  This measures latency and throughput *at a given concurrency*, and the offered load adapts to the
  service rate. It is **not** an arrival-rate (open-loop) test. Queueing under bursty arrivals is
  outside the current scope. An open-loop mode must be labelled separately, with its arrival
  process, timeouts and measurement window.
- Warm-up: ≥ 5 requests (validated) at the same concurrency, run to completion before measured requests
  start. Warm-up prompts are distinct from measured prompts.
- Sample sizes: 30 measured requests for exploratory cells, ≥ 200 for published p95, and p99 only at ≥ 1,000
  (these are config defaults; `summary.json` carries `p95_reportable`). Every table shows n.
- Each cell gets a **fresh server process**. Model memory is released between cells, cells run strictly
  sequentially, and load time is observed per cell. Independent repeats (`repeats:`) relaunch the server.
- Client and server run on the same host.

## 3. Workloads

**Performance suite** (`workloads/perf.py`): for each model, prompts are built with the model's pinned
tokenizer and pinned chat template, thinking disabled. The *rendered* prompt, template overhead included,
hits the target input length exactly. Every generated workload so far has 35/35 exact hits. Each prompt
begins with a unique tag, so no two prompts share more than the template's fixed prefix. The template
overhead is recorded per item.

- `input_tokens` is everything the model prefills. For Qwen3 with thinking disabled, that includes the
  empty `<think></think>` block the template inserts.
- The SmolLM3 template injects `Today Date: …`, so its token count can change with the calendar. The
  render date is part of that workload's identity.

**Output length.** `max_tokens` is a cap, not a promise. For fixed-length performance runs we send
`ignore_eos: true` where the backend verifiably supports it (llama.cpp). mlx-lm 0.31.3 and Ollama have no such control: there, early EOS is recorded,
`early_stops` is reported, and a *full-length-only* subset is reported separately. Nothing is padded, and
no early-stopped request is counted as having produced the target length. Application-quality runs always
stop naturally.

**Application suite** (`datasets/app-v1`): synthetic, seeded, versioned. Dev (18) and test (60) splits
are disjoint. See §7.

## 4. Template and prompt verification

- llama.cpp: `/apply-template` renders the server-side prompt for the first measured item, and `/tokenize`
  counts it. Both are compared with the pinned HF template: `rendered_prompt_identical`, token counts.
- All backends: server-reported `usage.prompt_tokens` (Ollama: `prompt_eval_count`) is compared with the
  expected count for every measured request (`prompt_token_check`).
- Ollama runs by default in `generate_raw` mode, with the prompt rendered from the pinned template. This
  isolates runtime performance from Ollama's template auto-detection for imported GGUFs. The chat mode
  exists but is a different configuration.
- Chat-template hashes of the source and converted repos are pinned in `configs/models.yaml`. Several
  conversions ship a *different* template from their source (for example Qwen3-8B-MLX-4bit), which is why
  the verification is per artifact.

## 5. Cache controls

The baseline profile is `cache_policy: no_reuse`:

| Backend | Control | Verification |
|---|---|---|
| llama.cpp | `cache_prompt: false` per request; `--cache-ram 0` (the b11246 default is an 8 GiB host prompt cache) | native `timings.cache_n`, `usage` |
| mlx-lm | `--prompt-cache-size 0` | `usage.prompt_tokens_details.cached_tokens` |
| Ollama | none available | `prompt_eval_count` vs prompt tokens (lower means reuse) |

Intentional prefix reuse belongs in a separate profile (`cache_policy: prefix_reuse`).

## 6. Timing and metric definitions

All timestamps come from `time.perf_counter_ns()` (monotonic). Raw stream payloads are stored with
their arrival time relative to dispatch in `events.jsonl.gz`, so every metric below can be recomputed
without rerunning inference.

| Metric | Definition | Caveats |
|---|---|---|
| dispatch | immediately before httpx sends the request | includes connection reuse from a keep-alive pool |
| **TTFT** (`ttft_s`) | dispatch → arrival of the first payload with non-empty generated text (content *or* reasoning) | client-observed first-content latency. It includes HTTP, queueing, tokenization and prefill. Role-only, metadata and usage-only events are excluded. `ttft_content_s` excludes reasoning. |
| **E2E** (`e2e_s`) | dispatch → end of stream (the client read the final byte after `[DONE]` / `done:true`) | only for `status=ok` |
| **TPOT** (`tpot_s`) | (t_last_text − t_first_text) / (output_tokens − 1) | `token_timing=token_level` only if the number of text payloads equals the server's token count (one token per payload); otherwise `chunk_derived`. Null if output < 2 tokens or the token count is unknown. |
| ITL vs inter-chunk gap | gaps between successive text payloads | called **ITL** only when `token_level`, otherwise **inter-chunk gap** |
| decode tok/s per request | (output_tokens − 1) / (t_last_text − t_first_text) | same labelling as TPOT |
| aggregate output tok/s | Σ output tokens of successful measured requests / window | window = earliest measured dispatch → latest measured stream end. Includes the closed-loop ramp-down tail, and `mean_inflight` shows how full the window was. |
| successful RPS | successful measured requests / window | failures, timeouts, cancellations and not-dispatched requests are counted separately |
| native prefill tok/s | llama.cpp `timings.prompt_n / prompt_ms`; Ollama `prompt_eval_count / prompt_eval_duration` | server-side, excludes HTTP |
| prefill proxy tok/s | input_tokens / TTFT | labelled *proxy*: includes HTTP, queueing and scheduling |
| readiness | launch → health OK; launch → first completed 1-token generation | the OS file cache is not controlled, so weights may be warm. Download time is never included. |

Token counts come from server `usage` when present (`output_tokens_method=server_usage`). A fallback to
re-tokenizing text is supported but labelled, and streaming chunks are never counted as tokens.
Percentiles use linear interpolation (numpy's default). Median CIs are percentile-bootstrap
(1,000 resamples, seed 0) and are reported only for n ≥ 10.

## 7. Quality scoring

No model judge is used, paid or local.

- **Extraction**: strict parse (the whole response is JSON), lenient parse (first balanced `{…}` after
  stripping fences), schema validity (exact keys, types, date format, enum), per-field semantic accuracy,
  exact-all, and null-handling accuracy on absent or ambiguous fields.
- **Routing**: accuracy, macro-F1 over 6 labels (invalid output counts as wrong), unnecessary tool calls
  (gold `none`, predicted a tool), missed tool calls.
- **CV incident report**: per-field accuracy on 9 deterministic fields (sets compared as sets), exact-all,
  hallucinated event references. The `summary` free text is exported for human rating against
  `docs/rubrics/cv-summary.md`, and never auto-scored.
- **Composite** = mean(extraction field accuracy, routing macro-F1, CV field accuracy), per language and overall.
- Prompted JSON is the baseline. Constrained decoding (grammars, JSON schema) would be a separate
  configuration, and schema conformance alone is not semantic correctness.

## 7b. Proofreading scoring

- **Grammar (JFLEG test, 748 sentences, 4 human corrections each):** ERRANT extracts edits from source → output and
  source → each reference, using spaCy tokenization on all three strings (JFLEG references are pre-tokenized, so
  whitespace splitting would turn detokenized punctuation into false edits). An edit is a true positive when a
  reference contains the same (span, correction). Each sentence uses the reference with the best sentence-level F0.5;
  TP/FP/FN are summed over the corpus and reported as precision, recall and F0.5 (precision weighted twice as much as recall).
- **Over-correction:** 150 already-correct sentences (JFLEG reference 1 fed back as input); keep rate = share with zero edits.
- **Tone rewrites:** 40 synthetic messages; a rewrite passes when every fact is preserved (equivalent forms accepted:
  24h/12h times, thousands separators, a leading article; the verbatim rate is reported too), the text changed, the
  writing system is unchanged, and formal rewrites contain no listed slang, emoji or contractions.
- **Comparisons:** paired bootstrap over the same items (3,000 resamples); grammar F0.5 is recomputed at corpus level
  on every resample. Items a run never sent (deadline) are excluded from scoring and completed with `bench --fill-missing`.

## 8. Memory on Apple Silicon

Reported side by side, never summed:

| Field | Meaning |
|---|---|
| `gpu_mem_delta_gib` | system-wide AGX "In use system memory" peak minus the pre-launch baseline. It is the closest proxy for "GPU memory this run added", but other processes can move it. |
| `peak_rss_gib` | resident pages of the server process tree, including clean mmap'd weight pages. It can be *smaller* than the weights if pages were evicted under pressure. |
| `peak_footprint_gib` | `phys_footprint` lifetime max from `proc_pid_rusage`: dirty/anonymous memory. It **excludes** mmap'd weights, so llama.cpp shows about 0.4 GiB for a 4.7 GiB model. |
| `mlx_peak_gib` | MLX allocator `get_peak_memory()`, sampled in-process by our wrapper (MLX only) |

Every run also records memory pressure (1/2/4), swap in/out deltas, available memory before launch and
after shutdown, power source and low-power mode. `memory-pressure flag = yes` (pressure above normal
at any point, or more than 256 MiB swapped) marks timings that may include paging.

## 9. Comparison views

1. **Runtime comparison**: same hardware, model revision, precision, workload and effective generation
   settings wherever the formats allow. GGUF vs MLX on the Mac is *not* a pure runtime comparison, because
   the quantization schemes differ. llama.cpp vs Ollama on the byte-identical GGUF is: Ollama's blob digest
   is checked against the GGUF's sha256.
2. **Practical deployment**: the best measured configuration per device, with every differing setting visible.
3. **Model selection**: task quality, complete-task latency, memory and throughput. A candidate must meet an
   explicit quality threshold *and* a latency budget.

## 10. Result states

`measured`, `untested`, `unsupported`, `gated`, `out_of_memory`, `timeout`, `failed`, `synthetic`. Pre-run
states come from the planner (format mismatch, gated access, wrong platform, backend unavailable).
Runtime states come from the run (OOM is detected from logs and errors). Synthetic runs come from the
mock server, are excluded from reports by default, and are labelled SYNTHETIC wherever included.

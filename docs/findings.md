# Findings

Every statement here links to evidence in `results-public/` or to a pinned upstream source. Performance
numbers are quoted from `results-public/report/results.md`, which `llm-sheet report` regenerates from
raw events. **All Mac runs so far were taken under memory pressure** (flagged per run), so absolute
Mac numbers are pessimistic.

## A. Setup and methodology findings (verified, not performance claims)

1. **The agent/terminal shell ran under Rosetta on an M3 Pro.** `sysctl.proc_translated=1` and `uname -m=x86_64`.
   Any universal binary launched from it would run as x86_64. All Mac backends are launched with
   `arch -arm64`, their binaries are arm64-only (`lipo -archs`), and every manifest records the shell state.
2. **Converted artifacts ship different chat templates from their sources.** Pinned sha256 (from the HF API):
   Qwen3-8B source `a55ee1b1…`, Qwen3-8B-MLX-4bit `57f1fd00…`; Qwen3-1.7B-MLX-4bit `be3e45a0…`, which does
   not even reference `enable_thinking`; SmolLM3 source `3e0e275a…` vs mlx-community `0dab472a…`. Verifying the
   effective prompt per backend is therefore necessary, not paranoia.
3. **llama.cpp served exactly the intended prompt.** For Qwen3-8B Q4_K_M, the server-side `/apply-template`
   render was byte-identical to the pinned HF template with `enable_thinking=false`, and `usage.prompt_tokens`
   matched the expected 1,024 for every measured request (`template_verification`, `prompt_token_check` in
   the mac-smoke and mac-comparison manifests).
4. **SmolLM3's chat template injects today's date** (`Today Date: 29 September 2026`) and a 68-token metadata
   block, so its prompt length is calendar-dependent. The render date is part of that workload's identity.
5. **llama.cpp b11246 defaults would silently change a benchmark:** `--fit on` (auto-adjusts unset arguments to
   fit memory), `--cache-ram 8192` (8 GiB host-RAM prompt cache on an 18 GB machine), `-np auto` with a unified
   KV cache. The adapter sets `--fit off`, `--cache-ram 0` (no-reuse profile), an explicit `-np`, and
   `--no-kv-unified`, then verifies them from the log (`n_ctx_seq`, `kv_unified = false`).
6. **mlx-lm 0.31.3 serializes requests that carry a `seed`** (`_is_batchable` requires `seed is None`), and its
   `/health` answers before the model has loaded. The adapter never sends `seed` to MLX and defines readiness
   as a completed 1-token generation.
7. **On Apple Silicon, `phys_footprint` is the wrong memory metric for llama.cpp.** Weights are mmap'd and
   wrapped as Metal buffers, so footprint showed ~0.4 GiB for a 4.7 GiB model. llama.cpp's own exit breakdown
   is authoritative: MTL0 self 5,094 MiB = 4,789 model + 216 KV + 89 compute, against Metal's 13,639 MiB
   working-set limit. The measured KV buffer (216 MiB at 1,536 ctx) matches the geometric estimate
   (2 × 36 layers × 8 KV heads × 128 × 2 B = 144 KiB/token) exactly.
8. **The streaming layer was verified at token level for llama.cpp:** the number of text-bearing SSE payloads equals
   `usage.completion_tokens` for every request, so TPOT and ITL there are genuine per-token measurements.
   Client TTFT agreed with llama.cpp's native prefill time to within ~1% (see §B).

## B. Mac performance findings: Qwen3-8B, 1,024 in / 256 out (`results-public/mac-comparison`)

All six cells: n = 30 measured requests, greedy, thinking disabled, no prefix reuse. Every cell is
**flagged**: memory pressure reached level 2 and 7.9–18.1 GiB was swapped in *during* each run.
Treat the absolute numbers as pessimistic and the relative ordering as provisional. p95 is not
reportable at n = 30.

| backend (weights) | C | TTFT p50 | TPOT p50 | E2E p50 | decode tok/s/req | aggregate tok/s | GPU mem Δ |
|---|---|---|---|---|---|---|---|
| llama.cpp b11246 (Q4_K_M) | 1 | 3.74 s | 39.7 ms | 13.86 s | 25.2 | 18.4 | 5.78 GiB |
| MLX-LM 0.31.3 (MLX 4-bit g128) | 1 | 3.81 s | 37.6 ms | 13.41 s | 26.6 | 18.8 | 6.11 GiB |
| Ollama 0.9.0 (same Q4_K_M file) | 1 | 3.98 s | 41.9 ms | 14.65 s | 23.9 | 17.2 | 6.13 GiB |
| llama.cpp | 4 | 7.68 s | 161.7 ms | 48.91 s | 6.2 | 21.0 | 6.29 GiB |
| MLX-LM | 4 | 14.83 s | 52.2 ms | 28.11 s | 19.2 | **35.5** | 7.16 GiB |
| Ollama | 4 | 16.05 s | 144.0 ms | 52.83 s | 6.9 | 18.3 (28/30 ok) | 6.38 GiB |

What the data supports:

1. **At C=1 the three runtimes are within ~6–11% of each other** (TTFT spread 6%, E2E 9%, TPOT and decode 11%). Single-stream decode
   on this chip is memory-bandwidth-bound, and all three read ~4.4–4.7 GiB of weights per token. The ordering
   (MLX ≳ llama.cpp > Ollama) is small relative to the swap noise and needs clean, repeated runs before it counts as a finding.
2. **At C=4 the runtimes make opposite scheduling trade-offs.** llama.cpp prefills newly arriving requests
   promptly (TTFT 7.7 s), which stalls every in-flight decode (TPOT 162 ms, 4× C=1), so aggregate throughput barely
   moves (18.4 → 21.0). MLX-LM lets new prompts wait (TTFT 14.8 s), keeps decoding batched (TPOT 52 ms), and nearly
   doubles aggregate throughput (18.8 → 35.5). Which is "better" depends on the latency budget: interactive users feel
   TTFT, while batch jobs care about throughput. Part of llama.cpp's C=4 behaviour is explained by §C.
3. **Ollama vs llama.cpp uses byte-identical weights** (blob sha256 = GGUF sha256). Ollama 0.9.0 runs its vendored
   llama.cpp runner (mid-2025 ggml) behind its own server. At C=1 it is ~5% slower on decode and ~6% slower on TTFT. At C=4 its native
   prefill rate fell to 64 tok/s (vs 259 at C=1): prompts were processed while other sequences decoded.
4. **Ollama aborted 2 of 30 C=4 requests** with `prediction aborted, token repeat limit reached`. That is a built-in
   repetition guard that closed the stream without `done:true` on greedy output over filler text. The harness counts
   these as errors; they are not silently truncated successes.
5. **Readiness** (launch → first 1-token generation): llama.cpp 1.4–3.5 s, MLX 4.2–8.0 s. Ollama's 23.3 s at C=1 includes
   the one-off `ollama create` import (copying the 4.7 GiB GGUF into its blob store). Its C=4 relaunch took 5.9 s.
6. **Streams were token-level on all three backends** (payload count = server token count for 29–30 of 30 requests per cell, 26 of 28 in the Ollama C=4 cell;
   the rest are labelled `chunk_derived`). Server prompt-token counts equalled the pinned-template expectation (1,024) for every measured request on all
   three backends, including MLX (whose repo ships a different template file) and Ollama raw mode.

## C. Profiling investigation: why llama.cpp stops scaling at 4 concurrent requests on Metal

**Symptom** (§B.2): at C=4, llama-server's TPOT was 4× the C=1 value, and aggregate throughput rose only 14%.

**Step 1: remove HTTP and scheduling.** llama.cpp's own `llama-batched-bench` (same b11246 build, same GGUF,
flash attention on) measures decode throughput for B parallel sequences directly. Raw output is in
`results-public/investigations/llama-cpp-batching/`. Split vs unified KV made no difference (±2%), so KV layout is not the cause.

| B (parallel sequences) | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 12 | 16 |
|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3-8B Q4_K_M, decode tok/s (npp 128) | 25.1 | 38.8 | 39.5 | **31.4** | 28.6 | 33.7 | 27.4 | 29.3 | 78.1 | 105.7 |

Aggregate decode throughput is non-monotonic: it gains at B=2–3, *drops* for B=4–8, then jumps above 8.
A bandwidth-bound decode should improve smoothly with batch size, so this pointed to discrete code-path changes.

**Step 2: read the kernel selection at the pinned commit** (`139997d8e`, `ggml/src/ggml-metal/ggml-metal-ops.cpp`
`ggml_metal_op_mul_mat`, and `ggml-metal-common.cpp:35-42`). For a weight matrix × activations with `ne11` = batch columns:

- `mul_mv_ext` ("small-batch mat-mv kernels … efficient for BS [2, ~8]") is used for Q4_0/Q8_0/F16/… when `2 ≤ ne11 ≤ 8`,
  but for **K-quants (Q4_K, Q5_K, Q6_K, Q2_K, Q3_K) only when `4 ≤ ne11 ≤ 8`**. It runs with a fixed `nsg = 2`, and an in-source
  TODO questions that choice.
- `mul_mm` (simdgroup matrix-multiply) is used when `ne11 > 8`.
- Otherwise, including K-quants at B=2–3, plain `mul_mv`.

Q4_K_M is made of Q4_K and Q6_K tensors, so B=1–3 → `mul_mv`, B=4–8 → `mul_mv_ext`, B≥9 → `mul_mm`. The throughput
changes sit exactly on those boundaries.

**Step 3: controlled counter-experiment.** If the thresholds are the mechanism, a non-K-quant model should switch at
different B. Same tool, same settings:

| B | 1 | 2 | 3 | 4 | 5 | 6 | 8 | **9** | 12 | 16 |
|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3-4B Q4_K_M | 45.4 | 65.3 | 67.6 | **57.1** | 52.5 | 61.6 | 54.1 | **87.6** | 114.2 | 145.1 |
| Qwen3-1.7B Q8_0 | 58.2 | 118.5 | 161.2 | 159.6 | 151.6 | 170.0 | 165.0 | **232.1** | 302.3 | 383.5 |

- The **B=8 → 9 step appears for both weight types** (+62% for 4B Q4_K_M, +41% for 1.7B Q8_0), exactly at the `mul_mm` threshold.
- **Q4_K_M regresses when entering B=4** (67.6 → 57.1), the K-quant `mul_mv_ext` entry point. Q8_0 enters `mul_mv_ext`
  already at B=2 and does *not* regress: it scales to B=3 and then plateaus until B=8.
- Conclusion: on this M3 Pro, B=4–8 is a **plateau for all tested weight types** and a **regression for K-quants**. Throughput
  scales again only once the batch exceeds 8 and `mul_mm` takes over.

**Why it matters for serving:** in llama-server, the decode batch equals the number of sequences decoding in that step, so
concurrency 4–8 with K-quant GGUFs lands in the slow band. Practical options: stay at C ≤ 3, jump to ≥ 9 slots when memory
allows (KV grows linearly with slots), use a non-K quant, or use MLX, whose batching showed no such dip at C=4.

**Caveats:** the batched-bench sweeps ran at memory pressure level 1–2, and each sweep is a within-run relative comparison.
Kernel-level timings (Metal System Trace / Instruments) were not captured, so the attribution to specific kernels rests on
the source thresholds plus the counter-experiment, not on per-kernel profiles. End-to-end confirmation through llama-server
is below.

### End-to-end check through llama-server (`results-public/mac-investigation-batching`)

Qwen3-4B, 128 in / 256 out (decode-heavy, so prefill interference is small), n = 30 per cell, closed loop. All cells are
flagged for memory pressure; the comparison is within one session.

| C | 1 | 2 | 3 | **4** | 8 | **9** | 12 |
|---|---|---|---|---|---|---|---|
| llama-server Q4_K_M, aggregate out tok/s | 41.2 | 57.6 | 58.9 | **51.9** | 50.6 | **82.3** | 87.5 |
| llama-server TPOT p50 (ms) | 22.9 | 32.7 | 48.9 | 73.9 | 144.6 | 94.7 | 106.8 |
| MLX-LM 4-bit, aggregate out tok/s | 42.4 | 75.6 | 91.0 | 104.9 | 123.0 | _see report_ | – |

- The prediction from Steps 2–3 holds through the full serving path. Throughput **falls** from C=3 to C=4 (−12%), stays
  flat to C=8, and **jumps +63% at C=9**, where the decode batch crosses 8 and `mul_mm` is used. TPOT at C=9 (94.7 ms) is
  *lower* than at C=8 (144.6 ms) despite one more concurrent sequence.
- `llama-batched-bench` predicted the same shape for this model (67.6 → 57.1 at B=4, 54.1 → 87.6 at B=8 → 9). The server
  numbers are lower in absolute terms because they include HTTP, sampling and prefill of arriving requests.
- MLX-LM on the same model scales monotonically (2.9× at C=8), so the dip is specific to llama.cpp's Metal kernel selection,
  not a property of the chip.
- `mean_inflight` confirms the closed loop kept the intended concurrency (3.88 at C=4, 8.12 at C=9). At C=12, 30 requests
  leave a tail where fewer than 12 are in flight (mean 9.97).

**Result:** for K-quant GGUFs on this M3 Pro, llama-server concurrency 4–8 buys no throughput over C=2–3; C ≥ 9
(memory permitting) does. This is a concrete, reproducible configuration rule backed by source, a controlled
counter-experiment and an end-to-end measurement. Worth re-testing on newer llama.cpp builds, since the in-source
TODO suggests the `mul_mv_ext` parameters are not final.

## D. Application quality and model selection (`results-public/quality`)

60 held-out synthetic items (JSON extraction, request routing, CV incident report × 20), greedy, thinking disabled,
prompted JSON, natural stop, C=1. Latency columns come from the same runs (flagged for memory pressure).

| model / weights | backend | composite | extraction field acc | null handling | routing macro-F1 | CV field acc | strict JSON | E2E p50 | GPU mem Δ (perf run) |
|---|---|---|---|---|---|---|---|---|---|
| Qwen3-4B MLX 4-bit | MLX-LM | **0.864** | 0.975 | 0.885 | 1.000 | 0.617 | 1.00 | 2.18 s | 3.37 GiB |
| Qwen3-4B Q4_K_M | llama.cpp | 0.858 | 0.975 | 0.769 | 1.000 | 0.600 | 1.00 | 1.85 s | 3.27 GiB |
| Qwen3-4B Q4_K_M | Ollama | 0.855 | 0.975 | 0.769 | 1.000 | 0.589 | 1.00 | 1.76 s | 3.32 GiB |
| Qwen3-8B Q4_K_M | llama.cpp | 0.819 | 0.975 | 0.885 | 1.000 | 0.483 | 1.00 | 3.51 s | 5.78 GiB |
| Qwen3-8B MLX 4-bit | MLX-LM | 0.815 | 0.983 | 0.923 | 1.000 | 0.461 | 1.00 | 3.58 s | 6.11 GiB |
| Qwen3-8B Q4_K_M | Ollama | 0.806 | 0.975 | 0.885 | 1.000 | 0.444 | 1.00 | 3.18 s | 6.13 GiB |
| SmolLM3-3B Q4_K_M | llama.cpp | 0.775 | 0.842 | 0.308 | 1.000 | 0.483 | 0.33 | 1.67 s | 2.30 GiB |
| SmolLM3-3B MLX 4-bit | MLX-LM | 0.739 | 0.800 | 0.154 | 1.000 | 0.417 | 0.67 | 1.32 s | 3.12 GiB |
| SmolLM3-3B | Ollama 0.9.0 | **unsupported** (`unknown model architecture: 'smollm3'`) | | | | | | | |
| Llama-3.1-8B-Instruct | all | **gated** | | | | | | | |

Paired bootstrap (4,000 resamples over the same items; mean per-item score difference, 95% CI):

| comparison | overall (n = 60) | CV task (n = 20) | extraction (n = 20) |
|---|---|---|---|
| Qwen3-4B − Qwen3-8B, GGUF on llama.cpp | +0.039 (−0.013, +0.091) | +0.117 (−0.033, +0.267) | +0.000 |
| Qwen3-4B − Qwen3-8B, MLX | +0.049 (−0.013, +0.108) | +0.156 (−0.011, +0.317) | −0.008 (−0.025, 0.000) |
| Qwen3-8B GGUF − MLX | +0.005 (−0.019, +0.035) | +0.022 (−0.044, +0.106) | −0.008 (−0.025, 0.000) |
| Qwen3-4B GGUF − MLX | −0.006 (−0.039, +0.024) | −0.017 (−0.117, +0.072) | +0.000 |
| llama.cpp − Ollama, same Qwen3-4B file | +0.004 (−0.004, +0.011) | +0.011 (−0.011, +0.033) | +0.000 |
| Qwen3-4B − SmolLM3-3B, GGUF | **+0.083 (+0.018, +0.152)** | +0.117 (−0.067, +0.306) | **+0.133 (+0.075, +0.200)** |

What this supports:

1. **Qwen3-8B shows no measurable quality gain over Qwen3-4B on this suite**, at about twice the memory and latency.
   The point estimates favour 4B (driven by the CV task), but no 4B−8B difference is significant at n = 60. So the
   supported claim is "8B buys nothing measurable here", not "4B is better".
2. **GGUF Q4_K_M vs MLX 4-bit showed no quality difference** for either model. The schemes differ, but on this suite
   the difference is below what 60 items can resolve.
3. **llama.cpp and Ollama on the byte-identical GGUF score the same** (Δ 0.004). That is a useful end-to-end sanity check
   of the scoring pipeline and of Ollama raw-mode prompting.
4. **SmolLM3-3B is clearly behind Qwen3-4B** (−0.083 overall, −0.133 on extraction). It wraps JSON in Markdown fences
   (strict parse 33–67%) and often fills absent fields with nearby text (null handling 0.15–0.31).
5. **Extraction and routing are nearly saturated** for all Qwen3 configurations (extraction field accuracy 0.975–0.983,
   routing macro-F1 1.0 with zero unnecessary tool calls). The CV task is the one that discriminates. Harder extraction
   and routing sets are needed.
6. **Dataset defect found by error analysis:** the CV prompt never defines `vehicle_involved`. The gold means "the incident
   involves a vehicle", while Qwen3-8B often marked any vehicle in the log as `true` (vehicle_involved accuracy 0.55 vs
   0.90 for 4B on llama.cpp). `app-v1` is kept frozen with this caveat; a corrected `app-v2` should define the field
   explicitly. The unambiguous CV gaps remain: 8B misapplies the explicit `uncertain` rule more often (0.35 vs 0.75).
7. **Ollama's quality-run TTFT is not like-for-like.** On the same 4B file, Ollama's native prefill rate in this suite was
   far above llama.cpp's, while the two were at parity with unique prompts (259 vs 275 tok/s in §B). The suite's prompts
   share long system prompts, and Ollama reuses cached prefixes with no control to disable it, while `prompt_eval_count`
   still reports the full prompt length. llama.cpp (`cache_prompt:false`) and MLX (cache size 0) did no reuse.

### Selection rule (explicit threshold and budget)

Budget used for this recommendation: **composite ≥ 0.80**, **E2E p50 ≤ 3 s per task at C=1**, and **GPU memory Δ ≤ 6 GiB** on
the 18 GB M3 Pro (leaving room for the OS and apps).

| candidate | composite ≥ 0.80 | E2E ≤ 3 s | mem ≤ 6 GiB | verdict |
|---|---|---|---|---|
| Qwen3-4B Q4_K_M on llama.cpp | 0.858 ✓ | 1.85 s ✓ | 3.27 ✓ | **selected** |
| Qwen3-4B MLX 4-bit on MLX-LM | 0.864 ✓ | 2.18 s ✓ | 3.37 ✓ | meets budget; best when serving ≥ 4 concurrent users (§B, §C) |
| Qwen3-4B Q4_K_M on Ollama | 0.855 ✓ | 1.76 s ✓ (cache-assisted) | 3.32 ✓ | meets budget; version must support the model |
| Qwen3-8B (any) | 0.81–0.82 ✓ | 3.2–3.6 s ✗ | 5.8–6.1 ≈ | fails latency; no measurable quality gain |
| SmolLM3-3B (any) | 0.74–0.78 ✗ | ✓ | ✓ | fails quality |

**Recommendation for this Mac:** Qwen3-4B at ~4-bit. Choose llama.cpp for single-user interactive work (lowest TTFT, ≤ 3
concurrent requests). Choose MLX-LM when several requests run at once, because its batching scales where llama.cpp's
K-quant path plateaus at 4–8. Re-validate after a clean-memory run.

## E. Capacity sweep and KV-cache quantization (`results-public/mac-capacity`, `results-public/mac-kvquant-q8`)

C=1, 128 output tokens, n = 5 measured per cell (exploratory capacity check, not a latency distribution). All cells
are flagged for memory pressure. GPU memory for llama.cpp is its own exit breakdown ("self" = weights + KV + compute);
for MLX it is the allocator peak.

| model / weights | backend | input | TTFT p50 | prefill tok/s | TPOT p50 | GPU memory | KV (llama.cpp log) |
|---|---|---|---|---|---|---|---|
| Qwen3-1.7B Q8_0 | llama.cpp | 1k / 4k / 16k | 0.82 / 3.63 / 28.3 s | 1259 / 1134 / 579 | 18 / 20 / 34 ms | 1.9 / 2.2 / 3.6 GiB | 140 / 476 / 1820 MiB |
| Qwen3-1.7B MLX 4-bit | MLX-LM | 1k / 4k / 16k | 0.93 / 4.49 / 21.4 s | 1102 / 912 / 766 (proxy) | 11 / 17 / 27 ms | 1.7 / 2.2 / 4.9 GiB (peak) | – |
| Qwen3-8B Q4_K_M | llama.cpp | 1k / 4k / 16k | 3.92 / 18.6 / 100.3 s | 262 / 221 / 163 | 41 / 51 / 63 ms | 4.9 / 5.4 / 7.1 GiB | 180 / 612 / 2340 MiB |
| Qwen3-8B MLX 4-bit | MLX-LM | 1k / 4k / 16k | 3.93 / 17.2 / 80.1 s | 261 / 239 / 204 (proxy) | 38 / 46 / 72 ms | 5.0 / 5.6 / 8.9 GiB (peak) | – |
| Qwen3-14B Q4_K_M | llama.cpp | 1k / 4k / 16k | 7.45 / 37.9 / 181 s | 138 / 108 / 91 | 77 / 96 / 110 ms | 8.7 / 9.2 / 11.1 GiB | 200 / 680 / 2600 MiB |
| Qwen3-14B MLX 4-bit | MLX-LM | 1k / 4k / 16k | 7.53 / 30.8 / 143 s | 136 / 133 / 114 (proxy) | 70 / 74 / 111 ms | 8.3 / 8.9 / 12.7 GiB (peak) | – |

(MLX prefill is the TTFT proxy, input/TTFT, because mlx_lm.server exposes no native timings. For llama.cpp the proxy
and native values agreed within 1% in every cell.)

1. **Everything fit, including Qwen3-14B at 16k context.** llama.cpp's 14B/16k cell used 11,317 MiB of Metal's 13,639 MiB
   working-set limit, and MLX peaked at 12.7 GiB. No OOM, but swap-out grew to 435–717 MiB in the 14B/16k cells, so this
   is the edge of what 18 GB can do alongside other apps. The memory-fit hypotheses in `plan` are now measurements.
2. **Fitting is not the same as being usable.** 14B has 7.5 s TTFT at 1k input and 13–14 tok/s decode. At 16k input,
   TTFT is 2.4–3 minutes. Prefill throughput falls as context grows (8B llama.cpp: 262 → 163 tok/s from 1k to 16k)
   because attention cost grows with sequence length, while decode slows more gently (41 → 63 ms/token).
3. **Long-context trade-off between runtimes:** MLX prefilled long prompts faster (8B at 16k: 80 s vs 100 s TTFT;
   14B: 143 s vs 181 s) but used more memory (8B at 16k: 8.9 GiB allocator peak vs llama.cpp's 7.1 GiB), and its decode
   at 16k was slower for 8B (72 vs 63 ms). The two memory numbers come from different instruments (allocator peak vs
   buffer breakdown), so compare them directionally.
4. **KV geometry estimate = measurement.** The registry's estimate for Qwen3-8B at the 16,640-token slot
   (144 KiB/token × 16,640 = 2,340 MiB) equals llama.cpp's reported KV buffer exactly. The same holds for every other cell.

### KV-cache quantization (llama.cpp, Qwen3-8B Q4_K_M weights fixed; `-ctk/-ctv q8_0` vs f16)

| input | KV f16 | KV q8_0 | TTFT f16 → q8_0 | TPOT f16 → q8_0 | llama.cpp self |
|---|---|---|---|---|---|
| 4,096 | 612 MiB | 325 MiB (53%) | 18.6 → 16.7 s | 50.6 → 45.1 ms | 5,493 → 5,214 MiB |
| 16,384 | 2,340 MiB | 1,243 MiB (53%) | 100.3 → 87.7 s | 63.2 → 55.3 ms | 7,252 → 6,166 MiB |

- **q8_0 KV nearly halves KV memory** (53% of f16, consistent with q8_0's 8.5 bits per element vs 16), saving ~1.1 GiB at 16k.
- It was also ~11–13% *faster* on both TTFT and TPOT at long context, plausibly because attention reads half the KV
  bytes. The runs were in different sessions, with n = 5 and memory-pressure flags, so treat the speed gain as a hypothesis
  worth a clean repeat. The memory saving is exact.
- **Quality impact was not measured.** A q8_0-KV run of the application suite (or a long-context retrieval test) is needed
  before recommending it. The AGX "GPU mem Δ" for the 4k q8_0 cell (12.4 GiB) is an outlier caused by other processes
  moving the system-wide counter. The per-process llama.cpp breakdown is the reliable figure here.

## F. Proofreading and style rewriting (`results-public/proofread`)

The Grammarly-style use case: fix grammar and spelling without rewriting correct text, and change a message's tone
without losing its facts. Greedy decoding, thinking disabled, C=1. n = 748 grammar sentences + 150 controls + 40 style
items per configuration.

- **Grammar:** JFLEG test set (jhu-clsp/jfleg, pinned revision; CC BY-NC-SA 4.0), 748 learner sentences with 4 human
  corrections each, scored at the edit level with **ERRANT** (P/R/F0.5, best reference per sentence), the standard GEC scorer.
- **Over-correction:** 150 already-correct sentences (JFLEG human corrections fed back as input). Keep rate = share left
  with zero edits.
- **Style:** 40 synthetic messages rewritten casual→formal or formal→casual, checked deterministically: facts preserved
  (names, times, numbers, places; equivalent forms such as 14:45 / 2:45 PM accepted, with the strict verbatim rate also
  reported), text changed, same writing system, and for formal: no slang, no emoji, no contractions.

| model / weights | backend | grammar F0.5 | precision | recall | keep rate | style pass | facts verbatim | E2E p50 |
|---|---|---|---|---|---|---|---|---|
| Qwen3-4B Q4_K_M | llama.cpp | 0.675 | 0.719 | 0.542 | 0.653 | 0.925 | 0.80 | 0.70 s |
| Qwen3-8B Q4_K_M | llama.cpp | **0.718** | 0.752 | 0.609 | 0.633 | 1.000 | 0.60 | 1.17 s |
| Qwen3-4B MLX 4-bit | MLX-LM | 0.689 | 0.728 | 0.568 | 0.600 | 0.975 | 0.80 | 0.79 s |
| Qwen3-8B MLX 4-bit | MLX-LM | 0.707 | 0.744 | 0.592 | 0.653 | 0.875 | 0.70 | 1.28 s |

Paired bootstrap (3,000 resamples; grammar F0.5 resamples sentences and recomputes corpus F0.5):

| comparison | grammar F0.5 | keep rate | style pass |
|---|---|---|---|
| Qwen3-8B − Qwen3-4B, llama.cpp | **+0.043 (+0.025, +0.061)** | −0.020 (−0.073, +0.033) | +0.075 (0.000, +0.150) |
| Qwen3-8B − Qwen3-4B, MLX | **+0.018 (+0.001, +0.036)** | +0.053 (−0.007, +0.113) | −0.100 (−0.225, 0.000) |
| Qwen3-4B GGUF − MLX | −0.014 (−0.027, −0.001) | **+0.053 (+0.013, +0.100)** | −0.050 (−0.150, +0.050) |

What this supports:

1. **For grammar correction, 8B is measurably better**, unlike on the JSON-task suite (§D). It finds more errors (recall
   +0.07 on llama.cpp) at slightly higher precision, which gives +0.043 F0.5. The cost is 1.7× the latency per sentence
   (1.17 s vs 0.70 s), still interactive for single sentences.
2. **Both sizes over-edit correct text.** About 35–40% of already-correct sentences received at least one edit, with no
   significant 4B/8B difference. The controls are human corrections of learner text, so some of those edits are legitimate
   alternatives, but for a "don't touch what's fine" proofreader this is the main weakness. It is the first thing to target
   with prompting (e.g. asking for minimal edits only) and to re-measure with the same controls.
3. **Tone rewriting works for both sizes** (88–100% pass once equivalent time and number formats are accepted), and the 4B/8B
   differences are not significant at n = 40. The failure modes differ: 8B more often reformats times (verbatim fact rate
   0.60–0.70 vs 0.80), and the 8B MLX build kept "awesome" in 5 formal rewrites.
4. **GGUF vs MLX at 4B:** MLX scored slightly higher F0.5 (+0.014) but edited more correct sentences (keep 0.600 vs 0.653).
   Both differences are small, and this is the only suite where the quantization format showed any measurable quality effect.

Scoring notes (fixed before these numbers were produced; all results recompute from saved outputs):
- JFLEG references are space-tokenized while models write normal text, so ERRANT runs with spaCy tokenization on all
  three strings. Whitespace tokenization would count every detokenized punctuation mark as a false edit.
- The 8B cells first hit the 60-minute cell limit and skipped their last items. Those items were completed with
  `bench --fill-missing` and merged; skipped items are never scored as failures.
- The first style checker required facts verbatim and penalized 24h→12h time conversion. It now accepts equivalent forms,
  and the verbatim rate is reported alongside.

**Proofreading recommendation:** Qwen3-8B when correction quality matters most (≈1.2 s per sentence on this Mac); Qwen3-4B
for snappier interactive use with ~4 F0.5 points less. Either way, add a minimal-edit instruction and check the keep rate,
because both models touch about a third of already-correct sentences.

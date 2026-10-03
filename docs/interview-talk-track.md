# Demo / interview talk track (≈5 minutes)

Only claims that the repository can back up. Quote numbers from the regenerated report, never from memory.

## 1. Problem (30 s)
"I wanted to know which open-weight models and runtimes are actually usable on my 18 GB M3 Pro,
measured in a way I'd trust at work: pinned artifacts, raw data, and numbers I can regenerate."

## 2. Design (60 s)
- Backend adapters own process control only; one async closed-loop client owns measurement.
- Every SSE/NDJSON payload is stored with `perf_counter_ns` arrival times, so all metrics are recomputable.
- The parser is byte-level and incremental. Tests split the stream at *every* byte boundary, including inside
  UTF-8 code points and CRLF pairs, against a deterministic mock server that can also drop, hang, or error mid-stream.
- Prompts hit exact token counts *after* chat templating, using each model's pinned tokenizer and template.
- Planner states (`unsupported`, `gated`, `untested`) are first-class results, not missing rows.

## 3. Three things that would have silently corrupted results (90 s)
1. **Rosetta:** my agent shell was x86_64-translated on Apple Silicon. Without `arch -arm64` I would have benchmarked emulation.
2. **Hidden defaults:** llama.cpp b11246 ships `--fit on` and an 8 GiB host prompt cache; mlx-lm stops batching
   whenever a request carries a seed. Each was found in `--help` or the source and pinned down explicitly.
3. **Wrong memory metric:** `phys_footprint` excludes mmap'd weights, so llama.cpp "used" 0.4 GiB for a 4.7 GiB
   model. I switched to llama.cpp's exit breakdown and the system-wide AGX delta, and the KV figure matched the
   geometric estimate to the MiB.

## 4. Honesty mechanisms (45 s)
- TPOT is labelled `token_level` only when payload count equals the server's token count; otherwise `chunk_derived`.
- Runs under memory pressure or swap are auto-flagged. My machine was under pressure, and the report says so.
- p95 appears in parentheses below n = 200. Early EOS is counted, never padded.
- The prompt is verified per backend (llama.cpp renders byte-identically to the pinned HF template).

## 5. Results (60 s)
All Mac numbers were measured under memory pressure. Say so first, then:
- **The investigation (the story to tell):** llama.cpp at 4 concurrent users had 4× worse per-token latency and almost no
  throughput gain. `llama-batched-bench` reproduced it without HTTP, so it wasn't the server. Reading `ggml_metal_op_mul_mat`
  at the pinned commit showed batch thresholds: K-quants switch to a small-batch kernel at 4 and to simdgroup matmul above 8.
  A Q8_0 model predicted a different pattern and showed it. Then through the real server: throughput *fell* from C=3 to C=4
  and jumped +63% at C=9. The result is a configuration rule, not a vibe.
- **Selection:** Qwen3-8B showed no measurable quality gain over Qwen3-4B on my 60-item suite (paired bootstrap; no
  significant difference) at twice the memory and latency. So for this Mac: 4B at ~4-bit, llama.cpp for one user, MLX for several.
- **Integrity checks that paid off:** llama.cpp and Ollama on the byte-identical file scored within 0.004; Ollama's
  suspiciously fast TTFT turned out to be hidden prefix-cache reuse; Ollama 0.9.0 couldn't load SmolLM3, which the harness
  initially mislabeled as a timeout until permanent load errors were classified explicitly.

## 6. What's next (30 s)
Re-run the headline Mac cells without memory pressure, then grow the application suite with harder items.

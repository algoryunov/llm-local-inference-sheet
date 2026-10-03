# Implementation plan

Written 2026-09-29 after inspecting the environment and current upstream docs.
This is the working plan. For what was actually delivered, see the README and
`docs/findings.md`.

## Environment findings that shape the design

| Observation | Consequence |
|---|---|
| Apple M3 Pro, 18 GB (19,327,352,832 B), 11 CPU (5P+6E), 14 GPU cores, Metal 4, macOS 26.6.2 | Mac experiments limited to ≤8B at 4-bit for comfortable headroom. 14B Q4_K_M (9.0 GB) is a capacity experiment. |
| The Claude Code shell runs **under Rosetta** (`sysctl.proc_translated=1`, `uname -m` = x86_64) | Universal binaries launched from this shell would run as x86_64 and lose Metal/NEON performance. Backends are launched with `arch -arm64`, and `doctor` plus every run manifest record the architecture of the server process. |
| Only ~43 GB free disk | Download one model family at a time and measure before downloading more. Ollama import duplicates the GGUF into its blob store, so the estimator counts that copy. |
| No HF token | `meta-llama/Llama-3.1-8B-Instruct` (gated=manual) is recorded as `gated` until the user supplies `HF_TOKEN`. |
| Ollama 0.9.0 installed (current stable is 0.34.4), not running | Record the installed version as-is and document the upgrade path. The installed version is not upgraded silently. |
| llama.cpp `b11246` provides `llama-b11246-bin-macos-arm64.tar.gz` with a published sha256 | A pinned official binary is installed into `.tools/` and verified by digest. Homebrew is not used, so the build number is exact. |
| mlx-lm 0.31.3 server: continuous batching only when `seed` is unset. `/health` returns OK before the model has loaded. Reports `cached_tokens`. | Never send `seed` to MLX in throughput runs. Readiness means a completed 1-token generation, not `/health`. `cached_tokens` is used to verify there was no prefix-cache reuse. |
| Qwen3-8B source: weights uploaded 2025-04-28; `tokenizer_config.json` changed 2025-05-19. GGUF repo created 2025-05-21. MLX repo 2025-07-07. | Weight provenance is plausible but unproven. Embedded chat templates are hashed and compared rather than assumed equal. |
| TensorRT-LLM supported-hardware page (commit fca831e, 2026-09-21) lists Ada as **L20, L40/L40S only**. RTX 4090 is not listed. | RTX 4090 TRT-LLM is an *attempt* labelled "not in official support matrix". L40S is the documented fallback, and the fallback repeats the baselines on the same GPU. |
| TRT-LLM latest container is `release:1.3.0rc28` (pre-release) | The chosen tag and digest are resolved at rental time. The LLM API / `trtllm-serve` PyTorch path is preferred. We never claim that a TensorRT engine was built. |

## Architecture

```
CLI (argparse) ─┬─ config (pydantic, YAML) ── registry (models.yaml)
                ├─ plan: expands matrix → cells, estimates storage/requests, no inference
                ├─ serve/bench: Backend adapter (process control) ──► server process
                │                 │
                │                 └─ Runner (asyncio, httpx) ── SSE/NDJSON parser ── raw events
                │                        └─ telemetry sampler (thread)
                ├─ evaluate: application suite → scorers
                └─ report: saved observations → metrics → CSV/MD/HTML (no inference)
```

Backend process control is separate from the HTTP client. The client talks to
any OpenAI-compatible endpoint, plus Ollama's native NDJSON endpoint, which is
the only way to get Ollama's server-side timings.

## Milestones and status targets for this session

1. CLI, registry, deterministic mock server, streaming parser, metrics, report pipeline, tests. **Do now.**
2. Real Mac slice: Qwen3-8B Q4_K_M via llama.cpp Metal. **Do now.**
3. MLX-LM and Ollama. Initial comparison of 3 models (Llama gated). Quality suite. **Do now, within disk limits.**
4. Dockerfiles (HF/vLLM/TRT-LLM), Vast recipes, preflight, smoke. **Author and lint only.** No GPU, no builds of multi-GB images on a 43 GB disk.
5. NVIDIA measurements. **Blocked: requires a rented GPU.** Cells stay `untested`.
6. 1.7B/14B capacity sweep on Mac, plus one quantization experiment. **Attempt if disk allows.**
7. Findings, profiling investigation, talk track. **Only from real data.**

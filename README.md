# LLM Local Inference Sheet

Which local LLM and runtime should you run on a MacBook? This repo answers that with measurements, not
vendor numbers: real llama.cpp, MLX-LM and Ollama servers, one shared streaming client, and every metric
recomputed from raw stream events.

## Recommendation

For an 18 GB M3 Pro, with a budget of quality ≥ 0.80, ≤ 3 s per task and ≤ 6 GiB GPU memory:

- **Model:** Qwen3-4B at ~4-bit. Qwen3-8B scores no better on the task suite at nearly twice the latency and memory; SmolLM3-3B misses the quality bar.
- **One user, interactive:** llama.cpp (`llama-server`). Lowest time to first token, fine up to 3 concurrent requests.
- **Several concurrent requests:** MLX-LM. Throughput scales 2.5× at 4 requests and 2.9× at 8; llama.cpp plateaus at 4–8.
- **Ollama:** convenient and runs the same kernels as llama.cpp, but 0.9.0 cannot load SmolLM3 and reuses prefix caches you cannot turn off.
- **Proofreading:** Qwen3-8B corrects grammar better (F0.5 0.718 vs 0.675). Neither model is safe for blind auto-correct.

## Results

Apple M3 Pro, 18 GB, macOS 26.6.2. All runs were taken under memory pressure, so absolute speeds are
pessimistic; use them to compare rows, not as absolute numbers. Full tables and confidence intervals: [docs/findings.md](docs/findings.md).

**Speed:** Qwen3-8B, 1,024 tokens in / 256 out, n = 30 per cell.

| Runtime | Time to first token | Time per output token | Throughput, 1 → 4 requests |
|---|---|---|---|
| llama.cpp b11246, Q4_K_M | 3.74 s | 39.7 ms | 18.4 → 21.0 tok/s |
| MLX-LM 0.31.3, MLX 4-bit | 3.81 s | 37.6 ms | 18.8 → **35.5** tok/s |
| Ollama 0.9.0, same Q4_K_M file | 3.98 s | 41.9 ms | 17.2 → 18.3 tok/s |

**Task quality:** 60 items of JSON extraction, request routing and incident reports.

| Model | Score | Latency p50 | GPU memory |
|---|---|---|---|
| Qwen3-4B, MLX-LM | **0.864** | 2.18 s | 3.4 GiB |
| Qwen3-4B, llama.cpp | 0.858 | 1.85 s | 3.3 GiB |
| Qwen3-8B, llama.cpp / MLX-LM | 0.819 / 0.815 | 3.5–3.6 s | 5.8–6.1 GiB |
| SmolLM3-3B, llama.cpp / MLX-LM | 0.775 / 0.739 | 1.3–1.7 s | 2.3–3.1 GiB |

**Findings**

- **Why llama.cpp stops scaling:** on Metal with K-quant models, throughput *drops* from 3 to 4 concurrent requests
  and jumps 63% at 9. The cause is batch-size thresholds in its matmul kernel selection, confirmed from source,
  a counter-experiment and an end-to-end run ([§C](docs/findings.md#c-profiling-investigation-why-llamacpp-stops-scaling-at-4-concurrent-requests-on-metal)).
- GGUF Q4_K_M and MLX 4-bit give the same quality. llama.cpp and Ollama on the identical file differ by 0.004.
- **Proofreading** (748 JFLEG sentences, ERRANT F0.5): 8B beats 4B at 1.7× the latency. Both models edit about a
  third of already-correct sentences; both handle formal/casual tone rewrites well ([§F](docs/findings.md#f-proofreading-and-style-rewriting-results-publicproofread)).
- **Capacity:** Qwen3-14B at 16k context fits in 18 GB (11.3 of 13.6 GiB Metal limit) but takes 3 minutes to the
  first token. A q8_0 KV cache halves KV memory.

All results as an interactive page: [results-public/dashboard/index.html](results-public/dashboard/index.html) (open it locally in a browser).

## Quick start

```bash
uv venv --python /opt/homebrew/opt/python@3.12/bin/python3.12 .venv   # native arm64 Python, not Rosetta
uv pip install --python .venv/bin/python -e ".[dev,workloads,proofread]"
scripts/install_llama_cpp.sh          # pinned llama.cpp b11246 into .tools/
scripts/install_mlx.sh                # pinned mlx-lm in .venvs/mlx
source .venv/bin/activate

llm-sheet doctor                      # hardware, memory pressure, Rosetta, available backends
llm-sheet plan  --config configs/experiments/mac-smoke.yaml
llm-sheet bench --config configs/experiments/mac-smoke.yaml --download
llm-sheet report --runs results/mac-smoke
pytest                                # CPU-only tests
```

Other experiments are in [configs/experiments/](configs/experiments/). Quality suites run with `llm-sheet evaluate`;
the proofreading suite first needs `python scripts/prepare_proofread.py` to download JFLEG.

## How it works

1. `configs/models.yaml` pins every model file by revision and sha256; an experiment config lists which models,
   runtimes, workloads and concurrency levels to combine.
2. For each combination, the harness starts a fresh server, waits until it is ready, and sends requests from one
   shared async client, so time to first token means the same thing for every runtime.
3. Every stream chunk is stored with a monotonic timestamp. Speed, quality scores and reports are computed from
   those raw events and can be recomputed at any time.

Diagrams: [docs/architecture.md](docs/architecture.md). Metric definitions: [docs/methodology.md](docs/methodology.md).

## Published data

Local runs go to the git-ignored `results/`. `python scripts/export_public.py` filters them into `results-public/`
and rebuilds the report, compatibility matrix and dashboard. Git keeps the compact part (manifests, configs,
summaries); the raw events, telemetry and server logs ship as a release asset:

```bash
curl -LO https://github.com/algoryunov/llm-local-inference-sheet/releases/latest/download/results-public-full.tar.gz
tar -xzf results-public-full.tar.gz        # restores the raw files under results-public/
```

## Documentation

- [docs/findings.md](docs/findings.md): all findings with evidence, including the profiling investigation
- [docs/methodology.md](docs/methodology.md): metric definitions, load model, cache and template controls
- [docs/architecture.md](docs/architecture.md): diagrams of the data flow and code layout
- [docs/compatibility.md](docs/compatibility.md): model × runtime matrix
- [docs/macos.md](docs/macos.md): native Mac backends, the Rosetta trap, unified-memory metrics
- [results-public/README.md](results-public/README.md): layout of the published runs

## Limitations

- Measured under memory pressure: good for comparisons, pessimistic in absolute terms.
- llama.cpp vs MLX compares runtime *and* quantization (GGUF Q4_K_M vs MLX 4-bit); only llama.cpp vs Ollama
  uses byte-identical weights.
- MLX-LM and Ollama cannot force a fixed output length; early stops are recorded and reported separately.
- The task suite is small (60 synthetic items), so differences of a few points are noise.
- Closed-loop load only; no thermal or power telemetry.
- Llama-3.1-8B is gated and was not run.

## License

[MIT](LICENSE) for the code and the synthetic datasets. Model weights keep their own licences. JFLEG
(CC BY-NC-SA 4.0) is downloaded by `scripts/prepare_proofread.py` and not redistributed.

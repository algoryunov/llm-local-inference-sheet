# Results tables

Generated from saved observations by `llm-sheet report`. Latest run per cell. p95 is shown for every run but is only *reportable* at n ≥ 200 (see methodology). Rows marked SYNTHETIC come from the mock server and are not model performance. `memory-pressure flag = yes` means pressure rose above normal or >256 MiB was swapped during the run: those timings may include paging and are not headline-quality.

## Performance

TTFT p95 in parentheses: n < 200, exploratory only.

| experiment | backend | artifact | quant | hardware | workload | C | state | n ok | TTFT p50 s | TTFT p95 s | TPOT p50 ms | E2E p50 s | decode tok/s/req p50 | agg out tok/s | native prefill tok/s | GPU mem Δ GiB | peak RSS GiB | MLX peak GiB | llama.cpp GPU self MiB | ready s | early stops | timing basis | memory-pressure flag |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mac-capacity | llama-cpp | qwen3-1.7b-mlx4 | mlx-affine-4bit | – | ctx-1024-128 | 1 | unsupported | – | – | – | – | – | – | – | – | – | – | – | – | – | – | – | – |
| mac-capacity | llama-cpp | qwen3-1.7b-mlx4 | mlx-affine-4bit | – | ctx-16384-128 | 1 | unsupported | – | – | – | – | – | – | – | – | – | – | – | – | – | – | – | – |
| mac-capacity | llama-cpp | qwen3-1.7b-mlx4 | mlx-affine-4bit | – | ctx-4096-128 | 1 | unsupported | – | – | – | – | – | – | – | – | – | – | – | – | – | – | – | – |
| mac-capacity | llama-cpp | qwen3-1.7b-q8 | Q8_0 | Apple M3 Pro 18GB | ctx-1024-128 | 1 | measured | 5 | 0.82 | (0.86) | 18.19 | 3.16 | 54.98 | 40.61 | 1258.60 | 1.94 | 1.93 | – | 1929 | 1.90 | 0 | token_level | yes |
| mac-capacity | llama-cpp | qwen3-1.7b-q8 | Q8_0 | Apple M3 Pro 18GB | ctx-16384-128 | 1 | measured | 5 | 28.32 | (29.28) | 34.21 | 32.90 | 29.23 | 3.97 | 579.26 | 4.31 | 3.60 | – | 3667 | 1.11 | 0 | token_level | yes |
| mac-capacity | llama-cpp | qwen3-1.7b-q8 | Q8_0 | Apple M3 Pro 18GB | ctx-4096-128 | 1 | measured | 5 | 3.63 | (4.03) | 20.27 | 6.20 | 49.32 | 20.09 | 1134.15 | 2.78 | 2.32 | – | 2268 | 0.82 | 0 | token_level | yes |
| mac-capacity | llama-cpp | qwen3-14b-mlx4 | mlx-affine-4bit | – | ctx-1024-128 | 1 | unsupported | – | – | – | – | – | – | – | – | – | – | – | – | – | – | – | – |
| mac-capacity | llama-cpp | qwen3-14b-mlx4 | mlx-affine-4bit | – | ctx-16384-128 | 1 | unsupported | – | – | – | – | – | – | – | – | – | – | – | – | – | – | – | – |
| mac-capacity | llama-cpp | qwen3-14b-mlx4 | mlx-affine-4bit | – | ctx-4096-128 | 1 | unsupported | – | – | – | – | – | – | – | – | – | – | – | – | – | – | – | – |
| mac-capacity | llama-cpp | qwen3-14b-q4 | Q4_K_M | Apple M3 Pro 18GB | ctx-1024-128 | 1 | measured | 5 | 7.45 | (7.54) | 76.46 | 17.25 | 13.08 | 7.43 | 137.69 | 9.23 | 6.67 | – | 8902 | 12.17 | 0 | token_level | yes |
| mac-capacity | llama-cpp | qwen3-14b-q4 | Q4_K_M | Apple M3 Pro 18GB | ctx-16384-128 | 1 | measured | 5 | 181.05 | (185.39) | 109.80 | 195.01 | 9.11 | 0.66 | 90.75 | 11.59 | 10.50 | – | 11317 | 3.00 | 0 | token_level | yes |
| mac-capacity | llama-cpp | qwen3-14b-q4 | Q4_K_M | Apple M3 Pro 18GB | ctx-4096-128 | 1 | measured | 5 | 37.93 | (41.96) | 95.83 | 49.82 | 10.44 | 2.52 | 108.09 | 9.55 | 7.73 | – | 9385 | 6.15 | 0 | token_level | yes |
| mac-capacity | llama-cpp | qwen3-8b-mlx4 | mlx-affine-4bit | – | ctx-1024-128 | 1 | unsupported | – | – | – | – | – | – | – | – | – | – | – | – | – | – | – | – |
| mac-capacity | llama-cpp | qwen3-8b-mlx4 | mlx-affine-4bit | – | ctx-16384-128 | 1 | unsupported | – | – | – | – | – | – | – | – | – | – | – | – | – | – | – | – |
| mac-capacity | llama-cpp | qwen3-8b-mlx4 | mlx-affine-4bit | – | ctx-4096-128 | 1 | unsupported | – | – | – | – | – | – | – | – | – | – | – | – | – | – | – | – |
| mac-capacity | llama-cpp | qwen3-8b-q4 | Q4_K_M | Apple M3 Pro 18GB | ctx-1024-128 | 1 | measured | 5 | 3.92 | (4.17) | 41.37 | 9.46 | 24.17 | 13.71 | 261.94 | 5.69 | 4.21 | – | 5058 | 4.72 | 0 | token_level | yes |
| mac-capacity | llama-cpp | qwen3-8b-q4 | Q4_K_M | Apple M3 Pro 18GB | ctx-16384-128 | 1 | measured | 5 | 100.34 | (106.43) | 63.24 | 108.38 | 15.81 | 1.18 | 163.35 | 8.03 | 6.89 | – | 7252 | 2.00 | 0 | token_level | yes |
| mac-capacity | llama-cpp | qwen3-8b-q4 | Q4_K_M | Apple M3 Pro 18GB | ctx-4096-128 | 1 | measured | 5 | 18.57 | (18.79) | 50.56 | 25.21 | 19.78 | 5.17 | 220.71 | 6.06 | 5.26 | – | 5493 | 1.43 | 0 | token_level | yes |
| mac-capacity | mlx-lm | qwen3-1.7b-mlx4 | mlx-affine-4bit | Apple M3 Pro 18GB | ctx-1024-128 | 1 | measured | 5 | 0.93 | (0.93) | 11.35 | 2.37 | 88.12 | 53.94 | – | 2.05 | 1.16 | 1.74 | – | 2.70 | 0 | chunk_derived | yes |
| mac-capacity | mlx-lm | qwen3-1.7b-mlx4 | mlx-affine-4bit | Apple M3 Pro 18GB | ctx-16384-128 | 1 | measured | 5 | 21.37 | (21.39) | 27.29 | 24.86 | 36.64 | 5.15 | – | 6.92 | 1.17 | 4.86 | – | 2.14 | 0 | chunk_derived | yes |
| mac-capacity | mlx-lm | qwen3-1.7b-mlx4 | mlx-affine-4bit | Apple M3 Pro 18GB | ctx-4096-128 | 1 | measured | 5 | 4.49 | (4.73) | 17.43 | 6.70 | 57.37 | 19.05 | – | 3.10 | 1.17 | 2.15 | – | 1.89 | 0 | chunk_derived | yes |
| mac-capacity | mlx-lm | qwen3-1.7b-q8 | Q8_0 | – | ctx-1024-128 | 1 | unsupported | – | – | – | – | – | – | – | – | – | – | – | – | – | – | – | – |
| mac-capacity | mlx-lm | qwen3-1.7b-q8 | Q8_0 | – | ctx-16384-128 | 1 | unsupported | – | – | – | – | – | – | – | – | – | – | – | – | – | – | – | – |
| mac-capacity | mlx-lm | qwen3-1.7b-q8 | Q8_0 | – | ctx-4096-128 | 1 | unsupported | – | – | – | – | – | – | – | – | – | – | – | – | – | – | – | – |
| mac-capacity | mlx-lm | qwen3-14b-mlx4 | mlx-affine-4bit | Apple M3 Pro 18GB | ctx-1024-128 | 1 | measured | 5 | 7.53 | (7.76) | 70.17 | 16.42 | 14.25 | 7.72 | – | 9.02 | 6.75 | 8.32 | – | 4.87 | 0 | token_level | yes |
| mac-capacity | mlx-lm | qwen3-14b-mlx4 | mlx-affine-4bit | Apple M3 Pro 18GB | ctx-16384-128 | 1 | measured | 5 | 143.42 | (162.55) | 110.86 | 156.77 | 9.02 | 0.78 | – | 12.40 | 5.80 | 12.72 | – | 4.90 | 0 | token_level | yes |
| mac-capacity | mlx-lm | qwen3-14b-mlx4 | mlx-affine-4bit | Apple M3 Pro 18GB | ctx-4096-128 | 1 | measured | 5 | 30.80 | (32.12) | 74.31 | 40.22 | 13.46 | 3.15 | – | 10.42 | 6.48 | 8.91 | – | 4.98 | 0 | token_level | yes |
| mac-capacity | mlx-lm | qwen3-14b-q4 | Q4_K_M | – | ctx-1024-128 | 1 | unsupported | – | – | – | – | – | – | – | – | – | – | – | – | – | – | – | – |
| mac-capacity | mlx-lm | qwen3-14b-q4 | Q4_K_M | – | ctx-16384-128 | 1 | unsupported | – | – | – | – | – | – | – | – | – | – | – | – | – | – | – | – |
| mac-capacity | mlx-lm | qwen3-14b-q4 | Q4_K_M | – | ctx-4096-128 | 1 | unsupported | – | – | – | – | – | – | – | – | – | – | – | – | – | – | – | – |
| mac-capacity | mlx-lm | qwen3-8b-mlx4 | mlx-affine-4bit | Apple M3 Pro 18GB | ctx-1024-128 | 1 | measured | 5 | 3.93 | (3.94) | 37.81 | 8.73 | 26.45 | 14.66 | – | 5.36 | 3.68 | 4.99 | – | 3.17 | 0 | token_level | yes |
| mac-capacity | mlx-lm | qwen3-8b-mlx4 | mlx-affine-4bit | Apple M3 Pro 18GB | ctx-16384-128 | 1 | measured | 5 | 80.13 | (83.92) | 71.66 | 89.07 | 13.96 | 1.42 | – | 11.98 | 3.68 | 8.88 | – | 3.55 | 0 | token_level | yes |
| mac-capacity | mlx-lm | qwen3-8b-mlx4 | mlx-affine-4bit | Apple M3 Pro 18GB | ctx-4096-128 | 1 | measured | 5 | 17.16 | (17.38) | 46.34 | 23.05 | 21.58 | 5.59 | – | 7.19 | 3.68 | 5.57 | – | 3.19 | 0 | token_level | yes |
| mac-capacity | mlx-lm | qwen3-8b-q4 | Q4_K_M | – | ctx-1024-128 | 1 | unsupported | – | – | – | – | – | – | – | – | – | – | – | – | – | – | – | – |
| mac-capacity | mlx-lm | qwen3-8b-q4 | Q4_K_M | – | ctx-16384-128 | 1 | unsupported | – | – | – | – | – | – | – | – | – | – | – | – | – | – | – | – |
| mac-capacity | mlx-lm | qwen3-8b-q4 | Q4_K_M | – | ctx-4096-128 | 1 | unsupported | – | – | – | – | – | – | – | – | – | – | – | – | – | – | – | – |
| mac-comparison | llama-cpp | llama3.1-8b-mlx4 | mlx-affine-4bit | – | balanced-1024-256 | 1 | unsupported | – | – | – | – | – | – | – | – | – | – | – | – | – | – | – | – |
| mac-comparison | llama-cpp | llama3.1-8b-mlx4 | mlx-affine-4bit | – | balanced-1024-256 | 4 | unsupported | – | – | – | – | – | – | – | – | – | – | – | – | – | – | – | – |
| mac-comparison | llama-cpp | llama3.1-8b-q4 | Q4_K_M | – | balanced-1024-256 | 1 | gated | – | – | – | – | – | – | – | – | – | – | – | – | – | – | – | – |
| mac-comparison | llama-cpp | llama3.1-8b-q4 | Q4_K_M | – | balanced-1024-256 | 4 | gated | – | – | – | – | – | – | – | – | – | – | – | – | – | – | – | – |
| mac-comparison | llama-cpp | qwen3-4b-mlx4 | mlx-affine-4bit | – | balanced-1024-256 | 1 | unsupported | – | – | – | – | – | – | – | – | – | – | – | – | – | – | – | – |
| mac-comparison | llama-cpp | qwen3-4b-mlx4 | mlx-affine-4bit | – | balanced-1024-256 | 4 | unsupported | – | – | – | – | – | – | – | – | – | – | – | – | – | – | – | – |
| mac-comparison | llama-cpp | qwen3-4b-q4 | Q4_K_M | Apple M3 Pro 18GB | balanced-1024-256 | 1 | measured | 30 | 2.05 | (2.06) | 22.87 | 7.88 | 43.72 | 32.52 | 501.87 | 3.27 | 2.65 | – | 2660 | 1.91 | 0 | token_level | yes |
| mac-comparison | llama-cpp | qwen3-4b-q4 | Q4_K_M | Apple M3 Pro 18GB | balanced-1024-256 | 4 | measured | 30 | 4.34 | (8.29) | 92.38 | 27.97 | 10.82 | 36.70 | 462.04 | 4.82 | 3.32 | – | 3376 | 0.61 | 0 | token_level | yes |
| mac-comparison | llama-cpp | qwen3-8b-mlx4 | mlx-affine-4bit | – | balanced-1024-256 | 1 | unsupported | – | – | – | – | – | – | – | – | – | – | – | – | – | – | – | – |
| mac-comparison | llama-cpp | qwen3-8b-mlx4 | mlx-affine-4bit | – | balanced-1024-256 | 4 | unsupported | – | – | – | – | – | – | – | – | – | – | – | – | – | – | – | – |
| mac-comparison | llama-cpp | qwen3-8b-q4 | Q4_K_M | Apple M3 Pro 18GB | balanced-1024-256 | 1 | measured | 30 | 3.74 | (3.84) | 39.66 | 13.86 | 25.22 | 18.43 | 274.62 | 5.78 | 4.44 | – | 5094 | 3.49 | 0 | chunk_derived,token_level | yes |
| mac-comparison | llama-cpp | qwen3-8b-q4 | Q4_K_M | Apple M3 Pro 18GB | balanced-1024-256 | 4 | measured | 30 | 7.68 | (15.12) | 161.73 | 48.91 | 6.18 | 21.05 | 264.27 | 6.29 | 5.45 | – | 5775 | 1.44 | 0 | token_level | yes |
| mac-comparison | llama-cpp | smollm3-3b-mlx4 | mlx-affine-4bit | – | balanced-1024-256 | 1 | unsupported | – | – | – | – | – | – | – | – | – | – | – | – | – | – | – | – |
| mac-comparison | llama-cpp | smollm3-3b-mlx4 | mlx-affine-4bit | – | balanced-1024-256 | 4 | unsupported | – | – | – | – | – | – | – | – | – | – | – | – | – | – | – | – |
| mac-comparison | llama-cpp | smollm3-3b-q4 | Q4_K_M | Apple M3 Pro 18GB | balanced-1024-256 | 1 | measured | 30 | 1.56 | (1.63) | 17.80 | 6.10 | 56.19 | 41.55 | 659.01 | 2.30 | 1.82 | – | 2001 | 2.26 | 0 | token_level | yes |
| mac-comparison | llama-cpp | smollm3-3b-q4 | Q4_K_M | Apple M3 Pro 18GB | balanced-1024-256 | 4 | measured | 30 | 3.29 | (6.35) | 63.10 | 20.77 | 15.85 | 49.31 | 605.93 | 2.73 | 2.08 | – | 2325 | 1.47 | 0 | token_level | yes |
| mac-comparison | mlx-lm | llama3.1-8b-mlx4 | mlx-affine-4bit | – | balanced-1024-256 | 1 | gated | – | – | – | – | – | – | – | – | – | – | – | – | – | – | – | – |
| mac-comparison | mlx-lm | llama3.1-8b-mlx4 | mlx-affine-4bit | – | balanced-1024-256 | 4 | gated | – | – | – | – | – | – | – | – | – | – | – | – | – | – | – | – |
| mac-comparison | mlx-lm | llama3.1-8b-q4 | Q4_K_M | – | balanced-1024-256 | 1 | unsupported | – | – | – | – | – | – | – | – | – | – | – | – | – | – | – | – |
| mac-comparison | mlx-lm | llama3.1-8b-q4 | Q4_K_M | – | balanced-1024-256 | 4 | unsupported | – | – | – | – | – | – | – | – | – | – | – | – | – | – | – | – |
| mac-comparison | mlx-lm | qwen3-4b-mlx4 | mlx-affine-4bit | Apple M3 Pro 18GB | balanced-1024-256 | 1 | measured | 30 | 2.36 | (2.38) | 25.17 | 8.76 | 39.72 | 29.10 | – | 3.37 | 1.62 | 2.97 | – | 3.16 | 0 | token_level | yes |
| mac-comparison | mlx-lm | qwen3-4b-mlx4 | mlx-affine-4bit | Apple M3 Pro 18GB | balanced-1024-256 | 4 | measured | 30 | 8.63 | (14.21) | 38.58 | 18.31 | 25.92 | 48.97 | – | 10.93 | 1.66 | 5.97 | – | 2.80 | 0 | token_level | yes |
| mac-comparison | mlx-lm | qwen3-4b-q4 | Q4_K_M | – | balanced-1024-256 | 1 | unsupported | – | – | – | – | – | – | – | – | – | – | – | – | – | – | – | – |
| mac-comparison | mlx-lm | qwen3-4b-q4 | Q4_K_M | – | balanced-1024-256 | 4 | unsupported | – | – | – | – | – | – | – | – | – | – | – | – | – | – | – | – |
| mac-comparison | mlx-lm | qwen3-8b-mlx4 | mlx-affine-4bit | Apple M3 Pro 18GB | balanced-1024-256 | 1 | measured | 30 | 3.81 | (4.29) | 37.61 | 13.41 | 26.59 | 18.75 | – | 6.11 | 3.65 | 4.99 | – | 8.04 | 0 | token_level | yes |
| mac-comparison | mlx-lm | qwen3-8b-mlx4 | mlx-affine-4bit | Apple M3 Pro 18GB | balanced-1024-256 | 4 | measured | 30 | 14.83 | (15.77) | 52.17 | 28.11 | 19.17 | 35.47 | – | 7.16 | 3.68 | 6.05 | – | 4.18 | 0 | token_level | yes |
| mac-comparison | mlx-lm | qwen3-8b-q4 | Q4_K_M | – | balanced-1024-256 | 1 | unsupported | – | – | – | – | – | – | – | – | – | – | – | – | – | – | – | – |
| mac-comparison | mlx-lm | qwen3-8b-q4 | Q4_K_M | – | balanced-1024-256 | 4 | unsupported | – | – | – | – | – | – | – | – | – | – | – | – | – | – | – | – |
| mac-comparison | mlx-lm | smollm3-3b-mlx4 | mlx-affine-4bit | Apple M3 Pro 18GB | balanced-1024-256 | 1 | measured | 30 | 1.62 | (1.64) | 18.33 | 6.30 | 54.56 | 40.62 | – | 3.12 | 2.00 | 2.56 | – | 3.11 | 0 | token_level | yes |
| mac-comparison | mlx-lm | smollm3-3b-mlx4 | mlx-affine-4bit | Apple M3 Pro 18GB | balanced-1024-256 | 4 | measured | 30 | 6.22 | (9.59) | 25.64 | 12.81 | 39.01 | 71.03 | – | 5.60 | 2.01 | 4.25 | – | 2.07 | 0 | token_level | yes |
| mac-comparison | mlx-lm | smollm3-3b-q4 | Q4_K_M | – | balanced-1024-256 | 1 | unsupported | – | – | – | – | – | – | – | – | – | – | – | – | – | – | – | – |
| mac-comparison | mlx-lm | smollm3-3b-q4 | Q4_K_M | – | balanced-1024-256 | 4 | unsupported | – | – | – | – | – | – | – | – | – | – | – | – | – | – | – | – |
| mac-comparison | ollama | llama3.1-8b-mlx4 | mlx-affine-4bit | – | balanced-1024-256 | 1 | unsupported | – | – | – | – | – | – | – | – | – | – | – | – | – | – | – | – |
| mac-comparison | ollama | llama3.1-8b-mlx4 | mlx-affine-4bit | – | balanced-1024-256 | 4 | unsupported | – | – | – | – | – | – | – | – | – | – | – | – | – | – | – | – |
| mac-comparison | ollama | llama3.1-8b-q4 | Q4_K_M | – | balanced-1024-256 | 1 | gated | – | – | – | – | – | – | – | – | – | – | – | – | – | – | – | – |
| mac-comparison | ollama | llama3.1-8b-q4 | Q4_K_M | – | balanced-1024-256 | 4 | gated | – | – | – | – | – | – | – | – | – | – | – | – | – | – | – | – |
| mac-comparison | ollama | qwen3-4b-mlx4 | mlx-affine-4bit | – | balanced-1024-256 | 1 | unsupported | – | – | – | – | – | – | – | – | – | – | – | – | – | – | – | – |
| mac-comparison | ollama | qwen3-4b-mlx4 | mlx-affine-4bit | – | balanced-1024-256 | 4 | unsupported | – | – | – | – | – | – | – | – | – | – | – | – | – | – | – | – |
| mac-comparison | ollama | qwen3-4b-q4 | Q4_K_M | Apple M3 Pro 18GB | balanced-1024-256 | 1 | measured | 30 | 2.20 | (2.20) | 24.78 | 8.52 | 40.36 | 30.03 | 468.50 | 3.32 | 2.68 | – | – | 6.39 | 0 | token_level | yes |
| mac-comparison | ollama | qwen3-4b-q4 | Q4_K_M | Apple M3 Pro 18GB | balanced-1024-256 | 4 | measured | 24 | 9.02 | (9.14) | 86.08 | 30.89 | 11.62 | 29.15 | 113.71 | 3.82 | 3.41 | – | – | 2.83 | 0 | chunk_derived,token_level | yes |
| mac-comparison | ollama | qwen3-8b-mlx4 | mlx-affine-4bit | – | balanced-1024-256 | 1 | unsupported | – | – | – | – | – | – | – | – | – | – | – | – | – | – | – | – |
| mac-comparison | ollama | qwen3-8b-mlx4 | mlx-affine-4bit | – | balanced-1024-256 | 4 | unsupported | – | – | – | – | – | – | – | – | – | – | – | – | – | – | – | – |
| mac-comparison | ollama | qwen3-8b-q4 | Q4_K_M | Apple M3 Pro 18GB | balanced-1024-256 | 1 | measured | 30 | 3.98 | (4.38) | 41.88 | 14.65 | 23.88 | 17.17 | 258.61 | 6.13 | 4.97 | – | – | 23.27 | 0 | token_level | yes |
| mac-comparison | ollama | qwen3-8b-q4 | Q4_K_M | Apple M3 Pro 18GB | balanced-1024-256 | 4 | measured | 28 | 16.05 | (16.31) | 144.02 | 52.83 | 6.94 | 18.28 | 63.92 | 6.38 | 5.62 | – | – | 5.93 | 0 | chunk_derived,token_level | yes |
| mac-comparison | ollama | smollm3-3b-mlx4 | mlx-affine-4bit | – | balanced-1024-256 | 1 | unsupported | – | – | – | – | – | – | – | – | – | – | – | – | – | – | – | – |
| mac-comparison | ollama | smollm3-3b-mlx4 | mlx-affine-4bit | – | balanced-1024-256 | 4 | unsupported | – | – | – | – | – | – | – | – | – | – | – | – | – | – | – | – |
| mac-comparison | ollama | smollm3-3b-q4 | Q4_K_M | Apple M3 Pro 18GB | balanced-1024-256 | 1 | unsupported | – | – | – | – | – | – | – | – | 0.00 | 0.04 | – | – | – | – | – | yes |
| mac-comparison | ollama | smollm3-3b-q4 | Q4_K_M | Apple M3 Pro 18GB | balanced-1024-256 | 4 | unsupported | – | – | – | – | – | – | – | – | 0.01 | 0.03 | – | – | – | – | – | no |
| mac-investigation-batching | llama-cpp | qwen3-4b-mlx4 | mlx-affine-4bit | – | decode-128-256 | 1 | unsupported | – | – | – | – | – | – | – | – | – | – | – | – | – | – | – | – |
| mac-investigation-batching | llama-cpp | qwen3-4b-mlx4 | mlx-affine-4bit | – | decode-128-256 | 12 | unsupported | – | – | – | – | – | – | – | – | – | – | – | – | – | – | – | – |
| mac-investigation-batching | llama-cpp | qwen3-4b-mlx4 | mlx-affine-4bit | – | decode-128-256 | 2 | unsupported | – | – | – | – | – | – | – | – | – | – | – | – | – | – | – | – |
| mac-investigation-batching | llama-cpp | qwen3-4b-mlx4 | mlx-affine-4bit | – | decode-128-256 | 3 | unsupported | – | – | – | – | – | – | – | – | – | – | – | – | – | – | – | – |
| mac-investigation-batching | llama-cpp | qwen3-4b-mlx4 | mlx-affine-4bit | – | decode-128-256 | 4 | unsupported | – | – | – | – | – | – | – | – | – | – | – | – | – | – | – | – |
| mac-investigation-batching | llama-cpp | qwen3-4b-mlx4 | mlx-affine-4bit | – | decode-128-256 | 8 | unsupported | – | – | – | – | – | – | – | – | – | – | – | – | – | – | – | – |
| mac-investigation-batching | llama-cpp | qwen3-4b-mlx4 | mlx-affine-4bit | – | decode-128-256 | 9 | unsupported | – | – | – | – | – | – | – | – | – | – | – | – | – | – | – | – |
| mac-investigation-batching | llama-cpp | qwen3-4b-q4 | Q4_K_M | Apple M3 Pro 18GB | decode-128-256 | 1 | measured | 30 | 0.28 | (0.31) | 22.95 | 6.13 | 43.58 | 41.22 | 486.95 | 2.96 | 2.46 | – | 2515 | 1.91 | 0 | token_level | yes |
| mac-investigation-batching | llama-cpp | qwen3-4b-q4 | Q4_K_M | Apple M3 Pro 18GB | decode-128-256 | 12 | measured | 30 | 1.77 | (3.36) | 106.84 | 29.07 | 9.36 | 87.54 | 76.92 | 3.82 | 3.28 | – | 3521 | 1.15 | 0 | token_level | yes |
| mac-investigation-batching | llama-cpp | qwen3-4b-q4 | Q4_K_M | Apple M3 Pro 18GB | decode-128-256 | 2 | measured | 30 | 0.53 | (0.57) | 32.69 | 8.87 | 30.59 | 57.56 | 421.71 | 3.00 | 2.47 | – | 2598 | 1.14 | 0 | token_level | yes |
| mac-investigation-batching | llama-cpp | qwen3-4b-q4 | Q4_K_M | Apple M3 Pro 18GB | decode-128-256 | 3 | measured | 30 | 0.54 | (0.82) | 48.92 | 13.02 | 20.44 | 58.92 | 249.03 | 3.10 | 2.68 | – | 2691 | 0.62 | 0 | token_level | yes |
| mac-investigation-batching | llama-cpp | qwen3-4b-q4 | Q4_K_M | Apple M3 Pro 18GB | decode-128-256 | 4 | measured | 30 | 1.04 | (1.11) | 73.91 | 19.89 | 13.53 | 51.88 | 164.67 | 3.39 | 2.72 | – | 2783 | 0.88 | 0 | token_level | yes |
| mac-investigation-batching | llama-cpp | qwen3-4b-q4 | Q4_K_M | Apple M3 Pro 18GB | decode-128-256 | 8 | measured | 30 | 1.91 | (2.25) | 144.59 | 39.09 | 6.92 | 50.55 | 118.94 | 3.51 | 3.01 | – | 3152 | 0.88 | 0 | token_level | yes |
| mac-investigation-batching | llama-cpp | qwen3-4b-q4 | Q4_K_M | Apple M3 Pro 18GB | decode-128-256 | 9 | measured | 30 | 2.00 | (2.43) | 94.74 | 26.18 | 10.56 | 82.33 | 64.70 | 3.91 | 3.01 | – | 3244 | 1.13 | 0 | token_level | yes |
| mac-investigation-batching | mlx-lm | qwen3-4b-mlx4 | mlx-affine-4bit | Apple M3 Pro 18GB | decode-128-256 | 1 | measured | 30 | 0.42 | (0.49) | 21.35 | 5.88 | 46.83 | 42.45 | – | 2.72 | 1.62 | 2.40 | – | 2.84 | 0 | token_level | yes |
| mac-investigation-batching | mlx-lm | qwen3-4b-mlx4 | mlx-affine-4bit | Apple M3 Pro 18GB | decode-128-256 | 12 | measured | 30 | 2.66 | (3.99) | 84.55 | 24.23 | 11.83 | 124.14 | – | 5.38 | 2.38 | 4.04 | – | 2.08 | 0 | token_level | yes |
| mac-investigation-batching | mlx-lm | qwen3-4b-mlx4 | mlx-affine-4bit | Apple M3 Pro 18GB | decode-128-256 | 2 | measured | 30 | 0.77 | (0.86) | 23.51 | 6.78 | 42.54 | 75.60 | – | 3.14 | 1.62 | 2.70 | – | 3.15 | 0 | token_level | yes |
| mac-investigation-batching | mlx-lm | qwen3-4b-mlx4 | mlx-affine-4bit | Apple M3 Pro 18GB | decode-128-256 | 3 | measured | 30 | 1.07 | (1.40) | 28.85 | 8.42 | 34.67 | 90.97 | – | 3.22 | 1.64 | 2.77 | – | 2.83 | 0 | token_level | yes |
| mac-investigation-batching | mlx-lm | qwen3-4b-mlx4 | mlx-affine-4bit | Apple M3 Pro 18GB | decode-128-256 | 4 | measured | 30 | 1.38 | (1.66) | 30.66 | 9.18 | 32.62 | 104.91 | – | 3.43 | 1.61 | 2.99 | – | 2.93 | 0 | token_level | yes |
| mac-investigation-batching | mlx-lm | qwen3-4b-mlx4 | mlx-affine-4bit | Apple M3 Pro 18GB | decode-128-256 | 8 | measured | 30 | 2.52 | (2.55) | 54.76 | 16.50 | 18.26 | 122.96 | – | 3.45 | 1.97 | 3.47 | – | 2.52 | 0 | token_level | yes |
| mac-investigation-batching | mlx-lm | qwen3-4b-mlx4 | mlx-affine-4bit | Apple M3 Pro 18GB | decode-128-256 | 9 | measured | 30 | 2.64 | (2.79) | 68.42 | 20.05 | 14.62 | 113.45 | – | 3.93 | 2.00 | 3.45 | – | 2.38 | 0 | token_level | yes |
| mac-investigation-batching | mlx-lm | qwen3-4b-q4 | Q4_K_M | – | decode-128-256 | 1 | unsupported | – | – | – | – | – | – | – | – | – | – | – | – | – | – | – | – |
| mac-investigation-batching | mlx-lm | qwen3-4b-q4 | Q4_K_M | – | decode-128-256 | 12 | unsupported | – | – | – | – | – | – | – | – | – | – | – | – | – | – | – | – |
| mac-investigation-batching | mlx-lm | qwen3-4b-q4 | Q4_K_M | – | decode-128-256 | 2 | unsupported | – | – | – | – | – | – | – | – | – | – | – | – | – | – | – | – |
| mac-investigation-batching | mlx-lm | qwen3-4b-q4 | Q4_K_M | – | decode-128-256 | 3 | unsupported | – | – | – | – | – | – | – | – | – | – | – | – | – | – | – | – |
| mac-investigation-batching | mlx-lm | qwen3-4b-q4 | Q4_K_M | – | decode-128-256 | 4 | unsupported | – | – | – | – | – | – | – | – | – | – | – | – | – | – | – | – |
| mac-investigation-batching | mlx-lm | qwen3-4b-q4 | Q4_K_M | – | decode-128-256 | 8 | unsupported | – | – | – | – | – | – | – | – | – | – | – | – | – | – | – | – |
| mac-investigation-batching | mlx-lm | qwen3-4b-q4 | Q4_K_M | – | decode-128-256 | 9 | unsupported | – | – | – | – | – | – | – | – | – | – | – | – | – | – | – | – |
| mac-kvquant-q8 | llama-cpp | qwen3-8b-q4 | Q4_K_M | Apple M3 Pro 18GB | ctx-16384-128 | 1 | measured | 5 | 87.71 | (90.38) | 55.26 | 94.72 | 18.10 | 1.34 | 186.84 | 6.49 | 6.02 | – | 6166 | 0.92 | 0 | token_level | yes |
| mac-kvquant-q8 | llama-cpp | qwen3-8b-q4 | Q4_K_M | Apple M3 Pro 18GB | ctx-4096-128 | 1 | measured | 5 | 16.72 | (16.94) | 45.12 | 22.45 | 22.16 | 5.66 | 245.18 | 12.37 | 5.09 | – | 5214 | 3.74 | 0 | token_level | yes |
| mac-smoke | llama-cpp | qwen3-8b-q4 | Q4_K_M | Apple M3 Pro 18GB | balanced-1024-256 | 1 | measured | 10 | 3.73 | (3.77) | 40.57 | 14.10 | 24.65 | 18.13 | 275.36 | 5.44 | 3.46 | – | – | 10.09 | 0 | token_level | yes |

## Application quality

| backend | artifact | quant | state | n | composite | extract field acc | null handling | routing macro-F1 | unneeded tool calls | CV field acc | strict JSON | E2E p50 s | GPU mem Δ GiB | memory-pressure flag |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| llama-cpp | qwen3-4b-mlx4 | mlx-affine-4bit | unsupported | – | – | – | – | – | – | – | – | – | – | – |
| llama-cpp | qwen3-4b-q4 | Q4_K_M | measured | 938 | 0.80 | – | – | – | – | – | – | 0.70 | 3.32 | yes |
| llama-cpp | qwen3-8b-mlx4 | mlx-affine-4bit | unsupported | – | – | – | – | – | – | – | – | – | – | – |
| llama-cpp | qwen3-8b-q4 | Q4_K_M | measured | 938 | 0.86 | – | – | – | – | – | – | 1.17 | 6.03 | yes |
| mlx-lm | qwen3-4b-mlx4 | mlx-affine-4bit | measured | 938 | 0.83 | – | – | – | – | – | – | 0.79 | 3.08 | yes |
| mlx-lm | qwen3-4b-q4 | Q4_K_M | unsupported | – | – | – | – | – | – | – | – | – | – | – |
| mlx-lm | qwen3-8b-mlx4 | mlx-affine-4bit | measured | 938 | 0.79 | – | – | – | – | – | – | 1.28 | 5.73 | yes |
| mlx-lm | qwen3-8b-q4 | Q4_K_M | unsupported | – | – | – | – | – | – | – | – | – | – | – |
| llama-cpp | llama3.1-8b-mlx4 | mlx-affine-4bit | unsupported | – | – | – | – | – | – | – | – | – | – | – |
| llama-cpp | llama3.1-8b-q4 | Q4_K_M | gated | – | – | – | – | – | – | – | – | – | – | – |
| llama-cpp | qwen3-4b-mlx4 | mlx-affine-4bit | unsupported | – | – | – | – | – | – | – | – | – | – | – |
| llama-cpp | qwen3-4b-q4 | Q4_K_M | measured | 60 | 0.86 | 0.97 | 0.77 | 1.00 | 0 | 0.60 | 1.00 | 1.85 | 3.48 | yes |
| llama-cpp | qwen3-8b-mlx4 | mlx-affine-4bit | unsupported | – | – | – | – | – | – | – | – | – | – | – |
| llama-cpp | qwen3-8b-q4 | Q4_K_M | measured | 60 | 0.82 | 0.97 | 0.88 | 1.00 | 0 | 0.48 | 1.00 | 3.51 | 5.48 | yes |
| llama-cpp | smollm3-3b-mlx4 | mlx-affine-4bit | unsupported | – | – | – | – | – | – | – | – | – | – | – |
| llama-cpp | smollm3-3b-q4 | Q4_K_M | measured | 60 | 0.77 | 0.84 | 0.31 | 1.00 | 0 | 0.48 | 0.33 | 1.67 | 3.36 | yes |
| mlx-lm | llama3.1-8b-mlx4 | mlx-affine-4bit | gated | – | – | – | – | – | – | – | – | – | – | – |
| mlx-lm | llama3.1-8b-q4 | Q4_K_M | unsupported | – | – | – | – | – | – | – | – | – | – | – |
| mlx-lm | qwen3-4b-mlx4 | mlx-affine-4bit | measured | 60 | 0.86 | 0.97 | 0.88 | 1.00 | 0 | 0.62 | 1.00 | 2.18 | 3.25 | yes |
| mlx-lm | qwen3-4b-q4 | Q4_K_M | unsupported | – | – | – | – | – | – | – | – | – | – | – |
| mlx-lm | qwen3-8b-mlx4 | mlx-affine-4bit | measured | 60 | 0.81 | 0.98 | 0.92 | 1.00 | 0 | 0.46 | 1.00 | 3.58 | 6.10 | yes |
| mlx-lm | qwen3-8b-q4 | Q4_K_M | unsupported | – | – | – | – | – | – | – | – | – | – | – |
| mlx-lm | smollm3-3b-mlx4 | mlx-affine-4bit | measured | 60 | 0.74 | 0.80 | 0.15 | 1.00 | 0 | 0.42 | 0.67 | 1.32 | 4.03 | yes |
| mlx-lm | smollm3-3b-q4 | Q4_K_M | unsupported | – | – | – | – | – | – | – | – | – | – | – |
| ollama | llama3.1-8b-mlx4 | mlx-affine-4bit | unsupported | – | – | – | – | – | – | – | – | – | – | – |
| ollama | llama3.1-8b-q4 | Q4_K_M | gated | – | – | – | – | – | – | – | – | – | – | – |
| ollama | qwen3-4b-mlx4 | mlx-affine-4bit | unsupported | – | – | – | – | – | – | – | – | – | – | – |
| ollama | qwen3-4b-q4 | Q4_K_M | measured | 60 | 0.85 | 0.97 | 0.77 | 1.00 | 0 | 0.59 | 1.00 | 1.76 | 4.23 | yes |
| ollama | qwen3-8b-mlx4 | mlx-affine-4bit | unsupported | – | – | – | – | – | – | – | – | – | – | – |
| ollama | qwen3-8b-q4 | Q4_K_M | measured | 60 | 0.81 | 0.97 | 0.88 | 1.00 | 0 | 0.44 | 1.00 | 3.18 | 6.23 | yes |
| ollama | smollm3-3b-mlx4 | mlx-affine-4bit | unsupported | – | – | – | – | – | – | – | – | – | – | – |
| ollama | smollm3-3b-q4 | Q4_K_M | unsupported | – | – | – | – | – | – | – | – | – | 0.00 | yes |

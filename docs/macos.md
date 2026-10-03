# macOS / Apple Silicon guide

## What runs where

All Mac backends run **natively** (arm64, Metal). None runs in Docker: a Linux container on
macOS has no Metal access, so it would measure CPU inference inside a VM.

| Backend | Install | Pinned version | Weights format |
|---|---|---|---|
| llama.cpp `llama-server` (Metal) | `scripts/install_llama_cpp.sh`: official release tarball, sha256-verified, into `.tools/` | b11246 (139997d8e) | GGUF |
| MLX-LM `mlx_lm.server` | `scripts/install_mlx.sh`: isolated venv `.venvs/mlx` from `configs/backends/mlx-requirements.txt` | mlx-lm 0.31.3, mlx 0.32.3 | MLX safetensors |
| Ollama | existing Homebrew install, used as-is (see below) | whatever `ollama --version` reports; recorded per run | imports the *same* GGUF file |

## Rosetta trap (observed on this machine)

The terminal or agent shell can itself run under Rosetta (`sysctl.proc_translated = 1`,
`uname -m` = `x86_64` on an M3 Pro). A universal binary started from such a shell runs as
x86_64: no NEON, and no Metal-optimized path in some builds. The harness therefore:

- launches every Mac backend with `arch -arm64`;
- shows `WARNING: this shell runs under Rosetta` in `llm-sheet doctor`;
- records `lipo -archs` of each backend binary (all arm64-only here) and `shell_translated_rosetta` in every manifest.

## Before measuring

1. `llm-sheet doctor`: check the memory pressure level (must be 1 = normal), swap, and available memory.
2. Quit memory-heavy apps. Docker Desktop's VM, browsers and Electron apps are the usual culprits.
3. Plug in AC power and disable Low Power Mode (both are recorded).
4. Keep the lid open and avoid thermal throttling from other loads (not measured directly; macOS
   exposes no unprivileged thermal telemetry. `powermetrics` needs sudo and is not used).

Runs made under pressure are still saved, but they are flagged (`memory-pressure flag = yes`) and
must not be used as headline results.

## Unified memory, stated precisely

- Metal reports a recommended max working set of **13,639 MiB** on this 18 GiB machine
  (`llama-server` device log). GPU-resident weights + KV + compute buffers must fit under it, and
  everything else on the system shares the remaining RAM.
- llama.cpp memory-maps GGUF weights and wraps them as Metal buffers without copying. They count
  toward RSS and the AGX "in use" figure, but not toward `phys_footprint`.
- MLX loads weights into its own allocator. Its peak is read in-process (`mlx_peak_gib`).
- Swap on macOS is dynamic. `swap_in_bytes_delta` > 0 during a run means pages were read back from
  disk, and timing is then not representative.

## Ollama notes

- The harness starts a **private** `ollama serve` on a random localhost port, with
  `OLLAMA_MODELS=models/ollama`. The desktop Ollama app and its models are not touched.
- Import copies the GGUF into Ollama's blob store, which doubles disk usage for that model. The
  blob digest is compared with the GGUF's sha256 (`blob_matches_artifact_sha256`).
- Installed here: 0.9.0 (June 2025). Current upstream stable is 0.34.4 (2026-09-23). The harness does
  not upgrade it silently. To upgrade: `brew upgrade ollama`, then re-run the Ollama cells (the version is
  part of every manifest).
- Ollama is a model manager and server built on ggml/llama.cpp code. Its results are reported as
  "Ollama (version, runner as logged)", never as an independent kernel implementation.

## Updating pinned runtimes

- llama.cpp: pick a new build, add its asset sha256 to `scripts/install_llama_cpp.sh`, bump
  `PINNED_BUILD` in `backends/llama_cpp.py`, and re-check flags against `llama-server --help`
  (b11246 changed defaults: `--fit on`, `--cache-ram 8192`, `-np auto`, `--kv-unified` when auto).
- mlx-lm: edit `configs/backends/mlx-requirements.in`, recompile (command in the file header), then
  run `scripts/install_mlx.sh`. Check the server's request fields (batching, `seed`, `chat_template_kwargs`).

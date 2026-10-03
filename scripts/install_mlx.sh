#!/usr/bin/env bash
# Create the isolated, pinned MLX-LM venv used by the mlx-lm backend (native arm64 Python).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV="${LLM_SHEET_MLX_VENV:-$ROOT/.venvs/mlx}"
PY="${MLX_PYTHON:-/opt/homebrew/opt/python@3.12/bin/python3.12}"
if [[ "$(uname -s)" != "Darwin" ]]; then echo "MLX backend is macOS/Apple Silicon only" >&2; exit 2; fi
file "$PY" | grep -q arm64 || { echo "$PY is not an arm64 binary" >&2; exit 2; }
uv venv --python "$PY" "$VENV"
uv pip install --python "$VENV/bin/python" -r "$ROOT/configs/backends/mlx-requirements.txt"
arch -arm64 "$VENV/bin/python" -c "import mlx.core as mx, mlx_lm, platform; print('mlx', mx.__version__, 'mlx-lm', mlx_lm.__version__, platform.machine(), mx.default_device())"

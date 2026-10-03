"""Run ``mlx_lm.server`` unchanged and sample MLX allocator metrics from inside the process.

MLX's allocator counters (active/peak/cache memory) are only readable in-process.
This wrapper starts a daemon thread that appends them to a JSONL file every
``MLX_MEM_INTERVAL_S`` seconds, then hands control to ``mlx_lm.server.main()``,
so serving behavior is unchanged. Runs inside the isolated MLX venv, and uses only stdlib + mlx.

    python mlx_server_wrapper.py --mem-log path.jsonl -- <mlx_lm.server args>
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time


def _sampler(path: str, interval: float) -> None:
    import mlx.core as mx

    get_active = getattr(mx, "get_active_memory", None) or mx.metal.get_active_memory
    get_peak = getattr(mx, "get_peak_memory", None) or mx.metal.get_peak_memory
    get_cache = getattr(mx, "get_cache_memory", None) or mx.metal.get_cache_memory
    with open(path, "a", buffering=1) as f:
        try:
            info = mx.device_info() if hasattr(mx, "device_info") else mx.metal.device_info()
            f.write(json.dumps({"device_info": {k: (v if isinstance(v, (int, float, str)) else str(v))
                                                for k, v in info.items()}}) + "\n")
        except Exception as e:
            f.write(json.dumps({"device_info_error": repr(e)}) + "\n")
        while True:
            try:
                f.write(json.dumps({
                    "t_wall": time.time(),
                    "t_ns": time.perf_counter_ns(),
                    "mlx_active": int(get_active()),
                    "mlx_peak": int(get_peak()),
                    "mlx_cache": int(get_cache()),
                }) + "\n")
            except Exception as e:
                f.write(json.dumps({"error": repr(e)}) + "\n")
            time.sleep(interval)


def main() -> None:
    argv = sys.argv[1:]
    mem_log = None
    if "--mem-log" in argv:
        i = argv.index("--mem-log")
        mem_log = argv[i + 1]
        argv = argv[:i] + argv[i + 2 :]
    if argv and argv[0] == "--":
        argv = argv[1:]
    if mem_log:
        interval = float(os.environ.get("MLX_MEM_INTERVAL_S", "0.5"))
        threading.Thread(target=_sampler, args=(mem_log, interval), daemon=True).start()
    from mlx_lm import server

    sys.argv = ["mlx_lm.server", *argv]
    server.main()


if __name__ == "__main__":
    main()

"""Backend adapter registry."""

from __future__ import annotations

from .base import Availability, Backend, BackendError, LaunchSpec, ServerHandle


def get_backend(name: str) -> Backend:
    from .llama_cpp import LlamaCppBackend
    from .mlx_lm import MlxLmBackend
    from .mock import MockBackend
    from .ollama import OllamaBackend

    factories = {
        "llama-cpp": lambda: LlamaCppBackend("metal"),
        "mlx-lm": MlxLmBackend,
        "ollama": OllamaBackend,
        "mock": MockBackend,
    }
    if name not in factories:
        raise KeyError(f"unknown backend {name!r}; known: {sorted(factories)}")
    return factories[name]()


BACKEND_NAMES = ["llama-cpp", "mlx-lm", "ollama", "mock"]

__all__ = ["BACKEND_NAMES", "Availability", "Backend", "BackendError", "LaunchSpec", "ServerHandle", "get_backend"]

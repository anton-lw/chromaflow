# src/chromaflow/config.py
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Literal

from .exceptions import BackendConfigurationError

Backend = Literal["numpy", "jax", "numba"]

_CURRENT_BACKEND: Backend = "numpy"


def set_backend(backend: Backend) -> None:
    """
    Sets the global computation backend for ChromaFlow.

    Args:
        backend: The name of the backend to use ('numpy', 'jax', or 'numba').
    """
    global _CURRENT_BACKEND
    if backend not in ("numpy", "jax", "numba"):
        raise BackendConfigurationError(f"Unknown backend: '{backend}'")
    _CURRENT_BACKEND = backend


def get_backend() -> Backend:
    """Returns the name of the currently active backend."""
    return _CURRENT_BACKEND


@contextmanager
def backend(name: Backend) -> Iterator[None]:
    """
    A context manager to temporarily switch the backend.

    Example:
        >>> with chromaflow.config.backend('jax'):
        ...     # Code inside this block will use the JAX backend
        ...     color.to('lab-d65')
    """
    original_backend = get_backend()
    try:
        set_backend(name)
        yield
    finally:
        set_backend(original_backend)

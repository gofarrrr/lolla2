from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator


@contextmanager
def traced_span(name: str) -> Iterator[None]:
    """Placeholder tracing span context manager."""
    try:
        yield
    finally:
        pass

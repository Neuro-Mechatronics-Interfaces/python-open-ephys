from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional, Tuple
import numpy as np
import threading


class BaseClient(ABC):
    """Minimal lifecycle and access surface shared by live clients."""

    def __init__(self) -> None:
        self.ready_event = threading.Event()
        self.streaming = False

    @abstractmethod
    def start(self) -> None:
        ...

    @abstractmethod
    def stop(self, timeout: Optional[float] = None) -> None:
        ...

    @abstractmethod
    def get_latest(self, n: int) -> Tuple[np.ndarray, np.ndarray]:
        """Return latest n samples as (channels x samples, timestamps)."""
        ...

    def get_latest_window(self, window_ms: int) -> np.ndarray:
        """Return the latest ``window_ms`` milliseconds of data.

        Convenience wrapper around :meth:`get_latest` that converts a
        duration in milliseconds to a sample count using ``self.fs``.

        Parameters
        ----------
        window_ms : int
            Duration of the desired window in milliseconds.

        Returns
        -------
        np.ndarray, shape (channels, samples)
        """
        if not hasattr(self, "fs") or self.fs is None:
            raise RuntimeError(
                "Sample rate (self.fs) not set; cannot convert ms to samples."
            )
        n = int(round(self.fs * window_ms / 1000.0))
        return self.get_latest(n)[0]

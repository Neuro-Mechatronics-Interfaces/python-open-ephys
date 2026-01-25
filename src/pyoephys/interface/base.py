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

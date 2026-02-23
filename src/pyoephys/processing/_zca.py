"""
pyoephys.processing._zca
=========================
Online ZCA (Zero-phase Component Analysis) whitening for multi-channel EMG.

Ported and extended from MindRove-EMG (nml/processing/zca.py).

ZCA whitening decorrelates channels while preserving the original signal
orientation as much as possible, improving classifier performance on
high-density EMG by removing shared noise components.

Pipeline
--------
1. Accumulate ``buffer_duration_sec`` seconds of resting EMG  (warmup).
2. Compute channel-wise mean (μ), global std (σ), and ZCA matrix W:

       C = corrcoef(X_normalised)
       D, E = eig(C)
       W = E · diag(1 / sqrt(D + ε)) · E'

3. For subsequent windows apply:

       Z = ((X – μ) / σ) @ W

Usage
-----
>>> from pyoephys.processing import ZcaParams, ZcaHandler
>>> params = ZcaParams(num_channels=8, buffer_duration_sec=5.0)
>>> zca = ZcaHandler(fs=2000.0, params=params)
>>> zca.update_buffer(rest_data)    # (8, N) – call until zca.trained
>>> whitened = zca.apply(window)    # apply to every subsequent window
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# Parameter container
# ---------------------------------------------------------------------------

class ZcaParams:
    """Configuration for :class:`ZcaHandler`.

    Parameters
    ----------
    num_channels : int
        Number of EMG channels to whiten.
    buffer_duration_sec : float
        Duration of the resting-EMG buffer used to compute the whitening
        transform (seconds).
    tikhonov_epsilon : float
        Regularisation constant added to eigenvalues before inversion.
        Larger values → less aggressive whitening, more numerical stability.
    enable : bool
        Whether ZCA whitening is active.  When ``False``, :meth:`ZcaHandler.apply`
        is a no-op.
    """

    def __init__(
        self,
        num_channels: int = 8,
        buffer_duration_sec: float = 5.0,
        tikhonov_epsilon: float = 1.5,
        enable: bool = True,
    ) -> None:
        if num_channels <= 0:
            raise ValueError("ZcaParams.num_channels must be > 0.")
        if buffer_duration_sec <= 0:
            raise ValueError("ZcaParams.buffer_duration_sec must be > 0.")
        if tikhonov_epsilon <= 0:
            raise ValueError("ZcaParams.tikhonov_epsilon must be > 0.")

        self.num_channels = int(num_channels)
        self.buffer_duration_sec = float(buffer_duration_sec)
        self.tikhonov_epsilon = float(tikhonov_epsilon)
        self.enable = bool(enable)

    @classmethod
    def from_dict(cls, d: dict) -> "ZcaParams":
        """Construct from a plain dict (e.g. loaded from JSON config)."""
        return cls(
            num_channels=d.get("num_channels", 8),
            buffer_duration_sec=d.get("buffer_duration_sec", 5.0),
            tikhonov_epsilon=d.get("tikhonov_epsilon", 1.5),
            enable=d.get("enable", True),
        )

    def to_dict(self) -> dict:
        return {
            "num_channels": self.num_channels,
            "buffer_duration_sec": self.buffer_duration_sec,
            "tikhonov_epsilon": self.tikhonov_epsilon,
            "enable": self.enable,
        }

    def copy(self) -> "ZcaParams":
        return ZcaParams(
            num_channels=self.num_channels,
            buffer_duration_sec=self.buffer_duration_sec,
            tikhonov_epsilon=self.tikhonov_epsilon,
            enable=self.enable,
        )

    def __repr__(self) -> str:
        return (
            f"ZcaParams(num_channels={self.num_channels}, "
            f"buffer_duration_sec={self.buffer_duration_sec}, "
            f"tikhonov_epsilon={self.tikhonov_epsilon}, "
            f"enable={self.enable})"
        )


# ---------------------------------------------------------------------------
# Handler
# ---------------------------------------------------------------------------

class ZcaHandler:
    """Online single-shot ZCA whitening for multi-channel EMG.

    Call :meth:`update_buffer` with resting EMG until :attr:`trained` is
    ``True``, then call :meth:`apply` on every subsequent window.

    Parameters
    ----------
    fs : float
        Sampling frequency in Hz.
    params : ZcaParams
        Configuration object.

    Examples
    --------
    Offline (whole recording):

    >>> zca = ZcaHandler(fs=2000.0, params=ZcaParams(num_channels=8))
    >>> zca.update_buffer(rest_emg)       # shape (8, N_rest)
    >>> assert zca.trained
    >>> whitened = zca.apply(task_emg)    # shape (8, N_task)

    Online (streaming, chunk by chunk):

    >>> for chunk in stream:
    ...     if not zca.trained:
    ...         zca.update_buffer(chunk)
    ...     else:
    ...         whitened = zca.apply(chunk)
    """

    def __init__(self, fs: float, params: ZcaParams) -> None:
        self._fs = float(fs)
        self._num_channels = params.num_channels
        self._num_buffer_samples = int(params.buffer_duration_sec * fs)
        self._eps = params.tikhonov_epsilon
        self._enabled = params.enable

        self._buffer = np.zeros(
            (self._num_channels, self._num_buffer_samples), dtype=np.float64
        )
        self._buffer_count = 0

        self._trained = False
        self._mu: NDArray[np.float64] | None = None
        self._sigma: float | None = None
        self._W: NDArray[np.float64] | None = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def trained(self) -> bool:
        """``True`` once the whitening matrix has been computed."""
        return self._trained

    @property
    def buffer_fill_fraction(self) -> float:
        """Fraction of the warmup buffer that has been filled (0 – 1.0)."""
        return self._buffer_count / self._num_buffer_samples

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update_buffer(self, data: NDArray[np.float64]) -> None:
        """Feed resting-EMG into the warmup buffer.

        When the buffer is full the whitening matrix is computed automatically.
        Has no effect once :attr:`trained` is ``True``.

        Parameters
        ----------
        data : NDArray, shape (C, N)
            Multi-channel EMG samples.
        """
        if self._trained or not self._enabled:
            return

        ns = data.shape[1]
        remaining = self._num_buffer_samples - self._buffer_count
        take = min(ns, remaining)

        self._buffer[:, self._buffer_count: self._buffer_count + take] = data[:, :take]
        self._buffer_count += take

        if self._buffer_count >= self._num_buffer_samples:
            self._compute_whitener()

    def apply(self, data: NDArray[np.float64]) -> NDArray[np.float64]:
        """Apply whitening to *data*.

        Returns *data* unchanged if not yet trained or if disabled.

        Parameters
        ----------
        data : NDArray, shape (C, N)

        Returns
        -------
        NDArray, shape (C, N)
            Whitened EMG (or the original if not yet trained / disabled).
        """
        if not self._trained or not self._enabled:
            return data

        X = data.T                              # (N, C)
        Xn = (X - self._mu) / self._sigma
        Z = Xn @ self._W

        return Z.T                              # (C, N)

    def process(self, data: NDArray[np.float64]) -> NDArray[np.float64]:
        """Alias for :meth:`apply` — compatible with the filter-chain interface."""
        if not self._trained:
            self.update_buffer(data)
            return data
        return self.apply(data)

    def enable(self) -> None:
        """Enable ZCA whitening."""
        self._enabled = True

    def disable(self) -> None:
        """Disable ZCA whitening (apply becomes a no-op)."""
        self._enabled = False

    def reset(
        self,
        new_num_channels: int | None = None,
        new_buffer_duration_sec: float | None = None,
    ) -> None:
        """Clear the trained state and restart the warmup buffer.

        Parameters
        ----------
        new_num_channels : int, optional
            Override channel count before reset.
        new_buffer_duration_sec : float, optional
            Override buffer duration before reset.
        """
        if new_num_channels is not None:
            self._num_channels = int(new_num_channels)
        if new_buffer_duration_sec is not None:
            self._num_buffer_samples = int(new_buffer_duration_sec * self._fs)

        self._trained = False
        self._buffer_count = 0
        self._buffer = np.zeros(
            (self._num_channels, self._num_buffer_samples), dtype=np.float64
        )
        self._mu = None
        self._sigma = None
        self._W = None

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _compute_whitener(self) -> None:
        """Compute μ, σ, and the ZCA matrix W from the warmup buffer."""
        X = self._buffer.T          # (N, C)

        mu = X.mean(axis=0)
        sigma = X.std()

        Xn = (X - mu) / sigma

        C = np.corrcoef(Xn, rowvar=False)
        # Use eigh (symmetric-matrix variant): guarantees real eigenvalues/vectors,
        # more numerically stable than eig for correlation matrices.
        D, E = np.linalg.eigh(C)

        D_inv_sqrt = np.diag(1.0 / np.sqrt(np.abs(D) + self._eps))
        W = E @ D_inv_sqrt @ E.T  # E is real-orthogonal from eigh, no .real needed

        self._mu = mu
        self._sigma = float(sigma)
        self._W = W
        self._trained = True

        del self._buffer            # free warmup memory

    def __repr__(self) -> str:
        status = "trained" if self._trained else f"{self.buffer_fill_fraction:.0%} filled"
        return (
            f"ZcaHandler(fs={self._fs}, n_channels={self._num_channels}, "
            f"status={status!r})"
        )

"""
pyoephys.processing._spatial
==============================
Spatial referencing for circular multi-electrode EMG arrays.

Ported and extended from MindRove-EMG (nml/processing/spatial.py).

Modes
-----
MONOPOLAR         Raw signal with no inter-electrode subtraction.
SINGLE_DIFFERENTIAL  Each channel minus its clockwise neighbour.
LAPLACIAN         Each channel minus the mean of its two nearest neighbours.
CAR               Common Average Reference (subtract global mean).
"""

from __future__ import annotations

from enum import IntEnum
import numpy as np
from numpy.typing import NDArray


class MontageMode(IntEnum):
    """Spatial-referencing mode for a circular electrode array."""

    MONOPOLAR = 0
    """No spatial subtraction — raw (HPF) signal."""

    SINGLE_DIFFERENTIAL = 1
    """Each channel minus its immediate clockwise neighbour."""

    LAPLACIAN = 2
    """Each channel minus the mean of its two nearest neighbours."""

    CAR = 3
    """Common Average Reference — subtract the mean of all channels."""


class SpatialReference:
    """Apply spatial referencing to multi-channel EMG.

    Designed for circular electrode arrays (e.g. 8-electrode ring cuffs or
    HD-EMG patches), but works for any channel count ≥ 2.

    Parameters
    ----------
    n_channels : int
        Number of EMG channels.  Must be ≥ 2.
    montage_mode : MontageMode | int
        One of :class:`MontageMode` (or its integer equivalent).

    Examples
    --------
    >>> sr = SpatialReference(8, MontageMode.LAPLACIAN)
    >>> emg_ref = sr.process(raw_emg)          # shape (8, N)

    >>> sr.set_mode(MontageMode.SINGLE_DIFFERENTIAL)
    >>> emg_sd = sr.process(raw_emg)
    """

    def __init__(
        self,
        n_channels: int = 8,
        montage_mode: MontageMode | int = MontageMode.MONOPOLAR,
    ) -> None:
        if n_channels < 2:
            raise ValueError("n_channels must be ≥ 2.")
        self.n_channels = n_channels
        self.montage_mode = MontageMode(montage_mode)
        self._adjacency: list[list[int]] = self._build_adjacency(self.montage_mode)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_mode(self, mode: MontageMode | int) -> None:
        """Change the montage mode and regenerate the adjacency table."""
        self.montage_mode = MontageMode(mode)
        self._adjacency = self._build_adjacency(self.montage_mode)

    def process(self, data: NDArray[np.float64]) -> NDArray[np.float64]:
        """Apply spatial referencing to *data*.

        Parameters
        ----------
        data : NDArray, shape (C, N)
            Raw (or pre-filtered) EMG.  ``C`` must equal ``self.n_channels``.

        Returns
        -------
        NDArray, shape (C, N)
            Spatially referenced EMG.
        """
        C, N = data.shape
        if C != self.n_channels:
            raise ValueError(
                f"data has {C} channels but SpatialReference was built for "
                f"{self.n_channels} channels."
            )

        if self.montage_mode == MontageMode.MONOPOLAR:
            return data.copy()

        if self.montage_mode == MontageMode.CAR:
            return data - data.mean(axis=0, keepdims=True)

        out = np.zeros_like(data)
        for ch, neighbours in enumerate(self._adjacency):
            neigh_mean = np.mean([data[n] for n in neighbours], axis=0)
            out[ch] = data[ch] - neigh_mean
        return out

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_adjacency(self, mode: MontageMode) -> list[list[int]]:
        C = self.n_channels
        if mode == MontageMode.SINGLE_DIFFERENTIAL:
            # Each channel's only neighbour is the next one (circular)
            return [[(ch + 1) % C] for ch in range(C)]

        if mode == MontageMode.LAPLACIAN:
            # Two nearest neighbours (circular)
            return [
                [(ch - 1) % C, (ch + 1) % C]
                for ch in range(C)
            ]

        # MONOPOLAR / CAR — no per-channel adjacency needed
        return [[] for _ in range(C)]

    def __repr__(self) -> str:
        return (
            f"SpatialReference(n_channels={self.n_channels}, "
            f"montage_mode={self.montage_mode.name})"
        )

from __future__ import annotations

import numpy as np
from scipy.signal import butter, iirnotch, sosfilt, sosfilt_zi, tf2sos

class RealtimeFilter:
    """
    Stateful realtime filter cascade for multi-channel streams.

    Order of stages (if enabled): BANDPASS -> NOTCH (one or more) -> LOWPASS

    Input/Output blocks are shaped (C, N): C = channels, N = samples.
    """

    def __init__(
        self,
        fs: float,
        n_channels: int,
        # bandpass
        bp_low: float = 20.0,
        bp_high: float = 498.0,
        bp_order: int = 4,
        enable_bandpass: bool = True,
        # notch
        notch_freqs=(60.0,),        # can be (60., 120., 180., ...) if desired
        notch_q: float = 30.0,
        enable_notch: bool = True,
        # optional post low-pass
        lp_cut: float | None = None,
        lp_order: int = 4,
        enable_lowpass: bool = False,
    ):
        self.fs = float(fs)
        self.C = int(n_channels)

        # config
        self.enable_bandpass = bool(enable_bandpass)
        self.bp_low = float(bp_low)
        self.bp_high = float(bp_high)
        self.bp_order = int(bp_order)

        self.enable_notch = bool(enable_notch)
        self.notch_freqs = tuple(float(f) for f in notch_freqs)
        self.notch_q = float(notch_q)

        self.enable_lowpass = bool(enable_lowpass)
        self.lp_cut = None if lp_cut is None else float(lp_cut)
        self.lp_order = int(lp_order)

        # designed filter sections and per-channel states
        self._stages = []          # list of dicts: {name, sos, zi} ; zi shape (n_sections, C, 2)
        self._design_filters()

    # ------------- public API -------------

    def reconfigure(self, **kwargs):
        """
        Update parameters (e.g., bp_low=30, notch_freqs=(60,120)) and redesign filters.
        Resets filter state (safer for realtime); set preserve_state=True if you really
        want to keep old states (not recommended when topology changes).
        """
        preserve_state = kwargs.pop("preserve_state", False)
        for k, v in kwargs.items():
            if not hasattr(self, k):
                raise AttributeError(f"Unknown config: {k}")
            setattr(self, k, v)
        self._design_filters(preserve_state=preserve_state)

    def reset(self):
        """Zero the internal filter state (zi) for all stages/channels."""
        for st in self._stages:
            st["zi"][:] = np.tile(sosfilt_zi(st["sos"])[:, None, :], (1, self.C, 1))

    def process(self, block: np.ndarray) -> np.ndarray:
        """
        Filter a block of shape (C, N). Returns filtered block of same shape.
        Maintains internal state across calls.
        """
        if block.ndim != 2 or block.shape[0] != self.C:
            raise ValueError(f"Expected block shape (C,N)=({self.C}, N); got {block.shape}")
        y = block.astype(np.float32, copy=True)

        # Apply each stage in sequence, maintaining per-channel zi
        for st in self._stages:
            sos = st["sos"]
            zi = st["zi"]  # (n_sections, C, 2)
            # process each channel with its own zi
            for ch in range(self.C):
                y[ch, :], zi[:, ch, :] = sosfilt(sos, y[ch, :], zi=zi[:, ch, :])
        return y

    # ------------- internals -------------

    def _design_filters(self, preserve_state: bool = False):
        """(Re)build SOS stages and initialize per-channel zi."""
        old_states = None
        if preserve_state and self._stages:
            # shallow capture of previous zi to try to keep continuity if topology identical
            old_states = [(st["name"], st["sos"].copy(), st["zi"].copy()) for st in self._stages]

        self._stages = []
        nyq = 0.5 * self.fs

        # helper to clamp and validate edges
        def _bounded(f):
            return max(1e-6, min(f, nyq * 0.999))

        # BANDPASS
        if self.enable_bandpass:
            lo = _bounded(self.bp_low)
            hi = _bounded(self.bp_high)
            if hi <= lo:
                hi = lo + 1.0
            sos_bp = butter(self.bp_order, [lo, hi], btype="band", fs=self.fs, output="sos")
            self._stages.append(self._mk_stage("bandpass", sos_bp))

        # NOTCH (one or more)
        if self.enable_notch and len(self.notch_freqs) > 0:
            for f0 in self.notch_freqs:
                f0b = _bounded(f0)
                # design second-order notch and convert to sos
                b, a = iirnotch(w0=f0b, Q=self.notch_q, fs=self.fs)  # returns (b, a)
                sos_n = tf2sos(b, a)
                self._stages.append(self._mk_stage(f"notch_{int(round(f0))}", sos_n))

        # LOWPASS (optional)
        if self.enable_lowpass and self.lp_cut is not None:
            cut = _bounded(self.lp_cut)
            sos_lp = butter(self.lp_order, cut, btype="low", fs=self.fs, output="sos")
            self._stages.append(self._mk_stage("lowpass", sos_lp))

        # initialize zi arrays (n_sections, C, 2)
        for st in self._stages:
            st["zi"] = np.tile(sosfilt_zi(st["sos"])[:, None, :], (1, self.C, 1)).astype(np.float32)

        # attempt to preserve state if topology identical
        if old_states is not None:
            for name, sos_old, zi_old in old_states:
                for st in self._stages:
                    if st["name"] == name and st["sos"].shape == sos_old.shape and np.allclose(st["sos"], sos_old):
                        st["zi"] = zi_old  # keep prior state

    def _mk_stage(self, name, sos):
        return {"name": name, "sos": np.asarray(sos, dtype=np.float64), "zi": None}

from __future__ import annotations

import time
import threading
import numpy as np
from typing import Optional, Tuple, Sequence

from pylsl import StreamInlet, resolve_byprop, resolve_streams, local_clock
from pyoephys.logging import get_logger

log = get_logger("interface.lsl")


class NotReadyError(RuntimeError):
    pass


# A generic LSL handler that subscribes to LSL streams and provides basic functionality
class OldLSLClient:
    """
    LSL timebase client (ring buffer) that pulls CHUNKS with timestamps and serves
    a rolling window on request.

    Choose the stream with exactly one of:
      - stream_name="CosineWave"
      - stream_type="EMG"
    """
    def __init__(
        self,
        stream_name=None,
        stream_type=None,
        channels=None,
        window_secs=5.0,
        fallback_fs=2000.0,
        auto_start=False,
        verbose=False,
    ):
        if (stream_name is None) == (stream_type is None):
            raise ValueError("Provide exactly one of stream_name or stream_type")

        if stream_name is not None:
            streams = resolve_byprop("name", stream_name, timeout=5)
        else:
            streams = resolve_byprop("type", stream_type, timeout=5)

        if not streams:
            sel = f"name='{stream_name}'" if stream_name else f"type='{stream_type}'"
            raise RuntimeError(f"No LSL stream found with {sel}")

        self.inlet = StreamInlet(streams[0])
        info = self.inlet.info()
        self.name = info.name()
        self.type = info.type()
        self.fs = float(info.nominal_srate() or 0.0)
        self.n_channels_total = int(info.channel_count())
        if self.n_channels_total < 1:
            raise RuntimeError("Stream reports zero channels.")

        if channels is None:
            self.channel_index = list(range(self.n_channels_total))
        else:
            self.channel_index = [int(c) for c in channels]
            for c in self.channel_index:
                if not (0 <= c < self.n_channels_total):
                    raise ValueError(f"Channel index {c} out of range [0,{self.n_channels_total-1}]")

        self.N_channels = len(self.channel_index)
        self.window_secs = float(window_secs)
        fs_for_alloc = self.fs if self.fs > 0 else float(fallback_fs)
        self.N_samples = int(max(1, round(fs_for_alloc * self.window_secs)))

        # ring buffers
        self.t = np.zeros(self.N_samples, dtype=np.float64)   # absolute LSL time (sec)
        self.y = np.zeros((self.N_channels, self.N_samples), dtype=np.float32)   # selected channel
        self.widx = 0
        self.count = 0
        self.lock = threading.Lock()
        self._stop = False
        self._thread = None
        self.streaming = False
        self.verbose = verbose

        print(f"[FastLSLClient] Connected to '{self.name}' ({self.fs} Hz, {self.N_channels} channels)")

        if auto_start:
            if self.verbose:
                print("Auto-starting LSL client...")
            self.start()

    # --- worker loop: chunked pulls into ring buffer ---
    def _worker(self):
        while not self._stop:
            data, ts = self.inlet.pull_chunk(timeout=0.03, max_samples=4096)
            if not data:
                continue
            arr = np.asarray(data, dtype=np.float32)      # shape: (n, n_channels_total)
            ts = np.asarray(ts, dtype=np.float64)         # shape: (n,)
            x = arr[:, self.channel_index]                # shape: (n, n_channels)
            n = x.shape[0]

            with self.lock:
                dst = self.widx % self.N_samples
                first = min(n, self.N_samples - dst)
                self.y[:, dst:dst + first] = x[:first].T
                self.t[dst:dst + first] = ts[:first]
                rem = n - first
                if rem > 0:
                    self.y[:, :rem] = x[first:].T  # NOTE the colon before rem
                    self.t[:rem] = ts[first:]
                self.widx = (self.widx + n) % self.N_samples
                self.count = min(self.count + n, self.N_samples)

    def start(self):
        if not self.streaming:
            self._stop = False
            self.streaming = True
            self._thread = threading.Thread(target=self._worker, daemon=True)
            self._thread.start()
            print("[FastLSLClient] Streaming started.")

    def stop(self):
        self._stop = True
        self.streaming = False
        if self._thread:
            self._thread.join()

    def latest(self):
        """
        Returns (t_rel, y) where:
          - t_rel: (M,) seconds, ends at 0 (most recent sample)
          - y:     (N_channels, M)
        """
        with self.lock:
            if self.count == 0 or np.all(self.t == 0):
                return None, None

            end = self.widx
            if self.count < self.N_samples:
                # take first 'count' samples along the TIME axis
                t = self.t[:self.count].copy()
                y = self.y[:, :self.count].copy()
            else:
                # stitch TIME axis: [end:]+[:end]
                t = np.hstack((self.t[end:], self.t[:end])).copy()
                y = np.hstack((self.y[:, end:], self.y[:, :end])).copy()

        # relative time so X fits [-window_secs, 0]
        t_last = t[-1]
        t_rel = t - t_last

        # window by time (mask along TIME axis)
        mask = t_rel >= -self.window_secs
        t_rel = t_rel[mask]
        y = y[:, mask]

        # avoid drawing with 0/1 points (e.g., right after (re)connect)
        if t_rel.size < 2:
            return None, None

        return t_rel, y

    def drain_new(self):
        """Return only the new samples since last call: (t_abs_new, y_new) with shapes (M,), (C,M)."""
        with self.lock:
            if self.count == 0 or np.all(self.t == 0):
                return None, None

            # init read index to the oldest available sample
            if not hasattr(self, "_ridx"):
                self._ridx = (self.widx - self.count) % self.N_samples

            n_new = (self.widx - self._ridx) % self.N_samples
            if n_new == 0:
                return None, None

            start = self._ridx
            end = (self._ridx + n_new) % self.N_samples
            if start < end:
                t = self.t[start:end].copy()
                y = self.y[:, start:end].copy()
            else:
                t = np.hstack((self.t[start:], self.t[:end])).copy()
                y = np.hstack((self.y[:, start:], self.y[:, :end])).copy()

            self._ridx = end
        return t, y

    def fs_estimate(self, n_last=2000):
        """
        Median 1/dt over the last n_last timestamps (robust to outliers).
        """
        with self.lock:
            if self.count < 5:
                return float("nan")
            end = self.widx
            if self.count < self.N_samples:
                t = self.t[:self.count].copy()
            else:
                t = np.hstack((self.t[end:], self.t[:end])).copy()

        t = t[-n_last:]
        if t.size < 2:
            return float("nan")
        dt = np.diff(t)
        dt = dt[dt > 0]
        if dt.size == 0:
            return float("nan")
        return 1.0 / np.median(dt)


class LSLClient:
    """
    Subscribe to an LSL stream and keep a thread-safe ring buffer.

    Parameters
    ----------
    stream_name : Optional[str]
        Exact LSL stream name to match. If None, matches by type only.
    stream_type : str
        LSL stream type to match (e.g., "EMG").
    timeout_s : float
        How long to wait for a matching stream on start().
    buffer_seconds : float
        Size of the internal ring buffer in seconds.
    pull_timeout : float
        Timeout per pull_chunk() in seconds.
    """

    def __init__(
        self,
        stream_name: Optional[str] = None,
        stream_type: str = "EMG",
        timeout_s: float = 5.0,
        buffer_seconds: float = 30.0,
        pull_timeout: float = 0.2,
        verbose: bool = False,
    ) -> None:
        self.stream_name = stream_name
        self.stream_type = stream_type
        self.timeout_s = float(timeout_s)
        self.buffer_seconds = float(buffer_seconds)
        self.pull_timeout = float(pull_timeout)
        self.verbose = bool(verbose)

        self._inlet: Optional[StreamInlet] = None
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self.ready_event = threading.Event()

        # buffer (C x N), timestamps (N)
        self._buf_y: Optional[np.ndarray] = None
        self._buf_t: Optional[np.ndarray] = None
        self._widx: int = 0
        self._count: int = 0
        self._lock = threading.Lock()

        # meta
        self.fs_hz: Optional[float] = None
        self.n_channels: Optional[int] = None
        self.channel_labels: Optional[Sequence[str]] = None

    # -------------- lifecycle ----------------

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return

        info = self._resolve_stream()
        self._inlet = StreamInlet(info, max_buflen=int(self.buffer_seconds) + 5)

        # meta
        self.fs_hz = float(info.nominal_srate())
        self.n_channels = int(info.channel_count())

        # allocate ring
        n = max(int(self.buffer_seconds * self.fs_hz), 1)
        self._buf_y = np.zeros((self.n_channels, n), dtype=np.float32)
        self._buf_t = np.zeros((n,), dtype=np.float64)
        self._widx = 0
        self._count = 0

        self._stop.clear()
        self._thread = threading.Thread(target=self._run_loop, name="LSLClient", daemon=True)
        self._thread.start()
        if self.verbose:
            log.info(
                "LSL client started: name=%s type=%s fs=%.2fHz channels=%d buffer=%ds",
                self.stream_name or "<any>",
                self.stream_type,
                self.fs_hz,
                self.n_channels,
                self.buffer_seconds,
            )

    def stop(self, timeout: Optional[float] = 2.0) -> None:
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=timeout)
        if self.verbose:
            log.info("LSL client stopped.")

    # -------------- data access ----------------

    def get_latest(self, n: int) -> Tuple[np.ndarray, np.ndarray]:
        """Return the latest `n` samples as (C×n, n_t). Raises if not ready."""
        if not self.ready_event.is_set():
            raise NotReadyError("LSLClient not ready; no data received yet.")
        with self._lock:
            assert self._buf_y is not None and self._buf_t is not None
            n = int(max(1, min(n, self._buf_t.size, self._count)))
            end = self._widx
            start = (end - n) % self._buf_t.size
            if start < end:
                y = self._buf_y[:, start:end]
                t = self._buf_t[start:end]
            else:
                y = np.hstack((self._buf_y[:, start:], self._buf_y[:, :end]))
                t = np.concatenate((self._buf_t[start:], self._buf_t[:end]))
        return y.copy(), t.copy()

    def get_window(self, seconds: float) -> Tuple[np.ndarray, np.ndarray]:
        """Return the last `seconds` of data."""
        if self.fs_hz is None:
            raise NotReadyError("LSLClient not ready; fs unknown.")
        n = int(max(1, seconds * self.fs_hz))
        return self.get_latest(n)

    # -------------- internals ----------------

    def _resolve_stream(self):
        """
        Resolve an LSL stream using pylsl.resolve_byprop/resolve_streams so it works
        across pylsl versions that don't export `resolve_stream`.
        """
        pred_bits = []
        if self.stream_name:
            pred_bits.append(f"name='{self.stream_name}'")
        if self.stream_type:
            pred_bits.append(f"type='{self.stream_type}'")
        pred_str = " and ".join(pred_bits) if pred_bits else "<any>"

        if self.verbose:
            log.info("Resolving LSL stream with: %s", pred_str)

        deadline = time.perf_counter() + self.timeout_s
        while time.perf_counter() < deadline:
            if self.stream_name and self.stream_type:
                # First resolve by name, then filter by type
                matches = resolve_byprop("name", self.stream_name, timeout=0.5)
                matches = [m for m in matches if m.type() == self.stream_type]
            elif self.stream_name:
                matches = resolve_byprop("name", self.stream_name, timeout=0.5)
            elif self.stream_type:
                matches = resolve_byprop("type", self.stream_type, timeout=0.5)
            else:
                matches = resolve_streams(0.5)  # returns all streams

            if matches:
                return matches[0]

        raise TimeoutError(f"LSL stream not found within {self.timeout_s:.1f}s: {pred_str}")

    def _run_loop(self) -> None:
        inlet = self._inlet
        assert inlet is not None

        # First pull to mark ready
        while not self._stop.is_set():
            chunk, ts = inlet.pull_chunk(timeout=self.pull_timeout)
            if chunk:
                self._append_chunk(np.asarray(chunk, dtype=np.float32).T, np.asarray(ts, dtype=np.float64))
                self.ready_event.set()
                break

        # Main loop
        while not self._stop.is_set():
            chunk, ts = inlet.pull_chunk(timeout=self.pull_timeout)
            if not chunk:
                continue
            y = np.asarray(chunk, dtype=np.float32).T  # (C×B)
            t = np.asarray(ts, dtype=np.float64)       # (B,)
            self._append_chunk(y, t)

    def _append_chunk(self, y: np.ndarray, t: np.ndarray) -> None:
        assert self._buf_y is not None and self._buf_t is not None
        C, B = y.shape
        assert C == self._buf_y.shape[0], f"Channel mismatch: got {C}, expected {self._buf_y.shape[0]}"
        N = self._buf_t.size

        with self._lock:
            end = (self._widx + B) % N
            if self._widx + B <= N:
                self._buf_y[:, self._widx : self._widx + B] = y
                self._buf_t[self._widx : self._widx + B] = t
            else:
                k1 = N - self._widx
                self._buf_y[:, self._widx:] = y[:, :k1]
                self._buf_t[self._widx:] = t[:k1]
                self._buf_y[:, :end] = y[:, k1:]
                self._buf_t[:end] = t[k1:]
            self._widx = end
            self._count = min(N, self._count + B)

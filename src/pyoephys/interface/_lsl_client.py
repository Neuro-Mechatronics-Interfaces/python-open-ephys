
import threading
import numpy as np
from pylsl import StreamInlet, resolve_byprop

#from pyoephys.processing import RealtimeEMGFilter  # or your custom _filters import


# A generic LSL handler that subscribes to LSL streams and provides basic functionality
class LSLClient:
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

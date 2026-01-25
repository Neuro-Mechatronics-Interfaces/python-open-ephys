import os
import time
import argparse
import threading
import numpy as np
from pylsl import StreamInfo, StreamOutlet, local_clock
from pyoephys.io import load_open_ephys_session, find_oebin_files, parse_numeric_args


class OEBinPlaybackClient:
    """
    Realtime playback of Open Ephys .oebin recordings with clean LSL timestamps.

    Publishes chunks as (N_samples, N_channels) with one timestamp per row.
    Guarantees strictly increasing timestamps and correct data orientation.
    """

    def __init__(
            self,
            oebin_path,
            channels=None,
            block_size=1024,
            stream_name="OEBinData",
            stream_type="EMG",
            units="uV",
            speed_factor=1.0,
            loopback=False,
            enable_lsl=True,
            use_recorded_timestamps=True,
            auto_start=False,
            verbose=False,
    ):
        self.verbose = bool(verbose)
        self.loopback = bool(loopback)
        self.block_size = int(block_size)
        self.enable_lsl = bool(enable_lsl)
        self.use_recorded_timestamps = bool(use_recorded_timestamps)
        self.speed_factor = float(speed_factor)

        # Resolve .oebin path (file or directory)
        if os.path.isdir(oebin_path):
            if self.verbose:
                print(f"Searching for OEBIN files in: {oebin_path}")
            oebin_files = find_oebin_files(oebin_path)
            if not oebin_files:
                raise FileNotFoundError("No .oebin files found in the specified directory.")
            oebin_path = oebin_files[0]
        if not os.path.isfile(oebin_path):
            raise FileNotFoundError(f"No file found: {oebin_path}")

        if self.verbose:
            print(f"|  Loading session from: {oebin_path}")

        sess = load_open_ephys_session(os.path.dirname(oebin_path), verbose=self.verbose)
        data = np.asarray(sess["amplifier_data"], dtype=np.float32)  # unknown orientation
        names = list(sess["channel_names"])
        fs_file = float(sess["sample_rate"])
        t_file = sess.get("t_amplifier", None)  # seconds, shape (N,) if present

        # --- Normalize orientation to (C, N) ---
        # If first dim << second dim AND equals len(names), we likely already have (C, N)
        # If first dim == N samples (>> channels), flip.
        if data.shape[0] == len(names) and data.shape[0] < data.shape[1]:
            data_cn = data  # (C, N)
        elif data.shape[1] == len(names) and data.shape[1] < data.shape[0]:
            data_cn = data.T  # (C, N)
            if self.verbose:
                print("[playback] Transposed data to (C, N).")
        else:
            # Heuristic: pick the smaller as channels
            if data.shape[0] <= data.shape[1]:
                data_cn = data
            else:
                data_cn = data.T
                if self.verbose:
                    print("[playback] Heuristic transpose to (C, N).")

        # Channel selection
        if channels is None:
            self.data = data_cn
            self.channel_names = names
        else:
            if isinstance(channels, int):
                channels = [channels]
            bad = [ch for ch in channels if ch < 0 or ch >= data_cn.shape[0]]
            if bad:
                raise ValueError(f"Channel index out of range: {bad}")
            self.data = data_cn[np.asarray(channels, dtype=int), :]
            self.channel_names = [names[ch] for ch in channels]

        self.n_channels, self.n_samples = self.data.shape
        self.sampling_rate = fs_file  # file nominal
        self.playback_rate = fs_file * self.speed_factor  # actual timing rate

        # timestamps from file if available
        self.t_file = np.asarray(t_file, dtype=np.float64) if t_file is not None else None
        if self.t_file is not None and self.t_file.ndim != 1:
            self.t_file = self.t_file.ravel()

        # local mirror (optional)
        self.buffer = np.zeros_like(self.data, dtype=np.float32)
        self.total_samples = 0
        self.current_index = 0

        # threading / state
        self.streaming = False
        self.thread = None
        self.lock = threading.Lock()
        self._lsl_start_time = None  # wall time anchor for sample 0

        # LSL outlets
        self.lsl_outlet = None
        self.marker_outlet = None
        self.stream_name = stream_name
        self.stream_type = stream_type
        self.units = units
        if self.enable_lsl:
            self._initialize_lsl_stream()
            self._initialize_marker_stream()

        if auto_start:
            if self.verbose:
                print("Auto-starting playback client...")
            self.start()

    # ---------- LSL init ----------
    def _initialize_lsl_stream(self):
        info = StreamInfo(
            name=self.stream_name,
            type=self.stream_type,
            channel_count=self.n_channels,
            nominal_srate=self.playback_rate,  # match actual playback speed
            channel_format="float32",
            source_id="OEBinPlaybackClient",
        )
        info.desc().append_child_value("file_sample_rate", str(self.sampling_rate))
        info.desc().append_child_value("playback_rate", str(self.playback_rate))
        info.desc().append_child_value("created_at", time.strftime("%Y-%m-%d %H:%M:%S"))
        info.desc().append_child_value("manufacturer", "Open Ephys")

        chns = info.desc().append_child("channels")
        for ch_name in self.channel_names:
            ch = chns.append_child("channel")
            ch.append_child_value("label", str(ch_name))
            ch.append_child_value("unit", self.units)

        self.lsl_outlet = StreamOutlet(info, chunk_size=self.block_size, max_buffered=360)

    def _initialize_marker_stream(self):
        info = StreamInfo(
            name=f"{self.stream_name}Markers",
            type="Markers",
            channel_count=1,
            nominal_srate=0.0,
            channel_format="string",
            source_id="OEBinPlaybackClientMarkers",
        )
        self.marker_outlet = StreamOutlet(info)

    # ---------- control ----------
    def start(self):
        if self.streaming:
            return
        self.streaming = True
        self._lsl_start_time = local_clock()  # align file sample 0 to "now"
        self.thread = threading.Thread(target=self._stream_loop, daemon=True)
        self.thread.start()
        #if self.verbose:
        #    print(f"Streaming {self.n_samples} samples @ {self.playback_rate:.2f} Hz, "
        #          f"block={self.block_size}, channels={self.n_channels}")
        print(f"Streaming {self.n_samples} samples at {self.sampling_rate} Hz,"
              f"block size={self.block_size}, speed factor={self.speed_factor:.2f}x")
        print(f"Stream name: {self.stream_name}, type: {self.stream_type}, "
              f"channels: {self.n_channels}, units: {self.units}")

    def stop(self):
        if not self.streaming:
            return
        self.streaming = False
        if self.thread:
            self.thread.join()
            self.thread = None
        if self.verbose:
            print("Streaming stopped.")

    def reset(self):
        with self.lock:
            self.total_samples = 0
            self.current_index = 0
            self.buffer.fill(0)
            self._lsl_start_time = local_clock()

    def is_done(self):
        return self.current_index >= self.n_samples

    # ---------- main loop ----------
    def _stream_loop(self):
        start_mono = time.monotonic()
        dt_nom = 1.0 / max(1e-9, self.sampling_rate * self.speed_factor)

        # anchor for file timestamps
        file_t0 = 0.0
        if self.use_recorded_timestamps and self.t_file is not None and self.t_file.size >= self.n_samples:
            file_t0 = float(self.t_file[0])

        while self.streaming:
            # handle end-of-file first, so loopback doesn't fall through to pacing
            if self.current_index >= self.n_samples:
                if self.loopback:
                    if self.marker_outlet:
                        self.marker_outlet.push_sample(["LoopReset"])
                    with self.lock:
                        self.current_index = 0
                        self.total_samples = 0
                        self.buffer.fill(0)
                        self._lsl_start_time = local_clock()
                    start_mono = time.monotonic()
                    if self.t_file is not None:
                        file_t0 = float(self.t_file[0])
                    continue  # <-- critical: skip sleeping on old t_rel
                else:
                    break

            # ---- build next chunk ----
            with self.lock:
                end = min(self.current_index + self.block_size, self.n_samples)
                chunk = self.data[:, self.current_index:end]  # (C, N)
                n = chunk.shape[1]

                # mirror (optional)
                if self.total_samples + n > self.buffer.shape[1]:
                    extra = self.total_samples + n - self.buffer.shape[1]
                    self.buffer = np.pad(self.buffer, ((0, 0), (0, extra)), mode="constant")
                self.buffer[:, self.total_samples:self.total_samples + n] = chunk
                self.total_samples += n

                # LSL timestamps for this chunk (length n), relative to playback start
                if self.use_recorded_timestamps and self.t_file is not None:
                    t_rel = (self.t_file[self.current_index:end] - file_t0) / self.speed_factor
                    # enforce monotonicity; fallback to synthetic if needed
                    if np.any(np.diff(t_rel) <= 0):
                        idx = np.arange(self.current_index, end, dtype=np.float64)
                        t_rel = (idx - self.current_index) * dt_nom + (self.current_index * dt_nom)
                else:
                    idx = np.arange(self.current_index, end, dtype=np.float64)
                    t_rel = (idx - self.current_index) * dt_nom + (self.current_index * dt_nom)

                ts = (self._lsl_start_time + t_rel).tolist()

                if self.enable_lsl and self.lsl_outlet is not None:
                    self.lsl_outlet.push_chunk(chunk.T.tolist(), ts)  # (N, C) + per-sample ts

                self.current_index = end

            # ---- pacing (use sample count, not file ts) ----
            played_seconds = (self.current_index - 1) * dt_nom
            target = start_mono + played_seconds
            sleep_time = target - time.monotonic()
            if sleep_time > 0.0:
                time.sleep(float(sleep_time))

    # ---------- convenience ----------
    def close(self):
        self.stop()
        if self.verbose:
            print("Client closed.")

    def get_latest_window(self, window_ms=500):
        samples_per_window = int(max(1, round(window_ms / 1000.0 * self.sampling_rate)))
        with self.lock:
            start_index = max(0, self.total_samples - samples_per_window)
            end_index = self.total_samples
            return self.buffer[:, start_index:end_index]


def playback_cli():
    ap = argparse.ArgumentParser("oe-lsl-playback")
    ap.add_argument("oebin_path", type=str, help="Path to .oebin file or directory")
    ap.add_argument("--channels", nargs="+", default=None, help="e.g. 0 1 2 or 0:64 or all")
    ap.add_argument("--name", default="OEBinData")
    ap.add_argument("--type", default="EMG")
    ap.add_argument("--block", type=int, default=1024)
    ap.add_argument("--speed", type=float, default=1.0, help="Playback speed factor")
    ap.add_argument("--synthetic-ts", action="store_true", help="Ignore file timestamps")
    ap.add_argument("--loop", action="store_true")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    ch = parse_numeric_args(args.channels)
    client = OEBinPlaybackClient(
        oebin_path=args.oebin_path,
        channels=ch if ch != [0] else None,  # allow empty for all
        block_size=args.block,
        stream_name=args.name,
        stream_type=args.type,
        speed_factor=args.speed,
        loopback=args.loop,
        enable_lsl=True,
        use_recorded_timestamps=not args.synthetic_ts,
        verbose=args.verbose,
    )
    try:
        while True:
            time.sleep(0.25)
    except KeyboardInterrupt:
        pass
    finally:
        client.close()
        if args.verbose:
            print("Playback client stopped.")


# Run as CLI
if __name__ == "__main__":
    playback_cli()

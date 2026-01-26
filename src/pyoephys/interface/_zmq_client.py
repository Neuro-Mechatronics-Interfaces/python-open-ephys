from __future__ import annotations

import time
import json
import uuid
import numpy as np
import zmq
from typing import Iterable, Optional, List, Tuple
from pyoephys.logging import get_logger
from collections import deque
import threading
from ._gui_client import GUIClient
from ._gui_events import Event, Spike

log = get_logger("ZMQClient")


def _norm_name(s: str) -> str:
    return "".join(str(s).split()).upper()


def _parse_int(v) -> Optional[int]:
    if v is None:
        return None
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, str):
        return int(v) if v.isdigit() else None
    try:
        return int(v)
    except Exception:
        return None


class NotReadyError(RuntimeError):
    """Raised when getters are called before required channels are ready."""
    pass


def _addr(host: str, endpoint: str | int) -> str:
    """Return a valid ZMQ endpoint. If endpoint already has '://', return as-is.
       Otherwise, treat it as a port and build 'tcp://{host}:{port}'."""
    ep = str(endpoint)
    if "://" in ep:
        return ep

    # if host already has '://', don't prepend another 'tcp://'
    if "://" in host:
        return f"{host}:{int(ep)}"

    return f"tcp://{host}:{int(ep)}"


class ZMQClient:
    """
    Open Ephys–compatible ZMQ client with per-channel ring buffers (deque-based).

    """

    def __init__(self, host_ip: str = "127.0.0.1", data_port: str = "5556", heartbeat_port: Optional[str] = None,
                 buffer_seconds: float = 30.0, expected_channel_count: Optional[int] = None, expected_channel_names: Optional[Iterable[str]] = None,
                 required_fraction: float = 1.0, max_channels: int = 256, auto_start: bool = False, set_index_looping: bool = True,
                 align_to_header_index: bool = False, fill_value: float = np.nan, verbose: bool = False):

        # config
        self.host_ip = str(host_ip)
        self.data_port = str(data_port)
        self.hb_endpoint = str(int(data_port) + 1) if heartbeat_port is None else str(heartbeat_port)
        self.buffer_seconds = float(buffer_seconds)
        self.expected_count = int(expected_channel_count) if expected_channel_count else None
        self.expected_names = list(expected_channel_names) if expected_channel_names else None
        self.required_fraction = float(required_fraction)
        self.verbose = bool(verbose)
        self.max_channels = int(max_channels)
        self.name = "ZMQClient"
        self.type = "ZMQClient"
        self.expected_names = list(expected_channel_names) if expected_channel_names else None
        self.expected_names_norm = [_norm_name(n) for n in self.expected_names] if self.expected_names else None
        self.seen_names_norm = set()  # normalized names seen so far

        # zmq
        self._ctx = zmq.Context.instance()
        self._poller = zmq.Poller()
        self._data_sock: Optional[zmq.Socket] = None
        self._hb_sock: Optional[zmq.Socket] = None

        # state/threading
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.ready_event = threading.Event()
        self.channels_ready_event = threading.Event()
        self._lock = threading.Lock()

        # stream / buffers
        self.fs: float = 2000.0  # default, updated from headers
        self._deque_len = self._target_deque_len(self.fs)
        self.buffers: List[deque] = [deque(maxlen=self._deque_len) for _ in range(self.max_channels)]
        self._name_by_index: dict[int, str] = {}
        self.seen_nums: set[int] = set()
        self.seen_names: set[str] = set()
        self._ref_clock_ch: Optional[int] = None
        self.total_samples_written: int = 0
        self.channel_index: Optional[List[int]] = None  # selection
        self.N_channels: int = 0
        self.N_samples = self._deque_len  # Support older code
        self._drain_last_total = 0  # for drain_new()

        # Explicit sample index tracking
        self.sample_index = 0
        self.global_sample_index = 0  # Total across all channels
        self._last_index_log = 0.0
        self.index_log_interval_s = 0.25
        self._last_header_index = {}
        self.loop_global_index = False
        self.loop_cycle = 0
        self.loop_sample_index = 0
        self._index_offset = 0
        self._last_ref_s0 = None
        self._last_ref_end = None
        self.set_index_looping(enabled=set_index_looping)

        self.align_to_header_index = bool(align_to_header_index)
        self.fill_value = np.float32(fill_value)
        if self.align_to_header_index:
            self._ring = np.full((self.max_channels, self._deque_len), self.fill_value, dtype=np.float32)
            self._valid = np.zeros((self.max_channels, self._deque_len), dtype=np.bool_)

        # heartbeat tracking
        self._last_hb_send = 0.0
        self._waiting_hb_reply = False

        # https connection (if enabled)
        self.gui = GUIClient(host=self.host_ip)

        if auto_start:
            self.start()
            # Optional: let first frame arrive (non-blocking feel but helpful for UI)
            self.ready_event.wait(timeout=5.0)

    # --- internal ----
    def _setup(self) -> None:
        self._teardown()
        # data SUB socket: subscribe-all like the working client
        data_addr = _addr(self.host_ip, self.data_port)
        self._data_sock = self._ctx.socket(zmq.SUB)
        self._data_sock.connect(data_addr)
        self._data_sock.setsockopt(zmq.SUBSCRIBE, b"")
        self._data_sock.setsockopt(zmq.RCVTIMEO, 1000)
        self._poller.register(self._data_sock, zmq.POLLIN)

        # optional heartbeat REQ socket
        if self.hb_endpoint:
            hb_addr = _addr(self.host_ip, self.hb_endpoint)
            self._hb_sock = self._ctx.socket(zmq.REQ)
            self._hb_sock.connect(hb_addr)
            self._hb_sock.setsockopt(zmq.RCVTIMEO, 2000)
            self._poller.register(self._hb_sock, zmq.POLLIN)

        self._last_hb_send = 0.0
        self._waiting_hb_reply = False

    def _teardown(self) -> None:
        try:
            if self._data_sock:
                self._poller.unregister(self._data_sock)
        except Exception:
            pass
        try:
            if self._hb_sock:
                self._poller.unregister(self._hb_sock)
        except Exception:
            pass
        try:
            if self._data_sock:
                self._data_sock.close(0)
        finally:
            self._data_sock = None
        try:
            if self._hb_sock:
                self._hb_sock.close(0)
        finally:
            self._hb_sock = None

    def _target_deque_len(self, fs: float) -> int:
        return max(1, int(round(fs * self.buffer_seconds)))

    def _rebuild_deques_if_needed(self, new_fs: float) -> None:
        new_len = self._target_deque_len(new_fs)
        if new_len == self._deque_len:
            return
        # rebuild each deque to the new maxlen, keeping most recent samples
        for ch in range(self.max_channels):
            old = self.buffers[ch]
            if len(old) == 0:
                self.buffers[ch] = deque(maxlen=new_len)
                continue
            take = min(len(old), new_len)
            recent = list(old)[-take:]
            self.buffers[ch] = deque(recent, maxlen=new_len)
        self._deque_len = new_len
        self.N_samples = new_len

        if getattr(self, "align_to_header_index", False):
            self._ring = np.full((self.max_channels, self._deque_len), self.fill_value, dtype=np.float32)
            self._valid = np.zeros((self.max_channels, self._deque_len), dtype=np.bool_)

    def _send_heartbeat_if_due(self) -> None:
        if not self._hb_sock:
            return
        now = time.time()
        if self._waiting_hb_reply:
            return
        if now - self._last_hb_send >= 2.0:
            try:
                msg = json.dumps({"application": "NewZMQClient", "type": "heartbeat"})
                self._hb_sock.send(msg.encode("utf-8"))
                self._last_hb_send = now
                self._waiting_hb_reply = True
               #if self.verbose:
               #     print("[HB] sent")
            except Exception as e:
                if self.verbose:
                    print(f"[HB] send error: {e}")

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self._send_heartbeat_if_due()
                socks = dict(self._poller.poll(10))

                # heartbeat reply
                if self._hb_sock and self._hb_sock in socks and self._waiting_hb_reply:
                    try:
                        _ = self._hb_sock.recv(flags=zmq.NOBLOCK)
                        self._waiting_hb_reply = False
                        #if self.verbose:
                        #    print("[HB] reply")
                    except zmq.Again:
                        pass
                    except Exception as e:
                        if self.verbose:
                            print(f"[HB] recv error: {e}")

                # data frames
                if self._data_sock and self._data_sock in socks:
                    try:
                        frames = self._data_sock.recv_multipart(flags=zmq.NOBLOCK)
                        if len(frames) < 2:
                            continue
                        # frames[1]: JSON header; frames[2]: payload (if present)
                        try:
                            header = json.loads(frames[1].decode("utf-8", errors="ignore"))
                        except Exception:
                            continue

                        typ = header.get("type", "")
                        if typ == "data":
                            content = header.get("content", {})
                            ch = int(content.get("channel_num", -1))
                            ch_name = content.get("channel_name", f"CH{ch + 1}")
                            rate = float(content.get("sample_rate", self.fs))
                            s0 = _parse_int(content.get("sample_num"))  # first sample index in packet
                            ns = _parse_int(content.get("num_samples"))  # number of samples in packet

                            payload = frames[2] if len(frames) >= 3 else b""
                            samples = np.frombuffer(payload, dtype=np.float32)

                            # Update fs & buffer sizes if the stream rate changes
                            if rate > 0.0 and rate != self.fs:
                                with self._lock:
                                    self.fs = rate
                                    self._rebuild_deques_if_needed(self.fs)

                            if 0 <= ch < self.max_channels and samples.size:
                                with self._lock:
                                    # --- buffer/meta updates ---
                                    self.buffers[ch].extend(samples.tolist())
                                    self._name_by_index[ch] = ch_name
                                    self.seen_nums.add(ch)
                                    self.seen_names.add(ch_name)
                                    self.seen_names_norm.add(_norm_name(ch_name))
                                    if self._ref_clock_ch is None:
                                        self._ref_clock_ch = ch
                                    if ch == self._ref_clock_ch:
                                        self.total_samples_written += samples.size

                                    # --- header-based clocks with loop detection ---
                                    end_idx = None
                                    if (s0 is not None) and (ns is not None):
                                        end_idx = s0 + ns  # end-of-packet (exclusive)

                                    # Detect playback loop on the reference channel
                                    if (ch == self._ref_clock_ch) and (end_idx is not None):
                                        if (self._last_ref_s0 is not None) and (s0 < self._last_ref_s0):
                                            # loop just started
                                            self.loop_cycle += 1
                                            if self._last_ref_end is not None:
                                                self._index_offset += int(self._last_ref_end)
                                            # Optional: epoch reset on loop
                                            # for dq in self.buffers: dq.clear()
                                            # self._drain_last_total = 0
                                        self._last_ref_s0 = s0
                                        self._last_ref_end = end_idx

                                    if s0 is not None:
                                        self._last_header_index[ch] = s0

                                    # Maintain both clocks: loop-relative and monotonic
                                    if end_idx is not None:
                                        self.loop_sample_index = int(end_idx)
                                        if self.loop_global_index:
                                            # exported clock loops with playback
                                            if ch == self._ref_clock_ch:
                                                self.global_sample_index = self.loop_sample_index
                                        else:
                                            # exported clock is monotonic across loops
                                            mono = self._index_offset + int(end_idx)
                                            if mono > self.global_sample_index:
                                                self.global_sample_index = mono
                                    else:
                                        # No header indices; fall back
                                        if self.loop_global_index:
                                            if ch == self._ref_clock_ch:
                                                self.loop_sample_index += int(samples.size)
                                                self.global_sample_index = self.loop_sample_index
                                        else:
                                            self.global_sample_index += int(samples.size)

                                    # --- index-aligned ring write (optional) ---
                                    if getattr(self, "align_to_header_index", False):
                                        # Absolute start index for this packet (monotonic)
                                        if s0 is not None:
                                            abs_start = int(self._index_offset) + int(s0)
                                        else:
                                            # fallback: estimate from current end
                                            abs_end = (int(self._index_offset) + int(self.loop_sample_index)) \
                                                if self.loop_global_index else int(self.global_sample_index)
                                            abs_start = abs_end - int(samples.size)

                                        n = int(samples.size)
                                        if n > 0:
                                            L = self._deque_len
                                            pos0 = abs_start % L
                                            if pos0 + n <= L:
                                                self._ring[ch, pos0:pos0 + n] = samples
                                                self._valid[ch, pos0:pos0 + n] = True
                                            else:
                                                first = L - pos0
                                                self._ring[ch, pos0:L] = samples[:first]
                                                self._valid[ch, pos0:L] = True
                                                rest = n - first
                                                self._ring[ch, 0:rest] = samples[first:]
                                                self._valid[ch, 0:rest] = True

                                    # Mark ready and channels-ready
                                    if not self.ready_event.is_set():
                                        self.ready_event.set()
                                    if self._channels_complete_enough_unlocked():
                                        self.channels_ready_event.set()

                        elif typ == "event":
                            evt = Event(header.get("content", {}), frames[2] if len(frames) >= 3 else None)
                            if self.verbose:
                                print(evt)

                        elif typ == "spike":
                            spk = Spike(header.get("spike", {}), frames[2] if len(frames) >= 3 else None)
                            if self.verbose:
                                print(spk)

                    except zmq.Again:
                        pass
                    except Exception as e:
                        if self.verbose:
                            print(f"[Data] error: {e}")
                        time.sleep(0.01)

                # periodic global index logger
                now = time.time()
                if (now - self._last_index_log) >= self.index_log_interval_s and self.ready_event.is_set():
                    with self._lock:
                        gidx = int(self.global_sample_index)
                        lidx = int(self.loop_sample_index)
                        cycle = int(self.loop_cycle)
                        fs_local = float(self.fs)
                    gt = (gidx / fs_local) if fs_local > 0 else float('nan')
                    lt = (lidx / fs_local) if fs_local > 0 else float('nan')
                    if self.loop_global_index:
                        print(f"[IDX] (LOOP) global={gidx} t={gt:.3f}s | cycle={cycle} | fs={fs_local:.2f}Hz")
                    else:
                        print(
                            f"[IDX] (MONO) global={gidx} t={gt:.3f}s | loop_idx={lidx} loop_t={lt:.3f}s cycle={cycle} | fs={fs_local:.2f}Hz")
                    self._last_index_log = now

            except Exception as e:
                if self.verbose:
                    print(f"[Loop] error: {e}")
                time.sleep(0.1)

    def wait_for_expected_channels(self, timeout: float = 15.0) -> bool:
        """Block until expected channels (by name) have been seen (per required_fraction)."""
        if not self.expected_names_norm:
            return True
        return self.channels_ready_event.wait(timeout=timeout)

    def set_index_looping(self, enabled: bool) -> None:
        """If True, global_sample_index restarts at each playback loop; else stays monotonic."""
        self.loop_global_index = bool(enabled)

    def _channels_complete_enough_unlocked(self) -> bool:
        if self.expected_names_norm:
            need = set(self.expected_names_norm)
            have = self.seen_names_norm
            frac = len(need & have) / max(1, len(need))
            return frac >= self.required_fraction
        if self.expected_count:
            frac = len(self.seen_nums) / float(self.expected_count)
            return frac >= self.required_fraction
        return len(self.seen_nums) > 0

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._setup()
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="ZMQClient", daemon=True)
        self._thread.start()
        if self.verbose:
            print(
                f"[ZMQClient] started; data=tcp://{self.host_ip}:{self.data_port} hb=tcp://{self.host_ip}:{self.hb_endpoint or 'None'}")

    def stop(self, timeout: Optional[float] = 2.0) -> None:
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=timeout)
        self._teardown()
        if self.verbose:
            print("[ZMQClient] stopped")

    def close(self) -> None:
        self.stop()
        # do not terminate the shared Context (instance()) globally

    # ---------- selection & info ----------
    def set_channel_index(self, indices: Iterable[int]) -> None:
        with self._lock:
            idx = [int(i) for i in indices]
            for i in idx:
                if i < 0 or i >= self.max_channels:
                    raise ValueError(f"Channel index {i} out of range [0,{self.max_channels - 1}]")
            self.channel_index = idx
            self.N_channels = len(idx)

    @property
    def channel_names(self) -> List[str]:
        with self._lock:
            n_tot = max(self.seen_nums) + 1 if self.seen_nums else 0
            return [self._name_by_index.get(i, f"CH{i + 1}") for i in range(n_tot)]

    def fs_estimate(self, n_last: int = 2000) -> float:
        return float(self.fs)

    def get_latest_window(self, window_ms: int = 500) -> np.ndarray:
        nsamples = int(round(self.fs * window_ms / 1000.0))
        if nsamples < 1:
            raise ValueError("Window size must be at least 1 ms.")

        if not self.ready_event.is_set():
            raise NotReadyError("NewZMQClient not ready; no data received yet.")

        return self.get_latest(nsamples)[1]

    def get_latest(self, n: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return the latest n samples and their absolute timestamps.
        Y: (C_selected, n), t: (n,)
        """
        if not self.ready_event.is_set():
            raise NotReadyError("NewZMQClient not ready; no data received yet.")

        with self._lock:
            if not self.channel_index:
                # default to all seen channels, numeric order
                self.channel_index = sorted(self.seen_nums)
                self.N_channels = len(self.channel_index)

            n = max(1, int(n))

            if getattr(self, "align_to_header_index", False):
                # Index-aligned read from the circular ring using absolute indices
                total_abs = (int(self._index_offset) + int(self.loop_sample_index)) \
                    if self.loop_global_index else int(self.global_sample_index)
                start_abs = total_abs - n
                L = self._deque_len
                idxs = (np.arange(start_abs, total_abs, dtype=np.int64) % L)

                Y = np.empty((self.N_channels, n), dtype=np.float32)
                for i, ch in enumerate(self.channel_index):
                    Y[i, :] = self._ring[ch, idxs]
            else:
                # Legacy: read from per-channel deques (tail)
                Y = np.zeros((self.N_channels, n), dtype=np.float32)
                have = 0
                for i, ch in enumerate(self.channel_index):
                    buf = self.buffers[ch]
                    if not buf:
                        continue
                    m = min(len(buf), n)
                    Y[i, -m:] = list(buf)[-m:]
                    have = max(have, m)
                if have < 1:
                    raise NotReadyError("No samples yet for selected channels.")

            # timestamps from absolute index
            total = (int(self._index_offset) + int(self.loop_sample_index)) \
                if self.loop_global_index else int(self.global_sample_index)
            idx = np.arange(total - n, total, dtype=np.float64)
            t = idx / self.fs

        return Y, t

    def latest(self) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Return (t_rel, Y) like the old client:

        - t_rel: (M,) seconds ending at 0 ([-window, 0])
        - Y:     (N_channels, M)

        Uses self.N_samples as the window size.
        """
        if self.channel_index is None or len(self.channel_index) == 0:
            return None, None

        M = int(self.N_samples)
        with self._lock:
            Y = np.zeros((len(self.channel_index), M), dtype=np.float32)
            have = 0
            for i, ch in enumerate(self.channel_index):
                buf = list(self.buffers[ch])
                n = min(len(buf), M)
                if n > 0:
                    Y[i, -n:] = buf[-n:]
                    have = max(have, n)

        if have < 2:
            return None, None

        t_rel = (np.arange(-M, 0, dtype=np.float64) / self.fs)[-have:]
        return t_rel, Y[:, -have:]

    def drain_new(self) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Return only NEW samples since last call:
          - t_abs: (K,) seconds since stream start (based on total_samples_written / fs)
          - Y_new: (N_channels, K)
        """
        if self.channel_index is None or len(self.channel_index) == 0:
            return None, None

        with self._lock:
            # total = int(self.total_samples_written)
            total = int(self.global_sample_index)
            n_new = total - self._drain_last_total
            if n_new <= 0:
                return None, None

            K = n_new
            Y_new = np.zeros((len(self.channel_index), K), dtype=np.float32)
            for i, ch in enumerate(self.channel_index):
                lst = list(self.buffers[ch])
                if len(lst) >= K:
                    Y_new[i, :] = lst[-K:]
                elif len(lst) > 0:
                    pad = K - len(lst)
                    Y_new[i, pad:] = lst
                # else: leave zeros for channels that don't have data yet

            t_new = np.arange(self._drain_last_total, total, dtype=np.float64) / self.fs
            self._drain_last_total = total

        return t_new, Y_new


class WorkingZMQClient:
    """
    Real-time Open Ephys ZMQ client with ring buffers and LSL-like API.

    Key features:
      - Robust JSON header handling (skips non-JSON frames).
      - Tracks seen channels and (optionally) blocks until a required set is present.
      - 'latest()' returns a rolling window; 'drain_new()' returns only new samples since last call.
      - Channel selection via 'channel_index' (e.g., [7, 8, 9, 10]).
      - Time base based on sample index and fs (Open Ephys does not send per-sample timestamps).

    Parameters
    ----------
    zqm_ip : str
        ZMQ endpoint prefix (e.g. 'tcp://localhost').
    http_ip : str
        Host/IP for the HTTP side of GUIClient (optional helper).
    data_port : int
        Data stream port.
    heartbeat_port : int
        Heartbeat REQ port.
    window_secs : float
        Default time window length for plotting helpers.
    channels : Iterable[int] | None
        Selected channel indices to expose via latest() / drain_new(). If None, we’ll fill
        once we know 'n_channels_total' or use all seen channels.
    auto_start : bool
        Start streaming worker on construction.
    verbose : bool
        Chatty logs.
    expected_channel_names : Iterable[str] | None
        If provided, we’ll consider the stream “ready” when these names have appeared.
    expected_channel_count : int | None
        If names are not provided, you can require a specific count instead (e.g., 128).
    require_complete : bool
        If True, latest()/drain_new() will yield nothing until readiness criteria are met.
    required_fraction : float
        Fraction of expected channels required to mark ready (1.0 = all, 0.95 = 95%).
    max_channels : int
        Hard ceiling for buffer allocation (avoid reallocations when late channels appear).
    """

    def __init__(
            self,
            zqm_ip: str = "tcp://localhost",
            http_ip: str = "127.0.0.1",
            data_port: str = "5556",
            heartbeat_port: Optional[str] = None,
            window_secs: float = 5.0,
            channels: Optional[Iterable[int]] = None,
            auto_start: bool = True,
            verbose: bool = False,
            expected_channel_names: Optional[Iterable[str]] = None,
            expected_channel_count: Optional[int] = None,
            require_complete: bool = False,
            required_fraction: float = 1.0,
            max_channels: int = 256,
    ):
        # --- config / state
        self.ip = zqm_ip
        self.data_port = int(data_port)
        if heartbeat_port is None:
            heartbeat_port = int(data_port) + 1
        self.heartbeat_port = int(heartbeat_port)
        self.window_secs = float(window_secs)
        self.verbose = bool(verbose)

        self.name = "ZMQClient"
        self.type = "ZMQ Data Stream"

        # readiness/connection
        self.connection_lost = True
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 5

        self.context = zmq.Context()
        self.poller = zmq.Poller()
        self.heartbeat_socket = None
        self.data_socket = None
        self.uuid = str(uuid.uuid4())
        self.last_reply_time = time.time()
        self.last_heartbeat_time = time.time()
        self.socket_waits_reply = False

        # threading/locks
        self.lock = threading.Lock()
        self.ready_event = threading.Event()  # first data packet seen
        self.channels_ready_event = threading.Event()  # enough channels have been seen
        self.streaming = False
        self.streaming_thread = None

        # sampling/time base
        self.sampling_rate = 2000.0  # default; updated from headers when available
        self.fs = float(self.sampling_rate)
        self.N_samples = int(max(1, round(self.fs * self.window_secs)))
        self.total_samples_written = 0
        self._drain_last_total = 0  # last drained sample index
        self._ref_clock_ch = None  # channel we use as a "clock" (first-seen, or 0)

        # channels / buffers
        self.max_channels = int(max_channels)
        self._name_by_index = {}  # {ch_idx: "CH#"}
        self.seen_nums = set()  # {0,1,2,...}
        self.seen_names = set()  # {"CH1", ...}

        # expected/required channel set
        self.expected_names = list(expected_channel_names) if expected_channel_names else None
        self.expected_count = int(expected_channel_count) if expected_channel_count else None
        self.require_complete = bool(require_complete)
        self.required_fraction = float(required_fraction)

        # Known channel count so far; will grow as data arrives
        self.n_channels_total = 0
        # Channel selection (indices into "physical" channels)
        self.channel_index = list(channels) if channels is not None else None
        self.N_channels = len(self.channel_index) if self.channel_index is not None else 0

        # ring buffers per physical channel index [0..max_channels-1]
        maxlen = int(self.fs * self.window_secs)
        self.buffers: List[deque] = [deque(maxlen=maxlen) for _ in range(self.max_channels)]

        # init ZMQ
        self._initialize_sockets()

        # GUI control
        self.gui = GUIClient(host=http_ip)

        if auto_start:
            self.start()
            # Optional: wait for first data frame so downstream code can proceed
            self.ready_event.wait(timeout=5.0)
            # Optionally wait for required channel set
            if self.require_complete:
                self.wait_for_channels(timeout_sec=15.0)

        if self.verbose:
            self._print_metadata()

    def _print_metadata(self):
        print(f"[ZMQClient] Connected to {self.ip}:{self.data_port}")
        print(f"  UUID: {self.uuid}")
        print(f"  Channels: {self.n_channels_total}")
        print(f"  Sampling Rate: {self.sampling_rate} Hz")
        print(f"  Channel Names: {self.channel_names}")
        print(f" Type: ZMQ Data Stream")

    # ------------------- sockets / heartbeat -------------------

    def _initialize_sockets(self):
        try:
            if not self.data_socket:
                addr = f"{self.ip}:{self.data_port}"
                self.data_socket = self.context.socket(zmq.SUB)
                self.data_socket.connect(addr)
                self.data_socket.setsockopt(zmq.SUBSCRIBE, b"")
                self.data_socket.setsockopt(zmq.RCVTIMEO, 1000)
                self.poller.register(self.data_socket, zmq.POLLIN)
                print(f"[ZMQClient] Initialized data socket on {addr}")

            if not self.heartbeat_socket:
                addr = f"{self.ip}:{self.heartbeat_port}"
                self.heartbeat_socket = self.context.socket(zmq.REQ)
                self.heartbeat_socket.connect(addr)
                self.heartbeat_socket.setsockopt(zmq.RCVTIMEO, 2000)
                self.poller.register(self.heartbeat_socket, zmq.POLLIN)
                print(f"[ZMQClient] Initialized heartbeat socket on {addr}")

            self.connection_lost = False
            self.reconnect_attempts = 0
        except Exception as e:
            print(f"[Socket Error] Failed to initialize sockets: {e}")
            self.connection_lost = True

    def _reconnect_sockets(self) -> bool:
        if self.reconnect_attempts >= self.max_reconnect_attempts:
            print(f"[Connection] Max reconnection attempts reached ({self.max_reconnect_attempts})")
            return False
        print(f"[Connection] Attempting reconnection ({self.reconnect_attempts + 1}/{self.max_reconnect_attempts})")
        try:
            if self.data_socket:
                self.data_socket.close()
                self.data_socket = None
            if self.heartbeat_socket:
                self.heartbeat_socket.close()
                self.heartbeat_socket = None
            time.sleep(0.5)
            self._initialize_sockets()
            self.reconnect_attempts += 1
            return not self.connection_lost
        except Exception as e:
            print(f"[Reconnection Error] {e}")
            self.reconnect_attempts += 1
            return False

    def _send_heartbeat(self):
        if self.connection_lost:
            return
        try:
            msg = json.dumps({"application": self.name, "uuid": self.uuid, "type": "heartbeat"})
            self.heartbeat_socket.send(msg.encode("utf-8"))
            self.last_heartbeat_time = time.time()
            self.socket_waits_reply = True
            if self.verbose:
                print("[Heartbeat] Sent")
        except Exception as e:
            print(f"[Heartbeat Error] {e}")
            self.connection_lost = True

    # ------------------- readiness helpers -------------------

    def _channels_complete_enough_unlocked(self) -> bool:
        """Check without acquiring lock."""
        if self.expected_names:
            need = set(self.expected_names)
            have = self.seen_names
            frac = len(need & have) / max(1, len(need))
            return frac >= self.required_fraction
        if self.expected_count:
            frac = len(self.seen_nums) / float(self.expected_count)
            return frac >= self.required_fraction
        # No expectations set: any channel seen marks readiness.
        return len(self.seen_nums) > 0

    def wait_for_channels(self, timeout_sec: float = 10.0) -> bool:
        # Fast path
        with self.lock:
            if self._channels_complete_enough_unlocked():
                self.channels_ready_event.set()
                return True
        # Block until enough channels have been seen
        end = time.time() + timeout_sec
        while time.time() < end:
            if self.channels_ready_event.wait(timeout=0.1):
                return True
        return False

    # ------------------- public controls -------------------

    def start(self):
        if self.streaming:
            if self.verbose:
                print("Already streaming")
            return
        self.streaming = True
        self.streaming_thread = threading.Thread(target=self._streaming_worker, daemon=True)
        self.streaming_thread.start()

    def stop(self):
        self.streaming = False
        if self.streaming_thread:
            self.streaming_thread.join()

    def close(self):
        self.stop()
        if self.data_socket:
            self.data_socket.close()
        if self.heartbeat_socket:
            self.heartbeat_socket.close()
        self.context.term()

    # ------------------- worker loop -------------------

    def _streaming_worker(self):
        while self.streaming:
            if self.connection_lost and not self._reconnect_sockets():
                continue

            if (time.time() - self.last_heartbeat_time) > 2.0:
                self._send_heartbeat()

            try:
                socks = dict(self.poller.poll(10))
                # heartbeat replies
                if self.heartbeat_socket in socks and self.socket_waits_reply:
                    try:
                        _ = self.heartbeat_socket.recv()
                        self.socket_waits_reply = False
                        self.last_reply_time = time.time()
                    except zmq.Again:
                        pass
                    except Exception as e:
                        print(f"[Heartbeat Error] {e}")
                        self.connection_lost = True

                # data stream
                if self.data_socket in socks:
                    try:
                        msg = self.data_socket.recv_multipart(zmq.NOBLOCK)
                        if len(msg) < 2:
                            continue
                        # Guard decode: some frames may not be JSON headers
                        try:
                            header = json.loads(msg[1].decode("utf-8"))
                        except Exception:
                            continue

                        typ = header.get("type", "")
                        if typ == "data":
                            if not self.ready_event.is_set():
                                self.ready_event.set()

                            content = header.get("content", {})
                            ch = int(content.get("channel_num", -1))
                            ch_name = content.get("channel_name", f"CH{ch + 1}")
                            rate = float(content.get("sample_rate", self.fs))

                            # Update fs if needed
                            if rate > 0 and rate != self.fs:
                                self.fs = float(rate)
                                # do not resize deques; just update sizes for latest()
                                self.N_samples = int(max(1, round(self.fs * self.window_secs)))

                            samples = np.frombuffer(msg[2], dtype=np.float32, count=-1)
                            if 0 <= ch < self.max_channels and samples.size:
                                with self.lock:
                                    # extend buffer for this physical channel
                                    self.buffers[ch].extend(samples.tolist())

                                    # track names/count
                                    self._name_by_index[ch] = ch_name
                                    self.seen_nums.add(ch)
                                    self.seen_names.add(ch_name)
                                    self.n_channels_total = max(self.n_channels_total, ch + 1)

                                    # choose a reference clock channel if we haven't yet
                                    if self._ref_clock_ch is None:
                                        self._ref_clock_ch = ch
                                    # advance "global" samples when reference channel arrives
                                    if ch == self._ref_clock_ch:
                                        self.total_samples_written += samples.size

                                    # selection defaulting
                                    if self.channel_index is None and self.n_channels_total > 0:
                                        # default to all seen channels in numeric order
                                        self.channel_index = sorted(list(self.seen_nums))
                                        self.N_channels = len(self.channel_index)

                                    # readiness criteria
                                    if self._channels_complete_enough_unlocked():
                                        self.channels_ready_event.set()

                        elif typ == "event":
                            evt = Event(header.get("content", {}), msg[2] if header.get("data_size", 0) > 0 else None)
                            if self.verbose:
                                print(evt)

                        elif typ == "spike":
                            spk = Spike(header.get("spike", {}), msg[2])
                            if self.verbose:
                                print(spk)

                    except zmq.Again:
                        pass
                    except Exception as e:
                        print(f"[Data Error] {e}")

            except Exception as e:
                print(f"Streaming worker error: {e}")
                self.connection_lost = True
                time.sleep(0.1)

    # ------------------- LSL-like API -------------------

    def fs_estimate(self, n_last: int = 2000) -> float:
        return float(self.fs)

    @property
    def channel_names(self) -> List[str]:
        # Derived from name map; fill unknowns as CH#
        return [self._name_by_index.get(i, f"CH{i + 1}") for i in range(self.n_channels_total)]

    def set_channel_index(self, indices: Iterable[int]):
        with self.lock:
            idx = [int(i) for i in indices]
            for i in idx:
                if i < 0 or i >= self.max_channels:
                    raise ValueError(f"Channel index {i} out of range [0,{self.max_channels - 1}]")
            self.channel_index = idx
            self.N_channels = len(idx)

    def latest(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Return (t_rel, Y) where:
          - t_rel: (M,) seconds, ending at 0 ([-window_secs, 0])
          - Y:     (N_channels, M)
        """
        if self.require_complete and not self.channels_ready_event.is_set():
            return None, None

        M = self.N_samples
        with self.lock:
            if not self.channel_index:
                return None, None
            Y = np.zeros((self.N_channels, M), dtype=np.float32)
            have = 0
            for i, ch in enumerate(self.channel_index):
                # if channel hasn't arrived yet, its buffer will be empty => zeros
                buf = list(self.buffers[ch])
                n = min(len(buf), M)
                if n > 0:
                    Y[i, -n:] = buf[-n:]
                    have = max(have, n)

        if have < 2:
            return None, None

        t_rel = (np.arange(-M, 0, dtype=np.float64) / self.fs)[-have:]
        return t_rel, Y[:, -have:]

    def drain_new(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Return only NEW samples since last call:
          - t_abs: (K,) seconds since stream start (based on reference channel sample index / fs)
          - Y_new: (N_channels, K)
        """
        if self.require_complete and not self.channels_ready_event.is_set():
            return None, None

        with self.lock:
            if not self.channel_index:
                return None, None

            total = int(self.total_samples_written)
            n_new = total - self._drain_last_total
            if n_new <= 0:
                return None, None

            K = n_new
            Y_new = np.zeros((self.N_channels, K), dtype=np.float32)
            for i, ch in enumerate(self.channel_index):
                lst = list(self.buffers[ch])
                if len(lst) >= K:
                    Y_new[i, :] = lst[-K:]
                elif len(lst) > 0:
                    pad = K - len(lst)
                    Y_new[i, pad:] = lst
                # else: keep zeros for channels with no data yet

            t_new = np.arange(self._drain_last_total, total, dtype=np.float64) / self.fs
            self._drain_last_total = total

        return t_new, Y_new

    # ------------------- convenience -------------------

    def get_latest_window(self, window_ms: int) -> np.ndarray:
        """Return most-recent window for SELECTED channels (C, N)."""
        n_samples = int(self.fs * window_ms / 1000.0)
        with self.lock:
            if not self.channel_index:
                return np.zeros((0, n_samples), dtype=np.float32)
            out = np.zeros((self.N_channels, n_samples), dtype=np.float32)
            for i, ch in enumerate(self.channel_index):
                buf = list(self.buffers[ch])
                if len(buf) >= n_samples:
                    out[i, :] = buf[-n_samples:]
                elif len(buf) > 0:
                    pad = n_samples - len(buf)
                    out[i, :pad] = 0.0
                    out[i, pad:] = buf
                else:
                    out[i, :] = 0.0
        return out

    def get_connection_status(self):
        with self.lock:
            return {
                "connected": not self.connection_lost,
                "streaming": self.streaming,
                "reconnect_attempts": self.reconnect_attempts,
                "total_samples": self.total_samples_written,
                "seen_channels": sorted(list(self.seen_nums)),
                "n_channels_total": self.n_channels_total,
            }

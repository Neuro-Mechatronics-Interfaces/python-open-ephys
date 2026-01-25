#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ZMQClient: high-performance Open Ephys ZMQ data client with per-channel ring buffers.

Features
- Background receiver thread using ZeroMQ SUB.
- Parses Open Ephys GUI ZMQ frames (header JSON in frame[1], float32 payload in frame[2]).
- Keeps a per-channel circular (ring) buffer sized by buffer_seconds * sample_rate.
- Fast "latest" reads: get_latest(n) or get_latest_window(seconds).
- Optional CONFLATE (latest-only) mode and tunable RCVHWM/RCVBUF.
- Stats: message rate, channel-sample rate, bytes/sec, per-channel drop counts (sequence gaps).

Dependencies
    pip install pyzmq numpy
"""

from __future__ import annotations

import json
import threading
import time
from typing import Dict, Iterable, List, Optional

import numpy as np
import zmq

from pyoephys.interface import GUIClient


def _human_mb_per_s(bps: float) -> str:
    return f"{bps/1_000_000:.2f} MB/s"


class _RingBuffer:
    """Lightweight 1D ring buffer for float32 samples."""

    __slots__ = ("buf", "cap", "widx", "filled", "lock")

    def __init__(self, capacity: int):
        self.buf = np.zeros(int(capacity), dtype=np.float32)
        self.cap = int(capacity)
        self.widx = 0
        self.filled = False
        self.lock = threading.Lock()

    def append(self, data: np.ndarray):
        """Append 1D float32 array, overwriting old data as necessary."""
        data = np.asarray(data, dtype=np.float32)
        n = int(data.size)
        if n == 0:
            return
        with self.lock:
            cap = self.cap
            w = self.widx
            if n >= cap:
                self.buf[...] = data[-cap:]
                self.widx = 0
                self.filled = True
                return
            end = w + n
            if end <= cap:
                self.buf[w:end] = data
                self.widx = end % cap
            else:
                k = cap - w
                self.buf[w:] = data[:k]
                self.buf[: end % cap] = data[k:]
                self.widx = end % cap
            if self.widx == 0:
                self.filled = True

    def get_latest(self, n: int) -> np.ndarray:
        """Return a copy of the latest n samples (or fewer if not available)."""
        with self.lock:
            if not self.filled and self.widx == 0:
                return np.empty(0, dtype=np.float32)
            cap = self.cap
            size = cap if self.filled else self.widx
            n = int(min(max(n, 0), size))
            if n == 0:
                return np.empty(0, dtype=np.float32)
            end = self.widx
            start = (end - n) % cap
            if start < end or not self.filled:
                return self.buf[start:end].copy()
            else:
                return np.concatenate((self.buf[start:], self.buf[:end])).astype(np.float32, copy=False)

    def size(self) -> int:
        with self.lock:
            return self.cap if self.filled else self.widx

    def capacity(self) -> int:
        return self.cap


class ZMQClient:
    """
    Open Ephys ZMQ client that maintains per-channel ring buffers and exposes fast "latest" access.

    Message format assumption (based on Open Ephys GUI's ZMQ plugin & your examples):
        frames[0] : (optional) topic
        frames[1] : UTF-8 JSON header with keys:
           - "type": "data"
           - "message_num": int
           - "content": { "num_samples": int, "channel_num": int, "sample_rate": float, ... }
        frames[2] : float32 payload of shape (num_samples,)

    Parameters
    ----------
    endpoint : str
        ZMQ SUB endpoint for data, e.g. "tcp://127.0.0.1:5556".
    topic : bytes
        Subscription topic (empty = all).
    buffer_seconds : float
        Target buffer window in seconds. Actual ring size = ceil(sample_rate * buffer_seconds).
        If the sample rate is unknown at start, uses sample_rate_hint until first header with 'sample_rate' arrives.
    sample_rate_hint : float
        Used to size initial buffers until the stream reports sample_rate.
    expected_channels : int
        Pre-allocate ring buffers lazily up to this many channel indices (created on first sight).
    rcvhwm : int
        ZMQ RCVHWM (high-water mark). Increase for burst tolerance (e.g., 20000 - 50000).
    conflate : bool
        If True, enable ZMQ.CONFLATE (keep only latest message). Useful for "latest-only" apps. Hides drops.
    rcvbuf_bytes : Optional[int]
        Optional OS receive buffer hint (ZMQ_RCVBUF). None = don't set.
    poll_timeout_ms : int
        Poller timeout in ms in the receiver loop.
    ctx_io_threads : int
        ZMQ context I/O threads (advanced; defaults to 1).
    """

    def __init__(
        self,
        endpoint: str = "tcp://127.0.0.1:5556",
        topic: bytes = b"",        buffer_seconds: float = 10.0,
        sample_rate_hint: float = 1000.0,
        expected_channels: int = 128,
        rcvhwm: int = 20000,
        conflate: bool = False,
        rcvbuf_bytes: Optional[int] = None,
        poll_timeout_ms: int = 10,
        ctx_io_threads: int = 1,
    ):
        self.endpoint = endpoint
        self.topic = topic
        self.buffer_seconds = float(buffer_seconds)
        self.sample_rate = float(sample_rate_hint)  # updated from stream on first header
        self.expected_channels = int(expected_channels)
        self.rcvhwm = int(rcvhwm)
        self.conflate = bool(conflate)
        self.rcvbuf_bytes = rcvbuf_bytes
        self.poll_timeout_ms = int(poll_timeout_ms)

        # Context & socket set up in start()
        self._ctx = None
        self._sub = None
        self._poller = None

        # State
        self._stop = threading.Event()
        self._thread = None

        # Per-channel buffers (allocated lazily on first channel sighting)
        self._buffers: Dict[int, _RingBuffer] = {}
        self._chan_lock = threading.Lock()

        # Per-channel last sequence number & drop counts
        self._seq_prev: Dict[int, int] = {}
        self._seq_gaps: Dict[int, int] = {}

        # Stats
        self._total_msgs = 0
        self._total_bytes = 0
        self._total_ch_samples = 0
        self._last_report_ts = time.time()
        self._msgs_since = 0
        self._bytes_since = 0
        self._chsamp_since = 0

        # Last header-derived values
        self._last_block_size = None  # samples per message
        self._last_seen_channels = set()  # type: ignore

        # Context I/O threads
        self._ctx_io_threads = int(ctx_io_threads)

        # GUI Control
        self.gui = GUIClient(host=self.endpoint.split("//")[-1].split(":")[0])

    # ---------- lifecycle ----------

    def start(self):
        """Start the background receiver thread."""
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._ctx = zmq.Context(io_threads=self._ctx_io_threads)
        self._sub = self._ctx.socket(zmq.SUB)

        # SUB options
        self._sub.setsockopt(zmq.RCVHWM, self.rcvhwm)
        if self.conflate:
            try:
                self._sub.setsockopt(zmq.CONFLATE, 1)
            except Exception:
                pass
        if self.rcvbuf_bytes is not None:
            try:
                self._sub.setsockopt(zmq.RCVBUF, int(self.rcvbuf_bytes))
            except Exception:
                pass
        self._sub.setsockopt(zmq.SUBSCRIBE, self.topic)
        self._sub.connect(self.endpoint)

        self._poller = zmq.Poller()
        self._poller.register(self._sub, zmq.POLLIN)

        self._thread = threading.Thread(target=self._rx_loop, name="ZMQClient-RX", daemon=True)
        self._thread.start()

    def stop(self, timeout: Optional[float] = 2.0):
        """Signal the background thread to stop and close sockets."""
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=timeout)
        try:
            if self._poller and self._sub is not None:
                self._poller.unregister(self._sub)
        except Exception:
            pass
        try:
            if self._sub is not None:
                self._sub.close(0)
        except Exception:
            pass
        try:
            if self._ctx is not None:
                self._ctx.term()
        except Exception:
            pass
        self._thread = None
        self._sub = None
        self._ctx = None
        self._poller = None

    # ---------- RX loop ----------

    def _rx_loop(self):
        next_resize_check = time.time() + 0.5
        while not self._stop.is_set():
            socks = dict(self._poller.poll(timeout=self.poll_timeout_ms))
            if socks.get(self._sub) != zmq.POLLIN:
                continue

            frames = self._sub.recv_multipart()
            self._total_msgs += 1
            self._total_bytes += sum(len(f) for f in frames)
            self._msgs_since += 1
            self._bytes_since += sum(len(f) for f in frames)

            if len(frames) < 2:
                continue  # unknown shape

            # Header
            header = None
            try:
                header = json.loads(frames[1].decode("utf-8"))
            except Exception:
                pass

            if isinstance(header, dict) and header.get("type") == "data":
                c = header.get("content", {})
                ch = int(c.get("channel_num", -1))
                ns = int(c.get("num_samples", 0))
                sr = c.get("sample_rate", None)
                if sr is not None:
                    try:
                        sr = float(sr)
                        if sr > 0 and abs(sr - self.sample_rate) / max(self.sample_rate, 1.0) > 0.01:
                            self.sample_rate = sr
                    except Exception:
                        pass
                if ns > 0 and len(frames) >= 3:
                    payload = np.frombuffer(frames[2], dtype=np.float32, count=ns)
                    self._last_block_size = ns
                    if ch >= 0:
                        self._last_seen_channels.add(ch)
                        msg_num = header.get("message_num")
                        if isinstance(msg_num, int):
                            prev = self._seq_prev.get(ch, None)
                            if prev is not None and msg_num != prev + 1:
                                self._seq_gaps[ch] = self._seq_gaps.get(ch, 0) + 1
                            self._seq_prev[ch] = msg_num

                        buf = self._ensure_buffer_for_channel(ch)
                        buf.append(payload)
                        self._total_ch_samples += ns
                        self._chsamp_since += ns

            if time.time() >= next_resize_check:
                next_resize_check = time.time() + 0.5
                self._maybe_resize_buffers()

    # ---------- buffers & resizing ----------

    def _ensure_buffer_for_channel(self, ch: int) -> _RingBuffer:
        buf = self._buffers.get(ch)
        if buf is not None:
            return buf
        with self._chan_lock:
            buf = self._buffers.get(ch)
            if buf is not None:
                return buf
            cap = max(1, int(np.ceil(self.sample_rate * self.buffer_seconds)))
            buf = _RingBuffer(capacity=cap)
            self._buffers[ch] = buf
            return buf

    def _maybe_resize_buffers(self):
        target_cap = max(1, int(np.ceil(self.sample_rate * self.buffer_seconds)))
        for ch, buf in list(self._buffers.items()):
            if abs(buf.capacity() - target_cap) / target_cap > 0.1:
                tail = buf.get_latest(min(buf.size(), target_cap))
                newbuf = _RingBuffer(target_cap)
                if tail.size:
                    newbuf.append(tail)
                self._buffers[ch] = newbuf

    # ---------- public API ----------

    def get_latest(self, n: int, channels: Optional[Iterable[int]] = None) -> Dict[int, np.ndarray]:
        out: Dict[int, np.ndarray] = {}
        if channels is None:
            channels = sorted(self._buffers.keys())
        for ch in channels:
            buf = self._buffers.get(int(ch))
            if buf is None:
                out[int(ch)] = np.empty(0, dtype=np.float32)
            else:
                out[int(ch)] = buf.get_latest(n)
        return out

    def get_latest_window(self, seconds: float, channels: Optional[Iterable[int]] = None) -> Dict[int, np.ndarray]:
        n = int(max(0, round(self.sample_rate * float(seconds))))
        return self.get_latest(n, channels=channels)

    def get_stats(self, reset_window: bool = True) -> Dict[str, object]:
        now = time.time()
        elapsed = max(1e-6, now - self._last_report_ts)
        msg_rate_w = self._msgs_since / elapsed
        byte_rate_w = self._bytes_since / elapsed
        chsamp_rate_w = self._chsamp_since / elapsed
        stats = {
            "sample_rate_hz": self.sample_rate,
            "seen_channels": sorted(self._last_seen_channels),
            "last_block_size": self._last_block_size,
            "seq_gaps_per_channel": dict(self._seq_gaps),
            "overall": {
                "total_msgs": self._total_msgs,
                "total_bytes": self._total_bytes,
                "total_channel_samples": self._total_ch_samples,
            },
            "window": {
                "seconds": elapsed,
                "msg_rate_hz": msg_rate_w,
                "byte_rate_Bps": byte_rate_w,
                "ch_samp_rate_hz": chsamp_rate_w,
            },
            "human_window": {
                "msg_rate": f"{msg_rate_w:,.1f} msg/s",
                "byte_rate": _human_mb_per_s(byte_rate_w),
                "ch_samp_rate": f"{chsamp_rate_w:,.0f} ch-samp/s",
            },
        }
        if reset_window:
            self._last_report_ts = now
            self._msgs_since = 0
            self._bytes_since = 0
            self._chsamp_since = 0
        return stats

    def channels(self) -> List[int]:
        return sorted(self._buffers.keys())

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.stop()


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="ZMQClient demo: prints live stats and fetches latest window")
    ap.add_argument("--endpoint", default="tcp://127.0.0.1:5556")
    ap.add_argument("--topic", default="", help="Subscription topic filter")
    ap.add_argument("--buffer-seconds", type=float, default=10.0)
    ap.add_argument("--rcvhwm", type=int, default=20000)
    ap.add_argument("--rcvbuf", type=int, default=4_194_304, help="ZMQ_RCVBUF hint in bytes (default 4MiB)")
    ap.add_argument("--conflate", action="store_true")
    ap.add_argument("--print-interval", type=float, default=2.0)
    ap.add_argument("--window-sec", type=float, default=1.0, help="Fetch this many seconds on each print")
    args = ap.parse_args()

    client = ZMQClient(
        endpoint=args.endpoint,
        topic=args.topic.encode("utf-8"),
        buffer_seconds=args.buffer_seconds,
        rcvhwm=args.rcvhwm,
        conflate=args.conflate,
        rcvbuf_bytes=args.rcvbuf,
    )
    client.start()
    print(f"[zmq] SUB -> {args.endpoint} conflate={args.conflate} rcvhwm={args.rcvhwm} rcvbuf={args.rcvbuf}")

    try:
        while True:
            time.sleep(args.print_interval)
            stats = client.get_stats(reset_window=True)
            human = stats["human_window"]
            print(
                f"[stats] {human['msg_rate']} | {human['byte_rate']} | {human['ch_samp_rate']} "
                f"| ch={len(stats['seen_channels'])} | fs~{stats['sample_rate_hz']:.1f} Hz "
                f"| blk={stats['last_block_size']}"
            )
            chs = client.channels()[:4]
            latest = client.get_latest_window(args.window_sec, channels=chs)
            shapes = {ch: arr.shape for ch, arr in latest.items()}
            print(f"[latest] {args.window_sec}s shapes: {shapes}")
    except KeyboardInterrupt:
        pass
    finally:
        client.stop()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Open Ephys ZMQ Data Port (5556) - Throughput/Latency Benchmark (Refactored)
# Key changes vs. the original:
#   * Fixed a small parenthesis bug in _maybe_json() that could block JSON detection.
#   * Reorganized into a Benchmark class for easier testing/extension.
#   * Added CSV logging (--csv path.csv) for later analysis.
#   * Added graceful shutdown (Ctrl+C) with a clean final summary.
#   * Added --linger and --rcvtimeo socket options for robustness.
#   * Clear separation of: socket setup, warmup, measure loop, and reporting.
#   * Optional topic filter, parsing toggle, report interval, recv HWM, conflate.
#
# Examples:
#   python oe_zmq_bench.py --endpoint tcp://127.0.0.1:5556 --duration 10 --csv bench.csv
#   python oe_zmq_bench.py --endpoint tcp://192.168.1.50:5556 --recv-hwm 2000 --report-interval 0.5
#   python oe_zmq_bench.py --conflate --no-parse
#
# Dependencies:
#   pip install pyzmq

import argparse
import csv
import json
import signal
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, List

import zmq

from pyoephys.interface import GUIClient


def human_mb_per_s(bytes_per_s: float) -> str:
    return f"{bytes_per_s/1_000_000:.2f} MB/s"


def _maybe_json(frame: bytes) -> Optional[Dict[str, Any]]:
    """Best-effort JSON parse if the frame looks like UTF-8 JSON."""
    if not frame:
        return None
    if not ((frame[:1] in (b"{", b"[") and (frame[-1:] in (b"}", b"]" ))) or (b'\n' in frame)):
        return None
    try:
        s = frame.decode("utf-8", errors="strict")
        return json.loads(s)
    except Exception:
        return None


def _extract_sampling_info(js: Dict[str, Any]) -> Tuple[Optional[float], Optional[int], Optional[int]]:
    """Try to extract sample_rate (Hz), num_channels, num_samples from JSON headers using common key variants."""
    sample_rate = None
    for k in ["sample_rate", "fs", "sampling_rate", "rate_hz", "rate"]:
        if k in js:
            try:
                sample_rate = float(js[k])
                break
            except Exception:
                pass

    num_channels = None
    for k in ["num_channels", "channels", "n_channels", "numCh", "nch"]:
        if k in js:
            try:
                num_channels = int(js[k])
                break
            except Exception:
                pass

    num_samples = None
    for k in ["num_samples", "samples_per_message", "samples", "n_samples", "nsamp", "block_size"]:
        if k in js:
            try:
                num_samples = int(js[k])
                break
            except Exception:
                pass

    return sample_rate, num_channels, num_samples


def _extract_timestamp(js: Dict[str, Any]) -> Optional[float]:
    """Try to extract a UNIX-like timestamp from JSON (seconds or milliseconds)."""
    for k in ["timestamp", "unix_time", "time", "t"]:
        if k in js:
            v = js[k]
            try:
                if isinstance(v, (int, float)) and v > 1e12:
                    return float(v) / 1000.0
                elif isinstance(v, (int, float)) and v > 1e9:
                    return float(v)
                return float(v)
            except Exception:
                pass
    return None


@dataclass
class BenchConfig:
    endpoint: str = "tcp://127.0.0.1:5556"
    topic: str = ""
    duration: float = 10.0
    warmup: float = 1.0
    recv_hwm: int = 100
    conflate: bool = False
    report_interval: float = 2.0
    no_parse: bool = False
    csv_path: Optional[str] = None
    linger_ms: int = 0
    rcvtimeo_ms: int = -1  # -1 = block forever


class Benchmark:
    def __init__(self, cfg: BenchConfig):
        self.cfg = cfg
        self.ctx = zmq.Context.instance()
        self.sock = self.ctx.socket(zmq.SUB)

        # socket options
        self.sock.setsockopt(zmq.RCVHWM, self.cfg.recv_hwm)
        if self.cfg.conflate:
            self.sock.setsockopt(zmq.CONFLATE, 1)
        self.sock.setsockopt(zmq.LINGER, self.cfg.linger_ms)
        self.sock.setsockopt(zmq.RCVTIMEO, self.cfg.rcvtimeo_ms)
        self.sock.setsockopt(zmq.SUBSCRIBE, self.cfg.topic.encode("utf-8"))

        self.poller = zmq.Poller()
        self.poller.register(self.sock, zmq.POLLIN)

        self._should_stop = False
        self._csv_writer = None
        self._csv_file = None

        # stats
        self.total_msgs = 0
        self.total_bytes = 0
        self.parsed_blocks = 0
        self.parsed_samples_total = 0  # samples across all channels
        self.last_latency_estimate = None
        self.last_sr = None
        self.last_nc = None
        self.last_ns = None

    def _open_csv(self):
        if not self.cfg.csv_path:
            return
        self._csv_file = open(self.cfg.csv_path, "w", newline="", encoding="utf-8")
        self._csv_writer = csv.writer(self._csv_file)
        self._csv_writer.writerow([
            "t_elapsed_s", "msg_rate_hz", "byte_rate_Bps", "samples_per_s",
            "total_msgs", "total_bytes", "parsed_blocks", "latency_ms", "fs_hz", "nch", "ns_block"
        ])

    def _close_csv(self):
        if self._csv_file:
            self._csv_file.flush()
            self._csv_file.close()
            self._csv_file = None
            self._csv_writer = None

    def connect(self):
        print(f"[bench] Connecting SUB -> {self.cfg.endpoint} (topic='{self.cfg.topic}')")
        self.sock.connect(self.cfg.endpoint)

    def warmup(self):
        if self.cfg.warmup <= 0:
            return
        print(f"[bench] Warmup for {self.cfg.warmup:.2f}s...")
        deadline = time.time() + self.cfg.warmup
        while time.time() < deadline and not self._should_stop:
            socks = dict(self.poller.poll(timeout=5))  # 5 ms
            if socks.get(self.sock) == zmq.POLLIN:
                try:
                    _ = self.sock.recv_multipart(flags=zmq.NOBLOCK)
                except zmq.Again:
                    pass

    def _handle_frames(self, frames: List[bytes], now: float):
        self.total_msgs += 1
        self.total_bytes += sum(len(f) for f in frames)

        if self.cfg.no_parse:
            return

        js = None
        for fr in frames:
            js = _maybe_json(fr)
            if js is not None:
                break

        if js is not None:
            sr, nc, ns = _extract_sampling_info(js)
            ts = _extract_timestamp(js)
            if sr is not None:
                self.last_sr = sr
            if nc is not None:
                self.last_nc = nc
            if ns is not None:
                self.last_ns = ns
                if nc is not None:
                    self.parsed_blocks += 1
                    self.parsed_samples_total += int(nc) * int(ns)
            if ts is not None:
                self.last_latency_estimate = max(0.0, now - ts)  # seconds

    def measure(self):
        self._open_csv()
        start = time.time()
        next_report = start + self.cfg.report_interval
        end = start + self.cfg.duration

        try:
            while not self._should_stop:
                now = time.time()
                if now >= end:
                    break
                timeout_ms = max(1, int((end - now) * 1000))
                socks = dict(self.poller.poll(timeout=timeout_ms))
                if socks.get(self.sock) == zmq.POLLIN:
                    frames = self.sock.recv_multipart()
                    self._handle_frames(frames, now)

                if time.time() >= next_report:
                    self._report_progress(start)
                    next_report += self.cfg.report_interval
        finally:
            self._report_final(start)
            self._close_csv()

    def _report_progress(self, start: float):
        elapsed = max(1e-6, time.time() - start)
        msg_rate = self.total_msgs / elapsed
        byte_rate = self.total_bytes / elapsed

        if self.parsed_blocks > 0:
            samples_per_s = self.parsed_samples_total / elapsed
            msg = (f"[bench] t={elapsed:5.2f}s | {msg_rate:8.1f} msg/s | " 
                   f"{human_mb_per_s(byte_rate):>12} | {samples_per_s:,.0f} samp/s")
            if self.last_latency_estimate is not None:
                msg += f" | ~latency={self.last_latency_estimate*1e3:.1f} ms"
            if (self.last_sr or self.last_nc or self.last_ns):
                msg += f" | fs~{self.last_sr or 0:g} Hz, ch~{self.last_nc}, ns/block~{self.last_ns}"
        else:
            msg = f"[bench] t={elapsed:5.2f}s | {msg_rate:8.1f} msg/s | {human_mb_per_s(byte_rate):>12}"

        print(msg)

        if self._csv_writer is not None:
            samples_per_s = (self.parsed_samples_total / elapsed) if self.parsed_blocks > 0 else 0.0
            self._csv_writer.writerow([
                f"{elapsed:.3f}", f"{msg_rate:.3f}", f"{byte_rate:.3f}", f"{samples_per_s:.3f}",
                self.total_msgs, self.total_bytes, self.parsed_blocks,
                f"{(self.last_latency_estimate*1e3):.3f}" if self.last_latency_estimate is not None else "",
                f"{self.last_sr:g}" if self.last_sr is not None else "",
                f"{self.last_nc}" if self.last_nc is not None else "",
                f"{self.last_ns}" if self.last_ns is not None else "",
            ])

    def _report_final(self, start: float):
        elapsed = max(1e-6, time.time() - start)
        msg_rate = self.total_msgs / elapsed
        byte_rate = self.total_bytes / elapsed

        print("\n=== Benchmark Summary ===")
        print(f"Endpoint:        {self.cfg.endpoint}")
        print(f"Duration:        {self.cfg.duration:.2f} s (warmup {self.cfg.warmup:.2f} s)")
        print(f"Recv HWM:        {self.cfg.recv_hwm}  | CONFLATE={'on' if self.cfg.conflate else 'off'}")
        print(f"RCVTIMEO:        {self.cfg.rcvtimeo_ms} ms | LINGER: {self.cfg.linger_ms} ms")
        print(f"Total messages:  {self.total_msgs}")
        print(f"Total bytes:     {self.total_bytes:,}  ({human_mb_per_s(byte_rate)} sustained)")
        print(f"Msg rate:        {msg_rate:,.1f} msg/s")

        if self.parsed_blocks > 0:
            samples_per_s = self.parsed_samples_total / elapsed
            print(f"Parsed blocks:   {self.parsed_blocks}")
            print(f"Samples (total): {self.parsed_samples_total:,}")
            print(f"Samples/sec:     {samples_per_s:,.0f} samp/s")
            if self.last_latency_estimate is not None:
                print(f"Last latency:    {self.last_latency_estimate*1e3:.1f} ms (best-effort from header timestamp)")
            if self.last_sr is not None:
                print(f"Sample rate:     ~{self.last_sr:g} Hz")
            if self.last_nc is not None:
                print(f"Channels:        ~{self.last_nc}")
            if self.last_ns is not None:
                print(f"Samps/block:     ~{self.last_ns}")
        else:
            print("Note: Could not extract sampling info (JSON header not found or keys unknown)." )
            print("      Throughput above still reflects real byte/msg rates.")

    def stop(self, *_):
        self._should_stop = True


def parse_args() -> BenchConfig:
    p = argparse.ArgumentParser(description="Open Ephys ZMQ data benchmark (refactored)")
    p.add_argument("--endpoint", default="tcp://127.0.0.1:5556", help="ZMQ SUB endpoint (default: tcp://127.0.0.1:5556)")
    p.add_argument("--topic", default="", help="Subscription topic filter (default: empty = all)")
    p.add_argument("--duration", type=float, default=10.0, help="Benchmark duration in seconds (default: 10)")
    p.add_argument("--warmup", type=float, default=1.0, help="Warmup time before measuring (sec)")
    p.add_argument("--recv-hwm", type=int, default=100, help="ZMQ RCVHWM (default: 100)")
    p.add_argument("--conflate", action="store_true", help="Enable ZMQ.CONFLATE (keep only last message)")
    p.add_argument("--report-interval", type=float, default=2.0, help="Live report interval (sec)")
    p.add_argument("--no-parse", action="store_true", help="Disable JSON parsing (pure byte/message rate only)")
    p.add_argument("--csv", dest="csv_path", default=None, help="Optional CSV path for logging (e.g., bench.csv)")
    p.add_argument("--linger", dest="linger_ms", type=int, default=0, help="ZMQ LINGER in ms (default: 0 for fast exit)")
    p.add_argument("--rcvtimeo", dest="rcvtimeo_ms", type=int, default=-1, help="ZMQ RCVTIMEO in ms (default: -1)")
    args = p.parse_args()
    return BenchConfig(**vars(args))


def main():
    cfg = parse_args()
    bench = Benchmark(cfg)

    # Handle Ctrl+C gracefully
    signal.signal(signal.SIGINT, bench.stop)
    try:
        import signal as _s
        if hasattr(_s, "SIGTERM"):
            signal.signal(_s.SIGTERM, bench.stop)
    except Exception:
        pass

    bench.connect()
    bench.warmup()
    bench.measure()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Open Ephys ZMQ Benchmark with GUIClient control (start/stop acquisition)
# Adds: --warmup, --rcvhwm, --conflate, per-channel accounting, CSV logging, instantaneous rates.

import argparse
import csv
import json
import os
import time
from collections import defaultdict
from typing import Optional, Dict, Any, Set

import zmq

# ---- Try to import GUIClient ----
_GUIClient = None
try:
    from pyoephys.interface._gui_client import GUIClient as _GUIClient  # type: ignore
except Exception:
    import importlib.util
    here = os.path.dirname(os.path.abspath(__file__))
    cand = os.path.join(here, "_gui_client.py")
    if os.path.exists(cand):
        spec = importlib.util.spec_from_file_location("user_gui_client", cand)
        if spec and spec.loader:
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)  # type: ignore
            if hasattr(mod, "GUIClient"):
                _GUIClient = getattr(mod, "GUIClient")


def human_mb_per_s(bps: float) -> str:
    return f"{bps/1_000_000:.2f} MB/s"


def send_gui_command(endpoint: str, command: str, **params) -> Optional[Dict[str, Any]]:
    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.REQ)
    sock.setsockopt(zmq.LINGER, 0)
    sock.setsockopt(zmq.RCVTIMEO, 2000)
    sock.setsockopt(zmq.SNDTIMEO, 2000)
    try:
        sock.connect(endpoint)
        payload = {"command": command}
        if params:
            payload.update(params)
        sock.send_json(payload)
        return sock.recv_json()
    except Exception as e:
        print(f"[bench] ZMQ command '{command}' failed: {e}")
        return None
    finally:
        sock.close(0)


def start_acq(use_http: bool, gui_host: str, cmd_endpoint: str) -> Optional[Dict[str, Any]]:
    if use_http and _GUIClient is not None:
        try:
            client = _GUIClient(host=gui_host)
            return client.start_acquisition()
        except Exception as e:
            print(f"[bench] GUIClient.start_acquisition failed: {e}")
    return send_gui_command(cmd_endpoint, "startAcquisition")


def stop_acq(use_http: bool, gui_host: str, cmd_endpoint: str) -> Optional[Dict[str, Any]]:
    if use_http and _GUIClient is not None:
        try:
            client = _GUIClient(host=gui_host)
            return client.stop_acquisition()
        except Exception as e:
            print(f"[bench] GUIClient.stop_acquisition failed: {e}")
    return send_gui_command(cmd_endpoint, "stopAcquisition")


def drain(socket, poller, seconds: float):
    if seconds <= 0:
        return
    deadline = time.time() + seconds
    drained = 0
    while time.time() < deadline:
        socks = dict(poller.poll(timeout=5))
        if socks.get(socket) == zmq.POLLIN:
            try:
                _ = socket.recv_multipart(flags=zmq.NOBLOCK)
                drained += 1
            except zmq.Again:
                pass
    print(f"[bench] Warmup drained ~{drained} messages in {seconds:.2f}s")


def run_benchmark(endpoint: str,
                  duration: float,
                  topic: bytes,
                  report_interval: float,
                  expect_ch: Optional[int],
                  expect_fs: Optional[float],
                  warmup: float,
                  rcvhwm: int,
                  conflate: bool,
                  csv_path: Optional[str]) -> None:
    ctx = zmq.Context.instance()
    sub = ctx.socket(zmq.SUB)
    sub.setsockopt(zmq.SUBSCRIBE, topic)
    sub.setsockopt(zmq.RCVHWM, rcvhwm)
    if conflate:
        try:
            sub.setsockopt(zmq.CONFLATE, 1)
            print("[bench] CONFLATE enabled on SUB (latest message only).")
        except Exception as e:
            print(f"[bench] WARNING: could not enable CONFLATE on SUB: {e}")
    sub.connect(endpoint)

    poller = zmq.Poller()
    poller.register(sub, zmq.POLLIN)

    if warmup > 0:
        drain(sub, poller, warmup)

    csv_file = None
    csv_writer = None
    chan_csv_file = None
    chan_csv_writer = None
    if csv_path:
        csv_file = open(csv_path, "w", newline="", encoding="utf-8")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow([
            "t_elapsed_s",
            "msg_rate_hz_overall", "byte_rate_Bps_overall", "ch_samp_rate_overall",
            "msg_rate_hz_window", "byte_rate_Bps_window", "ch_samp_rate_window",
            "total_msgs", "total_bytes", "total_ch_samples", "seq_gaps",
            "seen_channels", "last_sample_rate_hz"
        ])
        chan_csv_file = open(csv_path + ".channels.csv", "w", newline="", encoding="utf-8")
        chan_csv_writer = csv.writer(chan_csv_file)
        chan_csv_writer.writerow(["channel_num", "total_samples", "expected_samples", "percent_of_expected"])

    start = time.time()
    end = start + duration
    next_report = start + report_interval

    total_msgs = 0
    total_bytes = 0
    total_channel_samples = 0

    last_report_time = start
    last_msgs = 0
    last_bytes = 0
    last_chsamp = 0

    seq_prev = None
    seq_gaps = 0
    last_sample_rate = None
    seen_channels: Set[int] = set()
    per_channel_counts = defaultdict(int)

    print(f"[bench] SUB -> {endpoint} (topic={topic!r}); duration={duration}s; warmup={warmup}s; RCVHWM={rcvhwm}; conflate={conflate}")

    while True:
        now = time.time()
        if now >= end:
            break
        timeout_ms = max(1, int((end - now) * 1000))
        socks = dict(poller.poll(timeout=timeout_ms))
        if socks.get(sub) == zmq.POLLIN:
            frames = sub.recv_multipart()
            total_msgs += 1
            total_bytes += sum(len(f) for f in frames)

            if len(frames) >= 2:
                try:
                    header = json.loads(frames[1].decode("utf-8"))
                except Exception:
                    header = None

                if isinstance(header, dict) and "message_num" in header:
                    if seq_prev is not None and header["message_num"] != seq_prev + 1:
                        seq_gaps += 1
                    seq_prev = header["message_num"]

                if isinstance(header, dict) and header.get("type") == "data":
                    c = header.get("content", {})
                    num_samples = int(c.get("num_samples", 0))
                    channel_num = int(c.get("channel_num", -1))
                    if "sample_rate" in c:
                        try:
                            last_sample_rate = float(c.get("sample_rate"))
                        except Exception:
                            pass
                    if channel_num >= 0:
                        seen_channels.add(channel_num)
                        per_channel_counts[channel_num] += num_samples
                    if len(frames) >= 3 and num_samples > 0:
                        total_channel_samples += num_samples

        if time.time() >= next_report:
            t_now = time.time()
            elapsed = max(1e-6, t_now - start)
            window = max(1e-6, t_now - last_report_time)

            msg_rate = total_msgs / elapsed
            byte_rate = total_bytes / elapsed
            ch_samp_rate = total_channel_samples / elapsed

            d_msgs = total_msgs - last_msgs
            d_bytes = total_bytes - last_bytes
            d_chsamp = total_channel_samples - last_chsamp

            msg_rate_w = d_msgs / window
            byte_rate_w = d_bytes / window
            ch_samp_rate_w = d_chsamp / window

            est_ch = len(seen_channels) if seen_channels else None

            line = (f"[bench] t={elapsed:5.2f}s | "
                    f"{msg_rate:8.1f} msg/s (win {msg_rate_w:8.1f}) | "
                    f"{human_mb_per_s(byte_rate):>12} (win {human_mb_per_s(byte_rate_w)}) | "
                    f"{ch_samp_rate:,.0f} ch-samp/s (win {ch_samp_rate_w:,.0f})")
            if est_ch is not None:
                line += f" | seen_ch={est_ch}"
            if last_sample_rate:
                line += f" | fs~{last_sample_rate:g}"
            if seq_gaps:
                line += f" | seq_gaps={seq_gaps}"
            print(line)

            if csv_writer is not None:
                csv_writer.writerow([
                    f"{elapsed:.3f}",
                    f"{msg_rate:.3f}", f"{byte_rate:.3f}", f"{ch_samp_rate:.3f}",
                    f"{msg_rate_w:.3f}", f"{byte_rate_w:.3f}", f"{ch_samp_rate_w:.3f}",
                    total_msgs, total_bytes, total_channel_samples, seq_gaps,
                    len(seen_channels), f"{last_sample_rate or ''}"
                ])

            last_report_time = t_now
            last_msgs = total_msgs
            last_bytes = total_bytes
            last_chsamp = total_channel_samples

            next_report += report_interval

    elapsed = max(1e-6, time.time() - start)
    msg_rate = total_msgs / elapsed
    byte_rate = total_bytes / elapsed
    ch_samp_rate = total_channel_samples / elapsed
    est_ch = len(seen_channels) if seen_channels else None

    print("\n=== Benchmark Summary ===")
    print(f"Endpoint:        {endpoint}")
    print(f"Duration:        {duration:.2f} s (warmup {warmup:.2f} s)")
    print(f"Total messages:  {total_msgs}")
    print(f"Total bytes:     {total_bytes:,}  ({human_mb_per_s(byte_rate)} sustained)")
    print(f"Msg rate:        {msg_rate:,.1f} msg/s")
    print(f"Ch-samples/sec:  {ch_samp_rate:,.0f}")
    if est_ch is not None:
        print(f"Channels seen:   {est_ch}")
    if last_sample_rate is not None:
        print(f"Sample rate:     ~{last_sample_rate:g} Hz")
    if seq_gaps:
        print(f"Msg gaps:        {seq_gaps}")

    expect_per_chan = (expect_fs * duration) if expect_fs else None
    if per_channel_counts:
        print("\n--- Per-channel totals ---")
        for ch in sorted(per_channel_counts.keys()):
            total = per_channel_counts[ch]
            if expect_per_chan:
                pct = 100.0 * total / max(1.0, expect_per_chan)
                print(f"ch {ch:3d} -> {total:,} samp ({pct:5.1f}% of expected {int(expect_per_chan):,})")
                if chan_csv_writer:
                    chan_csv_writer.writerow([ch, total, int(expect_per_chan), f"{pct:.2f}"])
            else:
                print(f"ch {ch:3d} -> {total:,} samp")
                if chan_csv_writer:
                    chan_csv_writer.writerow([ch, total, "", ""])

    if expect_ch and expect_fs:
        target = expect_ch * expect_fs
        print(f"Target ch-samp/s: {target:,.0f} (expect_ch={expect_ch}, expect_fs={expect_fs})")
        if ch_samp_rate >= target:
            print("Result: ✅ Sustained rate meets or exceeds target.")
        else:
            pct = 100.0 * ch_samp_rate / max(1.0, target)
            print(f"Result: ⚠️  Sustained rate is {pct:.1f}% of target. Consider raising block size or RCVHWM.")

    if csv_file:
        csv_file.flush()
        csv_file.close()
    if chan_csv_file:
        chan_csv_file.flush()
        chan_csv_file.close()


def parse_args():
    ap = argparse.ArgumentParser(description="Open Ephys ZMQ Benchmark with GUIClient control")
    ap.add_argument("--endpoint", default="tcp://127.0.0.1:5556", help="ZMQ SUB endpoint for data")
    ap.add_argument("--cmd-endpoint", default="tcp://127.0.0.1:5557", help="ZMQ REQ endpoint for commands (fallback)")
    ap.add_argument("--gui-host", default="127.0.0.1", help="Host/IP for GUI HTTP control via GUIClient")
    ap.add_argument("--use-gui-http", action="store_true", help="Use GUIClient (HTTP) to start/stop acquisition")
    ap.add_argument("--auto-acq", action="store_true", help="Auto start/stop acquisition around the measurement")
    ap.add_argument("--duration", type=float, default=10.0, help="Benchmark duration (s)")
    ap.add_argument("--warmup", type=float, default=1.0, help="Warmup drain time before measuring (s)")
    ap.add_argument("--report-interval", type=float, default=2.0, help="Live report interval (s)")
    ap.add_argument("--topic", default="", help="ZMQ SUB topic (default empty = all)")
    ap.add_argument("--expect-ch", type=int, default=None, help="Expected total channels (for headroom check)")
    ap.add_argument("--expect-fs", type=float, default=None, help="Expected per-channel sample rate Hz (for headroom check)")
    ap.add_argument("--rcvhwm", type=int, default=4096, help="ZMQ SUB RCVHWM (high-water mark)")
    ap.add_argument("--conflate", action="store_true", help="Enable ZMQ.CONFLATE (latest message only); may hide drops")
    ap.add_argument("--csv", dest="csv_path", default=None, help="Optional CSV path for logging (plus .channels.csv)")
    return ap.parse_args()


def main():
    args = parse_args()

    if args.auto_acq:
        reply = start_acq(args.use_gui_http, args.gui_host, args.cmd_endpoint)
        print(f"[bench] start acquisition -> {reply}")

    try:
        run_benchmark(
            endpoint=args.endpoint,
            duration=args.duration,
            topic=args.topic.encode("utf-8"),
            report_interval=args.report_interval,
            expect_ch=args.expect_ch,
            expect_fs=args.expect_fs,
            warmup=args.warmup,
            rcvhwm=args.rcvhwm,
            conflate=args.conflate,
            csv_path=args.csv_path,
        )
    finally:
        if args.auto_acq:
            reply = stop_acq(args.use_gui_http, args.gui_host, args.cmd_endpoint)
            print(f"[bench] stop acquisition -> {reply}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# oe_plus_imu_monitor.py
# Monitor Open Ephys ZMQ data alongside the Sleeve IMU orientation.

import argparse
import time
from typing import Dict

# Expect zmq_client.py (the class we built) and sleeveimu.py (your IMU client) on PYTHONPATH
from zmq_client import ZMQClient
from sleeveimu import SleeveIMUClient


def main():
    ap = argparse.ArgumentParser(description="Open Ephys + Sleeve IMU monitor")
    # Open Ephys / ZMQClient
    ap.add_argument("--oe-endpoint", default="tcp://127.0.0.1:5556", help="ZMQ SUB endpoint for Open Ephys data")
    ap.add_argument("--buffer-seconds", type=float, default=10.0, help="Per-channel ring buffer duration (s)")
    ap.add_argument("--rcvhwm", type=int, default=50000, help="SUB socket RCVHWM (increase for burst tolerance)")
    ap.add_argument("--rcvbuf", type=int, default=4_194_304, help="OS receive buffer hint (ZMQ_RCVBUF bytes)")
    ap.add_argument("--conflate", action="store_true", help="Keep only latest message (hides drops; lowest latency)")
    # IMU
    ap.add_argument("--imu-host", default="192.168.4.1", help="Sleeve IMU host (Pico W IP or hostname)")
    ap.add_argument("--imu-port", type=int, default=5555, help="Sleeve IMU port")
    ap.add_argument("--imu-transport", default="UDP", choices=["UDP", "TCP"], help="Sleeve IMU transport")
    # Printing / preview
    ap.add_argument("--print-interval", type=float, default=1.0, help="Print period (s)")
    ap.add_argument("--preview-n", type=int, default=0, help="Number of channels to preview (0 = none)")
    ap.add_argument("--preview-window-sec", type=float, default=0.5, help="Window (s) fetched for channel preview")
    args = ap.parse_args()

    # Start ZMQ client
    zmq = ZMQClient(
        endpoint=args.oe_endpoint,
        buffer_seconds=args.buffer_seconds,
        rcvhwm=args.rcvhwm,
        conflate=args.conflate,
        rcvbuf_bytes=args.rcvbuf,
    )
    zmq.start()
    zmq.gui.start_acquisition()

    # Start IMU client
    imu = SleeveIMUClient(host=args.imu_host, port=args.imu_port, transport=args.imu_transport, auto_start=True)
    imu.wait_connected(timeout=3.0)

    print(f"[run] ZMQ -> {args.oe_endpoint} (rcvhwm={args.rcvhwm}, conflate={args.conflate}) | "
          f"IMU -> {args.imu_transport}@{args.imu_host}:{args.imu_port}")
    print("[run] Ctrl+C to stop")

    try:
        while True:
            time.sleep(args.print_interval)

            # ZMQ throughput window
            stats: Dict[str, object] = zmq.get_stats(reset_window=True)
            fs = float(stats["sample_rate_hz"] or 0.0)
            ch_samp_h = stats["human_window"]["ch_samp_rate"]
            seen_ch = len(stats["seen_channels"]) if stats.get("seen_channels") else 0

            # IMU orientation (roll, pitch, yaw) degrees
            rpy = imu.get_rpy_deg()
            if rpy is None:
                imu_txt = "IMU rpy=NA"
            else:
                r, p, y = rpy
                imu_txt = f"IMU rpy=({r:.2f}, {p:.2f}, {y:.2f})°"

            # Optional channel preview
            preview = {}
            if args.preview_n > 0:
                chs = zmq.channels()[:args.preview_n]
                latest = zmq.get_latest_window(args.preview_window_sec, channels=chs)
                preview = {ch: arr.shape for ch, arr in latest.items()}

            print(f"[tick] fs~{fs:.1f} Hz | ch={seen_ch} | {ch_samp_h} | {imu_txt} | preview_shapes={preview}")
    except KeyboardInterrupt:
        pass
    finally:
        try:
            imu.stop()
        finally:
            zmq.stop()
            zmq.gui.stop_acquisition()


if __name__ == "__main__":
    main()

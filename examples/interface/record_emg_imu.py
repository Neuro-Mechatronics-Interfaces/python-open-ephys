#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# record_oe_imu.py — record 10s EMG (ZMQClient) + IMU to CSV (timestamps, EMG channels, roll, pitch, yaw)

import argparse
import csv
import sys
import time
from typing import List, Dict

import numpy as np

from zmq_client import ZMQClient
from sleeveimu import SleeveIMUClient


def _interp_unwrapped(x_src: np.ndarray, y_deg_src: np.ndarray, x_tgt: np.ndarray) -> np.ndarray:
    if x_src.size < 2:
        return np.full_like(x_tgt, np.nan, dtype=float)
    y_rad = np.deg2rad(y_deg_src.astype(float))
    y_unw = np.unwrap(y_rad)
    y_tgt_unw = np.interp(x_tgt, x_src, y_unw, left=np.nan, right=np.nan)
    y_tgt_deg = np.rad2deg(y_tgt_unw)
    return ((y_tgt_deg + 180.0) % 360.0) - 180.0


def _interp_linear(x_src: np.ndarray, y_src: np.ndarray, x_tgt: np.ndarray) -> np.ndarray:
    if x_src.size < 2:
        return np.full_like(x_tgt, np.nan, dtype=float)
    return np.interp(x_tgt, x_src, y_src, left=np.nan, right=np.nan)


def _parse_channel_list(arg: str, observed: List[int]) -> List[int]:
    obs_set = set(observed)
    s = arg.strip().lower()
    if s in ("", "all", "*"):
        return sorted(observed)
    out = []
    for part in arg.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            try:
                lo = int(lo); hi = int(hi)
                for ch in range(min(lo, hi), max(lo, hi) + 1):
                    if ch in obs_set:
                        out.append(ch)
            except Exception:
                pass
        else:
            try:
                ch = int(part)
                if ch in obs_set:
                    out.append(ch)
            except Exception:
                pass
    return sorted(set(out))


def main():
    ap = argparse.ArgumentParser(description="Record EMG (Open Ephys via ZMQ) + IMU to CSV")
    ap.add_argument("--oe-endpoint", default="tcp://127.0.0.1:5556")
    ap.add_argument("--rcvhwm", type=int, default=50000)
    ap.add_argument("--rcvbuf", type=int, default=4_194_304)
    ap.add_argument("--buffer-seconds", type=float, default=20.0)
    ap.add_argument("--conflate", action="store_true")
    ap.add_argument("--channels", default="all")
    ap.add_argument("--auto-acq", action="store_true")
    ap.add_argument("--imu-host", default="192.168.4.1")
    ap.add_argument("--imu-port", type=int, default=5555)
    ap.add_argument("--imu-transport", default="UDP", choices=["UDP", "TCP"])
    ap.add_argument("--duration", type=float, default=10.0)
    ap.add_argument("--csv", dest="csv_path", default="record_oe_imu.csv")
    ap.add_argument("--csv-fs", type=int, default=1000, help="Target EMG rate for CSV (-1 = native rate)")
    args = ap.parse_args()

    zmq = ZMQClient(
        endpoint=args.oe_endpoint,
        buffer_seconds=max(args.buffer_seconds, args.duration + 5.0),
        rcvhwm=args.rcvhwm,
        conflate=args.conflate,
        rcvbuf_bytes=args.rcvbuf,
    )
    zmq.start()
    zmq.gui.start_acquisition()

    imu = SleeveIMUClient(host=args.imu_host, port=args.imu_port, transport=args.imu_transport, auto_start=True)
    imu.wait_connected(timeout=3.0)

    imu_times = []
    imu_r = []; imu_p = []; imu_y = []

    stop_ts = time.time() + args.duration
    print(f"[rec] Capturing for {args.duration:.2f} s")
    try:
        while time.time() < stop_ts:
            rpy = imu.get_rpy_deg()
            if rpy is not None:
                r, p, y = rpy
                imu_times.append(time.time()); imu_r.append(float(r)); imu_p.append(float(p)); imu_y.append(float(y))
            time.sleep(0.005)
    except KeyboardInterrupt:
        print("[rec] Aborted by user.")

    zmq.gui.stop_acquisition()

    fs = float(zmq.sample_rate or 0.0)
    if fs <= 0:
        print("[rec] ERROR: Could not determine EMG sample rate from stream.")
        return 1

    observed = zmq.channels()
    use_channels = _parse_channel_list(args.channels, observed)
    if not use_channels:
        print(f"[rec] ERROR: No channels selected. Observed={observed[:8]}... total={len(observed)}")
        return 1

    win = zmq.get_latest_window(args.duration, channels=use_channels)
    lengths = [arr.size for arr in win.values()]
    n = min(lengths) if lengths else 0
    if n == 0:
        print("[rec] ERROR: No EMG samples captured.")
        return 1

    emg_mat = np.stack([win[ch][-n:] for ch in use_channels], axis=1)
    n_ch = emg_mat.shape[1]

    t_end = time.time()
    t_start = t_end - (n - 1) / fs
    t_emg = t_start + np.arange(n) / fs

    target_fs = args.csv_fs
    if target_fs is None or target_fs <= 0:
        target_fs = int(round(fs))
    dec = max(1, int(round(fs / float(target_fs))))
    if dec > 1:
        t_emg = t_emg[::dec]
        emg_mat = emg_mat[::dec, :]
        eff_fs = fs / dec
    else:
        eff_fs = fs

    imu_times_np = np.array(imu_times, dtype=float)
    imu_r_np = np.array(imu_r, dtype=float) if imu_r else np.empty(0)
    imu_p_np = np.array(imu_p, dtype=float) if imu_p else np.empty(0)
    imu_y_np = np.array(imu_y, dtype=float) if imu_y else np.empty(0)

    roll_i = _interp_linear(imu_times_np, imu_r_np, t_emg) if imu_r_np.size else np.full_like(t_emg, np.nan)
    pitch_i = _interp_linear(imu_times_np, imu_p_np, t_emg) if imu_p_np.size else np.full_like(t_emg, np.nan)
    yaw_i = _interp_unwrapped(imu_times_np, imu_y_np, t_emg) if imu_y_np.size else np.full_like(t_emg, np.nan)

    t_rel = t_emg - t_emg[0]
    header = ["t_s"] + [f"ch{ch}" for ch in use_channels] + ["roll_deg", "pitch_deg", "yaw_deg"]

    print(f"[rec] Writing CSV -> {args.csv_path} @ ~{eff_fs:.1f} Hz ({len(t_rel)} rows, {len(header)} cols)...")
    with open(args.csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for i in range(len(t_rel)):
            row = [f"{t_rel[i]:.6f}"]
            row.extend(f"{v:.6f}" for v in emg_mat[i, :])
            row.extend((f"{roll_i[i]:.3f}", f"{pitch_i[i]:.3f}", f"{yaw_i[i]:.3f}"))
            w.writerow(row)

    print("[rec] Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

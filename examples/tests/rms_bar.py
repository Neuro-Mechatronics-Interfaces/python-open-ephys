#!/usr/bin/env python3
import sys, argparse, time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.signal import iirnotch, butter, sosfiltfilt, tf2sos

from pyoephys.interface import ZMQClient
from pyoephys.io import parse_numeric_args


# ---------- filters ----------
def design_filters(fs, notch_freq=60.0, notch_Q=30.0, bp_low=10.0, bp_high=500.0, order=4):
    b_notch, a_notch = iirnotch(w0=notch_freq, Q=notch_Q, fs=fs)
    sos_notch = tf2sos(b_notch, a_notch)
    sos_bp = butter(order, [bp_low / (0.5 * fs), bp_high / (0.5 * fs)], btype="band", output="sos")
    return sos_notch, sos_bp

def apply_filters_sos(x, sos_notch, sos_bp):
    # x: (C, N)
    y = sosfiltfilt(sos_notch, x, axis=1)
    y = sosfiltfilt(sos_bp,    y, axis=1)
    return y

def compute_rms_over_window(win_2d):
    return np.sqrt(np.mean(win_2d * win_2d, axis=1) + 1e-12)


def main():
    ap = argparse.ArgumentParser(description="Live EMG RMS per channel (ring-buffered)")
    ap.add_argument("--channels", nargs="+", default=["0:128"],
                    help="Channels to plot: e.g. 0 1 2 3  or 0:64  or all")
    ap.add_argument("--window_sec", type=float, default=0.5, help="RMS window (seconds)")
    ap.add_argument("--downsample", type=int, default=1, help="Optional time downsample for filter/RMS")
    ap.add_argument("--noise_threshold", type=float, default=100.0, help="Color red if RMS exceeds this")
    ap.add_argument("--host_ip", type=str, default="127.0.0.1")
    ap.add_argument("--data_port", type=str, default="5556")
    ap.add_argument("--notch", type=float, default=60.0, help="Notch frequency (Hz); 0 disables notch")
    ap.add_argument("--notch_Q", type=float, default=30.0)
    ap.add_argument("--bp_low", type=float, default=10.0)
    ap.add_argument("--bp_high", type=float, default=500.0)
    ap.add_argument("--order", type=int, default=4)
    ap.add_argument("--ui_ms", type=int, default=250, help="UI update period (ms)")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    # --- parse channels (replace your whole block with this) ---
    # Normalize to a list even if argparse handed us a bare string
    raw = args.channels if isinstance(args.channels, (list, tuple)) else [str(args.channels)]
    want_all = (len(raw) == 1 and str(raw[0]).lower() == "all")

    if want_all:
        channels = []
    else:
        # use parse_numeric_args on the normalized list
        channels = parse_numeric_args(raw)

    # --- ZMQ client ---
    client = ZMQClient(host_ip=args.host_ip, data_port=args.data_port, verbose=args.verbose)
    if not want_all and channels:
        client.set_channel_index(channels)
    client.start()
    try:
        client.ready_event.wait(timeout=5.0)
    except Exception:
        pass

    # sampling rate
    fs = getattr(client, "fs", None)
    if not fs or fs <= 0:
        fs_est = getattr(client, "fs_estimate", lambda: None)()
        fs = fs_est if fs_est else 2000.0
    if args.verbose:
        print(f"[RMS] fs={fs:.1f} Hz, window={args.window_sec:.3f}s")

    # sizes
    if want_all:
        C = getattr(client, "N_channels", None)
        if not C or C <= 0:
            C = 16  # fallback
    else:
        C = len(channels) if channels else 1
    N = max(1, int(round(args.window_sec * fs)))
    ring = np.zeros((C, N), dtype=np.float32)  # our rolling window
    have = 0  # how many valid columns currently buffered (<= N)

    # filters
    if args.notch > 0:
        sos_notch, sos_bp = design_filters(fs, notch_freq=args.notch, notch_Q=args.notch_Q,
                                           bp_low=args.bp_low, bp_high=args.bp_high, order=args.order)
    else:
        b, a = [1.0], [1.0]
        sos_notch = tf2sos(b, a)
        sos_bp = butter(args.order, [args.bp_low / (0.5 * fs), args.bp_high / (0.5 * fs)],
                        btype="band", output="sos")

    # matplotlib
    fig, ax = plt.subplots(figsize=(16, 6))
    chan_idx = np.arange(1, C + 1)
    bars = ax.bar(chan_idx, np.zeros(C), color="dodgerblue")
    ax.set_xlabel("Channel")
    ax.set_ylabel("RMS amplitude")
    ax.set_title(f"Live EMG RMS per Channel ({args.window_sec:.2f}s window)")
    ax.grid(True, axis="y")
    fig.tight_layout()
    status = ax.text(0.01, 0.98, "", transform=ax.transAxes, va="top", ha="left", fontsize=9)

    # dynamic y-scaling helper
    def autoscale_y(max_rms):
        # scale both up and down sensibly; never below 1.0
        lo, hi = ax.get_ylim()
        target = max(1.0, float(max_rms) * 1.2)
        # small smoothing to avoid flicker
        new_hi = 0.85 * hi + 0.15 * target
        if abs(new_hi - hi) / max(1.0, hi) > 0.02:
            ax.set_ylim(0, new_hi)

    # animation update: drain new, update ring, compute RMS, draw
    def update(_frame):
        nonlocal ring, have
        try:
            _, block = client.drain_new()  # (C, K) or (None, None)
        except Exception:
            block = None

        if block is not None and hasattr(block, "shape"):
            blk = np.asarray(block)
            # ensure (C,K)
            if blk.ndim == 1:
                blk = blk.reshape(C, -1)
            elif blk.shape[0] != C and blk.shape[1] == C:
                blk = blk.T

            if blk.size > 0:
                # optional downsample along time axis
                if args.downsample > 1:
                    blk = blk[:, ::args.downsample]

                K = blk.shape[1]
                if K >= N:
                    ring = blk[:, -N:]
                    have = N
                else:
                    # shift-left then append at end
                    if K > 0:
                        ring[:, :-K] = ring[:, K:]
                        ring[:, -K:] = blk
                        have = min(N, have + K)

        # enough data? still compute even if not full
        win = ring[:, -have:] if have > 0 else ring[:, -1:]
        try:
            yf = apply_filters_sos(win, sos_notch, sos_bp)
            rms = compute_rms_over_window(yf)
        except Exception:
            rms = compute_rms_over_window(win)

        # update bars
        for bar, val in zip(bars, rms):
            bar.set_height(float(val))
            bar.set_color("red" if val > args.noise_threshold else "dodgerblue")

        # y autoscale & status text
        autoscale_y(np.nanmax(rms) if rms.size else 0.0)
        status.set_text(f"buf: {have}/{N}  fs:{fs:.0f}Hz  maxRMS:{np.nanmax(rms):.3g}")

        return bars

    ani = FuncAnimation(fig, update, interval=args.ui_ms, blit=False, cache_frame_data=False)
    plt.show()


if __name__ == "__main__":
    main()

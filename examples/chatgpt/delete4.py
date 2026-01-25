#!/usr/bin/env python3
"""
0_compare_lsl_to_oebin_lslclient.py

Capture ~30s from an LSL EMG stream using YOUR pyoephys.interface.LSLClient,
align it to a reference OEBin recording by robust correlation, report the
best-alignment index/time, and plot comparisons.

Usage (example)
---------------
# Terminal 1: publish your OEBin over LSL
# python your_publisher.py --root_dir ... --lsl

# Terminal 2: run this comparison
python 0_compare_lsl_to_oebin_lslclient.py \
  --root_dir "G:/Shared drives/NML_shared/DataShare/HDEMG Human Healthy/.../2025_07_31" \
  --stream_name EMG \
  --duration_s 30 \
  --plot_channels 2 3 4 \
  --top_k 5 \
  --verbose
"""

from __future__ import annotations
import argparse
import logging
import os
import time
from typing import List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt

# Your package
from pyoephys.io import load_oebin_file
from pyoephys.interface import LSLClient


# -------------------------- LSL capture (via your client) --------------------------

def capture_with_lslclient(duration_s: float,
                           stream_name: str | None = None,
                           stream_type: str | None = None,
                           channels: Optional[List[int]] = None,
                           verbose: bool = False):
    """
    Grabs ~duration_s seconds from your LSL stream using your LSLClient.
    Returns (C x T array, fs_est).
    """
    if (stream_name is None) == (stream_type is None):
        raise ValueError("Provide exactly one of --stream_name or --stream_type")

    # Small cushion on the rolling window to ensure we can fill it
    client = LSLClient(stream_name=stream_name,
                       stream_type=stream_type,
                       channels=channels,
                       window_secs=float(duration_s) + 0.5,
                       auto_start=True,
                       verbose=verbose)

    try:
        t0 = time.time()
        t_rel, y = None, None

        # Wait until the window is ~filled (earliest relative time ~ -duration_s)
        while True:
            t_rel, y = client.latest()  # t_rel shape (M,), y shape (C, M), t_rel ends near 0
            if t_rel is not None and y is not None and t_rel.size > 2:
                coverage = -float(t_rel[0])  # seconds covered
                if coverage >= 0.98 * duration_s:
                    break
            if (time.time() - t0) > (duration_s * 3 + 5):
                logging.warning("Timed out waiting for LSL window to fill; proceeding with current buffer.")
                break
            time.sleep(0.05)

        # Sample rate: use nominal if known, otherwise estimate from timestamps
        fs_eff = float(client.fs_estimate(n_last=min(8192, t_rel.size)))

        # Final grab
        t_rel, y = client.latest()
        if t_rel is None or y is None or y.size == 0:
            raise RuntimeError("No LSL data received from LSLClient.")

        return y.astype(np.float32), float(fs_eff)

    finally:
        client.stop()


# -------------------------- Utilities: resample & envelope --------------------------

def resample_to_fs(X: np.ndarray, fs_from: float, fs_to: float) -> np.ndarray:
    """Resample (C,T) from fs_from to fs_to using linear interpolation."""
    if abs(fs_from - fs_to) < 1e-9:
        return X
    C, T = X.shape
    dur = T / fs_from
    t_from = np.linspace(0.0, dur, T, endpoint=False)
    T_to = max(1, int(round(dur * fs_to)))
    t_to = np.linspace(0.0, dur, T_to, endpoint=False)
    Y = np.empty((C, T_to), dtype=X.dtype)
    for c in range(C):
        Y[c] = np.interp(t_to, t_from, X[c])
    logging.info("Resampled from %.3f Hz (%d) -> %.3f Hz (%d)", fs_from, T, fs_to, T_to)
    return Y


def envelope_ma(X: np.ndarray, fs: float, win_s: float = 0.125) -> np.ndarray:
    """
    Simple rectified+moving-average envelope (C,T).
    win_s=0.125s ~ gentle LPF (~8 Hz-ish envelope feel).
    """
    K = max(1, int(round(win_s * fs)))
    if K == 1:
        return np.abs(X)
    kernel = np.ones(K, dtype=np.float32) / K
    Y = np.empty_like(X)
    for c in range(X.shape[0]):
        # full convolution then trim to same length (causal MA; no filtfilt)
        y = np.convolve(np.abs(X[c]), kernel, mode="same")
        Y[c] = y
    return Y


def resample_linear(X: np.ndarray, fs_from: float, fs_to: float) -> np.ndarray:
    """Alias of resample_to_fs for readability in coarse path."""
    return resample_to_fs(X, fs_from, fs_to)


# -------------------------- Normalized sliding correlation (NCC) --------------------------

def _next_pow2(n: int) -> int:
    m = 1
    while m < n:
        m <<= 1
    return m


def _sliding_ncc_1d(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Z-normalized sliding cross-correlation of query y (len M) over reference x (len N).
    Returns NCC of length N-M+1 in [-1, 1].

    Efficient via FFT for dot products + running mean/std for x windows.
    """
    N, M = x.size, y.size
    if N < M:
        raise ValueError("x must be >= y in length")

    # z-score y
    y = (y - y.mean()) / (y.std() + 1e-12)

    # Dot products of x with reversed y via FFT (valid part)
    nfft = _next_pow2(N + M)
    X = np.fft.rfft(x, nfft)
    Y = np.fft.rfft(y[::-1], nfft)
    dots = np.fft.irfft(X * Y, nfft)[:N + M - 1]
    dots = dots[M - 1:N]  # length N-M+1

    # Running mean/std for x windows of size M
    csum = np.cumsum(np.r_[0.0, x])
    csum2 = np.cumsum(np.r_[0.0, x * x])
    win_sum = csum[M:] - csum[:-M]
    win_sum2 = csum2[M:] - csum2[:-M]
    mu = win_sum / M
    var = np.maximum(win_sum2 / M - mu * mu, 1e-12)
    std = np.sqrt(var)

    # y is z-scored so std_y = 1
    ncc = dots / (std * M)
    return np.clip(ncc, -1.0, 1.0)


def _sliding_ncc_multich(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Average NCC across channels (X: CxN, Y: CxM)."""
    C, N = X.shape
    _, M = Y.shape
    acc = np.zeros(N - M + 1, dtype=np.float64)
    for c in range(C):
        acc += _sliding_ncc_1d(X[c], Y[c])
    return acc / C


# -------------------------- Coarse-to-fine best-lag search --------------------------

def mean_pearson_at(ref: np.ndarray, query: np.ndarray, start: int, allow_sign_flip: bool) -> float:
    """
    Mean Pearson r across channels at a given start index (ref segment vs query).
    If allow_sign_flip=True, per-channel sign is chosen to maximize r.
    """
    C, M = query.shape
    seg = ref[:, start:start + M]
    # z-score both
    Sz = (seg - seg.mean(axis=1, keepdims=True)) / (seg.std(axis=1, keepdims=True) + 1e-12)
    Qz = (query - query.mean(axis=1, keepdims=True)) / (query.std(axis=1, keepdims=True) + 1e-12)
    # per-channel correlation = mean of element-wise product
    r_per_ch = (Sz * Qz).mean(axis=1)
    if allow_sign_flip:
        r_per_ch = np.maximum(r_per_ch, -(r_per_ch))
    return float(r_per_ch.mean())


def coarse_to_fine_best_lag(X_ref: np.ndarray,
                            Y_query: np.ndarray,
                            fs: float,
                            allow_sign_flip: bool = True,
                            coarse_fs: float = 200.0,
                            envelope_win_s: float = 0.125,
                            top_k: int = 5,
                            refine_pad_s: float = 1.0):
    """
    1) Envelope + downsample -> coarse NCC (top-K peaks)
    2) Refine each candidate at full rate around ±refine_pad_s using mean Pearson r
    Returns: best_lag, best_r, candidates(list of (lag, r)), coarse_ncc (for plotting)
    """
    C, N = X_ref.shape
    _, M = Y_query.shape

    # --- Coarse: envelope + downsample to coarse_fs ---
    X_env = envelope_ma(X_ref, fs, win_s=envelope_win_s)
    Y_env = envelope_ma(Y_query, fs, win_s=envelope_win_s)
    Xc = resample_linear(X_env, fs, coarse_fs)
    Yc = resample_linear(Y_env, fs, coarse_fs)
    fs_c = float(coarse_fs)
    logging.info("Coarse stage: envelope win=%.3fs, downsample to %.1f Hz", envelope_win_s, coarse_fs)

    if Xc.shape[1] < Yc.shape[1]:
        raise ValueError("Coarse reference shorter than coarse query.")

    ncc = _sliding_ncc_multich(Xc, Yc)

    # Top-K candidate coarse indices
    K = min(top_k, ncc.size)
    cand_c_idx = np.argpartition(-ncc, K - 1)[:K]
    cand_c_idx = cand_c_idx[np.argsort(-ncc[cand_c_idx])]

    # Map to full-rate indices
    cand_full_idx = np.clip((cand_c_idx * (fs / fs_c)).astype(int), 0, N - M)

    # --- Fine: full-rate search around each candidate ---
    pad = int(round(refine_pad_s * fs))
    candidates: List[Tuple[int, float]] = []
    best_lag, best_r = 0, -np.inf
    logging.info("Fine stage: %d candidate(s), ±%.3fs window", len(cand_full_idx), refine_pad_s)

    for base in cand_full_idx:
        lo = max(0, base - pad)
        hi = min(N - M, base + pad)
        # quick vector probe every few samples to narrow it down
        probe = np.arange(lo, hi + 1, max(1, int(fs // 400)))  # ~2.5 ms steps at 1 kHz
        scores = [mean_pearson_at(X_ref, Y_query, i, allow_sign_flip) for i in probe]
        i_star = probe[int(np.argmax(scores))]
        # refine locally at single-sample resolution around i_star
        fine = np.arange(max(lo, i_star - 8), min(hi, i_star + 8) + 1)
        s_fine = [mean_pearson_at(X_ref, Y_query, i, allow_sign_flip) for i in fine]
        i_best = int(fine[int(np.argmax(s_fine))])
        r_best = float(np.max(s_fine))

        candidates.append((i_best, r_best))
        if r_best > best_r:
            best_lag, best_r = i_best, r_best

    candidates.sort(key=lambda x: -x[1])
    return best_lag, best_r, candidates, ncc


# -------------------------- Plotting --------------------------

def make_plots(corr_curve: np.ndarray, best_lag: int,
               oebin_seg: np.ndarray, lsl_seg: np.ndarray,
               fs: float, plot_channels: List[int], out_path: Optional[str]):
    """Two panels: (1) coarse NCC vs lag, (2) overlay of a few channels (z-scored)."""
    fig = plt.figure(figsize=(12, 7), constrained_layout=True)

    # Panel 1: coarse NCC curve
    ax1 = fig.add_subplot(2, 1, 1)
    lags = np.arange(len(corr_curve))
    ax1.plot(lags, corr_curve)
    ax1.axvline(best_lag, linestyle='--')
    ax1.set_title('Coarse multi-channel NCC (envelope, downsampled)')
    ax1.set_xlabel('Lag (coarse samples)')
    ax1.set_ylabel('NCC (avg across channels)')

    # Panel 2: overlay a few channels at full rate
    ax2 = fig.add_subplot(2, 1, 2)
    t = np.arange(lsl_seg.shape[1]) / fs
    for ch in plot_channels:
        if ch < 0 or ch >= oebin_seg.shape[0]:
            continue
        a = oebin_seg[ch]
        b = lsl_seg[ch]
        a = (a - a.mean()) / (a.std(ddof=1) + 1e-12)
        b = (b - b.mean()) / (b.std(ddof=1) + 1e-12)
        ax2.plot(t, a, label=f"OEBin ch{ch}")
        ax2.plot(t, b, alpha=0.7, label=f"LSL ch{ch}")
    ax2.set_title(f"Aligned overlay (best lag = {best_lag} samples)")
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Z-score amplitude')
    ax2.legend(ncol=2)

    if out_path:
        os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
        fig.savefig(out_path, dpi=150)
        logging.info("Saved figure to %s", out_path)
    else:
        plt.show()


def local_lag_slope(ref, qry, fs, best_lag, win_s=3.0, hop_s=1.0, search_pad_s=0.5,
                    coarse_fs=200.0, envelope_win_s=0.125):
    """
    Estimate linear drift between ref (CxN) and qry (CxM) around best_lag.
    Returns slope (samples per second), intercept (samples), and points (t, delta).
    """
    # envelope + downsample both to coarse_fs for speed/robustness
    X_env = envelope_ma(ref, fs, win_s=envelope_win_s)
    Y_env = envelope_ma(qry, fs, win_s=envelope_win_s)
    Xc = resample_linear(X_env, fs, coarse_fs)
    Yc = resample_linear(Y_env, fs, coarse_fs)
    fsc = float(coarse_fs)

    M = Yc.shape[1]
    win = int(round(win_s * fsc))
    hop = int(round(hop_s * fsc))
    pad = int(round(search_pad_s * fsc))

    deltas = []
    times  = []
    start_refs = []

    # for each query window, search in ref around (best_lag + q_idx) ± pad
    q0s = np.arange(0, max(1, M - win), hop, dtype=int)
    for q0 in q0s:
        q1 = q0 + win
        if q1 > M:
            break
        # map query index to ref coarse index
        r_center = int(round(best_lag * (fsc / fs))) + q0
        r0 = max(0, r_center - pad)
        r1 = min(Xc.shape[1] - win, r_center + pad)
        if r1 <= r0:
            continue

        # compute NCC across small search region
        segQ = Yc[:, q0:q1]
        searchN = r1 - r0 + 1
        scores = np.empty(searchN)
        for k, rs in enumerate(range(r0, r1 + 1)):
            segR = Xc[:, rs:rs+win]
            scores[k] = _sliding_ncc_multich(segR, segQ)[0]  # same-length, k=0
        kbest = int(np.argmax(scores))
        rbest = r0 + kbest

        delta = (rbest - (int(round(best_lag*(fsc/fs))) + q0))   # ref - (expected)
        deltas.append(delta)
        times.append(q0 / fsc)      # seconds into the query snippet
        start_refs.append(rbest)

    if len(times) < 2:
        return 0.0, float(best_lag), (np.array(times), np.array(deltas))

    t = np.asarray(times)
    d = np.asarray(deltas)
    # fit delta ≈ a + b * t  (b in coarse samples/sec)
    A = np.c_[np.ones_like(t), t]
    coeff, _, _, _ = np.linalg.lstsq(A, d, rcond=None)
    a, b = coeff
    # convert slope back to full-rate samples/sec
    slope = b * (fs / fsc)
    intercept = best_lag + a * (fs / fsc)
    return float(slope), float(intercept), (t, d)

def time_warp_resample(X, fs_from, fs_to, scale):
    """
    Resample with a small time-scale correction (scale ~ 1 +/- eps).
    Effective output fs = fs_to, but query is stretched by 'scale'.
    """
    C, T = X.shape
    dur_in = T / fs_from
    # new duration after scaling (so spikes match across the window)
    dur_scaled = dur_in * scale
    T_to = int(round(dur_scaled * fs_to))
    t_from = np.linspace(0.0, dur_in, T, endpoint=False)
    t_to   = np.linspace(0.0, dur_scaled, T_to, endpoint=False)
    Y = np.empty((C, T_to), dtype=X.dtype)
    for c in range(C):
        Y[c] = np.interp(t_to, t_from, X[c])
    return Y


# -------------------------- Main --------------------------

def main():
    ap = argparse.ArgumentParser(description="Compare ~30s LSL EMG (via LSLClient) to OEBin via robust correlation.")
    ap.add_argument('--root_dir', required=True, help='Root directory containing raw/ (with OEBin)')
    ap.add_argument('--oebin_subdir', default='raw', help='Subdir under root that holds OEBin (default: raw)')
    ap.add_argument('--stream_name', default=None, help='LSL stream name (preferred)')
    ap.add_argument('--stream_type', default=None, help='LSL stream type (use if no name provided)')
    ap.add_argument('--channels', nargs='+', type=int, default=None, help='Optional channel indices to capture from LSL')
    ap.add_argument('--duration_s', type=float, default=30.0, help='Seconds to capture from LSL (default: 30)')
    ap.add_argument('--plot_channels', nargs='+', type=int, default=[0], help='Channel indices to overlay in plot')
    ap.add_argument('--envelope_win_s', type=float, default=0.125, help='Envelope MA window (s) for coarse search')
    ap.add_argument('--coarse_fs', type=float, default=200.0, help='Coarse envelope sampling rate (Hz)')
    ap.add_argument('--top_k', type=int, default=5, help='Top-K coarse candidates to refine')
    ap.add_argument('--refine_pad_s', type=float, default=1.0, help='Fine-search half-width around each candidate (s)')
    ap.add_argument('--no_sign_flip', action='store_true', help='Disable per-channel sign flip during refinement')
    ap.add_argument('--save_fig', default=None, help='Path to save the comparison figure')
    ap.add_argument('--save_npz', default=None, help='Optional path to save aligned arrays and metrics')
    ap.add_argument('--verbose', action='store_true')
    args = ap.parse_args()

    logging.basicConfig(level=(logging.DEBUG if args.verbose else logging.INFO),
                        format='%(asctime)s [%(levelname)s] %(message)s')

    if (args.stream_name is None) == (args.stream_type is None):
        raise ValueError("Provide exactly one of --stream_name or --stream_type")

    # Load OEBin (reference)
    raw_dir = os.path.join(args.root_dir, args.oebin_subdir)
    rec = load_oebin_file(raw_dir, verbose=args.verbose)
    emg = np.asarray(rec['amplifier_data'], dtype=np.float32)  # (C, N)
    fs = float(rec['sample_rate'])
    logging.info("OEBin: fs=%.3f Hz, channels=%d, samples=%d", fs, emg.shape[0], emg.shape[1])

    # Capture via your LSLClient
    lsl_CxT, fs_lsl = capture_with_lslclient(args.duration_s, args.stream_name, args.stream_type, args.channels, args.verbose)
    logging.info("LSL (LSLClient): eff_fs=%.3f Hz, channels=%d, samples=%d", fs_lsl, lsl_CxT.shape[0], lsl_CxT.shape[1])

    # Channel matching: index-based (LSLClient doesn't expose labels); use min common channels
    C = min(emg.shape[0], lsl_CxT.shape[0])
    X_ref = emg[:C, :]
    Y_tpl = lsl_CxT[:C, :]

    # Resample LSL snippet to OEBin fs if needed
    if abs(fs_lsl - fs) / fs > 5e-5:  # ~50 ppm
        logging.info("Sample rates differ (LSL %.6f vs OEBin %.6f). Resampling LSL to OEBin fs.", fs_lsl, fs)
        Y_tpl = resample_to_fs(Y_tpl, fs_lsl, fs)

    if X_ref.shape[1] < Y_tpl.shape[1]:
        raise ValueError("OEBin recording is shorter than the captured LSL snippet; cannot compute valid correlation.")

    # Best lag via robust coarse-to-fine search
    best_lag, best_r, cands, coarse_ncc = coarse_to_fine_best_lag(
        X_ref=X_ref,
        Y_query=Y_tpl,
        fs=fs,
        allow_sign_flip=(not args.no_sign_flip),
        coarse_fs=args.coarse_fs,
        envelope_win_s=args.envelope_win_s,
        top_k=args.top_k,
        refine_pad_s=args.refine_pad_s
    )

    # Report results
    logging.info("Best alignment: lag=%d samples (%.3fs), mean Pearson r=%.4f", best_lag, best_lag / fs, best_r)
    if cands:
        pretty = ", ".join([f"{i} ({i/fs:.2f}s, r={r:.3f})" for i, r in cands[:min(8, len(cands))]])
        logging.info("Top candidates: %s", pretty)

    # Extract aligned segments for plotting/saving
    oebin_seg = X_ref[:, best_lag:best_lag + Y_tpl.shape[1]]
    lsl_seg = Y_tpl

    # --- 1) Per-channel correlation matrix to detect reorder/sign issues ---
    def zscore_rows(X):
        return (X - X.mean(axis=1, keepdims=True)) / (X.std(axis=1, keepdims=True) + 1e-12)

    A = zscore_rows(oebin_seg)  # C x T
    B = zscore_rows(lsl_seg)  # C x T

    # Correlation matrix (C x C)
    R = (A @ B.T) / A.shape[1]

    # Greedy channel mapping: for each OEBin ch i, pick LSL ch j with max |corr|
    C = A.shape[0]
    available = set(range(C))
    perm = np.empty(C, dtype=int)
    signs = np.ones(C, dtype=np.float32)
    for i in range(C):
        j_best = max(available, key=lambda j: abs(R[i, j]))
        perm[i] = j_best
        signs[i] = 1.0 if R[i, j_best] >= 0 else -1.0
        available.remove(j_best)

    # Quality before/after
    def mean_diag_corr(A, B):
        return float(np.mean([np.corrcoef(A[i], B[i])[0, 1] for i in range(min(32, A.shape[0]))]))

    diag_before = mean_diag_corr(A, B)
    B_mapped = zscore_rows(signs[:, None] * lsl_seg[perm])
    diag_after = mean_diag_corr(A, B_mapped)

    print(f"[diag] mean per-channel corr BEFORE remap: {diag_before:.3f}")
    print(f"[diag] mean per-channel corr AFTER  remap: {diag_after:.3f}")
    print(f"[diag] fraction of negative raw correlations: {np.mean(R.diagonal() < 0):.2f}")

    # Apply mapping for plotting/comparison going forward:
    lsl_seg = signs[:, None] * lsl_seg[perm, :]

    # --- 2) Scale/units sanity check (µV vs V etc.) ---
    std_ratio = np.median(np.std(oebin_seg, axis=1) / (np.std(lsl_seg, axis=1) + 1e-12))
    print(f"[diag] median std(oebin)/std(lsl) = {std_ratio:.3f}  (>>1 or <<1 suggests unit mismatch)")

    # --- 3) Quick spectral check for hidden filters ---
    def avg_periodogram(X, fs, nfft=4096):
        X0 = X[:, :min(X.shape[1], nfft)]
        F = np.fft.rfftfreq(nfft, 1 / fs)
        P = np.mean(np.abs(np.fft.rfft(zscore_rows(X0), n=nfft, axis=1)) ** 2, axis=0) / (fs * nfft)
        return F, P

    F, P_o = avg_periodogram(oebin_seg, fs)
    _, P_l = avg_periodogram(lsl_seg, fs)

    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 4))
    plt.semilogy(F, P_o + 1e-18, label="OEBin avg PSD")
    plt.semilogy(F, P_l + 1e-18, label="LSL avg PSD", alpha=0.8)
    plt.xlim(0, 400)  # EMG band
    plt.xlabel("Hz");
    plt.ylabel("Power");
    plt.legend();
    plt.title("Avg periodogram (z-scored)")
    plt.show()


    # Save arrays/metrics if requested
    if args.save_npz:
        np.savez(args.save_npz,
                 oebin_seg=oebin_seg,
                 lsl_seg=lsl_seg,
                 fs=fs,
                 best_lag=best_lag,
                 best_r=best_r,
                 candidates=np.array(cands, dtype=object))
        logging.info("Saved aligned data to %s", args.save_npz)

    # Plot
    make_plots(coarse_ncc, int(round(best_lag * (args.coarse_fs / fs))),  # mark approx position on coarse curve
               oebin_seg, lsl_seg, fs, args.plot_channels, args.save_fig)


if __name__ == '__main__':
    main()

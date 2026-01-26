# oebin_zmq_window_verify_cached.py
from __future__ import annotations
import os, json, time, argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Your package imports (unchanged)
from pyoephys.io import load_open_ephys_session
from pyoephys.interface import ZMQClient


def _dataset_dir(p: str | os.PathLike) -> Path:
    p = Path(p);  return p if p.is_dir() else p.parent

def _cache_dir(oebin_or_dir: str | os.PathLike) -> Path:
    d = _dataset_dir(oebin_or_dir)
    cdir = d / "_oe_cache"
    cdir.mkdir(exist_ok=True)
    return cdir

def _cache_path(oebin_or_dir: str | os.PathLike, tag: str | None = None) -> Path:
    d = _dataset_dir(oebin_or_dir)
    name = d.name if tag is None else f"{d.name}_{tag}"
    return _cache_dir(oebin_or_dir) / f"{name}_capture_winverify.npz"

def save_cache(cache_path: str | os.PathLike, payload: dict) -> None:
    cache_path = Path(cache_path)
    meta = dict(
        created_utc=time.time(),
        version=1,
        fs=float(payload["fs_hz"]),
        window_ms=float(payload.get("window_ms", 100.0)),
        lag_samples=int(payload.get("lag_samples", 0)),
        lag_seconds=float(payload.get("lag_seconds", 0.0)),
        source=str(payload.get("source_oebin", "")),
        channel_count_o=int(payload["o_emg"].shape[0]),
        sample_count_o=int(payload["o_emg"].shape[1]),
        channel_count_z=int(payload["z_emg"].shape[0]),
        sample_count_z=int(payload["z_emg"].shape[1]),
    )
    np.savez_compressed(
        cache_path,
        o_emg=payload["o_emg"].astype(np.float32, copy=False),
        o_ts=payload["o_ts"].astype(np.float64, copy=False),
        z_emg=payload["z_emg"].astype(np.float32, copy=False),
        z_ts=payload["z_ts"].astype(np.float64, copy=False),
        fs_hz=np.array(float(payload["fs_hz"]), dtype=np.float64),
        ch_names_o=np.array(payload["ch_names_o"], dtype=object),
        ch_names_z=np.array(payload["ch_names_z"], dtype=object),
        recon_o=payload["recon_o"].astype(np.float32, copy=False),
        recon_z=payload["recon_z"].astype(np.float32, copy=False),
        lags_curve=np.asarray(payload.get("lags_curve", []), dtype=np.int64),
        corr_curve=np.asarray(payload.get("corr_curve", []), dtype=np.float32),
        meta_json=np.array(json.dumps(meta)),
    )
    print(f"[cache] saved → {cache_path}")

def load_cache(cache_path: str | os.PathLike) -> dict:
    cache_path = Path(cache_path)
    if not cache_path.exists():
        raise FileNotFoundError(f"Cache not found: {cache_path}")
    with np.load(cache_path, allow_pickle=True) as F:
        meta = json.loads(str(F["meta_json"].item()))
        out = dict(
            o_emg=F["o_emg"],
            o_ts=F["o_ts"],
            z_emg=F["z_emg"],
            z_ts=F["z_ts"],
            fs_hz=float(F["fs_hz"]),
            ch_names_o=list(F["ch_names_o"]),
            ch_names_z=list(F["ch_names_z"]),
            recon_o=F["recon_o"],
            recon_z=F["recon_z"],
            lags_curve=F["lags_curve"],
            corr_curve=F["corr_curve"],
            meta=meta,
        )
    print(f"[cache] loaded ← {cache_path}")
    return out


# ==============================
# NCC & alignment helpers
# ==============================
def ncc_lag(a: np.ndarray, b: np.ndarray, max_lag: int | None = None):
    """Normalized cross-correlation peak. +lag means B is later; to align, shift B left by lag."""
    a = np.asarray(a, np.float64); b = np.asarray(b, np.float64)
    n = min(a.size, b.size); a = a[:n]; b = b[:n]
    finite = np.isfinite(a) & np.isfinite(b)
    if not np.any(finite):
        raise RuntimeError("No finite overlap for NCC.")
    a = (a[finite] - a[finite].mean()) / (a[finite].std() + 1e-12)
    b = (b[finite] - b[finite].mean()) / (b[finite].std() + 1e-12)

    N = len(a)
    L = 1 << (2*N-1).bit_length()
    fa = np.fft.rfft(a, L); fb = np.fft.rfft(b, L)
    corr = np.fft.irfft(fa * np.conj(fb), L)
    corr = np.concatenate([corr[-(N-1):], corr[:N]])
    lags = np.arange(-N+1, N, dtype=int)

    if max_lag is not None:
        keep = (lags >= -max_lag) & (lags <= max_lag)
        lags, corr = lags[keep], corr[keep]

    k = int(np.argmax(corr))
    return int(lags[k]), float(corr[k]), lags, corr

def align_1d_by_lag(a: np.ndarray, b: np.ndarray, lag: int):
    """Align using the NCC convention above. Returns overlap only."""
    n = min(a.size, b.size)
    if lag > 0:      # B later → shift B left
        L = n - lag
        return a[:L], b[lag:lag+L]
    elif lag < 0:    # A later → shift A left by -lag
        k = -lag; L = n - k
        return a[k:k+L], b[:L]
    else:
        return a[:n], b[:n]


# ==============================
# Ring buffer + windowing
# ==============================
class RingBuffer:
    """Minimal ring buffer to collect fixed-size windows and re-stitch for verification."""
    def __init__(self, n_channels: int, capacity: int, dtype=np.float32):
        self.C = int(n_channels)
        self.N = int(capacity)
        self.dtype = dtype
        self.buf = np.zeros((self.C, self.N), dtype=self.dtype)
        self.write_idx = 0
        self.total_written = 0
        self._linear = []  # store windows in arrival order

    def append_window(self, W: np.ndarray):
        W = np.asarray(W)
        assert W.ndim == 2 and W.shape[0] == self.C
        w = W.shape[1]
        self._linear.append(W.astype(self.dtype, copy=False))

        start = self.write_idx % self.N
        end = start + w
        if end <= self.N:
            self.buf[:, start:end] = W
        else:
            k = self.N - start
            self.buf[:, start:] = W[:, :k]
            self.buf[:, : (w - k)] = W[:, k:]
        self.write_idx = (self.write_idx + w) % self.N
        self.total_written += w

    def drain_all(self) -> np.ndarray:
        if not self._linear:
            return np.zeros((self.C, 0), dtype=self.dtype)
        return np.concatenate(self._linear, axis=1)

def iter_fixed_windows(Y: np.ndarray, W: int) -> list[np.ndarray]:
    """Slice (C,S) into non-overlapping windows of size W. Truncates tail."""
    C, S = Y.shape
    n_full = S // W
    return [Y[:, i*W:(i+1)*W] for i in range(n_full)]

def windowed_reconstruction_verify(
    Y: np.ndarray, fs: float, window_ms: float = 100.0, label: str = "signal",
    atol: float = 0.0, rtol: float = 0.0, verbose: bool = True
) -> dict:
    C, S = Y.shape
    W = max(1, int(round((window_ms / 1000.0) * fs)))
    windows = iter_fixed_windows(Y, W)
    S_lin = len(windows) * W

    rb = RingBuffer(n_channels=C, capacity=max(W * 8, W))
    for w in windows:
        rb.append_window(w)

    recon = rb.drain_all()  # (C, S_lin)
    ok_shape = recon.shape == (C, S_lin)
    ok_equal = np.allclose(recon, Y[:, :S_lin], atol=atol, rtol=rtol)

    if verbose:
        print(f"[{label}] C={C}, S={S}, fs={fs:.3f} Hz, window={W} samp (~{window_ms:.1f} ms)")
        print(f"[{label}] windows={len(windows)}, stitched_len={S_lin} samples")
        print(f"[{label}] shape match: {ok_shape}, value match: {ok_equal}")

    return {
        "reconstructed": recon,
        "window_samples": W,
        "n_windows": len(windows),
        "stitched_len": S_lin,
        "shape_ok": ok_shape,
        "equal_ok": ok_equal,
    }


# ==============================
# Data capture (from your working flow)
# ==============================
def capture_same_length_from_gui(
    oebin_or_dir: str, http_ip: str = "127.0.0.1", data_port: str = "5556",
    heartbeat_port: str | None = None, safety_margin_s: float = 2.0,
    ready_timeout_s: float = 5.0, collect_timeout_s: float = 420.0,
    verbose: bool = True
):
    sess = load_open_ephys_session(oebin_or_dir, verbose=verbose)
    Y_ref = sess["amplifier_data"]
    t_ref = sess["t_amplifier"]
    fs = float(sess["sample_rate"])
    names = sess["channel_names"]
    C_ref, S_ref = int(Y_ref.shape[0]), int(Y_ref.shape[1])
    if verbose:
        print(f"[OEBin] fs={fs} Hz, shape={Y_ref.shape}")

    need_seconds = (S_ref / fs) + safety_margin_s
    cli = ZMQClient(
        host_ip=http_ip, data_port=data_port, heartbeat_port=heartbeat_port,
        buffer_seconds=need_seconds, auto_start=False, expected_channel_count=None,
        set_index_looping=False, align_to_header_index=True, verbose=verbose,
    )

    cli.gui.idle(); time.sleep(0.2)
    cli.gui.start_acquisition()
    cli.start()

    if not cli.ready_event.wait(ready_timeout_s):
        cli.stop()
        raise TimeoutError("ZMQClient not ready; no data received.")

    start_idx = int(cli.global_sample_index); t0 = time.time()
    while (int(cli.global_sample_index) - start_idx) < S_ref:
        if (time.time() - t0) > collect_timeout_s:
            have = int(cli.global_sample_index) - start_idx
            cli.stop(); raise TimeoutError(f"Timed out: need {S_ref}, have {have}")
        time.sleep(0.01)

    cli.gui.idle()
    # settle to a stable last index
    last = int(cli.global_sample_index)
    for _ in range(3):
        time.sleep(0.05)
        now = int(cli.global_sample_index)
        if now == last: break
        last = now

    zmq_names = cli.channel_names or [f"CH{i+1}" for i in range(C_ref)]
    sel = list(range(min(C_ref, len(zmq_names))))
    cli.set_channel_index(sel)

    Y, t = cli.get_latest(S_ref)
    Y = Y[:, :S_ref].astype(np.float32, copy=False)
    t = t[:S_ref].astype(np.float64, copy=False)
    cli.stop()

    o = dict(emg=Y_ref.astype(np.float32, copy=False),
             timestamps=t_ref.astype(np.float64, copy=False),
             fs_hz=fs, channel_names=names, source_oebin=oebin_or_dir)
    z = dict(emg=Y, timestamps=t, fs_hz=fs,
             channel_names=[zmq_names[i] for i in sel])
    return o, z


# ==============================
# Cache-aware loader
# ==============================
def load_or_capture_with_cache(
    file_path: str | os.PathLike,
    tag: str | None = None,
    window_ms: float = 100.0,
    force_recapture: bool = False,
    verbose: bool = True
) -> tuple[dict, dict, dict, Path]:
    """
    Returns (o_dict, z_dict, aux_dict, cache_path)
    aux_dict includes: lag_samples, lag_seconds, lags_curve, corr_curve, recon_o, recon_z
    """
    cpath = _cache_path(file_path, tag)
    if (not force_recapture) and cpath.exists():
        C = load_cache(cpath)
        # Slim dicts shaped like capture output to keep downstream code unchanged
        o = dict(emg=C["o_emg"], timestamps=C["o_ts"], fs_hz=C["fs_hz"], channel_names=C["ch_names_o"], source_oebin=str(file_path))
        z = dict(emg=C["z_emg"], timestamps=C["z_ts"], fs_hz=C["fs_hz"], channel_names=C["ch_names_z"])
        aux = dict(
            lags_curve=C["lags_curve"], corr_curve=C["corr_curve"],
            recon_o=C["recon_o"], recon_z=C["recon_z"],
            lag_samples=int(round(C["meta"].get("lag_samples", 0))),
            lag_seconds=float(C["meta"].get("lag_seconds", 0.0)),
        )
        if verbose:
            print("[cache] Using cached capture + window verification.")
        return o, z, aux, cpath

    # Fresh capture
    o, z = capture_same_length_from_gui(file_path, verbose=verbose)

    # Compute lag on channel 0 as a quick alignment sanity-check
    lag_samples, score, lags_curve, corr_curve = ncc_lag(o["emg"][0], z["emg"][0], max_lag=None)
    if verbose:
        print(f"[lag] {lag_samples} samples ({lag_samples/float(o['fs_hz']):.6f} s), NCC={score:.4f}")

    # Windowed stitching verify for both sources
    res_o = windowed_reconstruction_verify(o["emg"], fs=float(o["fs_hz"]), window_ms=window_ms, label="OEBin", verbose=verbose)
    res_z = windowed_reconstruction_verify(z["emg"], fs=float(z["fs_hz"]), window_ms=window_ms, label="ZMQ", verbose=verbose)

    # Save to cache
    save_cache(
        cpath,
        dict(
            o_emg=o["emg"], o_ts=o["timestamps"],
            z_emg=z["emg"], z_ts=z["timestamps"],
            fs_hz=float(o["fs_hz"]),
            ch_names_o=o["channel_names"], ch_names_z=z["channel_names"],
            recon_o=res_o["reconstructed"], recon_z=res_z["reconstructed"],
            lags_curve=lags_curve, corr_curve=corr_curve,
            window_ms=float(window_ms),
            lag_samples=int(lag_samples),
            lag_seconds=float(lag_samples)/float(o["fs_hz"]),
            source_oebin=str(file_path),
        )
    )

    aux = dict(
        lags_curve=lags_curve, corr_curve=corr_curve,
        recon_o=res_o["reconstructed"], recon_z=res_z["reconstructed"],
        lag_samples=int(lag_samples),
        lag_seconds=float(lag_samples)/float(o["fs_hz"]),
    )
    return o, z, aux, cpath


# ==============================
# Plot from cache
# ==============================
def plot_from_cache(cache_path: str | os.PathLike, ch: int = 0):
    C = load_cache(cache_path)
    o_emg, o_ts = C["o_emg"], C["o_ts"]
    z_emg, z_ts = C["z_emg"], C["z_ts"]
    fs = float(C["fs_hz"])
    lags_curve, corr_curve = C["lags_curve"], C["corr_curve"]
    lag_samples = int(round(C["meta"].get("lag_samples", 0)))
    lag_s = lag_samples / fs

    a, ta = o_emg[ch], o_ts - o_ts[0]
    b, tb = z_emg[ch], z_ts - z_ts[0]
    a_al, b_al = align_1d_by_lag(a, b, -lag_samples)
    L = min(a_al.size, b_al.size)
    t_shared = np.arange(L, dtype=np.float64) / fs

    fig, axes = plt.subplots(3, 1, figsize=(15, 8), constrained_layout=True)
    axes[0].plot(ta, a, label="Open Ephys EMG")
    axes[0].plot(tb, b, label="ZMQ EMG", alpha=0.6)
    axes[0].set_title(f"Raw overlay (ch {ch})")
    axes[0].set_xlabel("Time (s)"); axes[0].legend(loc="upper left")

    if lags_curve.size and corr_curve.size and np.any(np.isfinite(corr_curve)):
        axes[1].plot(lags_curve / fs, corr_curve)
        axes[1].axvline(lag_s, ls="--")
        axes[1].set_title("Normalized cross-correlation")
        axes[1].set_xlabel("Lag (s)"); axes[1].set_ylabel("Score")
    else:
        axes[1].text(0.5, 0.5, "No correlation curve", ha="center", va="center")
        axes[1].set_axis_off()

    axes[2].plot(t_shared, a_al[:L], label="Open Ephys EMG (aligned)")
    axes[2].plot(t_shared, b_al[:L], label="ZMQ EMG (aligned)", alpha=0.6)
    axes[2].set_title("Aligned EMG (overlap)")
    axes[2].set_xlabel("Time (s)"); axes[2].legend(loc="upper left")
    plt.show()



if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Windowed (100ms) reconstruction verify for OEBin vs ZMQ with caching.")
    ap.add_argument("--file_path", type=str, required=True, help="Path to an Open Ephys .oebin file or its directory")
    ap.add_argument("--window_ms", type=float, default=100.0, help="Window size in ms (default 100)")
    ap.add_argument("--tag", type=str, default=None, help="Optional cache tag")
    ap.add_argument("--use_cache", action="store_true", help="Load from cache if present")
    ap.add_argument("--force_recapture", action="store_true", help="Force re-capture even if cache exists")
    ap.add_argument("--plot_cache", action="store_true", help="Plot directly from cache and exit")
    ap.add_argument("--plot_channel", type=int, default=0, help="Channel index for plots")
    ap.add_argument("--no_plots", action="store_true", help="Skip plots")
    args = ap.parse_args()

    cpath = _cache_path(args.file_path, args.tag)

    if args.plot_cache:
        plot_from_cache(cpath, ch=args.plot_channel)

    if args.use_cache and (not args.force_recapture) and cpath.exists():
        o, z, aux, cpath = load_or_capture_with_cache(
            args.file_path, tag=args.tag, window_ms=args.window_ms,
            force_recapture=False, verbose=True
        )
    else:
        o, z, aux, cpath = load_or_capture_with_cache(
            args.file_path, tag=args.tag, window_ms=args.window_ms,
            force_recapture=args.force_recapture, verbose=True
        )

    # Cross-check reconstructed OEBin vs ZMQ using cached/just-computed lag
    a_al, b_al = align_1d_by_lag(aux["recon_o"][0], aux["recon_z"][0], -aux["lag_samples"])
    L = min(a_al.size, b_al.size)
    agree = np.allclose(a_al[:L], b_al[:L], atol=0.0, rtol=0.0)
    print(f"[Reconstructed cross-check] channel 0 aligned equality: {agree} (L={L})")

    if not args.no_plots:
        plot_from_cache(cpath, ch=args.plot_channel)


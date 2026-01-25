# oebin_zmq_window_verify_cached.py
from __future__ import annotations
import os, json, time, argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Your package imports (unchanged)
from pyoephys.io import load_open_ephys_session
from pyoephys.interface import ZMQClient


# ==============================
# Path & caching utilities
# ==============================
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
        version=2,
        fs=float(payload["fs_hz"]),
        window_ms=float(payload.get("window_ms", 100.0)),
        lag_samples=int(payload.get("lag_samples", 0)),
        lag_seconds=float(payload.get("lag_seconds", 0.0)),
        source=str(payload.get("source_oebin", "")),
        channel_count_o=int(payload["o_emg"].shape[0]) if "o_emg" in payload else 0,
        sample_count_o=int(payload["o_emg"].shape[1]) if "o_emg" in payload else 0,
        channel_count_z=int(payload["z_emg"].shape[0]),
        sample_count_z=int(payload["z_emg"].shape[1]),
        streamed=bool(payload.get("streamed", False)),
    )
    np.savez_compressed(
        cache_path,
        o_emg=payload.get("o_emg", np.zeros((0,0), np.float32)).astype(np.float32, copy=False),
        o_ts=payload.get("o_ts", np.zeros((0,), np.float64)).astype(np.float64, copy=False),
        z_emg=payload["z_emg"].astype(np.float32, copy=False),
        z_ts=payload.get("z_ts", np.zeros((payload["z_emg"].shape[1],), np.float64)).astype(np.float64, copy=False),
        fs_hz=np.array(float(payload["fs_hz"]), dtype=np.float64),
        ch_names_o=np.array(payload.get("ch_names_o", []), dtype=object),
        ch_names_z=np.array(payload["ch_names_z"], dtype=object),
        recon_o=payload.get("recon_o", payload["z_emg"][:, :0]).astype(np.float32, copy=False),
        recon_z=payload.get("recon_z", payload["z_emg"]).astype(np.float32, copy=False),
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
# Data capture (batch path)
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
# NEW: True streaming capture of fixed windows (no prediction)
# ==============================
def _fetch_latest_by_new_samples(cli: ZMQClient, new_n: int):
    """Helper to get exactly the newest 'new_n' samples."""
    if new_n <= 0:
        return None, None
    try:
        Y, T = cli.get_latest(new_n)
        return Y, T
    except Exception:
        return None, None

def stream_capture_from_gui(
    oebin_or_dir: str,
    window_ms: float = 100.0,
    http_ip: str = "127.0.0.1",
    data_port: str = "5556",
    heartbeat_port: str | None = None,
    buffer_seconds: float = 120.0,
    ready_timeout_s: float = 5.0,
    stop_after_len: bool = True,   # stop automatically when we match the .oebin length
    verbose: bool = True,
):
    """Stream from GUI in non-overlapping windows and save concatenated data (no prediction)."""
    sess = load_open_ephys_session(oebin_or_dir, verbose=verbose)
    Y_ref = sess["amplifier_data"]
    t_ref = sess["t_amplifier"]
    fs = float(sess["sample_rate"])
    names_o = sess["channel_names"]
    C_ref, S_ref = int(Y_ref.shape[0]), int(Y_ref.shape[1])

    W = max(1, int(round((window_ms / 1000.0) * fs)))  # samples per window
    if verbose:
        print(f"[stream] fs={fs:.3f} Hz, window={W} samples (~{window_ms} ms)")

    cli = ZMQClient(
        host_ip=http_ip, data_port=data_port, heartbeat_port=heartbeat_port,
        buffer_seconds=buffer_seconds, auto_start=False, expected_channel_count=None,
        set_index_looping=False, align_to_header_index=True, verbose=verbose,
    )
    # Try to control GUI
    try:
        cli.gui.idle(); time.sleep(0.1); cli.gui.start_acquisition()
    except Exception:
        if verbose: print("[stream] GUI control not available; assuming running.")

    cli.start()
    if not cli.ready_event.wait(ready_timeout_s):
        cli.stop()
        raise TimeoutError("ZMQClient not ready; no data received.")

    # Map channel names: order ZMQ to .oebin order for apples-to-apples compare later
    zmq_names_all = cli.channel_names or []
    if not zmq_names_all:
        # give it a moment to populate
        for _ in range(50):
            time.sleep(0.02)
            zmq_names_all = cli.channel_names or []
            if zmq_names_all:
                break
    if not zmq_names_all:
        cli.stop(); raise RuntimeError("ZMQ stream has no channel names yet.")

    # Build mapping by *exact name match*
    idx_map = []
    for nm in names_o:
        try:
            idx_map.append(zmq_names_all.index(nm))
        except ValueError:
            cli.stop()
            raise RuntimeError(f"ZMQ is missing required channel '{nm}' from .oebin")

    cli.set_channel_index(idx_map)
    names_z = [zmq_names_all[i] for i in idx_map]

    # Streaming loop
    last_seen = int(getattr(cli, "global_sample_index", 0))
    carry = np.zeros((len(idx_map), 0), dtype=np.float32)
    windows = []
    ts_windows = []
    collected = 0
    target = S_ref if stop_after_len else None

    try:
        if verbose: print("[stream] capturing windows…  Ctrl+C to stop.")
        while True:
            total = int(getattr(cli, "global_sample_index", 0))
            new_n = total - last_seen
            if new_n <= 0:
                time.sleep(0.003)
                continue

            Y_new, T_new = _fetch_latest_by_new_samples(cli, new_n)
            if Y_new is None or Y_new.size == 0:
                time.sleep(0.002)
                continue

            # select already-locked channel order
            if Y_new.ndim == 1:
                Y_new = Y_new[np.array(range(len(idx_map))), None]
                T_new = T_new[None, :]
            # append to carry
            Y_new = Y_new.astype(np.float32, copy=False)
            carry = np.concatenate([carry, Y_new], axis=1)

            # timestamps are per-sample; keep last row (any channel), or average across channels
            if T_new is None or T_new.size == 0:
                t_cand = None
            else:
                # Use the first channel's timestamps (Open Ephys provides identical per-channel)
                t_cand = T_new[0].astype(np.float64, copy=False)

            # emit full windows
            while carry.shape[1] >= W:
                if target is not None and (collected + W) > target:
                    # emit only what keeps us <= target
                    need = target - collected
                    if need <= 0:
                        break
                    Wcut = need
                else:
                    Wcut = W

                w = carry[:, :Wcut]
                windows.append(w)
                if t_cand is not None:
                    ts_windows.append(t_cand[:Wcut])

                carry = carry[:, Wcut:]
                if t_cand is not None:
                    t_cand = t_cand[Wcut:]
                collected += Wcut

                if target is not None and collected >= target:
                    raise KeyboardInterrupt  # trigger save and exit cleanly

            last_seen = total
            time.sleep(0.001)

    except KeyboardInterrupt:
        if verbose:
            print(f"[stream] stopping. collected={collected} samples "
                  f"({collected/fs:.2f} s, {len(windows)} windows of {W} or final partial).")
    finally:
        try:
            cli.gui.idle()
        except Exception:
            pass
        cli.stop()

    if not windows:
        raise RuntimeError("No windows captured from ZMQ stream.")

    Z_recon = np.concatenate(windows, axis=1)
    Z_ts = np.concatenate(ts_windows, axis=0) if ts_windows else np.arange(Z_recon.shape[1]) / fs

    # For OEBin side, also reconstruct to window multiples for apples-to-apples
    res_o = windowed_reconstruction_verify(Y_ref, fs=fs, window_ms=window_ms, label="OEBin", verbose=verbose)

    # Compute lag on channel 0 between full OEBin and streamed ZMQ concatenation
    lag_samples, score, lags_curve, corr_curve = ncc_lag(Y_ref[0], Z_recon[0], max_lag=None)
    if verbose:
        print(f"[lag] {lag_samples} samples ({lag_samples/fs:.6f} s), NCC={score:.4f}")

    return dict(
        o_emg=Y_ref, o_ts=t_ref, z_emg=Z_recon, z_ts=Z_ts,
        fs_hz=fs, ch_names_o=names_o, ch_names_z=names_z,
        recon_o=res_o["reconstructed"], recon_z=Z_recon,
        lags_curve=lags_curve, corr_curve=corr_curve,
        lag_samples=int(lag_samples), lag_seconds=float(lag_samples)/fs
    )


# ==============================
# Cache-aware loader (batch path)
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

    # Fresh batch capture (gets whole buffer and windows it)
    o, z = capture_same_length_from_gui(file_path, verbose=verbose)

    # Compute lag on channel 0 as a quick alignment sanity-check
    lag_samples, score, lags_curve, corr_curve = ncc_lag(o["emg"][0], z["emg"][0], max_lag=None)
    if verbose:
        print(f"[lag] {lag_samples} samples ({lag_samples/float(o['fs_hz']):.6f} s), NCC={score:.4f}")

    # Windowed stitching verify for both sources
    res_o = windowed_reconstruction_verify(o["emg"], fs=float(o["fs_hz"]), window_ms=window_ms, label="OEBin", verbose=verbose)
    res_z = windowed_reconstruction_verify(z["emg"], fs=float(z["fs_hz"]), window_ms=window_ms, label="ZMQ", verbose=verbose)

    cpath.parent.mkdir(parents=True, exist_ok=True)
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
            streamed=False,
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

    if o_emg.size:
        a, ta = o_emg[ch], o_ts - (o_ts[0] if o_ts.size else 0.0)
    else:
        a, ta = None, None
    b, tb = z_emg[ch], z_ts - (z_ts[0] if z_ts.size else 0.0)

    fig, axes = plt.subplots(3, 1, figsize=(15, 8), constrained_layout=True)
    if a is not None:
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

    if a is not None:
        # aligned overlay (overlap only)
        a_al, b_al = align_1d_by_lag(a, b, -lag_samples)
        L = min(a_al.size, b_al.size)
        t_shared = np.arange(L, dtype=np.float64) / fs
        axes[2].plot(t_shared, a_al[:L], label="Open Ephys EMG (aligned)")
        axes[2].plot(t_shared, b_al[:L], label="ZMQ EMG (aligned)", alpha=0.6)
        axes[2].set_title("Aligned EMG (overlap)")
        axes[2].set_xlabel("Time (s)"); axes[2].legend(loc="upper left")
    else:
        axes[2].text(0.5, 0.5, "No OEBin data present", ha="center", va="center")
        axes[2].set_axis_off()

    plt.show()


# ==============================
# CLI
# ==============================
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Windowed (100ms) reconstruction verify for OEBin vs ZMQ with caching.")
    ap.add_argument("--file_path", type=str, required=True, help="Path to an Open Ephys .oebin file or its directory")
    ap.add_argument("--window_ms", type=float, default=100.0, help="Window size in ms (default 100)")
    ap.add_argument("--tag", type=str, default=None, help="Optional cache tag")
    ap.add_argument("--use_cache", action="store_true", help="Load from cache if present (batch mode)")
    ap.add_argument("--force_recapture", action="store_true", help="Force re-capture even if cache exists (batch mode)")
    ap.add_argument("--plot_cache", action="store_true", help="Plot directly from cache and exit")
    ap.add_argument("--plot_channel", type=int, default=0, help="Channel index for plots")
    ap.add_argument("--no_plots", action="store_true", help="Skip plots")
    # NEW streaming flags
    ap.add_argument("--stream_capture", action="store_true", help="Stream windows from GUI and save concatenated data (no prediction)")
    ap.add_argument("--no_stop_after_len", action="store_true", help="If set, DO NOT stop at .oebin length; stop with Ctrl+C")
    ap.add_argument("--host_ip", type=str, default="127.0.0.1")
    ap.add_argument("--data_port", type=str, default="5556")
    ap.add_argument("--heartbeat_port", type=str, default=None)
    ap.add_argument("--buffer_seconds", type=float, default=120.0)
    ap.add_argument("--ready_timeout", type=float, default=5.0)
    args = ap.parse_args()

    cpath = _cache_path(args.file_path, args.tag)

    if args.plot_cache:
        plot_from_cache(cpath, ch=args.plot_channel)
    elif args.stream_capture:
        # True streaming capture path (no prediction)
        packed = stream_capture_from_gui(
            args.file_path,
            window_ms=args.window_ms,
            http_ip=args.host_ip,
            data_port=args.data_port,
            heartbeat_port=args.heartbeat_port,
            buffer_seconds=args.buffer_seconds,
            ready_timeout_s=args.ready_timeout,
            stop_after_len=(not args.no_stop_after_len),
            verbose=True,
        )
        Path(cpath).parent.mkdir(parents=True, exist_ok=True)
        save_cache(
            cpath,
            dict(
                **packed,
                window_ms=float(args.window_ms),
                source_oebin=str(args.file_path),
                streamed=True,
            )
        )
        print(f"[stream] cache saved. You can now run prediction offline using this cache.")
        if not args.no_plots:
            plot_from_cache(cpath, ch=args.plot_channel)
    else:
        # Original batch capture/verify path
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

        # Working!
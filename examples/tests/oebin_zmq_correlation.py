from __future__ import annotations
import os, json, time
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from pyoephys.io import load_open_ephys_session
from pyoephys.interface import ZMQClient


def _dataset_dir(p: str | os.PathLike) -> Path:
    p = Path(p);  return p if p.is_dir() else p.parent


def _cache_path(oebin_or_dir: str | os.PathLike, tag: str | None = None) -> Path:
    d = _dataset_dir(oebin_or_dir)
    cdir = d / "_oe_cache"; cdir.mkdir(exist_ok=True)
    name = d.name if tag is None else f"{d.name}_{tag}"
    return cdir / f"{name}_capture.npz"


def save_cache(cf: str | os.PathLike, o: dict, z: dict,
               lag: int, score: float, lags_curve: np.ndarray, corr_curve: np.ndarray) -> None:
    meta = {
        "created_utc": time.time(),
        "version": 2,
        "o_shape": list(o["emg"].shape),
        "z_shape": list(z["emg"].shape),
        "fs": float(z["fs_hz"]),
        "lag_samples": int(lag),
        "lag_seconds": float(lag)/float(z["fs_hz"]),
        "convention": "positive lag => B later; shift B left by lag",
        "flipped": False,
    }
    np.savez_compressed(
        Path(cf),
        o_emg=o["emg"].astype(np.float32, copy=False),
        o_t=o["timestamps"].astype(np.float64, copy=False),
        o_names=np.asarray(o["channel_names"], dtype=object),
        o_fs=np.float64(o["fs_hz"]),
        z_emg=z["emg"].astype(np.float32, copy=False),
        z_t=z["timestamps"].astype(np.float64, copy=False),
        z_names=np.asarray(z["channel_names"], dtype=object),
        z_fs=np.float64(z["fs_hz"]),
        lag_samples=np.int64(lag),
        score=np.float64(score),
        lags_curve=np.asarray(lags_curve, dtype=np.int64),
        corr_curve=np.asarray(corr_curve, dtype=np.float32),
        meta_json=np.array(json.dumps(meta)),
    )


def load_cache(cf: str | os.PathLike):
    p = Path(cf)
    if not p.exists(): return None
    data = np.load(p, allow_pickle=True)
    meta = json.loads(str(np.asarray(data["meta_json"])))
    if int(meta.get("version", -1)) != 2:
        return None  # invalidate older caches automatically
    o = dict(
        emg=np.asarray(data["o_emg"]),
        timestamps=np.asarray(data["o_t"]),
        fs_hz=float(data["o_fs"]),
        channel_names=list(np.asarray(data["o_names"], dtype=object)),
    )
    z = dict(
        emg=np.asarray(data["z_emg"]),
        timestamps=np.asarray(data["z_t"]),
        fs_hz=float(data["z_fs"]),
        channel_names=list(np.asarray(data["z_names"], dtype=object)),
    )
    lag = int(data["lag_samples"])
    score = float(data["score"])
    lags_curve = np.asarray(data["lags_curve"])
    corr_curve = np.asarray(data["corr_curve"])
    return o, z, lag, float(lag)/z["fs_hz"], score, False, lags_curve, corr_curve, meta


def ncc_lag(a: np.ndarray, b: np.ndarray, max_lag: int | None = None):
    """Normalized cross-correlation peak (no flipping).
       Convention: +lag => B is later; to align, shift B left by lag.
    """
    a = np.asarray(a, np.float64); b = np.asarray(b, np.float64)
    n = min(a.size, b.size); a = a[:n]; b = b[:n]
    finite = np.isfinite(a) & np.isfinite(b)
    if not np.any(finite):
        raise RuntimeError("No finite overlap for NCC.")
    a = a[finite]; b = b[finite]
    a = (a - a.mean()) / (a.std() + 1e-12)
    b = (b - b.mean()) / (b.std() + 1e-12)

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


def capture_same_length_from_gui(
        oebin_or_dir: str, http_ip: str = "127.0.0.1", data_port: str = "5556", heartbeat_port: str | None = None,
        safety_margin_s: float = 2.0, ready_timeout_s: float = 5.0, collect_timeout_s: float = 420.0, verbose: bool = True):

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

    # Start cleanly
    cli.gui.idle(); time.sleep(0.2)
    cli.gui.start_acquisition()
    cli.start()

    if not cli.ready_event.wait(ready_timeout_s):
        cli.stop()
        raise TimeoutError("ZMQClient not ready; no data received.")

    # Let the client data buffer fill
    start_idx = int(cli.global_sample_index); t0 = time.time()
    while (int(cli.global_sample_index) - start_idx) < S_ref:
        if (time.time() - t0) > collect_timeout_s:
            have = int(cli.global_sample_index) - start_idx
            cli.stop(); raise TimeoutError(f"Timed out: need {S_ref}, have {have}")
        time.sleep(0.01)

    cli.gui.idle()
    # small settle
    last = int(cli.global_sample_index)
    for _ in range(3):
        time.sleep(0.05)
        now = int(cli.global_sample_index)
        if now == last: break
        last = now

    zmq_names = cli.channel_names
    if not zmq_names:
        zmq_names = [f"CH{i+1}" for i in range(C_ref)]
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


def load_or_capture_with_cache(oebin_or_dir: str, capture_fn, force: bool = False, tag: str | None = None):
    cpath = _cache_path(oebin_or_dir, tag)
    if not force:
        cached = load_cache(cpath)
        if cached is not None:
            return cached

    o, z = capture_fn(oebin_or_dir)
    fs = float(z["fs_hz"])
    lag, score, lags_curve, corr_curve = ncc_lag(o["emg"][0], z["emg"][0], max_lag=None)
    save_cache(cpath, o, z, lag, score, lags_curve, corr_curve)
    # return the freshly saved (keeps one code path)
    return load_cache(cpath)


def plot_from_cache(o, z, lag_samples: int, lags_curve: np.ndarray, corr_curve: np.ndarray, ch: int = 0):
    fs = float(z["fs_hz"]); lag_s = lag_samples / fs
    a, ta = o["emg"][ch], o["timestamps"]
    b, tb = z["emg"][ch], z["timestamps"]

    # for raw-overlay readability (start at 0)
    ta = ta - ta[0]
    tb = tb - tb[0]

    a_al, b_al = align_1d_by_lag(a, b, -lag_samples)
    L = min(a_al.size, b_al.size)
    t_shared = np.arange(L, dtype=np.float64) / fs

    fig, axes = plt.subplots(3, 1, figsize=(15, 8), constrained_layout=True)

    axes[0].plot(ta, a, label="Open Ephys EMG")
    axes[0].plot(tb, b, label="ZMQ EMG", alpha=0.6)
    axes[0].set_title("Raw overlay (native time bases)")
    axes[0].set_xlabel("Time (s)"); axes[0].legend(loc="upper left")

    if lags_curve.size and corr_curve.size and np.any(np.isfinite(corr_curve)):
        axes[1].plot(lags_curve / fs, corr_curve, lw=1)
        axes[1].axvline(lag_s, ls="--", color="red")
        ytxt = float(np.nanmax(corr_curve))
        axes[1].text(lag_s, ytxt, f"  lag = {lag_samples} samp ({lag_s:.3f}s)", va="top", ha="left", color="red")
        axes[1].set_title("Normalized cross-correlation (cached)")
        axes[1].set_xlabel("Lag (s)"); axes[1].set_ylabel("Score")
    else:
        axes[1].text(0.5, 0.5, "No cached correlation curve", ha="center", va="center")
        axes[1].set_axis_off()

    axes[2].plot(t_shared, a_al, label="Open Ephys EMG")
    axes[2].plot(t_shared, b_al, label="ZMQ EMG", alpha=0.6)
    axes[2].set_title("Aligned EMG (overlap only)")
    axes[2].set_xlabel("Time (s)"); axes[2].legend(loc="upper left")
    plt.show()


if __name__ == "__main__":

    # Example usage:
    #     OE_PATH = r"G:\Shared drives\NML_shared\DataShare\HDEMG Human Healthy\HD-EMG_Cuff\Jonathan\2025_07_31\raw\gestures"
    ap = argparse.ArgumentParser(description="Compare Open Ephys EMG with ZMQ EMG stream.")
    ap.add_argument("--file_path", type=str, required=True, help="Path to an Open Ephys .oebin file or directory")
    ap.add_argument("--force", action="store_true", help="Force re-capture and cache even if cache exists")
    ap.add_argument("--tag", type=str, default=None, help="Optional tag for cache file name")
    args = ap.parse_args()

    o, z, lag_samples, lag_seconds, score, flipped, lags_curve, corr_curve, meta = \
        load_or_capture_with_cache(args.file_path, capture_same_length_from_gui, force=args.force, tag=args.tag)

    print(f"Cache at {_cache_path(args.file_path, args.tag)}")
    print(f"Lag: {lag_samples} samp ({lag_seconds:.3f}s), NCC peak={score:.3f}")
    print("OEBin:", o["emg"].shape, "ZMQ:", z["emg"].shape)

    plot_from_cache(o, z, lag_samples, lags_curve, corr_curve)

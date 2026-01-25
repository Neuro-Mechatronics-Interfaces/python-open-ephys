#!/usr/bin/env python3
"""
predict_from_cache_or_capture.py

Unified flow:
1) Locate (or create) a cache NPZ containing ZMQ-captured data + computed lag.
2) Repair non-finite (NaN/Inf) in the cached EMG, apply cached lag.
3) Preprocess, extract features with training-locked params, run model, evaluate.

Typical:
python predict_from_cache_or_capture.py ^
  --file_path "G:/.../2025_07_31/raw/gestures/Record Node 111/experiment1/recording1/structure.oebin" ^
  --root_dir  "G:/.../2025_07_31" ^
  --label "sleeve_15ch_ring" ^
  --use_cache --capture_if_missing --fix_mode interp --verbose
"""

from __future__ import annotations
import os, time, json, argparse
from pathlib import Path
import numpy as np

# pyoephys deps you already have
from pyoephys.io import load_open_ephys_session, load_config_file, load_metadata_json, lock_params_to_meta, normalize_name
from pyoephys.interface import ZMQClient
from pyoephys.processing import EMGPreprocessor
from pyoephys.ml import ModelManager, EMGClassifier, evaluate_against_events


# -----------------------------
# Cache path helpers (same naming you used)
# -----------------------------
def _dataset_dir(p: str | os.PathLike) -> Path:
    p = Path(p)
    return p if p.is_dir() else p.parent

def _cache_dir(oebin_or_dir: str | os.PathLike) -> Path:
    d = _dataset_dir(oebin_or_dir)
    cdir = d / "_oe_cache"
    cdir.mkdir(exist_ok=True)
    return cdir

def _cache_path(oebin_or_dir: str | os.PathLike, tag: str | None = None) -> Path:
    d = _dataset_dir(oebin_or_dir)
    name = d.name if tag is None else f"{d.name}_{tag}"
    return _cache_dir(oebin_or_dir) / f"{name}_capture_winverify.npz"


# -----------------------------
# Cache I/O (minimal)
# -----------------------------
def load_cache(cache_path: str | os.PathLike) -> dict:
    with np.load(cache_path, allow_pickle=True) as F:
        meta = json.loads(str(F["meta_json"].item())) if "meta_json" in F else {}
        return dict(
            z_emg = F["recon_z"] if "recon_z" in F and F["recon_z"].size else F["z_emg"],
            z_ts  = F["z_ts"] if "z_ts" in F else np.arange(F["z_emg"].shape[1]) / float(F["fs_hz"]),
            o_emg = F["o_emg"] if "o_emg" in F else np.zeros((0,0), np.float32),
            o_ts  = F["o_ts"] if "o_ts" in F else np.zeros((0,), np.float64),
            fs_hz = float(F["fs_hz"]) if "fs_hz" in F else float(F["fs"]),
            ch_names_z = list(F["ch_names_z"]) if "ch_names_z" in F else list(F["ch_names"]),
            ch_names_o = list(F["ch_names_o"]) if "ch_names_o" in F else [],
            meta = meta,
        )

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
        channel_count_o=int(payload.get("o_emg", np.zeros((0,0))).shape[0]),
        sample_count_o=int(payload.get("o_emg", np.zeros((0,0))).shape[1]) if payload.get("o_emg") is not None else 0,
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


# -----------------------------
# Simple same-length capture from GUI (non-overlapping, no prediction)
# -----------------------------
def capture_same_length_from_gui(
    oebin_or_dir: str,
    http_ip: str = "127.0.0.1",
    data_port: str = "5556",
    heartbeat_port: str | None = None,
    safety_margin_s: float = 2.0,
    ready_timeout_s: float = 5.0,
    collect_timeout_s: float = 420.0,
    verbose: bool = True,
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

    try:
        cli.gui.idle(); time.sleep(0.2); cli.gui.start_acquisition()
    except Exception:
        pass

    cli.start()
    if not cli.ready_event.wait(ready_timeout_s):
        cli.stop(); raise TimeoutError("ZMQClient not ready; no data received.")

    start_idx = int(cli.global_sample_index); t0 = time.time()
    while (int(cli.global_sample_index) - start_idx) < S_ref:
        if (time.time() - t0) > collect_timeout_s:
            have = int(cli.global_sample_index) - start_idx
            cli.stop(); raise TimeoutError(f"Timed out: need {S_ref}, have {have}")
        time.sleep(0.01)

    try:
        cli.gui.idle()
    except Exception:
        pass

    last = int(cli.global_sample_index)
    for _ in range(3):
        time.sleep(0.05)
        now = int(cli.global_sample_index)
        if now == last: break
        last = now

    zmq_names = cli.channel_names or [f"CH{i+1}" for i in range(C_ref)]
    idx_map = []
    for nm in names:
        try:
            idx_map.append(zmq_names.index(nm))
        except ValueError:
            cli.stop()
            raise RuntimeError(f"ZMQ is missing required channel '{nm}' from .oebin")

    cli.set_channel_index(idx_map)
    Y, t = cli.get_latest(S_ref)
    Y = Y[:, :S_ref].astype(np.float32, copy=False)
    t = t[:S_ref].astype(np.float64, copy=False)
    cli.stop()

    # quick lag from start times (works because both are length-matched)
    start_o = 0  # both slices begin at their own first sample
    start_z = 0
    lag_samples = start_o - start_z  # 0 in this same-length capture
    return dict(
        o_emg=Y_ref.astype(np.float32, copy=False),
        o_ts=t_ref.astype(np.float64, copy=False),
        z_emg=Y.astype(np.float32, copy=False),
        z_ts=t.astype(np.float64, copy=False),
        fs_hz=fs,
        ch_names_o=names,
        ch_names_z=[zmq_names[i] for i in idx_map],
        lag_samples=int(lag_samples),
        lag_seconds=float(lag_samples)/fs,
        recon_o=Y_ref.astype(np.float32, copy=False),
        recon_z=Y.astype(np.float32, copy=False),
        lags_curve=np.array([], np.int64),
        corr_curve=np.array([], np.float32),
        source_oebin=str(oebin_or_dir),
        streamed=True,
    )


# -----------------------------
# Non-finite repair (interp / 1k jitter / zero / none)
# -----------------------------
def _runs_from_bool(b: np.ndarray):
    if b.ndim != 1: b = b.ravel()
    if b.size == 0: return []
    db = np.diff(b.astype(np.int8))
    starts = np.where(db == 1)[0] + 1
    ends   = np.where(db == -1)[0] + 1
    if b[0]:  starts = np.r_[0, starts]
    if b[-1]: ends   = np.r_[ends, b.size]
    return list(zip(starts, ends))

def _fill_nonfinite_channel(x: np.ndarray, mode: str = "interp", noise_scale: float = 1.0):
    bad = ~np.isfinite(x)
    if not bad.any() or mode == "none":
        return x
    runs = _runs_from_bool(bad); n = x.size
    for a, b in runs:
        L = b - a
        left, right = a - 1, b
        while left >= 0 and not np.isfinite(x[left]): left -= 1
        while right < n and not np.isfinite(x[right]): right += 1
        vL = float(x[left]) if left >= 0 else 0.0
        vR = float(x[right]) if right < n else vL
        if mode == "zero":
            x[a:b] = 0.0; continue
        env = np.linspace(vL, vR, L, dtype=np.float64)
        if mode == "interp":
            x[a:b] = env; continue
        if mode == "noise1k":
            alt = 1.0 - 2.0 * ((np.arange(L) & 1).astype(np.float64))
            spread = max(1e-6, abs(vR - vL))
            jitter = noise_scale * 0.25 * spread * alt
            fill = np.clip(env + jitter, min(vL, vR), max(vL, vR))
            x[a:b] = fill; continue
        x[a:b] = env
    return x

def _repair_nonfinite(arr: np.ndarray, mode: str = "interp", noise_scale: float = 1.0):
    A = np.array(arr, copy=True, dtype=np.float64)
    for c in range(A.shape[0]):
        if (~np.isfinite(A[c])).any():
            _fill_nonfinite_channel(A[c], mode=mode, noise_scale=noise_scale)
    return A


# -----------------------------
# Channel order + lag helpers
# -----------------------------
def _select_training_channels_by_name(emg: np.ndarray, raw_names: list[str], trained_names: list[str]):
    norm_raw = [normalize_name(n) for n in raw_names]
    norm_to_idx = {n: i for i, n in enumerate(norm_raw)}
    want_norm = [normalize_name(n) for n in trained_names]
    missing = [orig for orig, norm in zip(trained_names, want_norm) if norm not in norm_to_idx]
    if missing:
        raise RuntimeError(f"Recording missing channels required by model: {missing[:10]}{'...' if len(missing)>10 else ''}")
    idx = [norm_to_idx[n] for n in want_norm]
    return emg[idx, :], idx, [raw_names[i] for i in idx]

def _apply_lag_roll(emg: np.ndarray, lag_samples: int) -> np.ndarray:
    if not lag_samples: return emg
    return np.roll(emg, shift=int(lag_samples), axis=1)


# -----------------------------
# Predict from cache (uses cached lag automatically)
# -----------------------------
def predict_from_cache(
    cache_npz: str,
    root_dir: str,
    label: str,
    fix_mode: str = "interp",
    noise_scale: float = 1.0,
    prefer_recon: bool = True,
    window_ms: int | None = None,
    step_ms: int | None = None,
    verbose: bool = False,
):
    C = load_cache(cache_npz)
    fs = float(C["fs_hz"])
    z_emg_all = C["z_emg"]
    z_names_all = list(C["ch_names_z"])
    z_ts = np.asarray(C["z_ts"])
    o_ts = np.asarray(C["o_ts"])
    lag_from_cache = int(round(C["meta"].get("lag_samples", 0)))
    if verbose:
        print(f"[cache] fs={fs:.3f} Hz  z_emg={z_emg_all.shape}  ch={len(z_names_all)}  lag={lag_from_cache} samples")

    # training-locked params + channel order
    meta = load_metadata_json(root_dir, label=label)
    trained_names = meta.get("data", {}).get("channel_names") or meta.get("channel_names")
    if not trained_names:
        raise RuntimeError("metadata missing data.channel_names")
    window_ms, step_ms, _, env_cut = lock_params_to_meta(meta['data'], window_ms, step_ms, selected_channels=None)
    if verbose:
        print(f"[params] window_ms={window_ms} step_ms={step_ms} env_cut={env_cut}")

    z_emg, z_idx, z_names_sel = _select_training_channels_by_name(z_emg_all, z_names_all, trained_names)

    # repair → lag → preprocess → features
    z_emg_fixed = _repair_nonfinite(z_emg, mode=fix_mode, noise_scale=noise_scale)
    z_emg_shift = _apply_lag_roll(z_emg_fixed, lag_from_cache)

    prep = EMGPreprocessor(fs=fs, envelope_cutoff=env_cut, verbose=verbose)
    z_pp = prep.preprocess(z_emg_shift)
    X = prep.extract_emg_features(
        z_pp, window_ms=window_ms, step_ms=step_ms, progress=True,
        tqdm_kwargs={"desc": "NPZ features", "unit": "win", "leave": False}
    )

    # window starts on OEBin base if present, else NPZ base
    if o_ts.size:
        start_index = int(round(o_ts[0] * fs))
    else:
        start_index = int(round(z_ts[0] * fs)) if z_ts.size else 0
    step_samples = int(round(step_ms / 1000.0 * fs))
    window_starts = np.arange(X.shape[0], dtype=int) * step_samples + start_index

    # model
    manager = ModelManager(root_dir=root_dir, label=label, model_cls=EMGClassifier, config={"verbose": verbose})
    manager.load_model()
    nfeat = len(manager.scaler.mean_)
    if X.shape[1] != nfeat:
        raise ValueError(f"Feature dim mismatch (expected {nfeat}) → NPZ={X.shape[1]}")
    y_pred = manager.predict(X)

    try:
        evaluate_against_events(root_dir, window_starts, y_pred)
    except Exception as e:
        if verbose: print(f"[eval] skipped/failed: {e}")


# -----------------------------
# Main orchestration
# -----------------------------
def run(
    file_path: str,
    root_dir: str,
    label: str,
    tag: str | None = None,
    use_cache: bool = True,
    capture_if_missing: bool = False,
    http_ip: str = "127.0.0.1",
    data_port: str = "5556",
    heartbeat_port: str | None = None,
    fix_mode: str = "interp",
    noise_scale: float = 1.0,
    prefer_recon: bool = True,
    window_ms: int | None = None,
    step_ms: int | None = None,
    verbose: bool = False,
):
    cpath = _cache_path(file_path, tag)
    if use_cache and Path(cpath).exists():
        if verbose: print(f"[ok] using existing cache: {cpath}")
    else:
        if not capture_if_missing:
            raise FileNotFoundError(f"No cache at {cpath}. Re-run with --capture_if_missing to build it from the GUI.")
        if verbose: print("[info] capturing from GUI to build cache…")
        packed = capture_same_length_from_gui(
            file_path, http_ip=http_ip, data_port=data_port, heartbeat_port=heartbeat_port, verbose=verbose
        )
        Path(cpath).parent.mkdir(parents=True, exist_ok=True)
        save_cache(cpath, packed)

    predict_from_cache(
        cache_npz=str(cpath),
        root_dir=root_dir,
        label=label,
        fix_mode=fix_mode,
        noise_scale=noise_scale,
        prefer_recon=prefer_recon,
        window_ms=window_ms,
        step_ms=step_ms,
        verbose=verbose,
    )


def parse_args():
    p = argparse.ArgumentParser("Predict from cached ZMQ (auto-lag) or capture if missing.")
    p.add_argument("--config_file", type=str)
    p.add_argument("--file_path", type=str, required=True, help="Path to .oebin or its directory (used to locate cache)")
    p.add_argument("--root_dir", type=str, required=True)
    p.add_argument("--label", type=str, default="")

    p.add_argument("--tag", type=str, default=None)
    p.add_argument("--use_cache", action="store_true")
    p.add_argument("--capture_if_missing", action="store_true")

    p.add_argument("--http_ip", type=str, default="127.0.0.1")
    p.add_argument("--data_port", type=str, default="5556")
    p.add_argument("--heartbeat_port", type=str, default=None)

    p.add_argument("--fix_mode", choices=["interp","noise1k","zero","none"], default="interp")
    p.add_argument("--noise_scale", type=float, default=1.0)
    p.add_argument("--no_prefer_recon", action="store_true")

    p.add_argument("--window_ms", type=int, default=None)
    p.add_argument("--step_ms", type=int, default=None)

    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = {}
    if args.config_file:
        cfg = load_config_file(args.config_file) or {}
    cfg.update(vars(args))
    run(
        file_path=cfg["file_path"],
        root_dir=cfg["root_dir"],
        label=cfg.get("label",""),
        tag=cfg.get("tag", None),
        use_cache=bool(cfg.get("use_cache", False)),
        capture_if_missing=bool(cfg.get("capture_if_missing", False)),
        http_ip=cfg.get("http_ip","127.0.0.1"),
        data_port=cfg.get("data_port","5556"),
        heartbeat_port=cfg.get("heartbeat_port", None),
        fix_mode=cfg.get("fix_mode","interp"),
        noise_scale=float(cfg.get("noise_scale", 1.0)),
        prefer_recon=not cfg.get("no_prefer_recon", False),
        window_ms=cfg.get("window_ms", None),
        step_ms=cfg.get("step_ms", None),
        verbose=bool(cfg.get("verbose", False)),
    )

#!/usr/bin/env python3
"""
3c_predict_npz_cache_debug.py
Offline EMG prediction from streamed NPZ with detailed alignment logging.

Key controls to match OEBin evaluation space:
- --start_mode {oebin_t0,relative,sample0}
    oebin_t0 (default): start_index = round(o_t0 * fs)   ← mirrors 3b OEBin script
    relative:           start_index = round((z_t0 - o_t0) * fs)
    sample0:            start_index = 0

- --lag_sign {none,minus,plus}
    How to apply cached lag_samples to window starts.

Also prints:
- cache fields present, fs, shapes, channels
- o_t0, z_t0, Δt, rel_offset, chosen start_index
- window_starts [min..max] pre/post lag
- optional event bounds peek and overlap check
"""

import os, json, argparse, logging, glob, csv
import numpy as np

from pyoephys.io import load_config_file, lock_params_to_meta, load_metadata_json, normalize_name
from pyoephys.ml import ModelManager, EMGClassifier, evaluate_against_events
from pyoephys.processing import EMGPreprocessor


# ---------- helpers ----------
def _ncc_lag(a: np.ndarray, b: np.ndarray, max_lag: int | None = None):
    a = np.asarray(a, np.float64); b = np.asarray(b, np.float64)
    n = min(a.size, b.size); a = a[:n]; b = b[:n]
    finite = np.isfinite(a) & np.isfinite(b)
    if not np.any(finite):
        raise RuntimeError("No finite overlap for NCC.")
    a = (a[finite] - a[finite].mean()) / (a[finite].std() + 1e-12)
    b = (b[finite] - b[finite].mean()) / (b[finite].std() + 1e-12)
    N = len(a); L = 1 << (2*N-1).bit_length()
    fa = np.fft.rfft(a, L); fb = np.fft.rfft(b, L)
    corr = np.fft.irfft(fa * np.conj(fb), L)
    corr = np.concatenate([corr[-(N-1):], corr[:N]])
    lags = np.arange(-N+1, N, dtype=int)
    if max_lag is not None:
        keep = (lags >= -max_lag) & (lags <= max_lag)
        lags, corr = lags[keep], corr[keep]
    k = int(np.argmax(corr))
    return int(lags[k]), float(corr[k])


def _select_training_channels_by_name(emg: np.ndarray,
                                      raw_names: list[str],
                                      trained_names: list[str]) -> tuple[np.ndarray, list[int]]:
    norm_raw = [normalize_name(n) for n in raw_names]
    norm_to_idx = {n: i for i, n in enumerate(norm_raw)}
    want_norm = [normalize_name(n) for n in trained_names]
    missing = [orig for orig, norm in zip(trained_names, want_norm) if norm not in norm_to_idx]
    if missing:
        raise RuntimeError(f"Recording missing channels required by model: {missing[:10]}{'...' if len(missing)>10 else ''}")
    idx = [norm_to_idx[n] for n in want_norm]
    return emg[idx, :], idx


def _load_npz(cache_npz: str):
    with np.load(cache_npz, allow_pickle=True) as F:
        present = sorted(list(F.files))
        fs_hz = float(F["fs_hz"])
        ch_names_z = list(F["ch_names_z"])
        emg = F["recon_z"] if "recon_z" in F and F["recon_z"].size else F["z_emg"]
        z_ts = F["z_ts"] if "z_ts" in F else np.arange(emg.shape[1]) / fs_hz
        o_ts = F["o_ts"] if "o_ts" in F else np.array([], dtype=np.float64)
        o_emg = F["o_emg"] if "o_emg" in F else np.zeros((0,0), np.float32)
        meta = json.loads(str(F["meta_json"].item())) if "meta_json" in F else {}
    return dict(
        emg=emg, fs_hz=fs_hz, z_ts=z_ts, o_ts=o_ts, o_emg=o_emg,
        ch_names_z=ch_names_z, meta=meta, present=present
    )


def _peek_event_bounds(root_dir: str) -> tuple[int | None, int | None, str]:
    candidates = []
    patterns = [
        os.path.join(root_dir, "events", "*.csv"),
        os.path.join(root_dir, "events", "*.npz"),
        os.path.join(root_dir, "events", "*.json"),
        os.path.join(root_dir, "model", "*events*.csv"),
        os.path.join(root_dir, "*events*.csv"),
        os.path.join(root_dir, "events.csv"),
        os.path.join(root_dir, "events.npz"),
        os.path.join(root_dir, "events.json"),
    ]
    for pat in patterns: candidates.extend(glob.glob(pat))
    candidates = sorted(set(candidates), key=lambda p: (len(p), p.lower()))

    for path in candidates:
        try:
            if path.lower().endswith(".csv"):
                with open(path, "r", newline="") as f:
                    r = csv.DictReader(f)
                    cols = [c.strip().lower() for c in (r.fieldnames or [])]
                    sample_cols = [c for c in cols if "sample" in c or c in ("index","idx","start")]
                    if not sample_cols: continue
                    smin, smax = None, None
                    for row in r:
                        for c in sample_cols:
                            val = row.get(c)
                            if not val: continue
                            try: k = int(round(float(val)))
                            except: continue
                            smin = k if smin is None else min(smin, k)
                            smax = k if smax is None else max(smax, k)
                    if smin is not None and smax is not None:
                        return smin, smax, path

            elif path.lower().endswith(".npz"):
                with np.load(path, allow_pickle=True) as F:
                    for key in ("start_sample","sample","samples","indices","idx","start_idx"):
                        if key in F:
                            arr = np.asarray(F[key]).astype(int)
                            if arr.size: return int(arr.min()), int(arr.max()), path

            elif path.lower().endswith(".json"):
                with open(path,"r") as f: J = json.load(f)
                if isinstance(J, dict):
                    for key in ("start_sample","sample","samples","indices","idx","start_idx"):
                        if key in J:
                            arr = np.asarray(J[key]).astype(int)
                            if arr.size: return int(arr.min()), int(arr.max()), path
                    if "events" in J and isinstance(J["events"], list) and J["events"]:
                        vals = []
                        for ev in J["events"]:
                            for key in ("start_sample","sample","idx","index"):
                                if key in ev:
                                    try: vals.append(int(round(float(ev[key]))))
                                    except: pass
                        if vals: return min(vals), max(vals), path
        except Exception:
            continue
    return None, None, ""


# ---------- main ----------
def run(cache_npz: str,
        root_dir: str,
        label: str = "",
        window_ms: int | None = None,
        step_ms: int | None = None,
        start_mode: str = "oebin_t0",      # 'oebin_t0' | 'relative' | 'sample0'
        lag_sign: str = "none",           # 'none' | 'minus' | 'plus'
        verbose: bool = False,
        peek_events: bool = True):

    lvl = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(format='[%(levelname)s] %(message)s', level=lvl)

    # Load cache
    C = _load_npz(cache_npz)
    emg, emg_fs = C["emg"], float(C["fs_hz"])
    z_ts, o_ts, o_emg = np.asarray(C["z_ts"]), np.asarray(C["o_ts"]), C["o_emg"]
    raw_channel_names = list(C["ch_names_z"])
    lag_samples = int(round(C["meta"].get("lag_samples", 0))) if C.get("meta") else 0

    logging.info(f"Loaded cache: {cache_npz}")
    logging.info(f"  present fields: {C['present']}")
    logging.info(f"  fs={emg_fs:.3f} Hz  emg shape={emg.shape}  o_emg shape={o_emg.shape}  lag={lag_samples} samp")
    logging.info(f"  ch_names_z: {len(raw_channel_names)}")

    # Training metadata
    meta_train = load_metadata_json(root_dir, label=label)
    trained_names = meta_train.get("data", {}).get("channel_names") or meta_train.get("channel_names")
    if not trained_names:
        raise RuntimeError("metadata missing data.channel_names (training channel order).")
    window_ms, step_ms, _, env_cut = lock_params_to_meta(meta_train['data'], window_ms, step_ms, selected_channels=None)
    logging.info(f"Training-locked params: window_ms={window_ms}  step_ms={step_ms}  env_cut={env_cut}")

    # Channel order lock
    emg, sel_idx = _select_training_channels_by_name(emg, raw_channel_names, trained_names)
    logging.info(f"Using {len(sel_idx)} channels locked to training order.")

    # Preprocess + features
    pre = EMGPreprocessor(fs=emg_fs, envelope_cutoff=env_cut, verbose=verbose)
    emg_pp = pre.preprocess(emg)
    X = pre.extract_emg_features(
        emg_pp, window_ms=window_ms, step_ms=step_ms, progress=True,
        tqdm_kwargs={"desc": "Extracting features", "unit": "win", "leave": False},
    )
    logging.info(f"Extracted feature matrix X with shape {X.shape}")

    # Start-index policies
    z_t0 = float(z_ts[0]) if z_ts.size else 0.0
    o_t0 = float(o_ts[0]) if o_ts.size else 0.0
    delta_t = z_t0 - o_t0
    rel_offset = int(round(delta_t * emg_fs))

    if start_mode not in ("oebin_t0","relative","sample0"):
        raise ValueError("--start_mode must be one of: oebin_t0, relative, sample0")

    if start_mode == "oebin_t0":
        if o_ts.size == 0:
            logging.warning("o_ts missing/empty; falling back to sample0 start.")
            start_index = 0
            chosen = "sample0(fallback)"
        else:
            start_index = int(round(o_t0 * emg_fs))
            chosen = "oebin_t0"
    elif start_mode == "relative":
        start_index = rel_offset
        chosen = "relative"
    else:  # sample0
        start_index = 0
        chosen = "sample0"

    # Build window starts, then apply lag with explicit sign
    step_samples = int(round(step_ms / 1000.0 * emg_fs))
    window_starts = np.arange(X.shape[0], dtype=int) * step_samples + start_index

    if lag_sign == "minus":
        window_starts = window_starts - lag_samples
        lag_note = f"applied -{lag_samples}"
    elif lag_sign == "plus":
        window_starts = window_starts + lag_samples
        lag_note = f"applied +{lag_samples}"
    elif lag_sign == "none":
        lag_note = "not applied"
    else:
        raise ValueError("--lag_sign must be one of: none, minus, plus")

    logging.info(
        "Alignment debug:\n"
        f"  o_t0={o_t0:.6f}s  z_t0={z_t0:.6f}s  Δt={delta_t:+.6f}s  rel_offset={rel_offset:+d}\n"
        f"  start_mode={chosen}  start_index={start_index:+d}\n"
        f"  step={step_ms} ms ({step_samples} samp)  windows={X.shape[0]}  approx_dur={emg.shape[1]/emg_fs:.2f}s\n"
        f"  lag correction: {lag_note}\n"
        f"  window_starts [min..max]=[{window_starts.min()} .. {window_starts.max()}]"
    )

    # Optional: NCC sanity-check vs o_emg (if present)
    if o_emg.size and emg.size:
        try:
            l, score = _ncc_lag(o_emg[0], emg[0], max_lag=None)
            logging.info(f"NCC sanity-check (o_emg vs emg): lag={l} samp ({l/emg_fs:.6f}s) NCC={score:.4f}")
        except Exception as e:
            logging.info(f"NCC sanity-check skipped: {e}")

    # Optional: peek event bounds
    if peek_events:
        emin, emax, esrc = _peek_event_bounds(root_dir)
        if emin is not None:
            wmin, wmax = int(window_starts.min()), int(window_starts.max())
            overlaps = not (wmax < emin or wmin > emax)
            logging.info(f"Event bounds from {esrc}: min_sample={emin}  max_sample={emax}")
            logging.info(f"Window range overlaps events? {overlaps}  "
                         f"(windows[{wmin}..{wmax}] vs events[{emin}..{emax}])")
        else:
            logging.info("Event bounds: not found (peek skipped).")

    # Model + predict + evaluate
    manager = ModelManager(root_dir=root_dir, label=label, model_cls=EMGClassifier, config={"verbose": verbose})
    manager.load_model()
    n_features_expected = len(manager.scaler.mean_)
    if X.shape[1] != n_features_expected:
        raise ValueError(f"Feature dim {X.shape[1]} != scaler expectation {n_features_expected}")
    y_pred = manager.predict(X)

    try:
        evaluate_against_events(root_dir, window_starts, y_pred)
    except Exception as e:
        logging.info(f"Event evaluation skipped or failed: {e}")


def parse_args():
    p = argparse.ArgumentParser("3c debug: Offline EMG prediction from NPZ with detailed alignment logging")
    p.add_argument("--config_file", type=str)
    p.add_argument("--cache_npz",   type=str, required=True)
    p.add_argument("--root_dir",    type=str, required=True)
    p.add_argument("--label",       type=str, default="")
    p.add_argument("--window_ms",   type=int, default=None)
    p.add_argument("--step_ms",     type=int, default=None)
    p.add_argument("--start_mode",  type=str, default="oebin_t0",
                   choices=["oebin_t0","relative","sample0"],
                   help="Base index policy for window starts")
    p.add_argument("--lag_sign",    type=str, default="none",
                   choices=["none","minus","plus"],
                   help="How to apply cached lag_samples to window starts")
    p.add_argument("--no_peek_events", action="store_true")
    p.add_argument("--verbose",     action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = {}
    if args.config_file:
        cfg = load_config_file(args.config_file) or {}
    cfg.update(vars(args))
    run(
        cache_npz=cfg["cache_npz"],
        root_dir=cfg["root_dir"],
        label=cfg.get("label", ""),
        window_ms=cfg.get("window_ms", None),
        step_ms=cfg.get("step_ms", None),
        start_mode=cfg.get("start_mode", "oebin_t0"),
        lag_sign=cfg.get("lag_sign", "none"),
        verbose=cfg.get("verbose", False),
        peek_events=not cfg.get("no_peek_events", False),
    )

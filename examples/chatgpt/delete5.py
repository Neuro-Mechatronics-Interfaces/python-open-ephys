#!/usr/bin/env python3
"""
compare_oebin_vs_npz_features_preds.py

What it does
------------
1) Loads OEBin and NPZ (recon_z by default), locks channels to the *training name order*.
2) Uses the training-locked preprocessing/feature params (window/step/env_cut).
3) Builds window indices on the OEBin sample axis (same as your working OEBin script).
4) Extracts features, runs the SAME model on both X matrices, evaluates each vs events.
5) Compares features (mean/median/max |Δ|, corrcoef) and per-window prediction agreement.
6) NEW: Prints the first N raw samples and first N timestamps (and their uniques)
         side-by-side for a chosen preview channel (by name or index in training order).
   Optional: dump those previews to CSV.

Typical usage
-------------
python compare_oebin_vs_npz_features_preds.py ^
  --root_dir "G:/.../2025_07_31" ^
  --cache_npz "G:/.../_oe_cache/2025_07_31_sleeve_15ch_ring_capture_winverify.npz" ^
  --label "sleeve_15ch_ring" ^
  --window_ms 200 --step_ms 50 --verbose ^
  --preview_channel_name CH1 --preview_n 20

# or by index (0 = first channel in the training-locked order)
--preview_channel_idx 0
"""

import os, json, argparse, logging, csv
from pathlib import Path
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

from pyoephys.io import load_oebin_file, load_config_file, lock_params_to_meta, load_metadata_json, normalize_name
from pyoephys.processing import EMGPreprocessor
from pyoephys.ml import ModelManager, EMGClassifier, evaluate_against_events


# ---------------------------
# Helpers
# ---------------------------
def _select_training_channels_by_name(emg: np.ndarray,
                                      raw_names: list[str],
                                      trained_names: list[str]) -> tuple[np.ndarray, list[int], list[str]]:
    """Return (emg_reordered, idx_list, names_in_training_order)."""
    norm_raw = [normalize_name(n) for n in raw_names]
    norm_to_idx = {n: i for i, n in enumerate(norm_raw)}
    want_norm = [normalize_name(n) for n in trained_names]
    missing = [orig for orig, norm in zip(trained_names, want_norm) if norm not in norm_to_idx]
    if missing:
        raise RuntimeError(f"Recording missing channels required by model: {missing[:10]}{'...' if len(missing)>10 else ''}")
    idx = [norm_to_idx[n] for n in want_norm]
    names_ordered = [raw_names[i] for i in idx]
    return emg[idx, :], idx, names_ordered


def _load_npz(cache_npz: str, use_raw_z: bool):
    with np.load(cache_npz, allow_pickle=True) as F:
        fs_hz = float(F["fs_hz"])
        ch_names_z = list(F["ch_names_z"])
        if (not use_raw_z) and ("recon_z" in F) and F["recon_z"].size:
            emg = F["recon_z"]; stream_used = "recon_z"
        else:
            emg = F["z_emg"];   stream_used = "z_emg"
        z_ts = F["z_ts"] if "z_ts" in F else np.arange(emg.shape[1]) / fs_hz
        o_ts = F["o_ts"] if "o_ts" in F else np.array([], dtype=np.float64)
        meta = json.loads(str(F["meta_json"].item())) if "meta_json" in F else {}
    return dict(emg=emg, fs=fs_hz, z_ts=z_ts, o_ts=o_ts, ch_names=ch_names_z, meta=meta, stream_used=stream_used)


def _format_arr(a, maxlen=20, precision=6):
    """Pretty print small vectors with limited length."""
    a = np.asarray(a)
    if a.size > maxlen:
        a = a[:maxlen]
    return np.array2string(a, precision=precision, separator=", ", suppress_small=False)


# ---------------------------
# Main
# ---------------------------
def run(root_dir: str,
        cache_npz: str,
        label: str = "",
        window_ms: int | None = None,
        step_ms: int | None = None,
        use_raw_z: bool = False,
        verbose: bool = False,
        plot_small_diff: bool = False,
        preview_channel_name: str | None = None,
        preview_channel_idx: int = 0,
        preview_n: int = 20,
        dump_preview_csv: str | None = None):

    lvl = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(format='[%(levelname)s] %(message)s', level=lvl)

    # ---- Load training metadata / params ----
    meta = load_metadata_json(root_dir, label=label)
    trained_names = meta.get("data", {}).get("channel_names") or meta.get("channel_names")
    if not trained_names:
        raise RuntimeError("metadata missing data.channel_names")
    window_ms, step_ms, _, env_cut = lock_params_to_meta(meta['data'], window_ms, step_ms, selected_channels=None)
    logging.info(f"Training-locked params: window_ms={window_ms}  step_ms={step_ms}  env_cut={env_cut}")

    # ---- Load OEBin ----
    raw_dir = os.path.join(root_dir, "raw")
    D = load_oebin_file(raw_dir, verbose=verbose)
    o_fs = float(D["sample_rate"])
    o_t  = D["t_amplifier"]
    o_emg_all = D["amplifier_data"]
    o_names_all = list(D.get("channel_names", []))
    logging.info(f"OEBin: fs={o_fs:.3f} Hz  emg={o_emg_all.shape}  ch={len(o_names_all)}  t0={o_t[0]:.6f}s")

    # Lock channels by training names (OEBin)
    o_emg, o_idx, o_names_sel = _select_training_channels_by_name(o_emg_all, o_names_all, trained_names)
    logging.info(f"OEBin channel order (training-locked): {list(zip(o_idx, o_names_sel))}")

    # ---- Load NPZ ----
    C = _load_npz(cache_npz, use_raw_z=use_raw_z)
    z_fs = float(C["fs"]); z_ts = np.asarray(C["z_ts"]); o_ts_from_npz = np.asarray(C["o_ts"])
    z_emg_all = C["emg"]; z_names_all = list(C["ch_names"])
    logging.info(f"NPZ: stream={C['stream_used']}  fs={z_fs:.3f} Hz  emg={z_emg_all.shape}  ch={len(z_names_all)}")

    # Lock channels by training names (NPZ)
    z_emg, z_idx, z_names_sel = _select_training_channels_by_name(z_emg_all, z_names_all, trained_names)
    logging.info(f"NPZ channel order (training-locked): {list(zip(z_idx, z_names_sel))}")

    # ---- Pick preview channel (in training-locked order) ----
    if preview_channel_name is not None:
        want = normalize_name(preview_channel_name)
        norm_sel = [normalize_name(n) for n in o_names_sel]
        if want not in norm_sel:
            raise RuntimeError(f"Preview channel name '{preview_channel_name}' not found in training-locked set.")
        pidx = norm_sel.index(want)
    else:
        pidx = int(preview_channel_idx)
        if pidx < 0 or pidx >= len(o_names_sel):
            raise RuntimeError(f"--preview_channel_idx {pidx} out of range 0..{len(o_names_sel)-1}")
    pv_name = o_names_sel[pidx]
    logging.info(f"Preview channel (training-locked): idx={pidx}, name={pv_name}")

    # ---- RAW preview (BEFORE preprocessing) ----
    Np = int(preview_n)
    raw_o = o_emg[pidx]
    raw_z = z_emg[pidx]

    # Limit to available length for printing
    n_o = min(Np, raw_o.size)
    n_z = min(Np, raw_z.size)
    print("\n=== RAW preview (first N samples) ===")
    print(f"N requested = {Np}  →  OEBin N={n_o}, NPZ N={n_z}")
    print(f"OEBin raw[{pv_name}][:{n_o}] = { _format_arr(raw_o[:n_o], maxlen=n_o, precision=6) }")
    print(f"NPZ   raw[{pv_name}][:{n_z}] = { _format_arr(raw_z[:n_z], maxlen=n_z, precision=6) }")

    # ---- Timestamp preview + uniques ----
    t_o = np.asarray(o_t)
    t_z = np.asarray(z_ts)
    n_to = min(Np, t_o.size)
    n_tz = min(Np, t_z.size)

    uo = np.unique(t_o[:n_to])
    uz = np.unique(t_z[:n_tz])

    print("\n=== Timestamp preview (first N) ===")
    print(f"OEBin t[:{n_to}] = { _format_arr(t_o[:n_to], maxlen=n_to, precision=6) }")
    print(f"NPZ   t[:{n_tz}] = { _format_arr(t_z[:n_tz], maxlen=n_tz, precision=6) }")
    print(f"Unique OEBin t (first {n_to}) → count={uo.size} values: { _format_arr(uo, maxlen=uo.size, precision=6) }")
    print(f"Unique NPZ   t (first {n_tz}) → count={uz.size} values: { _format_arr(uz, maxlen=uz.size, precision=6) }")

    # Also show step stats for first N-1 deltas
    if n_to > 1:
        d_o = np.diff(t_o[:n_to])
        print(f"OEBin Δt stats (first {n_to-1}): mean={d_o.mean():.9f}, std={d_o.std():.9f}, min={d_o.min():.9f}, max={d_o.max():.9f}")
    if n_tz > 1:
        d_z = np.diff(t_z[:n_tz])
        print(f"NPZ   Δt stats (first {n_tz-1}): mean={d_z.mean():.9f}, std={d_z.std():.9f}, min={d_z.min():.9f}, max={d_z.max():.9f}")

    # Optional CSV dump of the preview
    if dump_preview_csv:
        outp = Path(dump_preview_csv)
        outp.parent.mkdir(parents=True, exist_ok=True)
        rows = []
        m = max(n_o, n_z, n_to, n_tz)
        for i in range(m):
            rows.append(dict(
                i=i,
                o_raw=float(raw_o[i]) if i < raw_o.size else np.nan,
                z_raw=float(raw_z[i]) if i < raw_z.size else np.nan,
                o_ts=float(t_o[i]) if i < t_o.size else np.nan,
                z_ts=float(t_z[i]) if i < t_z.size else np.nan,
            ))
        with open(outp, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["i","o_raw","z_raw","o_ts","z_ts"])
            w.writeheader(); w.writerows(rows)
        print(f"[preview CSV] saved → {outp}")

    # ---- Preprocessing + features (same settings) ----
    pre_o = EMGPreprocessor(fs=o_fs, envelope_cutoff=env_cut, verbose=verbose)
    pre_z = EMGPreprocessor(fs=z_fs, envelope_cutoff=env_cut, verbose=verbose)

    o_pp = pre_o.preprocess(o_emg)
    z_pp = pre_z.preprocess(z_emg)

    X_o = pre_o.extract_emg_features(o_pp, window_ms=window_ms, step_ms=step_ms, progress=True,
                                     tqdm_kwargs={"desc": "OEBin features", "unit": "win", "leave": False})
    X_z = pre_z.extract_emg_features(z_pp, window_ms=window_ms, step_ms=step_ms, progress=True,
                                     tqdm_kwargs={"desc": "NPZ features", "unit": "win", "leave": False})
    logging.info(f"Features: OEBin X={X_o.shape}  NPZ X={X_z.shape}")

    # ---- Window starts on OEBin sample axis (both flows) ----
    start_index_o = int(round(float(o_t[0]) * o_fs))
    step_samples_o = int(round(step_ms / 1000.0 * o_fs))
    ws_o = np.arange(X_o.shape[0], dtype=int) * step_samples_o + start_index_o

    # NPZ mirrors OEBin indexing using OEBin t0 from NPZ if present, else from the OEBin we just loaded
    o_t0_npz = float(o_ts_from_npz[0]) if o_ts_from_npz.size else float(o_t[0])
    start_index_z = int(round(o_t0_npz * z_fs))
    step_samples_z = int(round(step_ms / 1000.0 * z_fs))
    ws_z = np.arange(X_z.shape[0], dtype=int) * step_samples_z + start_index_z

    logging.info(
        "Index ranges:\n"
        f"  OEBin ws[min..max]=[{ws_o.min()} .. {ws_o.max()}]\n"
        f"  NPZ   ws[min..max]=[{ws_z.min()} .. {ws_z.max()}]  (start via OEBin t0)"
    )

    # ---- Model / predict ----
    manager = ModelManager(root_dir=root_dir, label=label, model_cls=EMGClassifier, config={"verbose": verbose})
    manager.load_model()
    nfeat = len(manager.scaler.mean_)
    if X_o.shape[1] != nfeat or X_z.shape[1] != nfeat:
        raise ValueError(f"Feature dim mismatch (expected {nfeat}) → OEBin={X_o.shape[1]} NPZ={X_z.shape[1]}")

    y_o = manager.predict(X_o)
    y_z = manager.predict(X_z)

    # ---- Evaluate each vs events ----
    print("\n=== OEBin evaluation ===")
    try:
        evaluate_against_events(root_dir, ws_o, y_o)
    except Exception as e:
        logging.info(f"OEBin evaluation skipped/failed: {e}")

    print("\n=== NPZ evaluation ===")
    try:
        evaluate_against_events(root_dir, ws_z, y_z)
    except Exception as e:
        logging.info(f"NPZ evaluation skipped/failed: {e}")

    # ---- Direct comparisons (without events) ----
    n = min(X_o.shape[0], X_z.shape[0])
    d = np.abs(X_o[:n] - X_z[:n])
    print("\n=== Feature diff stats (first n windows, OEBin vs NPZ) ===")
    print(f"n_windows_compared = {n}")
    print(f"mean(|Δ|) = {d.mean():.6g}   median(|Δ|) = {np.median(d):.6g}   max(|Δ|) = {d.max():.6g}")
    try:
        r = np.corrcoef(X_o[:n].ravel(), X_z[:n].ravel())[0,1]
        print(f"corrcoef(X_o, X_z) = {r:.6f}")
    except Exception:
        pass

    print("\n=== Prediction agreement (window-by-window) ===")
    agree = (y_o[:n] == y_z[:n])
    print(f"agree_count = {int(agree.sum())} / {n}  ({100.0*agree.mean():.2f}%)")

    # Optional tiny plot of per-window |Δ| (mean over features)
    if plot_small_diff:
        import matplotlib.pyplot as plt
        m = d.mean(axis=1)
        plt.figure(figsize=(10,3))
        plt.plot(m)
        plt.title("Per-window mean |Δfeature| (OEBin vs NPZ)")
        plt.xlabel("window #")
        plt.ylabel("mean |Δ|")
        plt.tight_layout()
        plt.show()


def parse_args():
    p = argparse.ArgumentParser("Compare OEBin vs NPZ features & predictions + raw/timestamp previews")
    p.add_argument("--config_file", type=str)
    p.add_argument("--root_dir",    type=str, required=True)
    p.add_argument("--cache_npz",   type=str, required=True)
    p.add_argument("--label",       type=str, default="")
    p.add_argument("--window_ms",   type=int, default=None)
    p.add_argument("--step_ms",     type=int, default=None)
    p.add_argument("--use_raw_z",   action="store_true", help="Use z_emg instead of recon_z from NPZ")
    p.add_argument("--verbose",     action="store_true")
    p.add_argument("--plot_small_diff", action="store_true")

    # NEW: raw/timestamp previews
    p.add_argument("--preview_channel_name", type=str, default=None,
                   help="Preview this channel name (training-locked). If omitted, uses --preview_channel_idx.")
    p.add_argument("--preview_channel_idx",  type=int, default=0,
                   help="Preview this channel index in the training-locked order (default 0).")
    p.add_argument("--preview_n",            type=int, default=20,
                   help="How many initial samples/timestamps to print (default 20).")
    p.add_argument("--dump_preview_csv",     type=str, default=None,
                   help="Optional path to save a CSV of the preview rows (i, o_raw, z_raw, o_ts, z_ts).")

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = {}
    if args.config_file:
        cfg = load_config_file(args.config_file) or {}
    cfg.update(vars(args))
    run(
        root_dir=cfg["root_dir"],
        cache_npz=cfg["cache_npz"],
        label=cfg.get("label",""),
        window_ms=cfg.get("window_ms", None),
        step_ms=cfg.get("step_ms", None),
        use_raw_z=cfg.get("use_raw_z", False),
        verbose=cfg.get("verbose", False),
        plot_small_diff=cfg.get("plot_small_diff", False),
        preview_channel_name=cfg.get("preview_channel_name", None),
        preview_channel_idx=cfg.get("preview_channel_idx", 0),
        preview_n=cfg.get("preview_n", 20),
        dump_preview_csv=cfg.get("dump_preview_csv", None),
    )

#!/usr/bin/env python
# cacheify_oebin_npz.py  (compat version)
import os, argparse, numpy as np

def to_list(x):
    try:
        return x.tolist()
    except Exception:
        return list(x)

def main(in_npz: str, out_npz: str | None = None, lag: int = 0):
    F = np.load(in_npz, allow_pickle=True)

    # Required arrays from the oebin-exported NPZ
    emg = np.asarray(F["amplifier_data"], dtype=np.float64)          # (C, S)
    fs = float(F["sample_rate"])                                      # scalar
    ch_names = (
        to_list(F["channel_names"]) if "channel_names" in F.files
        else [f"CH{i+1}" for i in range(emg.shape[0])]
    )
    z0 = float(F["t_amplifier"][0]) if "t_amplifier" in F.files else 0.0
    t0 = F['t_amplifier'] if 't_amplifier' in F.files else np.arange(emg.shape[1]) / fs
    t0 = np.asarray(t0, dtype=np.float64).ravel()  # <- ensure 1-D float vector

    # print out first 10 time indices
    print(f"[cacheify] first 10 time indices: {t0[:10]} seconds")

    # Meta (kept for forward/backward compatibility with other tools)
    meta = {
        "fs": fs,
        "fs_hz": fs,
        "ch_names": ch_names,
        "t0_seconds": t0,
        "lag_samples": int(lag),
        "stream": "z_emg",
        "source": "oebin_npz",
    }

    if out_npz is None:
        base, _ = os.path.splitext(in_npz)
        out_npz = base + "_cache_for_predict.npz"

    # Save with BOTH top-level fields and the meta dict
    np.savez_compressed(
        out_npz,
        z_emg=emg,
        # top-level scalars/arrays for delete13.py
        fs=fs, fs_hz=fs,
        ch_names=np.array(ch_names, dtype=object),
        t0_seconds=t0,
        o_ts=t0,  # original timestamps (same as t0)
        z_ts=z0,
        lag_samples=int(lag),
        stream=np.array("z_emg", dtype=object),
        # and the full meta dict
        meta=meta,
    )

    # Quick sanity print
    F2 = np.load(out_npz, allow_pickle=True)
    print("keys:", list(F2.files))
    print(f"[cacheify] wrote {out_npz}  C={emg.shape[0]}  S={emg.shape[1]}  fs={fs:.3f}Hz  lag={lag}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_npz", required=True)
    ap.add_argument("--out_npz", default=None)
    ap.add_argument("--lag", type=int, default=0)
    main(**vars(ap.parse_args()))

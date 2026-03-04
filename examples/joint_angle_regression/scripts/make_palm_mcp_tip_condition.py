import argparse
from pathlib import Path

import numpy as np


PALM_MCP_TIP_KEYS = [
    "thumb_palm_mcp_tip",
    "index_palm_mcp_tip",
    "middle_palm_mcp_tip",
    "ring_palm_mcp_tip",
    "pinky_palm_mcp_tip",
]


def _to_landmarks_xyz(arr: np.ndarray, hand_idx: int) -> np.ndarray:
    data = np.asarray(arr, dtype=np.float32)

    if data.ndim == 4 and data.shape[-2:] == (21, 3):
        if data.shape[1] <= hand_idx:
            raise ValueError(
                f"Requested hand_idx={hand_idx}, but only {data.shape[1]} hand(s) found"
            )
        return data[:, hand_idx, :, :]

    if data.ndim == 3 and data.shape[-2:] == (21, 3):
        return data

    if data.ndim == 3 and data.shape[-2:] == (42, 3):
        start = hand_idx * 21
        end = start + 21
        if data.shape[1] < end:
            raise ValueError(
                f"Requested hand_idx={hand_idx}, but landmarks shape is {data.shape}"
            )
        return data[:, start:end, :]

    if data.ndim == 2 and data.shape[1] in (63, 126):
        n = data.shape[0]
        if data.shape[1] == 63:
            return data.reshape(n, 21, 3)
        reshaped = data.reshape(n, 42, 3)
        start = hand_idx * 21
        end = start + 21
        return reshaped[:, start:end, :]

    raise ValueError(
        f"Unsupported landmark shape {data.shape}; expected (N,21,3), (N,2,21,3), (N,42,3), (N,63), or (N,126)."
    )


def _find_landmark_key(npz_obj):
    candidates = [
        "landmarks",
        "hand_landmarks",
        "landmark_xyz",
        "joint_xyz",
        "xyz",
    ]
    for key in candidates:
        if key in npz_obj.files:
            return key
    return None


def _angle_deg(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    ba = a - b
    bc = c - b
    denom = np.linalg.norm(ba, axis=-1) * np.linalg.norm(bc, axis=-1)
    denom = np.maximum(denom, 1e-8)
    cosang = np.sum(ba * bc, axis=-1) / denom
    cosang = np.clip(cosang, -1.0, 1.0)
    return np.degrees(np.arccos(cosang)).astype(np.float32)


def _compute_palm_mcp_tip_angles(landmarks_xyz: np.ndarray) -> np.ndarray:
    # MediaPipe landmark indices
    # wrist/palm center proxy: 0
    # thumb: mcp=2, tip=4
    # index: mcp=5, tip=8
    # middle: mcp=9, tip=12
    # ring: mcp=13, tip=16
    # pinky: mcp=17, tip=20
    palm = landmarks_xyz[:, 0, :]
    thumb = _angle_deg(palm, landmarks_xyz[:, 2, :], landmarks_xyz[:, 4, :])
    index = _angle_deg(palm, landmarks_xyz[:, 5, :], landmarks_xyz[:, 8, :])
    middle = _angle_deg(palm, landmarks_xyz[:, 9, :], landmarks_xyz[:, 12, :])
    ring = _angle_deg(palm, landmarks_xyz[:, 13, :], landmarks_xyz[:, 16, :])
    pinky = _angle_deg(palm, landmarks_xyz[:, 17, :], landmarks_xyz[:, 20, :])
    return np.stack([thumb, index, middle, ring, pinky], axis=1)


def _build_default_output_path(input_path: Path) -> Path:
    stem = input_path.stem
    if "_finger5" in stem:
        stem = stem.replace("_finger5", "_palm-mcp-tip5")
    else:
        stem = f"{stem}_palm-mcp-tip5"
    return input_path.with_name(f"{stem}{input_path.suffix}")


def _convert_one(input_path: Path, output_path: Path, hand_idx: int):
    data = np.load(str(input_path), allow_pickle=True)

    if "angles" not in data.files:
        raise KeyError(f"{input_path} has no 'angles' key")

    lm_key = _find_landmark_key(data)
    if lm_key is None:
        raise KeyError(
            f"{input_path} has no landmark key (tried: landmarks, hand_landmarks, landmark_xyz, joint_xyz, xyz)"
        )

    landmarks_xyz = _to_landmarks_xyz(np.asarray(data[lm_key]), hand_idx=hand_idx)
    angles_new = _compute_palm_mcp_tip_angles(landmarks_xyz)

    angles_old = np.asarray(data["angles"])
    if angles_old.shape[0] != angles_new.shape[0]:
        raise ValueError(
            f"Sample mismatch: old angles N={angles_old.shape[0]} vs landmarks-derived N={angles_new.shape[0]}"
        )

    payload = {k: data[k] for k in data.files if k not in ("angles", "angle_keys", "target_spec")}
    payload["angles"] = angles_new.astype(np.float32)
    payload["angle_keys"] = np.asarray(PALM_MCP_TIP_KEYS, dtype=object)
    payload["target_spec"] = np.asarray("palm_mcp_tip5")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(output_path), **payload)


def main():
    parser = argparse.ArgumentParser(
        description="Create palm-MCP-tip 5-angle condition datasets from landmark-based NPZ files."
    )
    parser.add_argument("--data", nargs="+", required=True, help="Input NPZ file(s)")
    parser.add_argument(
        "--out_dir",
        type=str,
        default="",
        help="Optional output directory. If omitted, writes next to each input file.",
    )
    parser.add_argument(
        "--hand_idx",
        type=int,
        default=0,
        help="Hand index when landmark arrays contain multiple hands (default: 0)",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else None
    paths = [Path(p) for p in args.data]

    for in_path in paths:
        if not in_path.exists():
            raise FileNotFoundError(f"Input not found: {in_path}")
        out_path = _build_default_output_path(in_path)
        if out_dir is not None:
            out_path = out_dir / out_path.name
        _convert_one(in_path, out_path, hand_idx=args.hand_idx)
        print(f"[DONE] {in_path} -> {out_path}")


if __name__ == "__main__":
    main()
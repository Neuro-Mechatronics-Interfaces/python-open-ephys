from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Optional

import numpy as np

from pyoephys.interface import LSLClient, NotReadyError
from pyoephys.logging import configure


def lsl2npz_cli(argv: Optional[list[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="Capture from an LSL stream to an .npz file.")
    ap.add_argument("--name", default=None, help="Exact LSL stream name")
    ap.add_argument("--type", default="EMG", help="LSL stream type (default: EMG)")
    ap.add_argument("--duration", type=float, default=10.0, help="Seconds to record")
    ap.add_argument("--out", required=True, help="Output .npz path")
    ap.add_argument("--timeout", type=float, default=5.0, help="Resolve timeout")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)

    configure("INFO" if args.verbose else "WARNING")

    client = LSLClient(
        stream_name=args.name,
        stream_type=args.type,
        timeout_s=args.timeout,
        buffer_seconds=max(args.duration * 1.2, 10.0),
        verbose=args.verbose,
    )
    client.start()
    t0 = time.time()

    try:
        while time.time() - t0 < args.duration:
            time.sleep(0.1)
        # pull everything we have
        y, t = client.get_window(args.duration + 1.0)
    except NotReadyError:
        client.stop()
        raise SystemExit("No data received from LSL stream.")
    finally:
        client.stop()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, emg=y, timestamps=t, fs_hz=client.fs_hz, stream_name=args.name, stream_type=args.type)
    print(f"Saved {y.shape[1]} samples x {y.shape[0]} channels to {out}")


if __name__ == "__main__":
    lsl2npz_cli()

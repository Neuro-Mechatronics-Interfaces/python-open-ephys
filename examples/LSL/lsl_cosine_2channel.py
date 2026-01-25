import time
import numpy as np
from pylsl import StreamInfo, StreamOutlet, local_clock
import argparse


def cosine_publisher(
    name="CosineWave",
    stype="EMG",
    fs=2000,
    f0=1.0,
    amp=0.5,
    offset=0.0,
    chunk=64,
    phase_deg=0.0,   # 0.0 => identical channels; set e.g. 90 for a phase shift on ch1
):
    """
    Publish a 2-channel cosine on LSL using chunked writes + explicit timestamps.
    Channel 0: base cosine. Channel 1: same cosine (optionally phase-shifted).
    """
    n_channels = 2
    info = StreamInfo(name, stype, n_channels, fs, 'float32', 'cosine-2ch')

    # (optional) add channel labels to metadata
    chs = info.desc().append_child("channels")
    for i, label in enumerate(("ch0", "ch1")):
        ch = chs.append_child("channel")
        ch.append_child_value("label", label)
        ch.append_child_value("unit", "a.u.")

    outlet = StreamOutlet(info, chunk_size=chunk, max_buffered=int(fs * 10))
    print(f"[publisher] '{name}': fs={fs} Hz, f0={f0} Hz, chunk={chunk}, channels={n_channels}")

    t0 = local_clock()
    n_sent = 0
    phase = np.deg2rad(phase_deg)

    try:
        while True:
            # Monotonic sample indices -> perfect timebase
            idx = np.arange(n_sent, n_sent + chunk, dtype=np.int64)
            t = idx / fs

            # Build the two channels
            y0 = offset + amp * np.cos(2 * np.pi * f0 * t)
            y1 = offset + amp * np.cos(2 * np.pi * f0 * t + phase)  # same if phase=0

            # LSL expects shape (n_samples, n_channels)
            outlet.push_chunk(
                np.column_stack((y0, y1)).astype(np.float32),
                (t0 + t).tolist()  # One timestamp per row
            )

            # Sleep until the theoretical time of the next block
            n_sent += chunk
            next_wakeup = t0 + (n_sent / fs)
            wait = next_wakeup - local_clock()
            if wait > 0:
                time.sleep(wait)
    except KeyboardInterrupt:
        print(f"[publisher] '{name}': stopped after sending {n_sent} samples.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Publish a 2-ch cosine over LSL.")
    ap.add_argument("--name", default="CosineWave")
    ap.add_argument("--type", default="EMG")
    ap.add_argument("--fs", type=int, default=2000)
    ap.add_argument("--f0", type=float, default=1.0)
    ap.add_argument("--amp", type=float, default=0.5)
    ap.add_argument("--offset", type=float, default=0.0)
    ap.add_argument("--chunk", type=int, default=64)
    ap.add_argument("--phase", type=float, default=0.0, help="deg phase shift on ch1")
    args = ap.parse_args()
    cosine_publisher(args.name, args.type, args.fs, args.f0, args.amp, args.offset, args.chunk, args.phase)
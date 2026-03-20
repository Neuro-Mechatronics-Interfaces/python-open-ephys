"""
Test script to verify Open Ephys ZMQ streaming works.
Connects to localhost:5556, logs channel info and data snapshots to console.

Usage:
    python test_zmq_stream.py
    python test_zmq_stream.py --verbose
"""

import sys
import time
import argparse
import numpy as np
from pyoephys.interface import ZMQClient, NotReadyError


def main():
    parser = argparse.ArgumentParser(description="Test Open Ephys ZMQ stream")
    parser.add_argument("--host", default="127.0.0.1", help="Open Ephys host IP")
    parser.add_argument("--port", default="5556", help="ZMQ data port")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose client output")
    args = parser.parse_args()

    print(f"Connecting to Open Ephys ZMQ at {args.host}:{args.port} ...")

    client = ZMQClient(
        host_ip=args.host,
        data_port=args.port,
        auto_start=False,
        verbose=args.verbose,
    )

    client.start()
    print("Client started. Waiting for data...\n")

    try:
        while True:
            time.sleep(1.0)

            if not client.ready_event.is_set():
                print("[waiting] No data received yet...")
                continue

            # Auto-detect channels on first data
            if not client.channel_index:
                client.set_channel_index(sorted(client.seen_nums))
                print(f"[setup] Detected {len(client.channel_index)} channels: {client.channel_index}")
                print(f"[setup] Sample rate: {client.fs} Hz")
                print(f"[setup] Channel names: {client.channel_names}")
                print()

            try:
                # t, Y = client.drain_new()
                Y, t = client.get_latest(1)
                if Y is None or t is None:
                    print("[drain] No new samples")
                    continue
                # Y shape: (n_channels, n_samples), t shape: (n_samples,)
                print(f"[snapshot] shape={Y.shape}  t=[{t[0]:.3f}s .. {t[-1]:.3f}s]")
                for i, ch in enumerate(client.channel_index):
                    row = Y[i]
                    print(f"  CH{ch:>3d}: min={row.min():>10.2f}  max={row.max():>10.2f}  mean={row.mean():>10.2f}  std={row.std():>8.2f}")
                print()

            except NotReadyError as e:
                print(f"[not ready] {e}")

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        client.stop()
        print("Client stopped.")


if __name__ == "__main__":
    main()

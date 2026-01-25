
#!/usr/bin/env python3
"""
3cv2_predict_gui.py
-------------------
Connect to a running Open Ephys GUI via ZMQ and print the shape of incoming EMG
data for the *training channels* every interval (default: 0.5s).

- Uses pyoephys.interface.ZMQClient (ring-buffered).
- Locks channel selection to the training order by channel *names* from metadata.
- Prints shapes of the *new* samples received since the last print.

Example:
    python 3cv2_predict_gui.py --root_dir /path/to/session --label myrun \
        --zmq tcp://127.0.0.1 --data-port 5556 --heartbeat-port 5557 --interval 0.5
"""

import argparse
import logging
import signal
import sys
import time
from typing import List, Dict

import numpy as np

from pyoephys.interface import ZMQClient
from pyoephys.io import load_metadata_json, normalize_name


def _map_training_names_to_indices(client: ZMQClient, trained_names: List[str]) -> List[int]:
    """
    Build channel indices in the *training order* by matching names seen on the live stream.
    Falls back gracefully if a subset is available.
    """
    # Access the name map the client learns as data arrives
    name_by_idx: Dict[int, str] = getattr(client, "_name_by_index", {})
    norm_to_idx = {normalize_name(nm): idx for idx, nm in name_by_idx.items()}

    indices = []
    missing = []
    for nm in trained_names:
        nrm = normalize_name(nm)
        if nrm in norm_to_idx:
            indices.append(norm_to_idx[nrm])
        else:
            missing.append(nm)

    if missing:
        logging.warning("Some training channels not (yet) present on the live stream: %s", missing)
    if not indices:
        raise RuntimeError("No training channels were found on the live stream yet. "
                           "Let the GUI stream run and try again.")

    return indices


def main():
    ap = argparse.ArgumentParser("Connect to Open Ephys via ZMQ and print NEW data shapes periodically.")
    ap.add_argument("--root_dir", required=True, help="Root directory that contains metadata.json from training.")
    ap.add_argument("--label", default="", help="Optional label used during training to select metadata variant.")
    ap.add_argument("--zmq", default="tcp://127.0.0.1", help="ZMQ ip/prefix (e.g., tcp://127.0.0.1).")
    ap.add_argument("--data-port", type=int, default=5556, help="ZMQ data port.")
    ap.add_argument("--heartbeat-port", type=int, default=5557, help="ZMQ heartbeat port.")
    ap.add_argument("--interval", type=float, default=0.5, help="Seconds between prints.")
    ap.add_argument("--wait", type=float, default=15.0, help="Seconds to wait for required channels.")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--allow-partial", action="store_true",
                    help="Proceed if only a subset of training channels are present.")
    args = ap.parse_args()

    # Logging
    lvl = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(format='[%(levelname)s] %(message)s', level=lvl)

    # Load training channel names (canonical order)
    meta = load_metadata_json(args.root_dir, label=args.label)
    trained_names = meta.get("data", {}).get("channel_names") or meta.get("channel_names")
    if not trained_names:
        raise RuntimeError("metadata missing data.channel_names (training channel order).")

    logging.info("Expecting %d training channels (by name).", len(trained_names))

    # Connect ZMQ client
    client = ZMQClient(
        zqm_ip=args.zmq,
        http_ip="127.0.0.1",
        data_port=args.data_port,
        heartbeat_port=args.heartbeat_port,
        window_secs=5.0,
        channels=None,  # we'll populate *after* we learn channel-name->index map
        auto_start=True,
        verbose=args.verbose,
        expected_channel_names=trained_names,
        expected_channel_count=None,
        require_complete=not args.allow_partial,  # block latest()/drain_new() until ready when not allowing partial
        required_fraction=1.0,
        max_channels=512,
    )

    # Graceful shutdown
    stop = False
    def _sigint(_sig, _frm):
        nonlocal stop
        stop = True
        print("\n[Info] Ctrl-C received; closing...")
    signal.signal(signal.SIGINT, _sigint)

    # Wait for channels (up to args.wait seconds)
    ready = client.wait_for_channels(timeout_sec=args.wait)
    if not ready:
        if args.allow_partial:
            logging.warning("Proceeding with subset of training channels (timeout reached).")
        else:
            raise RuntimeError("Required training channels did not appear within --wait seconds. "
                               "Use --allow-partial to proceed with a subset.")

    # Map names we see on the live stream -> indices, in the training order
    indices = _map_training_names_to_indices(client, trained_names)
    client.set_channel_index(indices)
    logging.info("Selected %d/%d channels in training order.", len(indices), len(trained_names))

    # Initial status
    stat = client.get_connection_status()
    logging.info("Connected: %s | fs≈%.1f Hz | seen_channels=%s | n_total=%d",
                 stat["connected"], client.fs, stat["seen_channels"], stat["n_channels_total"])

    # Periodic loop: print shape of NEW data since last print
    last_print = time.time()
    prints = 0
    while not stop:
        t_new, Y_new = client.drain_new()
        if Y_new is not None:
            # Only print at the requested interval
            now = time.time()
            if now - last_print >= args.interval:
                prints += 1
                last_print = now
                # Y_new shape is (C, K) with C selected channels and K new samples collected since last drain
                print(f"[{prints:04d}] new samples: Y_new.shape={Y_new.shape} | fs≈{client.fs:.1f} Hz")
        # modest sleep to avoid busy spin; the print interval controls reporting frequency
        time.sleep(0.01)

    # Cleanup
    try:
        client.stop()
        client.close()
    except Exception:
        pass
    print("[Info] Closed. Bye.")


if __name__ == "__main__":
    main()

import time
import argparse
from pyoephys.io import get_example_oebin_path
from pyoephys.interface import OEBinPlaybackClient


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Playback OEBin file and stream via LSL.")
    parser.add_argument("--file_path", type=str, default=get_example_oebin_path("gestures"), help="Path to the OEBin file or directory containing OEBin files")
    parser.add_argument("--channels", nargs="+", default=None, help="Channels to stream (e.g., 0 1 2 3); None means all channels")
    parser.add_argument("--block_size", type=int, default=32, help="Block size for streaming (default: 32)")
    parser.add_argument("--stream_name", type=str, default="OEBinData", help="Name of the LSL stream (default: 'OEBinData')")
    parser.add_argument("--stream_type", type=str, default="EMG", help="Type of the LSL stream (default: 'EMG')")
    parser.add_argument("--loopback", action="store_true", help="Enable loopback mode (default: False)")
    parser.add_argument("--enable_lsl", action="store_true", help="Enable LSL streaming (default: True)")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output (default: False)")
    args = parser.parse_args()

    client = OEBinPlaybackClient(
        oebin_path=args.file_path,
        channels=args.channels,
        block_size=args.block_size,
        stream_name=args.stream_name,
        stream_type=args.stream_type,
        loopback=args.loopback,
        enable_lsl=args.enable_lsl,
        verbose=args.verbose
    )
    client.start()

    try:
        while True:
            time.sleep(0.25)
    except KeyboardInterrupt:
        print("Stopped by user.")

    finally:
        client.stop()

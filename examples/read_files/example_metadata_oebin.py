"""
Print the metadata contents of an Open Ephys ``.oebin`` file.

This does *not* load the raw signal data — it only parses ``structure.oebin``
(JSON) and prints a readable summary of the recording: processors, continuous
streams, sample rates, channel counts, channel names, units, bit-volts, and any
event/spike streams.

Usage::

    python example_metadata_oebin.py path/to/structure.oebin
    python example_metadata_oebin.py            # opens a file-picker dialog
"""
import argparse
import json
import os
import sys

from pyoephys.io import prompt_file


def parse_cli_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print the metadata contents of an Open Ephys .oebin file."
    )
    parser.add_argument("path", nargs="?", help="Path to a .oebin file.")
    parser.add_argument(
        "--max-channels",
        type=int,
        default=16,
        help="How many channel names to list per stream before truncating (default 16).",
    )
    return parser.parse_args(sys.argv[1:] if argv is None else argv)


def resolve_oebin_path(path_arg: str | None = None) -> str:
    if path_arg:
        return path_arg

    path = prompt_file(
        title="Select a .oebin file",
        filetypes=[("Open Ephys metadata", "*.oebin"), ("All files", "*.*")],
    )
    if not path:
        raise SystemExit("No .oebin file selected.")
    return path


def _listify(value) -> list:
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        return list(value.values())
    return []


def _format_channel_names(channels: list[dict], max_channels: int) -> str:
    names = [str(ch.get("channel_name") or ch.get("name") or f"ch{i}") for i, ch in enumerate(channels)]
    if len(names) <= max_channels:
        return ", ".join(names)
    head = ", ".join(names[:max_channels])
    return f"{head}, ... (+{len(names) - max_channels} more)"


def _summarize_units(channels: list[dict]) -> str:
    units = sorted({str(ch.get("units") or "?") for ch in channels})
    bitvolts = sorted({float(ch.get("bit_volts")) for ch in channels if ch.get("bit_volts") is not None})
    parts = []
    if units:
        parts.append("units=" + "/".join(units))
    if bitvolts:
        bv = ", ".join(f"{b:g}" for b in bitvolts)
        parts.append(f"bit_volts={bv}")
    return "  (" + "; ".join(parts) + ")" if parts else ""


def print_oebin_metadata(oebin_path: str, max_channels: int = 16) -> dict:
    if not os.path.isfile(oebin_path):
        raise FileNotFoundError(f"Not a file: {oebin_path}")

    with open(oebin_path, "r", encoding="utf-8") as handle:
        meta = json.load(handle)

    print("=" * 70)
    print(f".oebin file : {os.path.abspath(oebin_path)}")
    print(f"top-level keys: {sorted(meta.keys())}")
    if "GUI version" in meta:
        print(f"GUI version : {meta['GUI version']}")
    print("=" * 70)

    continuous = _listify(meta.get("continuous"))
    print(f"\nContinuous streams: {len(continuous)}")
    for idx, stream in enumerate(continuous):
        channels = _listify(stream.get("channels"))
        fs = stream.get("sample_rate") or stream.get("sampleRate") or stream.get("rate")
        declared = stream.get("num_channels") or stream.get("channel_count")
        print(f"\n  [{idx}] {stream.get('folder_name') or stream.get('stream_name') or stream.get('name') or 'continuous'}")
        print(f"       source           : {stream.get('source_processor_name', '?')} "
              f"(id {stream.get('source_processor_id', '?')})")
        print(f"       sample_rate      : {fs} Hz")
        print(f"       num_channels     : declared={declared}, listed={len(channels)}")
        if declared is not None and len(channels) and int(declared) != len(channels):
            print(f"       ** WARNING: declared num_channels ({declared}) != listed channels "
                  f"({len(channels)}); the loader will trust the listed count.")
        print(f"       channel_names    : {_format_channel_names(channels, max_channels)}"
              f"{_summarize_units(channels)}")

    events = _listify(meta.get("events"))
    if events:
        print(f"\nEvent streams: {len(events)}")
        for idx, ev in enumerate(events):
            print(f"  [{idx}] {ev.get('folder_name') or ev.get('channel_name') or 'events'} "
                  f"(type={ev.get('type', '?')}, channels={ev.get('num_channels', '?')})")

    spikes = _listify(meta.get("spikes"))
    if spikes:
        print(f"\nSpike streams: {len(spikes)}")
        for idx, sp in enumerate(spikes):
            print(f"  [{idx}] {sp.get('folder_name') or 'spikes'} "
                  f"(channels={sp.get('num_channels', '?')})")

    print("\n" + "=" * 70)
    return meta


def main(argv: list[str] | None = None) -> None:
    args = parse_cli_args(argv)
    path = resolve_oebin_path(args.path)
    print_oebin_metadata(path, max_channels=args.max_channels)


if __name__ == "__main__":
    main()

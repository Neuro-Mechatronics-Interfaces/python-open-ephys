import argparse
import tkinter as tk
from pathlib import Path
from tkinter import messagebox
from typing import List, Union

from ._config_utils import (
    prompt_channel_grid,
    prompt_directory,
    prompt_file,
    prompt_options,
    prompt_save_file,
)
from ._dataset_utils import save_session_to_mat, save_session_to_npz
from ._session_loader import load_open_ephys_session


def convert_session_api(
    input_data: Union[str, dict],
    output_path: str = None,
    output_format: str = "mat",
    selected_channels: List[str] = None,
):
    """
    Programmatic interface for converting sessions.

    Args:
        input_data: Path (str) or loaded session dict
        output_path: Destination file path.
        output_format: "mat" or "npz" (default: "mat")
        selected_channels: List of channel names to include. If None, keep all.
    """
    if isinstance(input_data, (str, Path)):
        p_in = Path(input_data)
        if not p_in.exists():
            raise FileNotFoundError(f"Input path not found: {input_data}")

        print(f"Loading session: {input_data}")
        session = load_open_ephys_session(input_data)
        print("Session loaded successfully.")
    else:
        session = input_data
        p_in = None

    # Filter channels if requested
    if selected_channels is not None:
        all_names = session.get("channel_names", [])
        indices = [i for i, name in enumerate(all_names) if name in selected_channels]

        if not indices:
            print("Warning: No channels matched selection. Saving all or empty?")
            # Proceeding might save empty arrays

        print(f"Filtering {len(indices)} / {len(all_names)} channels...")

        # Slicing: amplifier_data is (C, S)
        session["amplifier_data"] = session["amplifier_data"][indices, :]
        session["channel_names"] = [all_names[i] for i in indices]

    if not output_path:
        if not p_in:
            raise ValueError("Output path required when passing session dict directly.")

        stem = p_in.stem
        # For folders, use parent dir logic
        if p_in.is_dir():
            stem = p_in.name
            out_dir = p_in.parent
        else:
            out_dir = p_in.parent

        output_path = out_dir / f"{stem}_converted.{output_format}"

    print(f"Converting to {output_format.upper()}...")
    if output_format.lower() == "mat":
        save_session_to_mat(session, str(output_path))
    elif output_format.lower() == "npz":
        save_session_to_npz(session, str(output_path))
    else:
        raise ValueError(f"Unknown format: {output_format}")

    print(f"Successfully saved to: {output_path}")
    return str(output_path)


def convert_session_ui():
    """
    Interactive UI for converting an Open Ephys session.
    """
    parser = argparse.ArgumentParser(
        description="Convert Open Ephys/XDF sessions to MAT/NPZ"
    )
    parser.add_argument("--input", "-i", type=str, help="Input file or folder")
    parser.add_argument("--output", "-o", type=str, help="Output file path (optional)")
    parser.add_argument(
        "--format", "-f", type=str, choices=["mat", "npz"], help="Output format"
    )
    parser.add_argument("--adc", action="store_true", help="Include ADC channels")
    parser.add_argument("--aux", action="store_true", help="Include AUX channels")
    parser.add_argument(
        "--channels",
        type=str,
        help="Select channels by range (e.g. '1-32') or name list",
    )

    args = parser.parse_args()

    input_path = args.input
    output_path = args.output
    fmt = args.format

    # 1. Select Input
    if not input_path:
        print("No input provided. Opening selection dialog...")

        # Ask Type
        try:
            choice = prompt_options(
                "Select Input Type",
                "Load session from File or Folder?",
                ["File (.xdf, .npz)", "Folder (Open Ephys)"],
            )
        except Exception:
            choice = None

        if not choice:
            print("Operation cancelled.")
            return

        if "Folder" in choice:
            input_path = prompt_directory("Select Open Ephys Session Folder")
        else:
            input_path = prompt_file(
                "Select Session File",
                filetypes=[
                    ("All Supported", "*.xdf *.npz *.oebin"),
                    ("LabRecorder XDF", "*.xdf"),
                    ("NPZ Archive", "*.npz"),
                ],
            )

        if not input_path:
            print("Operation cancelled.")
            return

    if not Path(input_path).exists():
        print(f"Error: Input path does not exist: {input_path}")
        return

    # Load session EARLY to allow channel selection
    try:
        print(f"Loading session metadata from: {input_path}")
        # Note: loading full session here. Optimization: read only metadata?
        # For now, consistent with API.
        session = load_open_ephys_session(input_path)
    except Exception as e:
        print(f"Failed to load session: {e}")
        messagebox.showerror("Load Error", str(e))
        return

    # 1.5 Channel Selection
    all_channels = session.get("channel_names", [])
    selected_channels = None

    # Pre-filter logic using args
    if args.adc or args.aux or args.channels:
        selected_channels = []

        # Handle --channels (ranges/indices)
        if args.channels:
            parts = args.channels.split(",")
            for part in parts:
                part = part.strip()
                if "-" in part:
                    # Range: 1-32
                    try:
                        start_str, end_str = part.split("-")
                        start = int(start_str)
                        end = int(end_str)
                        # Assume 1-based indexing for user cli
                        # 1-32 -> indices 0 to 31 -> slice [0:32]
                        # Python slice is [start-1 : end]
                        selected_channels.extend(all_channels[max(0, start - 1) : end])
                    except ValueError:
                        print(f"Warning: Invalid range format '{part}'. Ignoring.")
                elif part.isdigit():
                    # Single index
                    idx = int(part) - 1
                    if 0 <= idx < len(all_channels):
                        selected_channels.append(all_channels[idx])
                else:
                    # Literal name match?
                    if part in all_channels:
                        selected_channels.append(part)
                    else:
                        print(f"Warning: Channel '{part}' not found.")

        # Handle flags
        if args.adc or args.aux:
            for ch in all_channels:
                name_upper = ch.upper()
                if args.adc and "ADC" in name_upper:
                    if ch not in selected_channels:
                        selected_channels.append(ch)
                if args.aux and "AUX" in name_upper:
                    if ch not in selected_channels:
                        selected_channels.append(ch)

        if not selected_channels:
            print("Warning: Arguments provided but no channels matched.")
            # If explicit args given but fail, probably shouldn't default to all.
            # But let's ask user? Or failing is safer.

    else:
        # Interactive Selection with Grid View

        # Default checked: Everything?
        # Usually users want to select specific blocks if asking for grid.
        # But for backward compatibility, select all is safe default.
        chosen = prompt_channel_grid(
            "Select Channels (Drag to Select)",
            channels=all_channels,
            defaults=[True] * len(all_channels),
        )
        if not chosen:
            print("No channels selected. Operation cancelled.")
            return
        selected_channels = chosen

    # 2. Select Format
    if not fmt:
        if output_path and output_path.endswith(".mat"):
            fmt = "mat"
        elif output_path and output_path.endswith(".npz"):
            fmt = "npz"
        else:
            try:
                fmt_long = prompt_options(
                    "Select Output Format",
                    "Convert to which format?",
                    ["MATLAB (.mat)", "NumPy (.npz)"],
                )
                if fmt_long:
                    fmt = "mat" if "mat" in fmt_long.lower() else "npz"
                else:
                    fmt = "mat"  # Default
            except Exception:
                fmt = "mat"

    # 3. Determine Output Filename
    if not output_path:
        p = Path(input_path)
        stem = p.name if p.is_dir() else p.stem
        # For standard OE date strings, keep them

        default_name = f"{stem}_converted.{fmt}"
        initial_dir = p.parent

        # Ask user for save location
        output_path = prompt_save_file(
            title="Save Converted File",
            initial_dir=str(initial_dir),
            initial_file=default_name,
            defaultextension=f".{fmt}",
            filetypes=[(f"{fmt.upper()} file", f"*.{fmt}")],
        )

        if not output_path:
            print("Save cancelled.")
            return

    # 4. Perform Conversion via API (passing the loaded session dict)
    try:
        convert_session_api(
            session, output_path, fmt, selected_channels=selected_channels
        )

        # Show success GUI if interactive
        if not args.input:
            root = tk.Tk()
            root.withdraw()
            messagebox.showinfo("Success", f"Successfully saved to:\n{output_path}")
            root.destroy()

    except Exception as e:
        print(f"Error converting session: {e}")
        if not args.input:
            root = tk.Tk()
            root.withdraw()
            messagebox.showerror("Error", str(e))
            root.destroy()

import os
import sys
from pathlib import Path

# Try importing the newly created functions
try:
    from pyoephys.io import (
        load_open_ephys_session,
        save_session_to_mat,
        save_session_to_npz,
    )

    print("[SUCCESS] Imported pyoephys.io functions.")
except ImportError as e:
    print(f"[ERROR] Failed to import pyoephys.io: {e}")
    sys.exit(1)


def test_conversion(input_path, output_format="mat"):
    if not os.path.exists(input_path):
        print(f"[ERROR] Input path does not exist: {input_path}")
        return

    print(f"Loading session from: {input_path}")
    try:
        session = load_open_ephys_session(input_path)
    except Exception as e:
        print(f"[ERROR] Failed to load session: {e}")
        import traceback

        traceback.print_exc()
        return

    # Basic Checks
    print("-" * 40)
    print("Loaded Data:")
    print(f"  - Amplifier Data Shape: {session['amplifier_data'].shape}")
    print(f"  - Time Vector Shape: {session['t_amplifier'].shape}")
    print(f"  - Sampling Rate: {session['sample_rate']} Hz")
    print(f"  - Channels: {len(session['channel_names'])}")
    if "events" in session and session["events"]:
        print(f"  - Events Found: Yes ({len(session['events'])} keys)")
        for k, v in session["events"].items():
            if hasattr(v, "shape"):
                print(f"    * {k}: {v.shape}")
            elif isinstance(v, dict):
                print(f"    * {k}: dict with keys {list(v.keys())}")
            else:
                print(f"    * {k}: {type(v)}")
    else:
        print("  - Events Found: No")
    print("-" * 40)

    # Conversion
    input_stem = Path(input_path).stem
    if output_format == "mat":
        out_file = f"{input_stem}_converted.mat"
        print(f"Converting to MAT: {out_file}")
        try:
            save_session_to_mat(session, out_file)
            print("[SUCCESS] Conversion complete.")
        except Exception as e:
            print(f"[ERROR] MAT conversion failed: {e}")
            import traceback

            traceback.print_exc()

    elif output_format == "npz":
        out_file = f"{input_stem}_converted.npz"
        print(f"Converting to NPZ: {out_file}")
        try:
            save_session_to_npz(session, out_file)
            print("[SUCCESS] Conversion complete.")
        except Exception as e:
            print(f"[ERROR] NPZ conversion failed: {e}")


if __name__ == "__main__":
    # Default test file if none provided
    default_file = "ooo_2026-03-03_15-15-18-20260303T230615Z-3-001"

    if len(sys.argv) > 1:
        target = sys.argv[1]
    else:
        target = default_file

    print(f"Running conversion test on: {target}")
    test_conversion(target)

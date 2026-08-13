import sys
import argparse
from PyQt5.QtWidgets import QApplication
from pyoephys.interface._lsl_client import OldLSLClient
from pyoephys.plotting import StackedPlot
from pyoephys.io import parse_numeric_args


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch real-time EMG stacked plot from LSL stream.")
    parser.add_argument("--channels", nargs="+", default=["0", "1", "2", "3"],
                        help="Channels to plot: e.g., --channels 0 1 2 or --channels 0:64 or --channels all")
    parser.add_argument("--stream_name", type=str, default=None, help="LSL stream name to look for (default: None)")
    parser.add_argument("--stream_type", type=str, default=None, help="LSL stream type to look for (default: EMG when --stream_name is not set)")
    parser.add_argument("--ylim", type=float, nargs=2, default=[-1.0, 1.0], help="Y-axis limits for the plot (default: [-1.0, 1.0])")
    parser.add_argument("--downsample", type=int, default=1, help="Downsample factor (e.g., 2, 5, 10)")
    args = parser.parse_args()
    if args.stream_name is not None and args.stream_type is not None:
        parser.error("Use either --stream_name or --stream_type, not both.")

    # Parse channel selection
    channels = parse_numeric_args(args.channels)
    if channels == "all":
        channels = None
    print(f"Channels to plot: {channels}")
    stream_type = (args.stream_type or "EMG") if args.stream_name is None else None

    # Launch the Qt Application
    app = QApplication(sys.argv)

    client = OldLSLClient(stream_name=args.stream_name, stream_type=stream_type, channels=channels)
    client.start()

    # Create and launch the stacked plotter
    plotter = StackedPlot(
        client=client,
        auto_ylim=False,
        y_limits=tuple(args.ylim),
    )
    plotter.show()
    sys.exit(app.exec_())

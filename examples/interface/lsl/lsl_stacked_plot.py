import sys
import argparse
from PyQt5.QtWidgets import QApplication
from pyoephys.interface import LSLClient
from pyoephys.plotting import StackedPlot
from pyoephys.io import parse_numeric_args


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch real-time EMG stacked plot from LSL stream.")
    parser.add_argument("--channels", nargs="+", default=["0", "1", "2", "3"],
                        help="Channels to plot: e.g., --channels 0 1 2 or --channels 0:64 or --channels all")
    parser.add_argument("--stream_name", type=str, default=None, help="LSL stream name to look for (default: None)")
    parser.add_argument("--stream_type", type=str, default=None, help="LSL stream type to look for (default: None)")
    parser.add_argument("--ylim", type=float, nargs=2, default=[-1.0, 1.0], help="Y-axis limits for the plot (default: [-1.0, 1.0])")
    parser.add_argument("--downsample", type=int, default=1, help="Downsample factor (e.g., 2, 5, 10)")
    args = parser.parse_args()

    # Parse channel selection
    channels = parse_numeric_args(args.channels)
    print(f"Channels to plot: {channels}")

    # Launch the Qt Application
    app = QApplication(sys.argv)

    client = LSLClient(stream_name=args.stream_name, stream_type=args.stream_type, channels=channels)
    client.start()

    # Create and launch the stacked plotter
    plotter = StackedPlot(
        client=client,
        auto_ylim=False,
    )
    plotter.show()
    sys.exit(app.exec_())

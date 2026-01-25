import sys
import argparse
from PyQt5.QtWidgets import QApplication
from pyoephys.interface import ZMQClient
from pyoephys.plotting import StackedPlot
from pyoephys.io import parse_numeric_args


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch real-time EMG stacked plot from LSL stream.")
    parser.add_argument("--channels", nargs="+", default=["7", "8", "9", "10"],
                        help="Channels to plot: e.g., --channels 0 1 2 or --channels 0:64 or --channels all")
    parser.add_argument("--downsample", type=int, default=1, help="Downsample factor (e.g., 2, 5, 10)")
    args = parser.parse_args()

    # Parse channel selection
    channels_to_plot = parse_numeric_args(args.channels)
    print(f"Channels to plot: {channels_to_plot}")

    # Launch the Qt Application
    app = QApplication(sys.argv)

    client = ZMQClient()

    # Create and launch the stacked plotter
    plotter = StackedPlot(
        client=client,
        channels_to_plot=channels_to_plot,
        interval_ms=100,
        buffer_secs=4,
        downsample_factor=args.downsample,
    )
    plotter.start()
    sys.exit(app.exec_())

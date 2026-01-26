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
    channels = parse_numeric_args(args.channels)
    print(f"Channels to plot: {channels}")

    # Launch the Qt Application
    app = QApplication(sys.argv)

    client = ZMQClient(
        host_ip="127.0.0.1",
        data_port="5556",  # your Open Ephys ZMQ data port
        verbose=True,
    )
    client.set_channel_index(channels)
    client.start()

    # Create and launch the stacked plotter
    plotter = StackedPlot(
        client=client,
        auto_ylim=True
    )
    plotter.show()
    sys.exit(app.exec_())

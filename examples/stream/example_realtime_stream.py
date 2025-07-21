import argparse
from src.old_utils.ephys_utilities import OpenEphysClient
from pyoephys.plotting import RealtimePlotter

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", action='store_true', help="Print verbose output")
    args = parser.parse_args()

    # 1. Create OpenEphysClient instance
    client = OpenEphysClient(data_port=5556, verbose=args.verbose)

    # 2. Create the real-time plotter
    plotter = RealtimePlotter(
        client=client,
        sampling_rate=2000.0,       # Match your Open Ephys config
        plotting_interval=1.0,      # Display 1 second window
        samples_per_fetch=50,        # Control update resolution
        channels_to_plot=[8,9,10,11, 108,109,110,111]  # Customize this list
    )

    # 3. Run the interactive plot
    plotter.run()
    print("Plot window closed. Exiting.")

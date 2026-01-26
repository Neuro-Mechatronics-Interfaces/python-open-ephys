""" This example script demonstrates how to use the OpenEphysClient class to connect to the Open Ephys GUI and empty the
    buffer at a specified rate
"""

import sys
import time
import argparse
from pyoephys.interface import ZMQClient


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Open Ephys Client')
    parser.add_argument("--verbose", type=bool, default=False, help="Print verbose output")
    args = parser.parse_args()

    client = ZMQClient(
        data_port="5556",
        auto_start=False,
        set_index_looping=True,
        verbose=args.verbose,
    )

    try:
        client.gui.start_acquisition()
        client.start()
        while True:
            # Get the latest samples from the Open Ephys GUI
            time.sleep(1)
            # Empty the buffer
            #sample = client.get_latest(2000)
            #if sample:
            #    print(f"Number of samples: {len(sample)}")

    except KeyboardInterrupt as e:
        print("Keyboard interrupt")
        sys.exit(0)

    finally:
        client.stop()
        client.gui.stop_acquisition()
        print("Client stopped")



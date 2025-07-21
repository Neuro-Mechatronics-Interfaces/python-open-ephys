""" This example script demonstrates how to use the OpenEphysClient class to connect to the Open Ephys GUI and empty the
    buffer at a specified rate
"""

import sys
import time
import argparse
from pyoephys.interface import OpenEphysClient


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Open Ephys Client')
    parser.add_argument("--verbose", type=bool, default=False, help="Print verbose output")
    args = parser.parse_args()

    try:
        client = OpenEphysClient(buffer_len=1000, verbose=args.verbose)
        while True:
            # Empty the buffer
            sample = client.get_samples(channel=0)
            if sample:
                print(f"Number of samples: {len(sample)}")

            time.sleep(0.001)

    except KeyboardInterrupt as e:
        print("Keyboard interrupt")
        sys.exit(0)



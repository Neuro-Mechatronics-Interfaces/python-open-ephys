import os
import sys
import argparse
import numpy as np
import torch
import asyncio
from collections import deque

# Add the parent directory to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir)

# Get utilities
from src.old_utils import emg_processing as emg_proc, ephys_utilities as ephys_utils
from old_utils import emg_processing as emg_proc
from src.old_utils.ml_utilities import EMGCNN


class EphysEMGDecoder:
    """Real-time gesture classification using Open EPhys system """

    def __init__(self, config_path, use_serial=False, port='COM5', verbose=False):
        """
        Initialize the EphysEMGDecoder object

        Parameters
        ----------
            config_path  (str):  Path to the configuration file
            use_serial  (bool):  Use serial communication with PicoMessager
            port         (str):  COM port for PicoMessager. Windows machines use COMXX
            verbose     (bool):  Enable verbose output
        """
        self.verbose = verbose
        self.data_queue = asyncio.Queue()
        self.serial = PicoMessager(port=port, baudrate=9600) if use_serial else None

        # Load experiment configuration
        self.cfg = emg_proc.read_config_file(config_path)
        self.sample_rate = int(self.cfg.get("sample_rate", 30000))  # Default: 30kHz
        self.window_size = int(100/1000 * self.sample_rate)  # 100ms RMS window
        self.data_buffer = deque(maxlen=self.window_size)  # RMS buffer to store EMG data

        # Load the gesture labels
        self.gesture_labels_dict = self._load_gesture_labels()

        # Initialize the Open Ephys client
        self.client = ephys_utils.OpenEphysClient(verbose=self.verbose)

        # Load the trained model
        self.n_channels = None # Set when loading the model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model(os.path.join(self.cfg["root_directory"], self.cfg['model_filename']))

        print("✅ EphysEMGDecoder initialized.")

    def _load_model(self, model_path):
        """Loads a trained PyTorch model for gesture classification."""
        checkpoint = torch.load(model_path, map_location=self.device)
        self.n_channels = checkpoint['conv1.0.weight'].shape[1] # Number of input channels
        print(f"Model expects {self.n_channels} input channels. Updating configuration.")

        model = EMGCNN(num_classes=len(self.gesture_labels_dict), input_channels=self.n_channels)
        model.load_state_dict(checkpoint)
        model.to(self.device)
        model.eval()  # Set model to evaluation mode
        print("✅ Loaded PyTorch model successfully.")
        return model

    def _load_gesture_labels(self):
        """Load gesture labels from metrics file."""
        metrics_filepath = os.path.join(self.cfg["root_directory"], self.cfg["metrics_filename"])
        metrics_data, gesture_map = emg_proc.load_metrics_data(metrics_filepath)
        return {v: k for k, v in gesture_map.items()}  # Reverse mapping {index: gesture}

    async def start(self):
        """ Run the main routine with sampling, processing, and decoding tasks."""
        await asyncio.gather(
            self.sample_data(),
            self.decode_gesture()
        )

        if self.serial:
            self.serial.close()

    async def sample_data(self):
        """ Routing that continuously samples data from a connected Ephys system if available"""
        print("Starting data sampling with Ephys system...")
        while True:
            try:
                emg_samples = np.array(self.client.get_latest_sample())
                if emg_samples is not None:

                    # Reshape data and enforce the max channel count: (n_channels, n_samples)
                    emg_samples = np.expand_dims(emg_samples, axis=1)[:self.n_channels, :]

                    # Append to buffer
                    self.data_buffer.append(emg_samples)

                    # Only when we have enough data for the rms processing will we add data to the queue
                    if len(self.data_buffer) == self.window_size:
                        emg_samples = np.hstack(self.data_buffer)
                        await self.data_queue.put(emg_samples)
                        self.data_buffer.clear()

            except Exception as e:
                print(f"Error sampling data: {e}")

            await asyncio.sleep(0) # Prevents blocking. Can adjust sleep time for faster or slower sampling

    async def decode_gesture(self):
        """ Process EMG data from the queue and perform real-time predictions using the trained model"""
        print("Starting real-time gesture decoding...")

        while True:
            emg_data = await self.data_queue.get()
            if np.all(emg_data == 0):
                print("No EMG data detected. Check to see if the Ephys GUI is running.")
                continue

            # # Preprocess the EMG signal, output should be rms features
            #rms_features = emg_proc.preprocess_emg(emg_data, self.sample_rate, window_duration=100)
            # Using only the raw EMG data for now
            rms_features = emg_data

            # Ensure correct shape for CNN: (batch_size=1, channels, sequence_length)
            feature_tensor = torch.tensor(rms_features.T, dtype=torch.float32).unsqueeze(-1).to(self.device)

            # Predict the gesture
            with torch.no_grad():
                predictions = self.model(feature_tensor)
                pred_idx = torch.argmax(predictions, dim=1).item()
                gesture_str = self.gesture_labels_dict.get(pred_idx, "Unknown")
                print(f"Predicted Gesture: {gesture_str}")

                # Update PicoMessager with the detected gesture
                if self.serial:
                    self.serial.update_gesture(gesture_str)

            # Prevent blocking
            await asyncio.sleep(0.001)  # Lower sleep time for real-time performance

# Main Execution
if __name__ == "__main__":
    args = argparse.ArgumentParser(description='Real-time EMG gesture decoding using a trained model.')
    args.add_argument('--config_path', type=str, default='../config.txt', help='Path to the config file containing the directory of .rhd files.')
    args.add_argument('--use_serial', type=bool, default=False, help='Use serial communication with PicoMessager.')
    args.add_argument('--port', type=str, default='/dev/ttyACM0', help='COM port for PicoMessager. Windows machines use COMXX')
    args.add_argument('--verbose', action='store_true', help='Enable verbose output.')
    args = args.parse_args()

    # Initialze the decoder
    ephys_decoder = EphysEMGDecoder(config_path=args.config_path, use_serial=args.use_serial, port=args.port, verbose=args.verbose)
    asyncio.run(ephys_decoder.start())

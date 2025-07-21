""" Script to train a new classification model for gestures using EMG data.

Usage:
  python train.py --config_path "/mnt/c/Users/NML/Desktop/hdemg_test/Jonathan/2025_02_25/CONFIG.txt"
"""

import os
import sys
import torch
import argparse
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, random_split

# Add the parent directory to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir)

from src.old_utils import emg_processing as emg_proc, ml_utilities as ml_utils, ephys_utilities as ephys_utils


def process_emg_data(file_paths, metrics_data, gesture_map, n_channels, verbose):
    X_list, y_list = [], []
    for file in file_paths:
        result, data_present = ephys_utils.load_file(file, verbose=verbose)
        if not data_present:
            print(f"⚠️ Warning: No data found in {file}. Skipping.")
            continue

        emg_data = result['amplifier_data']
        print(f"EMG data shape: {emg_data.shape}")
        # If greater than 128 channels trim
        if emg_data.shape[1] > n_channels:
            emg_data = emg_data[:, :n_channels]

            # downsample the data by taking every 10th sample
            emg_data = emg_data[::50, :]
            print(f"| Trimmed EMG data shape: {emg_data.shape}")

        sample_rate = int(result['frequency_parameters']['amplifier_sample_rate'])
        print(f"| Sample rate: {sample_rate}")

        # # Preprocess the EMG signal, output should be rms features
        # rms_features = emg_proc.preprocess_emg(emg_data.T, sample_rate)
        # print(f"Processed file: {file}, RMS feature shape: {rms_features.shape}")
        # Instead of computing the RMS features, we will use the raw EMG data
        rms_features = emg_data.T

        # Retrieve gesture label
        file_name = os.path.basename(file)
        if file_name not in metrics_data['File Name'].values:
            print(f"⚠️ Warning: No entry found for {file_name} in metrics data. Skipping.")
            continue

        gesture = metrics_data[metrics_data['File Name'] == file_name]['Gesture'].values[0]

        # Append to lists
        X_list.append(rms_features.T)  # Shape (N_samples, 128)
        y_list.append(np.full(rms_features.shape[1], gesture_map[gesture]))

    return X_list, y_list

def train_emg_classifier(config_path, n_channels, epochs, verbose):
    """ Main function that trains a new classification model for gestures using EMG data. """
    # hard code the path to the configuration file. Will use the argparse module to pass this in the future
    cfg = emg_proc.read_config_file(config_path)

    # Load the metrics data
    metrics_filepath = os.path.join(cfg['root_directory'], cfg['metrics_filename'])
    metrics_data, gesture_map = emg_proc.load_metrics_data(metrics_filepath)
    valid_folderpaths = metrics_data['File Name'].values

    # Load and filter file paths
    folder_paths = emg_proc.get_file_paths(cfg['root_directory'], verbose=verbose)
    file_paths = [path for path in folder_paths if os.path.basename(path) in valid_folderpaths]

    print("Filtered file paths")
    for i in file_paths:
        print(i)

    # Process EMG data
    X_list, y_list = process_emg_data(file_paths, metrics_data, gesture_map, n_channels, verbose=verbose)

    # Convert EMG data and labels into tensors
    X_tensor, y_tensor = ml_utils.convert_lists_to_tensors(X_list, y_list)
    num_classes = torch.unique(y_tensor).shape[0]
    print(f"Final unique labels in y_tensor: {torch.unique(y_tensor)}")

    # Prepare Datasets into training and validation
    dataset = TensorDataset(X_tensor, y_tensor)
    train_size = int(0.8 * len(dataset))  # 80% training, 20% testing
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    # Model training
    print("Training model...")
    t_start = time.time()
    model = ml_utils.EMGCNN(num_classes=num_classes, input_channels=X_tensor.shape[1])
    train_losses, val_losses, val_accuracies = ml_utils.train_pytorch_model(model, train_loader, val_loader, num_epochs=epochs)

    print(f"Training completed in {time.time() - t_start:.2f} seconds.")

    # Save trained model and training metrics
    model_savepath = os.path.join(cfg['root_directory'], cfg["model_filename"])
    model.save(model_savepath)

    # training results file
    results_savepath = os.path.join(cfg['root_directory'], "training_results.txt")
    with open(results_savepath, 'w') as f:
        f.write(f"Train Losses: {train_losses}\n")
        f.write(f"Validation Losses: {val_losses}\n")
        f.write(f"Validation Accuracies: {val_accuracies}\n")
    print(f"Saved training results to {results_savepath}")

    # Plot the training and validation losses
    ml_utils.plot_training_metrics(train_losses, val_losses, val_accuracies)


if __name__ == "__main__":

    # Set up argument parser
    parser = argparse.ArgumentParser(description='Preprocess EMG data to extract gesture timings.')
    parser.add_argument('--config_path', type=str, default='config.txt', help='Path to the config file containing the directory of .rhd files.')
    parser.add_argument("--n_channels", type=int, default=128, help='Number of channels in the EMG data to use for training.')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train the model.')
    parser.add_argument('--verbose', type=bool, default=False, help='Whether to print verbose output.')
    args = parser.parse_args()

    # Perform training
    train_emg_classifier(args.config_path, args.n_channels, args.epochs, args.verbose)

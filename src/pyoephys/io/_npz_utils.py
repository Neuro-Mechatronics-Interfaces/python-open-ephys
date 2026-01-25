import os
import numpy as np


def save_as_npz(result: dict, file_path: str = None):
    """
    Save the Open Ephys session data as a .npz file.

    Args:
        result (dict): Dictionary containing the Open Ephys session data.
            Must contain keys: 'amplifier_data', 't_amplifier', 'sample_rate', 'recording_name'.
        file_path (str, optional): Path to save the .npz file. If None, uses the recording name.

    Returns:
        None
    """
    if not isinstance(result, dict):
        raise ValueError("Input must be a dictionary containing Open Ephys session data.")

    required_keys = ['amplifier_data', 't_amplifier', 'sample_rate', 'recording_name']
    if not all(key in result for key in required_keys):
        raise KeyError(f"Missing one of the required keys: {required_keys}")

    if file_path is None:
        # Save to local directory using file name
        file_path = result['recording_name'] + '.npz'
    elif not file_path.endswith('.npz'):
        file_path += '.npz'

    # Ensure the directory exists
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

    # Save the data to a .npz file
    print(f" Saving data to {file_path}...")

    # For every key in result, save it to the .npz file
    np.savez(file_path, **{key: result[key] for key in result.keys()})
    #np.savez(file_path,
    #         amplifier_data=result['amplifier_data'],
    #         t_amplifier=result['t_amplifier'],
    #         sample_rate=result['sample_rate']
    #         )
    print(f"Data saved to {file_path}")

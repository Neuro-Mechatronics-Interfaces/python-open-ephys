"""
Demo that shows how to load data from a .oebin file using pyoephys.
"""
import os
from pyoephys.io import load_oebin_file, save_as_npz

if __name__ == "__main__":

    save = True  # Set to True if you want to save the data as a .npz file

    # ================ Load the data ================
    #path = r'G:\Shared drives\NML_shared\DataShare\HDEMG Human Healthy\HD-EMG_Cuff\Jonathan\2025_06_13\raw\Dynamic5kHz\Record Node 101\experiment2\recording1\structure.oebin'
    #path = r'G:\Shared drives\NML_shared\DataShare\HDEMG Human Healthy\Open_Ephys\Jonathan\2025_05_07\raw\DynamicFingers\Record Node 105\experiment1\recording1\structure.oebin'
    #path = r"G:\Shared drives\NML_shared\DataShare\HDEMG Human Healthy\HD-EMG_Cuff\Jonathan\2025_06_13\raw\Dynamic1kHz\Record Node 101\experiment1\recording1\structure.oebin"
    path = r"G:\Shared drives\NML_shared\DataShare\HDEMG Human Healthy\HD-EMG_Cuff\Jonathan\2025_07_31\raw"
    result = load_oebin_file(path)  # We can use the file dialog to select the directory

    # ================ Print some info ================
    print(result.keys())
    print(f"Shape of emg_data: {result['amplifier_data'].shape}")
    print(f"Sampling frequency: {result['sample_rate']} Hz")
    if 'board_adc_data' in result and len(result['board_adc_data']) > 0:
        print(f"Shape of board_adc_data: {result['board_adc_data'].shape}")
    print(f"Time vector: {result.get('t_amplifier')[:10]}...")  # Print first 10 timestamps

    # =========== Save the pyoephys data to a numpy format ===========
    label = os.path.basename(result['source_path'])
    if save:
        save_as_npz(result, os.path.join(result['source_path'], f"{label}_emg_data.npz"))

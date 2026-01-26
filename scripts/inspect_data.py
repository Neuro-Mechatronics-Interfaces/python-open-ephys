import numpy as np
path = r'G:\Shared drives\NML_shared\DataShare\HDEMG Human Healthy\HD-EMG_Cuff\Jonathan\2025_07_31\raw\gestures\gestures_emg_data.npz'
data = np.load(path, allow_pickle=True)
print(f"Keys: {data.files}")
for f in data.files:
    try:
        val = data[f]
        if hasattr(val, 'shape'):
            print(f"{f}: {val.shape}")
        else:
            print(f"{f}: type {type(val)}")
    except Exception as e:
        print(f"Error loading {f}: {e}")

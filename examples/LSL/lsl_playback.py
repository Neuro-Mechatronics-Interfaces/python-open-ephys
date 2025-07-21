import time
from pyoephys.interface import OEBinPlaybackClient

#file_path = r"G:\Shared drives\NML_shared\DataShare\HDEMG Human Healthy\HD-EMG_Cuff\Jonathan\2025_05_07\raw\WristExtension_002"
#file_path = r"G:\Shared drives\NML_shared\DataShare\HDEMG Human Healthy\HD-EMG_Cuff\Jonathan\2025_06_13\raw\Dynamic1kHz"
#file_path = r"G:\Shared drives\NML_shared\DataShare\HDEMG Human Healthy\HD-EMG_Cuff\Jonathan\2025_06_13\raw\MVCHandCloseRamping"
file_path = r"G:\Shared drives\NML_shared\DataShare\HDEMG Human Healthy\HD-EMG_Cuff\Jonathan\2025_05_07\raw\WristFlexion_002"
client = OEBinPlaybackClient(file_path, block_size=50, loopback=True, enable_lsl=True)
client.start_streaming()

try:
    while True:
        time.sleep(0.1)
except KeyboardInterrupt:
    print("Stopped by user.")

finally:
    client.stop_streaming()

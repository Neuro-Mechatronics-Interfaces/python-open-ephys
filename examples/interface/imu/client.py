# any_script.py
from sleeveimu import SleeveIMUClient
import time

# Ensure your PC is connected to the Pico AP (HDEMG-SLEEVE-PICO) and has 192.168.4.x
client = SleeveIMUClient(host="192.168.4.1", port=5555, transport="UDP", auto_start=True)

# Wait for first packet (or bail after 5s)
if not client.wait_connected(timeout=2.0):
    print("Not connected yet… (still trying in background)")

pkt = client.get_imu_latest(timeout=5.0)
if pkt:
    print("First packet:", pkt)

# Loop reading the latest R/P/Y (non-blocking)
try:
    while True:
        rpy = client.get_rpy_deg()
        if rpy:
            r, p, y = rpy
            print(f"R={r:+6.2f}  P={p:+6.2f}  Y={y:+6.2f}")
        time.sleep(0.05)  # ~20 Hz display
except KeyboardInterrupt:
    pass
finally:
    client.stop()

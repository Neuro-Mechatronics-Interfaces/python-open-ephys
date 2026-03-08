"""CircuitPython 9.x example for streaming BNO085 IMU data over USB CDC.

Before using this on the Pico, enable the USB CDC data channel in boot.py:

    import usb_cdc
    usb_cdc.enable(console=True, data=True)

After saving boot.py, fully unplug/replug or power-cycle the Pico.
`CTRL-D` / soft reboot is not enough to create a new USB CDC data interface.

Then copy this file as code.py (or import from code.py) on the Pico.
The matching PC receiver is `examples/joint_angle_regression/imu_lsl_streamer.py`
with `--imu-transport SERIAL`.
"""

import math
import struct
import time

import board
import busio
import usb_cdc
from adafruit_bno08x import (
    BNO_REPORT_LINEAR_ACCELERATION,
    BNO_REPORT_ROTATION_VECTOR,
)
from adafruit_bno08x.i2c import BNO08X_I2C

TARGET_HZ = 100
BNO08X_ADDRESS = 0x4B
USB_MAGIC = b"IMU1"
PACK_FMT = "<4sII9f"
PACK_SIZE = struct.calcsize(PACK_FMT)
HELLO = b"HELLO_PICO_IMU_USB"
READY = b"PICO_IMU_READY\n"


def monotonic_us():
    try:
        return int(time.monotonic_ns() // 1_000)
    except AttributeError:
        return int(time.monotonic() * 1_000_000)


class PicoUSBIMUStreamer:
    def __init__(self):
        self.seq = 0
        self.period = 1.0 / TARGET_HZ
        self.yaw_offset = None
        self.buf = bytearray(PACK_SIZE)
        self.mv = memoryview(self.buf)
        self.cmd_buf = bytearray()

        self.stream = getattr(usb_cdc, "data", None)
        if self.stream is None:
            raise RuntimeError(
                "usb_cdc.data is unavailable. Enable it in boot.py with "
                "usb_cdc.enable(console=True, data=True), then fully unplug/replug the Pico. "
                "A soft reboot will not create the data port."
            )
        self.i2c = busio.I2C(board.GP5, board.GP4, frequency=400_000)
        self.bno = BNO08X_I2C(self.i2c, address=BNO08X_ADDRESS)

        report_us = int(1_000_000 / TARGET_HZ)
        time.sleep(0.3)
        self.bno.enable_feature(BNO_REPORT_ROTATION_VECTOR, report_interval=report_us)
        self.bno.enable_feature(
            BNO_REPORT_LINEAR_ACCELERATION, report_interval=report_us
        )
        time.sleep(0.2)

    @staticmethod
    def _parse_quat(q):
        if (
            hasattr(q, "i")
            and hasattr(q, "j")
            and hasattr(q, "k")
            and hasattr(q, "real")
        ):
            return (q.real, q.i, q.j, q.k)
        if isinstance(q, (tuple, list)) and len(q) >= 4:
            x, y, z, w = q[0], q[1], q[2], q[3]
            return (w, x, y, z)
        raise ValueError("Unexpected quaternion format")

    @staticmethod
    def _quat_to_rpy_deg(w, x, y, z):
        sinr_cosp = 2.0 * (w * x + y * z)
        cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
        roll = math.degrees(math.atan2(sinr_cosp, cosr_cosp))

        sinp = 2.0 * (w * y - z * x)
        sinp = max(-1.0, min(1.0, sinp))
        pitch = math.degrees(math.asin(sinp))

        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        yaw = math.degrees(math.atan2(siny_cosp, cosy_cosp))
        yaw = ((yaw + 180.0) % 360.0) - 180.0
        return roll, pitch, yaw

    def read_imu(self):
        q = self.bno.quaternion
        w, x, y, z = self._parse_quat(q)
        roll, pitch, yaw = self._quat_to_rpy_deg(w, x, y, z)

        if self.yaw_offset is None:
            self.yaw_offset = yaw
        yaw -= self.yaw_offset
        yaw = ((yaw + 180.0) % 360.0) - 180.0

        ax, ay, az = self.bno.linear_acceleration
        gx, gy, gz = (0.0, 0.0, 0.0)
        return roll, pitch, yaw, ax, ay, az, gx, gy, gz

    def _read_line(self):
        while self.stream.in_waiting:
            chunk = self.stream.read(self.stream.in_waiting)
            if not chunk:
                break
            self.cmd_buf.extend(chunk)
            if b"\n" in self.cmd_buf:
                line, _, rest = self.cmd_buf.partition(b"\n")
                self.cmd_buf = bytearray(rest)
                return bytes(line).strip()
        return None

    def _wait_for_host(self):
        print("Waiting for USB CDC data connection...")
        while not self.stream.connected:
            time.sleep(0.05)

        print("USB CDC connected. Waiting for host handshake...")
        while True:
            line = self._read_line()
            if line == HELLO:
                self.stream.write(READY)
                print("Host detected. Streaming IMU...")
                return
            time.sleep(0.01)

    def run(self):
        self._wait_for_host()
        next_t = time.monotonic()

        while True:
            line = self._read_line()
            if line == HELLO:
                self.stream.write(READY)

            if not self.stream.connected:
                self.cmd_buf = bytearray()
                self._wait_for_host()
                next_t = time.monotonic()
                continue

            now = time.monotonic()
            if now >= next_t:
                while next_t <= now:
                    next_t += self.period

                r, p, y, ax, ay, az, gx, gy, gz = self.read_imu()
                src_us = monotonic_us() & 0xFFFFFFFF
                struct.pack_into(
                    PACK_FMT,
                    self.buf,
                    0,
                    USB_MAGIC,
                    self.seq,
                    src_us,
                    r,
                    p,
                    y,
                    ax,
                    ay,
                    az,
                    gx,
                    gy,
                    gz,
                )
                self.stream.write(self.mv)
                self.seq = (self.seq + 1) & 0xFFFFFFFF

            time.sleep(0)


try:
    PicoUSBIMUStreamer().run()
except Exception as exc:
    print("Fatal:", exc)
    raise

# sleeve_imu_client.py
import json
import socket
import struct
import threading
import time
from collections import deque
from typing import Any, Dict, Optional, Tuple

try:
    import serial
except Exception as exc:
    serial = None
    SERIAL_IMPORT_ERROR = str(exc)
else:
    SERIAL_IMPORT_ERROR = None

try:
    from serial.tools import list_ports
except Exception:
    list_ports = None


class SleeveIMUClient:
    """
    PC client for the Sleeve IMU (Pico W).
    - UDP mode (default): sends periodic HELLOs, receives binary packets
    - TCP mode (optional): connects and reads newline-delimited JSON
    - SERIAL mode: reads framed binary packets from USB CDC / pyserial
    """

    def __init__(
        self,
        host: str = "192.168.4.1",
        port: int = 5555,
        transport: str = "UDP",  # "UDP" or "TCP"
        serial_port: Optional[str] = None,
        serial_baudrate: int = 115200,
        hello_interval: float = 3.0,  # UDP keep-alive interval (s)
        recv_timeout: float = 2.0,  # socket timeout (s)
        auto_start: bool = False,
    ) -> None:
        self.host = host
        self.port = port
        self.transport = transport.upper()
        self.serial_port = serial_port
        self.serial_baudrate = int(serial_baudrate)
        self.hello_interval = hello_interval
        self.recv_timeout = recv_timeout

        self._sock: Optional[Any] = None
        self._rx_thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._connected = threading.Event()  # set after initial handshake
        self._have_data = threading.Event()  # set after first packet received
        self._lock = threading.Lock()
        self._latest: Optional[Dict[str, Any]] = None  # last full JSON packet
        self._queue = deque(maxlen=4096)

        self.PACK_FMT = "<I9f"
        self.PACK_SIZE = struct.calcsize(self.PACK_FMT)
        self.USB_MAGIC = b"IMU1"
        self.USB_PACK_FMT = "<4sII9f"
        self.USB_PACK_SIZE = struct.calcsize(self.USB_PACK_FMT)
        self.SERIAL_HELLO = b"HELLO_PICO_IMU_USB\n"
        self.SERIAL_READY = b"PICO_IMU_READY"
        self.connected_port = None

        if auto_start:
            self.start()

    # ---------- public API ----------
    def start(self) -> None:
        """Start background I/O thread."""
        if self._rx_thread and self._rx_thread.is_alive():
            return
        self._stop.clear()
        if self.transport == "UDP":
            target = self._run_udp
        elif self.transport == "TCP":
            target = self._run_tcp
        elif self.transport in {"SERIAL", "USB", "CDC"}:
            target = self._run_serial
        else:
            raise ValueError(f"Unsupported transport: {self.transport}")
        self._rx_thread = threading.Thread(target=target, daemon=True)
        self._rx_thread.start()

    def stop(self) -> None:
        """Stop background I/O thread and close socket."""
        self._stop.set()
        self._connected.clear()
        try:
            if self._sock:
                self._sock.close()
        except Exception:
            pass
        self._sock = None
        if self._rx_thread:
            self._rx_thread.join(timeout=2.0)

    def is_running(self) -> bool:
        return bool(self._rx_thread and self._rx_thread.is_alive())

    def wait_connected(self, timeout: Optional[float] = 5.0) -> bool:
        """Wait until initial HELLO/OK (UDP) or connected (TCP)."""
        return self._connected.wait(timeout=timeout)

    def get_imu_latest(
        self, timeout: Optional[float] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Return the most recent IMU packet (dict) or None.
        If timeout is provided and no data yet, block up to timeout seconds.
        """
        if self._latest is None and timeout is not None:
            self._have_data.wait(timeout=timeout)
        with self._lock:
            # return a shallow copy so callers can mutate safely
            return dict(self._latest) if self._latest is not None else None

    def get_rpy_deg(self) -> Optional[Tuple[float, float, float]]:
        """
        Convenience accessor for roll/pitch/yaw (deg), or None if no data yet.
        """
        pkt = self.get_imu_latest()
        if not pkt:
            return None
        try:
            r, p, y = pkt["rpy"]
            return float(r), float(p), float(y)
        except Exception:
            return None

    def get_imu_packets(
        self,
        max_packets: Optional[int] = None,
        timeout: Optional[float] = None,
    ) -> list[Dict[str, Any]]:
        """
        Return and remove queued IMU packets in arrival order.
        If timeout is provided and no data is available yet, wait up to timeout seconds.
        """
        if not self._queue and timeout is not None:
            self._have_data.wait(timeout=timeout)

        with self._lock:
            if not self._queue:
                return []
            if max_packets is None or max_packets >= len(self._queue):
                packets = list(self._queue)
                self._queue.clear()
            else:
                packets = [self._queue.popleft() for _ in range(max_packets)]
            if not self._queue:
                self._have_data.clear()
            return packets

    # ---------- internals ----------
    def _run_udp(self) -> None:
        """UDP mode: HELLO_PICO/PICO_READY handshake + receive binary packets."""
        while not self._stop.is_set():
            try:
                self._close_sock()
                s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s.settimeout(self.recv_timeout)

                # Bind to the user-specified port (e.g. 5555)
                # If port=0, OS assigns random ephemeral port.
                local_port = self.port if self.port else 0
                try:
                    s.bind(("0.0.0.0", local_port))
                except OSError:
                    time.sleep(1.0)
                    continue

                _, bound_port = s.getsockname()

                # We expect packets FROM host:9000
                # We send TO host:9000
                try:
                    s.connect((self.host, 9000))
                except OSError:
                    pass
                self._sock = s

                # initial HELLO handshake
                msg = f"HELLO_PICO port={bound_port}".encode("utf-8")
                if not self._hello_loop(s, msg=msg):
                    time.sleep(0.5)
                    continue

                self._connected.set()
                last_hello = time.time()

                # receive loop
                while not self._stop.is_set():
                    now = time.time()
                    if now - last_hello > self.hello_interval:
                        try:
                            s.send(msg)
                        except OSError:
                            break
                        last_hello = now

                    try:
                        # 65535 is max UDP size
                        data = s.recv(65535)
                    except socket.timeout:
                        # Re-send HELLO to keep alive or re-discover
                        try:
                            s.send(msg)
                        except Exception:
                            pass
                        continue
                    except OSError:
                        break

                    self._handle_packet(data)

            except Exception:
                time.sleep(0.5)

            self._connected.clear()
            time.sleep(0.3)
        self._close_sock()

    def _hello_loop(self, s: socket.socket, msg: bytes, tries: int = 10) -> bool:
        """Send HELLO and wait for 'PICO_READY'."""
        for _ in range(tries):
            try:
                s.send(msg)
                data = s.recv(4096)
                # Pico sends "PICO_READY v1"
                if b"PICO_READY" in data:
                    return True
                # Or we might get a data packet if we re-connected
                if len(data) == self.PACK_SIZE:
                    self._handle_packet(data)
                    return True
            except socket.timeout:
                continue
            except OSError:
                return False
        return False

    def _run_tcp(self) -> None:
        """TCP mode: connect and read newline-delimited JSON packets."""
        buf = b""
        while not self._stop.is_set():
            try:
                self._close_sock()
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.settimeout(5.0)
                s.connect((self.host, self.port))
                s.settimeout(self.recv_timeout)
                self._sock = s
                self._connected.set()

                while not self._stop.is_set():
                    try:
                        chunk = s.recv(4096)
                        if not chunk:
                            break  # disconnected
                        buf += chunk
                        while b"\n" in buf:
                            line, buf = buf.split(b"\n", 1)
                            self._handle_packet(line)
                    except socket.timeout:
                        continue
            except Exception:
                pass

            self._connected.clear()
            time.sleep(0.5)

        self._close_sock()

    def _run_serial(self) -> None:
        """USB CDC / serial mode: open COM port and receive framed binary packets."""
        if serial is None:
            raise RuntimeError(f"pyserial import failed: {SERIAL_IMPORT_ERROR}")

        buf = bytearray()
        while not self._stop.is_set():
            try:
                self._close_sock()
                ser = self._open_serial_candidate()
                self._sock = ser
                self.connected_port = getattr(ser, "port", self.serial_port)
                self._connected.set()

                while not self._stop.is_set():
                    chunk = ser.read(max(1, self.USB_PACK_SIZE * 4))
                    if not chunk:
                        try:
                            ser.write(self.SERIAL_HELLO)
                        except Exception:
                            pass
                        continue
                    buf.extend(chunk)

                    if self.SERIAL_READY in buf:
                        ready_idx = buf.find(self.SERIAL_READY)
                        del buf[: ready_idx + len(self.SERIAL_READY)]
                        continue

                    while True:
                        start = buf.find(self.USB_MAGIC)
                        if start < 0:
                            if len(buf) > (self.USB_PACK_SIZE - 1):
                                del buf[: len(buf) - (self.USB_PACK_SIZE - 1)]
                            break
                        if start > 0:
                            del buf[:start]
                        if len(buf) < self.USB_PACK_SIZE:
                            break
                        packet = bytes(buf[: self.USB_PACK_SIZE])
                        del buf[: self.USB_PACK_SIZE]
                        self._handle_packet(packet)
            except Exception:
                self._connected.clear()
                self.connected_port = None
                time.sleep(0.5)

        self._close_sock()

    def _candidate_serial_ports(self) -> list[str]:
        if self.serial_port and self.serial_port.strip().upper() not in {"", "AUTO"}:
            return [self.serial_port.strip()]
        if list_ports is None:
            raise RuntimeError(
                "Automatic serial discovery requires pyserial list_ports support; set --imu-serial-port explicitly"
            )
        ports = [p.device for p in list_ports.comports()]
        if not ports:
            raise RuntimeError("No serial ports found for Pico auto-discovery")
        return ports

    def _open_serial_candidate(self):
        last_error = None
        for port_name in self._candidate_serial_ports():
            if self._stop.is_set():
                break
            try:
                ser = serial.Serial(
                    port=port_name,
                    baudrate=self.serial_baudrate,
                    timeout=0.2,
                    write_timeout=0.2,
                )
                try:
                    ser.reset_input_buffer()
                    ser.reset_output_buffer()
                except Exception:
                    pass

                if self._probe_serial_device(ser):
                    ser.timeout = self.recv_timeout
                    return ser
                ser.close()
            except Exception as exc:
                last_error = exc

        if last_error is not None:
            raise RuntimeError(f"Unable to detect Pico IMU serial device: {last_error}")
        raise RuntimeError("Unable to detect Pico IMU serial device")

    def _probe_serial_device(self, ser) -> bool:
        deadline = time.time() + max(0.5, self.recv_timeout)
        buf = bytearray()
        while time.time() < deadline and not self._stop.is_set():
            try:
                ser.write(self.SERIAL_HELLO)
            except Exception:
                return False

            inner_deadline = time.time() + 0.25
            while time.time() < inner_deadline and not self._stop.is_set():
                try:
                    chunk = ser.read(256)
                except Exception:
                    return False
                if chunk:
                    buf.extend(chunk)
                    if self.SERIAL_READY in buf or self.USB_MAGIC in buf:
                        return True
                else:
                    time.sleep(0.02)
        return False

    def _handle_packet(self, data: bytes) -> None:
        """Parse packet (binary or JSON) and store it as the latest reading."""
        try:
            # Try framed USB binary packet with source timestamp
            if len(data) == self.USB_PACK_SIZE:
                magic, seq, src_us, r, p, y, ax, ay, az, gx, gy, gz = struct.unpack(
                    self.USB_PACK_FMT, data
                )
                if magic == self.USB_MAGIC:
                    pkt = {
                        "seq": seq,
                        "src_us": src_us,
                        "rpy": [r, p, y],
                        "acc": [ax, ay, az],
                        "gyr": [gx, gy, gz],
                        "_received_at": time.time(),
                    }
                    with self._lock:
                        self._latest = pkt
                        self._queue.append(pkt)
                    self._have_data.set()
                    return

            # Try binary
            if len(data) == self.PACK_SIZE:
                seq, r, p, y, ax, ay, az, gx, gy, gz = struct.unpack(
                    self.PACK_FMT, data
                )
                pkt = {
                    "seq": seq,
                    "rpy": [r, p, y],
                    "acc": [ax, ay, az],
                    "gyr": [gx, gy, gz],
                    "_received_at": time.time(),
                }
                with self._lock:
                    self._latest = pkt
                    self._queue.append(pkt)
                self._have_data.set()
                return

            # Try JSON
            pkt = json.loads(data.decode("utf-8"))
            pkt["_received_at"] = time.time()
            with self._lock:
                self._latest = pkt
                self._queue.append(pkt)
            self._have_data.set()
        except Exception:
            pass

    def _close_sock(self) -> None:
        try:
            if self._sock:
                self._sock.close()
        except Exception:
            pass
        self._sock = None
        self.connected_port = None

    # context manager sugar
    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.stop()

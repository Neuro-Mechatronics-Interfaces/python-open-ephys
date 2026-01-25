# sleeve_imu_client.py
import json
import socket
import threading
import time
from typing import Optional, Tuple, Dict, Any

class SleeveIMUClient:
    """
    PC client for the Sleeve IMU (Pico W).
    - UDP mode (default): sends periodic HELLOs, receives JSON packets
    - TCP mode (optional): connects and reads newline-delimited JSON
    """

    def __init__(
        self,
        host: str = "192.168.4.1",
        port: int = 5555,
        transport: str = "UDP",          # "UDP" or "TCP"
        hello_interval: float = 3.0,     # UDP keep-alive interval (s)
        recv_timeout: float = 2.0,       # socket timeout (s)
        auto_start: bool = False,
    ) -> None:
        self.host = host
        self.port = port
        self.transport = transport.upper()
        self.hello_interval = hello_interval
        self.recv_timeout = recv_timeout

        self._sock: Optional[socket.socket] = None
        self._rx_thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._connected = threading.Event()  # set after initial handshake
        self._have_data = threading.Event()  # set after first packet received
        self._lock = threading.Lock()
        self._latest: Optional[Dict[str, Any]] = None  # last full JSON packet

        if auto_start:
            self.start()

    # ---------- public API ----------
    def start(self) -> None:
        """Start background I/O thread."""
        if self._rx_thread and self._rx_thread.is_alive():
            return
        self._stop.clear()
        target = self._run_udp if self.transport == "UDP" else self._run_tcp
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

    def get_imu_latest(self, timeout: Optional[float] = None) -> Optional[Dict[str, Any]]:
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

    # ---------- internals ----------
    def _run_udp(self) -> None:
        """UDP mode: HELLO/OK handshake + keepalive + packet receive."""
        while not self._stop.is_set():
            try:
                # fresh socket each connect attempt
                self._close_sock()
                s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                s.settimeout(self.recv_timeout)
                # "connect" a UDP socket to fix the remote and pick a local port/NIC
                s.connect((self.host, self.port))
                self._sock = s

                # initial HELLO handshake
                if not self._hello_loop(s, tries=10):
                    # wait a bit then retry from scratch
                    time.sleep(0.5)
                    continue

                self._connected.set()
                last_hello = time.time()

                # receive loop
                while not self._stop.is_set():
                    # periodic keepalive HELLO so Pico's watchdog stays happy
                    now = time.time()
                    if now - last_hello > self.hello_interval:
                        try:
                            s.send(b"HELLO")
                        except OSError:
                            # socket died; break to reconnect
                            break
                        last_hello = now

                    try:
                        data = s.recv(65535)
                    except socket.timeout:
                        # timeouts happen; send HELLO to re-register and continue
                        try:
                            s.send(b"HELLO")
                            # also try to receive the OK; ignore if none
                            _ = s.recv(4096)
                        except Exception:
                            pass
                        continue
                    except OSError:
                        # socket error; reconnect
                        break

                    self._handle_packet(data)

            except Exception:
                # swallow and retry outer loop
                pass

            # reconnect backoff
            self._connected.clear()
            time.sleep(0.3)

        self._close_sock()

    def _hello_loop(self, s: socket.socket, tries: int = 10) -> bool:
        """Send HELLO and wait for 'OK' reply a few times."""
        for _ in range(tries):
            try:
                s.send(b"HELLO")
                data = s.recv(4096)
                if data.strip() == b"OK":
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

    def _handle_packet(self, data: bytes) -> None:
        """Parse a JSON packet and store it as the latest reading."""
        try:
            pkt = json.loads(data.decode("utf-8"))
            pkt["_received_at"] = time.time()
            with self._lock:
                self._latest = pkt
            self._have_data.set()
        except Exception:
            # ignore malformed lines (e.g., partial UDP datagrams)
            pass

    def _close_sock(self) -> None:
        try:
            if self._sock:
                self._sock.close()
        except Exception:
            pass
        self._sock = None

    # context manager sugar
    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.stop()

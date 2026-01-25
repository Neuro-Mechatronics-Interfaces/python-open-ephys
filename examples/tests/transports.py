# transports.py
# Cross-transport helpers for sending LED commands to your Pico server.
# Supports UDP and USB Serial (pyserial). No newline required; commands end with ';'.
import time
import socket

class UDPTransport:
    def __init__(self, ip: str, port: int, max_payload=900):
        self.addr = (ip, port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.max_payload = max_payload

    def send_commands(self, cmds):
        """Batch a list of 'cmd;' strings into UDP-sized packets and append a single 'show;' at end."""
        buf = ""
        for c in cmds:
            c = c if c.endswith(';') else (c + ';')
            if c.strip().lower() == 'show;':
                continue
            if len(buf) + len(c) > self.max_payload:
                self.sock.sendto(buf.encode('utf-8'), self.addr)
                buf = ""
            buf += c
        buf += 'show;'
        if buf:
            self.sock.sendto(buf.encode('utf-8'), self.addr)

    def send_one(self, cmd):
        if not cmd.endswith(';'):
            cmd += ';'
        self.sock.sendto(cmd.encode('utf-8'), self.addr)

    def close(self):
        try:
            self.sock.close()
        except Exception:
            pass


class SerialTransport:
    def __init__(self, port: str, baud: int = 9600, write_chunk=256, write_pause=0.0):
        # Requires: pip install pyserial
        import serial
        self.ser = serial.Serial(port=port, baudrate=baud, timeout=0)
        self.write_chunk = write_chunk
        self.write_pause = write_pause  # seconds between chunks (0..0.005 typical)

    def _write_bytes(self, data: bytes):
        for i in range(0, len(data), self.write_chunk):
            self.ser.write(data[i:i+self.write_chunk])
            if self.write_pause:
                time.sleep(self.write_pause)
        self.ser.flush()

    def send_commands(self, cmds):
        parts = []
        for c in cmds:
            c = c if c.endswith(';') else (c + ';')
            if c.strip().lower() == 'show;':
                continue
            parts.append(c)
        parts.append('show;')
        payload = ''.join(parts).encode('utf-8')
        self._write_bytes(payload)

    def send_one(self, cmd):
        if not cmd.endswith(';'):
            cmd += ';'
        self._write_bytes(cmd.encode('utf-8'))

    def close(self):
        try:
            self.ser.close()
        except Exception:
            pass
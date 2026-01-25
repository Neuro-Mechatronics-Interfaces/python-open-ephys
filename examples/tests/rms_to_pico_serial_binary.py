#!/usr/bin/env python3
"""
rms_to_pico_serial_binary.py
Stream RMS power -> Pico LEDs using a compact **binary** frame over USB serial.

Packet:
    b'F' + uint16_be(length_bytes) + raw_rgb_bytes
where length_bytes = 3 * N_LEDS, RGB = R0 G0 B0 R1 G1 B1 ...

- One write() per frame, no ascii, no hex.
- On the Pico, a 'binary_frame_task' consumes these and calls pixels.show() once per frame.

Example:
  python rms_to_pico_serial_binary.py --serial COM8 --baud 460800 --channels 0:128 --brightness 35% --gain 1.2
"""
import sys, time, argparse
import numpy as np
from scipy.signal import iirnotch, butter, sosfiltfilt, tf2sos

from pyoephys.interface import ZMQClient
try:
    from pyoephys.io import parse_numeric_args
except Exception:
    parse_numeric_args = None

# --------------- serial ---------------
class PicoSerialBinaryTX:
    def __init__(self, port: str, baud: int = 460800, chunk=65536, verbose=False):
        import serial
        self.ser = serial.Serial(port=port, baudrate=baud, timeout=0)
        self.chunk = int(chunk)
        self.verbose = verbose

    def send_text(self, text: str):
        """Optional helper for ASCII cfg (pixels:, brightness:, etc.)."""
        if not text.endswith(';'):
            text += ';'
        data = text.encode('utf-8')
        self._write(data)

    def send_frame_rgb(self, rgb_bytes: bytes):
        """
        rgb_bytes: raw bytes of length 3*N (R G B ...).
        Writes: b'F' + uint16_be(len) + rgb_bytes
        """
        n = len(rgb_bytes)
        if n > 65535:
            raise ValueError("binary frame too large (>65535 bytes)")
        hdr = bytes((0x46, (n >> 8) & 0xFF, n & 0xFF))
        self._write(hdr + rgb_bytes)
        if self.verbose:
            print(f"[PC] sent frame N={n//3} bytes={n+3}")

    def _write(self, data: bytes):
        # write in a single call if possible; chunk only if needed
        if len(data) <= self.chunk:
            self.ser.write(data); self.ser.flush()
        else:
            for i in range(0, len(data), self.chunk):
                self.ser.write(data[i:i+self.chunk])
            self.ser.flush()

    def close(self):
        try: self.ser.close()
        except Exception: pass

# --------------- colors & DSP ---------------
def lerp(a, b, t): return a + (b - a) * t

def colormap(val):
    """val in [0,1] -> RGB via blue→cyan→green→yellow→red"""
    stops = [
        (0.00, (  0,   0, 255)),
        (0.25, (  0, 255, 255)),
        (0.50, (  0, 255,   0)),
        (0.75, (255, 255,   0)),
        (1.00, (255,   0,   0)),
    ]
    v = float(np.clip(val, 0.0, 1.0))
    for i in range(1, len(stops)):
        t0, c0 = stops[i-1]; t1, c1 = stops[i]
        if v <= t1:
            a = 0.0 if t1 == t0 else (v - t0) / (t1 - t0)
            r = int(lerp(c0[0], c1[0], a))
            g = int(lerp(c0[1], c1[1], a))
            b = int(lerp(c0[2], c1[2], a))
            return (r, g, b)
    return stops[-1][1]

def colors_to_bytes(colors):
    """[(r,g,b), ...] -> raw bytes length 3*N"""
    arr = np.asarray(colors, dtype=np.uint8).reshape(-1, 3)
    return arr.tobytes()

def design_filters(fs, notch_freq=60.0, notch_Q=30.0, bp_low=10.0, bp_high=500.0, order=4):
    b_notch, a_notch = iirnotch(w0=notch_freq, Q=notch_Q, fs=fs)
    sos_notch = tf2sos(b_notch, a_notch)
    sos_bp = butter(order, [bp_low / (0.5 * fs), bp_high / (0.5 * fs)], btype="band", output="sos")
    return sos_notch, sos_bp

def compute_rms(win_2d):
    return np.sqrt(np.mean(win_2d * win_2d, axis=1) + 1e-12)

# --------------- main ---------------
def main():
    ap = argparse.ArgumentParser(description="RMS -> Pico (binary frame)")
    ap.add_argument("--serial", required=True, help="COM port, e.g., COM8 or /dev/ttyACM0")
    ap.add_argument("--baud", type=int, default=460800, help="Baud (USB CDC ignores a lot, but keep high)")
    ap.add_argument("--channels", nargs="+", default=["0:128"], help="e.g., 0 1 2 or 0:64 or all")
    ap.add_argument("--window_sec", type=float, default=0.5)
    ap.add_argument("--downsample", type=int, default=1)
    ap.add_argument("--gain", type=float, default=1.0)
    ap.add_argument("--clamp", type=float, default=1.0)
    ap.add_argument("--brightness", type=str, default=None, help="e.g., 0.4 or 40% (sent once as ASCII)")
    ap.add_argument("--pixels", type=int, default=None, help="send 'pixels:N' once (ASCII)")
    ap.add_argument("--host_ip", type=str, default="127.0.0.1")
    ap.add_argument("--data_port", type=str, default="5556")
    ap.add_argument("--notch", type=float, default=60.0)
    ap.add_argument("--notch_Q", type=float, default=30.0)
    ap.add_argument("--bp_low", type=float, default=10.0)
    ap.add_argument("--bp_high", type=float, default=500.0)
    ap.add_argument("--order", type=int, default=4)
    ap.add_argument("--sleep", type=float, default=0.0)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    # parse channels
    raw = args.channels if isinstance(args.channels, (list, tuple)) else [str(args.channels)]
    want_all = (len(raw) == 1 and str(raw[0]).lower() == "all")
    if want_all:
        channels = []
    else:
        if parse_numeric_args is not None:
            channels = parse_numeric_args(raw)
        else:
            channels = []
            for tok in raw:
                if ":" in tok:
                    a, b = tok.split(":", 1); channels.extend(range(int(a), int(b)))
                else:
                    channels.append(int(tok))

    # ZMQ client
    client = ZMQClient(host_ip=args.host_ip, data_port=args.data_port, verbose=args.verbose)
    if not want_all and channels:
        client.set_channel_index(channels)
    client.start()
    try:
        client.ready_event.wait(timeout=5.0)
    except Exception:
        pass

    # fs
    fs = getattr(client, "fs", None)
    if not fs or fs <= 0:
        fs_est = getattr(client, "fs_estimate", lambda: None)()
        fs = fs_est if fs_est else 2000.0
    if args.verbose:
        print(f"[PC] fs={fs:.1f} Hz  window={args.window_sec:.3f}s  chans={len(channels) if channels else 'ALL'}")

    # C
    if want_all:
        C = getattr(client, "N_channels", None)
        if not C or C <= 0:
            C = 0; t0 = time.time()
            while time.time() - t0 < 2.0 and C == 0:
                try:
                    _, blk = client.drain_new()
                    if blk is None or getattr(blk, "size", 0) == 0:
                        blk = client.get_latest_window(window_ms=50)
                    if blk is not None:
                        arr = np.asarray(blk)
                        C = arr.shape[0] if arr.ndim == 2 else 1
                except Exception:
                    pass
                if C == 0: time.sleep(0.05)
            if C == 0: C = 16
    else:
        C = len(channels) if channels else 1

    # buffers & filters
    N = max(1, int(round(args.window_sec * fs)))
    ring = np.zeros((C, N), dtype=np.float32); have = 0
    b_notch, a_notch = iirnotch(w0=args.notch, Q=args.notch_Q, fs=fs) if args.notch > 0 else ([1.0],[1.0])
    sos_notch = tf2sos(b_notch, a_notch)
    sos_bp = butter(args.order, [args.bp_low / (0.5 * fs), args.bp_high / (0.5 * fs)], btype="band", output="sos")

    # serial
    tx = PicoSerialBinaryTX(args.serial, args.baud, verbose=args.verbose)
    # optional ASCII config (your Pico still understands text commands)
    if args.pixels is not None:
        tx.send_text(f"pixels:{int(args.pixels)};")
        time.sleep(0.01)
    if args.brightness is not None:
        s = str(args.brightness).strip()
        v = float(s[:-1])/100.0 if s.endswith('%') else float(s)
        if v > 1.0: v /= 100.0
        v = max(0.0, min(1.0, v))
        tx.send_text(f"brightness:{int(round(v*100))};")
        time.sleep(0.01)

    try:
        if args.verbose:
            print("[PC] streaming… Ctrl+C to stop.")
        while True:
            try:
                _, block = client.drain_new()
            except Exception:
                block = None

            if block is not None and hasattr(block, "shape"):
                blk = np.asarray(block)
                # coerce to (C, K)
                if blk.ndim == 1:
                    blk = blk.reshape(C, -1)
                elif blk.shape[0] != C and blk.shape[1] == C:
                    blk = blk.T
                elif blk.shape[0] != C and blk.shape[1] != C:
                    if blk.shape[0] < C:
                        pad = np.zeros((C - blk.shape[0], blk.shape[1]), dtype=blk.dtype)
                        blk = np.vstack((blk, pad))
                    else:
                        blk = blk[:C, :]

                if blk.size > 0:
                    if args.downsample > 1:
                        blk = blk[:, ::args.downsample]
                    K = blk.shape[1]
                    if K >= N:
                        ring = blk[:, -N:]; have = N
                    else:
                        if K > 0:
                            ring[:, :-K] = ring[:, K:]
                            ring[:, -K:] = blk
                            have = min(N, have + K)

            # compute colors
            win = ring[:, -have:] if have > 0 else ring[:, -1:]
            try:
                yf = sosfiltfilt(sos_notch, win, axis=1)
                yf = sosfiltfilt(sos_bp,    yf,  axis=1)
                rms = compute_rms(yf)
            except Exception:
                rms = compute_rms(win)

            m = float(np.max(rms)) if rms.size else 0.0
            vals = (rms / m) if m > 1e-12 else np.zeros_like(rms)
            vals = np.clip(vals * float(args.gain), 0.0, float(args.clamp))

            colors = [colormap(float(v)) for v in vals]
            rgb_bytes = colors_to_bytes(colors)
            tx.send_frame_rgb(rgb_bytes)

            if args.sleep > 0:
                time.sleep(args.sleep)

    except KeyboardInterrupt:
        pass
    finally:
        try: tx.close()
        except: pass
        try: client.close()
        except: pass

if __name__ == "__main__":
    main()

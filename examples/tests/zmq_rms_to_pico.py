#!/usr/bin/env python3
"""Stream RMS power -> Pico LEDs over UDP or Serial.

Matches the style of your zmq_stacked_plot.py (argparse, parse_numeric_args, ZMQClient).
Usage examples:
  python zmq_rms_to_pico.py --channels 0:64 --pico_ip 192.168.1.50 --pico_port 4444 --brightness 35%%
  python zmq_rms_to_pico.py --channels 0:32 --serial COM8 --baud 9600 --brightness 0.4
"""
import sys, time, argparse, math, socket
import numpy as np

# ---------- Dependencies from your stack ----------
try:
    from pyoephys.interface import ZMQClient
    from pyoephys.io import parse_numeric_args
except Exception as e:
    print("[zmq_rms_to_pico] Missing dependency from your environment (pyoephys).\n"
          "Install your package or adjust imports. Error: ", e, file=sys.stderr)
    print("You also need 'pyzmq' installed for ZMQ transport.", file=sys.stderr)

# Local transports (UDP/Serial)
from transports import UDPTransport, SerialTransport

# ---------- Helpers ----------
def lerp(a, b, t): return a + (b - a) * t

def colormap(val):
    """val in [0,1] -> RGB via blue->cyan->green->yellow->red"""
    stops = [
        (0.00, (  0,   0, 255)),
        (0.25, (  0, 255, 255)),
        (0.50, (  0, 255,   0)),
        (0.75, (255, 255,   0)),
        (1.00, (255,   0,   0)),
    ]
    v = float(np.clip(val, 0.0, 1.0))
    for i in range(1, len(stops)):
        t0, c0 = stops[i-1]
        t1, c1 = stops[i]
        if v <= t1:
            a = 0.0 if t1 == t0 else (v - t0) / (t1 - t0)
            r = int(lerp(c0[0], c1[0], a))
            g = int(lerp(c0[1], c1[1], a))
            b = int(lerp(c0[2], c1[2], a))
            return (r, g, b)
    return stops[-1][1]

class RollingRMS:
    def __init__(self, n_channels, fs, win_ms=100, hop_ms=20):
        self.n = n_channels
        self.fs = fs
        self.win = max(1, int(win_ms * fs / 1000))
        self.hop = max(1, int(hop_ms * fs / 1000))
        self.buf = np.zeros((n_channels, self.win), dtype=np.float32)
        self.pos = 0
        self.count = 0

    def push(self, block):
        """block: (n_channels, n_samples) -> list of rms frames (n_channels,)"""
        block = np.asarray(block, dtype=np.float32)
        assert block.shape[0] == self.n, f"expected {self.n} chans, got {block.shape[0]}"
        frames = []
        n_samp = block.shape[1]
        for t in range(n_samp):
            self.buf[:, self.pos] = block[:, t]
            self.pos = (self.pos + 1) % self.win
            self.count += 1
            if self.count >= self.win and ((self.count - self.win) % self.hop == 0):
                if self.pos == 0:
                    window = self.buf
                else:
                    window = np.hstack((self.buf[:, self.pos:], self.buf[:, :self.pos]))
                rms = np.sqrt(np.mean(window * window, axis=1) + 1e-12)
                frames.append(rms)
        return frames

def _choose_transport(args):
    if args.serial:
        return SerialTransport(args.serial, args.baud)
    if not args.pico_ip:
        raise SystemExit("Either --serial PORT or --pico_ip is required")
    return UDPTransport(args.pico_ip, args.pico_port)

def _send_brightness(tx, v01):
    v = max(0.0, min(1.0, float(v01)))
    tx.send_one(f"brightness:{int(round(v*100))};")

def _send_pixels(tx, n):
    tx.send_one(f"pixels:{int(n)};")

def _rms_to_led_cmds(rms, gain=1.0, clamp=1.0):
    m = float(np.max(rms)) if np.isfinite(rms).all() else 1.0
    if m <= 1e-12: m = 1.0
    vals = np.clip((rms / m) * gain, 0.0, clamp)
    cmds = []
    for i, v in enumerate(vals):
        r, g, b = colormap(float(v))
        cmds.append(f"led:{i}:{r},{g},{b};")
    cmds.append("show;")
    return cmds

def _recv_block(client):
    """Try common method names to get a (n_channels, n_samples) ndarray."""
    for m in ("recv", "read", "get_block", "get"):
        if hasattr(client, m):
            blk = getattr(client, m)()
            if blk is not None:
                return blk
    return None

def main():
    ap = argparse.ArgumentParser(description="Stream RMS power -> Pico LEDs over UDP or Serial")
    ap.add_argument("--channels", nargs="+", default=["0:8"], help="Channels (e.g., 0:64 or 0 1 2)")
    ap.add_argument("--downsample", type=int, default=1, help="Downsample factor across time axis")
    ap.add_argument("--fs", type=float, default=2000.0, help="Sample rate of incoming data (Hz)")
    ap.add_argument("--win_ms", type=float, default=100.0, help="RMS window (ms)")
    ap.add_argument("--hop_ms", type=float, default=25.0, help="RMS hop (ms)")
    ap.add_argument("--gain", type=float, default=1.0, help="Global gain applied after normalization")
    ap.add_argument("--clamp", type=float, default=1.0, help="Clamp (0..1) after gain")
    ap.add_argument("--sleep", type=float, default=0.0, help="Rate limit between LED frames (s)")
    ap.add_argument("--pico_ip", type=str, default=None, help="Pico IP (for UDP)")
    ap.add_argument("--pico_port", type=int, default=4444, help="Pico UDP port")
    ap.add_argument("--serial", type=str, default=None, help="Serial port (e.g., COM8 or /dev/ttyACM0)")
    ap.add_argument("--baud", type=int, default=9600, help="Serial baud (default 9600)")
    ap.add_argument("--brightness", type=str, default=None, help="e.g., 0.4 or 40%%")
    ap.add_argument("--pixels", type=int, default=None, help="Send pixels:N to Pico before streaming")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    # Parse channels (like in your plot script)
    try:
        channels = parse_numeric_args(args.channels)
    except Exception:
        # fallback: simple ranges like '0:8'
        chans = []
        for tok in args.channels:
            if ':' in tok:
                a,b = tok.split(':',1)
                chans.extend(list(range(int(a), int(b))))
            else:
                chans.append(int(tok))
        channels = chans
    if args.verbose:
        print(f"Channels: {channels}  (count={len(channels)})")

    # ZMQ client
    client = ZMQClient(host_ip="127.0.0.1", data_port="5556", verbose=args.verbose)
    if hasattr(client, 'set_channel_index'):
        client.set_channel_index(channels)

    # Transport
    tx = _choose_transport(args)

    # Initial Pico config
    if args.pixels is not None:
        _send_pixels(tx, args.pixels)
        time.sleep(0.02)
    if args.brightness is not None:
        s = args.brightness.strip()
        v = float(s[:-1])/100.0 if s.endswith('%') else float(s)
        if v > 1.0: v /= 100.0
        _send_brightness(tx, v)
        time.sleep(0.02)

    # RMS engine
    n_ch = len(channels)
    rms_engine = RollingRMS(n_ch, fs=args.fs, win_ms=args.win_ms, hop_ms=args.hop_ms)

    try:
        if args.verbose:
            print("Streaming… Ctrl+C to stop.")
        while True:
            block = _recv_block(client)  # expected shape (n_channels, n_samples)
            if block is None:
                time.sleep(0.001)
                continue
            block = np.asarray(block)
            # shape check & squeeze common layouts
            if block.ndim == 1:
                block = block.reshape(n_ch, -1)
            elif block.shape[0] != n_ch and block.shape[1] == n_ch:
                block = block.T

            if args.downsample > 1:
                block = block[:, ::args.downsample]

            frames = rms_engine.push(block)
            for rms in frames:
                cmds = _rms_to_led_cmds(rms, gain=args.gain, clamp=args.clamp)
                tx.send_commands(cmds)
                if args.sleep > 0:
                    time.sleep(args.sleep)

    except KeyboardInterrupt:
        pass
    finally:
        try: tx.close()
        except Exception: pass

if __name__ == "__main__":
    main()
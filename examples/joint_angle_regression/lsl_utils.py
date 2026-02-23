import time
import threading
from collections import deque

_LSL_IMPORT_ERROR = None
try:
    from pylsl import StreamInlet, StreamInfo, StreamOutlet, local_clock
    # pylsl API changed: resolve_stream was renamed in some versions
    try:
        from pylsl import resolve_stream as _resolve_stream
    except Exception:
        _resolve_stream = None
    try:
        from pylsl import resolve_byprop as _resolve_byprop
    except Exception:
        _resolve_byprop = None
    try:
        from pylsl import resolve_bypred as _resolve_bypred
    except Exception:
        _resolve_bypred = None
    # Prefer resolve_stream, fallback to resolve_byprop
    resolve_stream = _resolve_stream or _resolve_byprop
    HAS_LSL = True
    HAS_LSL_CLOCK = True
except Exception as exc:
    _LSL_IMPORT_ERROR = exc
    HAS_LSL = False
    HAS_LSL_CLOCK = False
    StreamInlet = None
    StreamInfo = None
    StreamOutlet = None
    resolve_stream = None
    local_clock = None


ANGLE_KEYS = [
    "thumb_cmc_mcp", "thumb_ip",
    "index_mcp", "index_pip", "index_dip",
    "middle_mcp", "middle_pip", "middle_dip",
    "ring_mcp", "ring_pip", "ring_dip",
    "pinky_mcp", "pinky_pip", "pinky_dip",
]

TARGET_SPECS = {
    "full14": ANGLE_KEYS,
    "finger5": ["thumb_cmc_mcp", "index_mcp", "middle_mcp", "ring_mcp", "pinky_mcp"],
    "index_only": ["index_mcp", "index_pip", "index_dip"],
}

DEFAULT_TARGET_SPEC = "finger5"


def get_target_keys(spec=None):
    if not spec:
        spec = DEFAULT_TARGET_SPEC
    return TARGET_SPECS.get(spec, ANGLE_KEYS)


def resolve_lsl_stream(name=None, stream_type=None, timeout=5.0):
    if not HAS_LSL:
        raise RuntimeError(f"pylsl import failed: {_LSL_IMPORT_ERROR!r}")
    if resolve_stream is None:
        raise RuntimeError("pylsl resolve function not available (resolve_stream/resolve_byprop missing).")
    if name:
        streams = resolve_stream("name", name, timeout=timeout)
    elif stream_type:
        streams = resolve_stream("type", stream_type, timeout=timeout)
    else:
        streams = resolve_stream("type", "MOCAP", timeout=timeout)
    if not streams:
        raise RuntimeError(f"LSL stream not found (name={name}, type={stream_type}).")
    return StreamInlet(streams[0])


def make_marker_outlet(stream_name="NML_TaskMarkers", source_id="nml_task_markers"):
    if not HAS_LSL:
        raise RuntimeError(f"pylsl import failed: {_LSL_IMPORT_ERROR!r}")
    info = StreamInfo(stream_name, "Markers", 1, 0, "string", source_id)
    return StreamOutlet(info)


class SampleBuffer:
    def __init__(self, maxlen=5000):
        self._buf = deque(maxlen=maxlen)
        self._lock = threading.Lock()

    def append(self, timestamp, sample):
        with self._lock:
            self._buf.append((timestamp, sample))

    def latest(self):
        with self._lock:
            return self._buf[-1] if self._buf else None

    def nearest(self, timestamp, max_age=0.5):
        with self._lock:
            if not self._buf:
                return None
            # Find nearest by linear scan (buffer is short)
            best = None
            best_dt = None
            for ts, sample in self._buf:
                dt = abs(ts - timestamp)
                if best_dt is None or dt < best_dt:
                    best_dt = dt
                    best = (ts, sample)
            if best is None:
                return None
            if best_dt is not None and best_dt > max_age:
                return None
            return best


class LSLPullThread:
    def __init__(self, inlet, buffer, stop_event, name="LSL"):
        self.inlet = inlet
        self.buffer = buffer
        self.stop_event = stop_event
        self.name = name
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self.thread.start()

    def _run(self):
        while not self.stop_event.is_set():
            sample, ts = self.inlet.pull_sample(timeout=0.1)
            if sample is None:
                continue
            # Use provided LSL timestamp if present; fallback to local time
            timestamp = ts if ts else time.time()
            self.buffer.append(timestamp, sample)

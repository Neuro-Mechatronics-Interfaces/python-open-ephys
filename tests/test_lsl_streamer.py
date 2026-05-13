from collections import deque
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import threading


MODULE_PATH = (
    Path(__file__).resolve().parent.parent
    / "examples"
    / "interface"
    / "lsl"
    / "open_ephys_lsl_streamer.py"
)


def _load_streamer_module():
    spec = spec_from_file_location("open_ephys_lsl_streamer", MODULE_PATH)
    module = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class _FakeOutlet:
    def __init__(self):
        self.chunks = []

    def push_chunk(self, data, timestamps):
        self.chunks.append((data, timestamps))


class _FakeClient:
    def __init__(self):
        self._lock = threading.Lock()
        self.global_sample_index = 15
        self.total_samples_written = 2
        self._deque_len = 64
        self.buffers = [deque([1.0, 2.0], maxlen=64)]


def test_poll_once_uses_payload_counter_over_header_index():
    module = _load_streamer_module()
    streamer = module.OpenEphysLSLStreamer(expected_fs=4000.0, emg_channels=1)
    streamer.running = True
    streamer.client = _FakeClient()
    streamer._cursor_source = "payload"
    streamer._prev_idx = 0
    streamer.emg_ch_idx = [0]
    streamer.adc_ch_idx = []
    streamer.emg_outlet = _FakeOutlet()
    streamer.imu_outlet = _FakeOutlet()
    streamer.adc_outlet = None
    streamer.detected_fs = 4000.0
    streamer.last_poll = module._now() - 0.05

    info = streamer.poll_once()

    assert info["chunk"] == 2
    assert info["emg_shape"] == (2, 1)
    assert streamer.total_emg == 2
    assert len(streamer.emg_outlet.chunks) == 1
    pushed_samples, pushed_timestamps = streamer.emg_outlet.chunks[0]
    assert len(pushed_samples) == 2
    assert len(pushed_timestamps) == 2


def test_poll_once_downsamples_source_rate_to_requested_output_rate():
    module = _load_streamer_module()
    streamer = module.OpenEphysLSLStreamer(expected_fs=4000.0, emg_channels=1)
    streamer.running = True
    streamer.client = _FakeClient()
    streamer.client.global_sample_index = 15
    streamer.client.total_samples_written = 15
    streamer.client.buffers = [deque([float(i) for i in range(15)], maxlen=64)]
    streamer._cursor_source = "payload"
    streamer._source_fs = 30000.0
    streamer._prev_idx = 0
    streamer.emg_ch_idx = [0]
    streamer.adc_ch_idx = []
    streamer.emg_outlet = _FakeOutlet()
    streamer.imu_outlet = _FakeOutlet()
    streamer.adc_outlet = None
    streamer.detected_fs = 4000.0
    streamer.last_poll = module._now() - 0.05

    info = streamer.poll_once()

    assert info["chunk"] == 2
    assert info["emg_shape"] == (2, 1)
    pushed_samples, pushed_timestamps = streamer.emg_outlet.chunks[0]
    assert len(pushed_samples) == 2
    assert len(pushed_timestamps) == 2
    assert pushed_samples[0][0] == 3.5
    assert pushed_samples[1][0] == 11.0


def test_fs_summary_distinguishes_requested_source_and_emitted_rates():
    module = _load_streamer_module()

    summary = module._fs_summary(
        4000.0,
        30000.0,
        4000.0,
        header_fs=30000.0,
        measured_fs=29984.0,
    )

    assert "requested=4000" in summary
    assert "source=30000" in summary
    assert "emitted=4000" in summary
    assert "header=30000" in summary
    assert "measured=29984" in summary
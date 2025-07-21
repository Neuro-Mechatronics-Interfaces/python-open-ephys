import time
import threading
from collections import deque
import numpy as np
from pyoephys.io import load_open_ephys_session
from pylsl import StreamInfo, StreamInlet, StreamOutlet, resolve_byprop


class OEBinPlaybackClient:
    def __init__(self, oebin_path, block_size=50, loopback=False, enable_lsl=False, verbose=False):
        self.session = load_open_ephys_session(oebin_path)

        print(self.session)

        self.data = self.session['amplifier_data']  # shape (n_channels, n_samples)
        self.sampling_rate = self.session['sample_rate']
        self.ch_names = self.session['channel_names']
        self.block_size = block_size
        self.enable_lsl = enable_lsl
        self.loopback = loopback
        self.verbose = verbose

        self.buffer = np.zeros_like(self.data)
        self.total_samples = 0
        self.current_index = 0  # persistent position tracker
        self.streaming = False
        self.thread = None
        self.lock = threading.Lock()

        if self.enable_lsl:
            self._initialize_lsl_stream()
            self._initialize_marker_stream()

    def _initialize_lsl_stream(self):
        """
        Initialize LSL stream for data
        """
        info = StreamInfo(
            name='OEBinData',
            type='EMG',
            channel_count=self.data.shape[0],
            nominal_srate=self.sampling_rate,
            channel_format='float32',
            source_id='OEBinPlaybackClient'
        )
        info.desc().append_child_value('sampling_rate', str(self.sampling_rate))
        info.desc().append_child_value('created_at', time.strftime('%Y-%m-%d %H:%M:%S'))
        info.desc().append_child_value('manufacturer', 'Open Ephys')
        # Add channel labels
        chns = info.desc().append_child('channels')
        for ch_name in self.ch_names:
            chns.append_child('channel').append_child_value('label', ch_name)
            chns.append_child('channel').append_child_value('unit', 'uV')  # Assuming microvoltage units

        self.lsl_outlet = StreamOutlet(info)

    def _initialize_marker_stream(self):
        """ Initialize LSL stream for markers
        """
        info = StreamInfo(
            name='OEBinMarkers',
            type='Markers',
            channel_count=1,
            nominal_srate=0,  # Markers don't have a sampling rate
            channel_format='string',
            source_id='OEBinPlaybackClientMarkers'
        )
        self.marker_outlet = StreamOutlet(info)

    def start_streaming(self):
        """ Start the streaming thread """
        if not self.streaming:
            self.streaming = True
            self.thread = threading.Thread(target=self._stream_loop, daemon=True)
            self.thread.start()
            if self.verbose:
                print("Streaming started...")

    def stop_streaming(self):
        """ Stop the streaming thread """
        if self.streaming:
            self.streaming = False
            if self.thread:
                self.thread.join()
                self.thread = None
            if self.verbose:
                print("Streaming stopped.")

    def reset_stream(self):
        """ Reset the stream to the beginning """
        with self.lock:
            self.total_samples = 0
            self.current_index = 0
            self.buffer.fill(0)
            if self.verbose:
                print("Stream reset.")

    def is_done(self):
        """ Check if the stream has finished """
        return self.current_index >= self.data.shape[1]

    def _stream_loop(self):
        """ Main loop for streaming data """
        interval = self.block_size / self.sampling_rate
        while self.streaming and self.current_index < self.data.shape[1]:
            with self.lock:
                end = min(self.current_index + self.block_size, self.data.shape[1])
                chunk = self.data[:, self.current_index:end]
                self.buffer[:, self.total_samples:end] = chunk
                self.total_samples = end
                self.current_index = end

                if self.enable_lsl and self.lsl_outlet:
                    # LSL expects shape: (samples, channels)
                    self.lsl_outlet.push_chunk(chunk.T.tolist())

                if self.verbose:
                    print(f"Streaming: index={self.current_index}, total_samples={self.total_samples}")

                # Handle loopback
                if self.current_index >= self.data.shape[1]:
                    if self.loopback:
                        if self.verbose:
                            print("Looping playback...")
                        if self.marker_outlet:
                            self.marker_outlet.push_sample(['LoopReset'])
                        self.current_index = 0
                        self.total_samples = 0
                        self.buffer.fill(0)
                    else:
                        break

            time.sleep(interval)

    def close(self):
        """ Close the client and release resources """
        self.stop_streaming()
        if self.verbose:
            print("Client closed.")

    def get_latest_window(self, window_ms):
        """ Get the latest window of data in milliseconds """
        samples_per_window = int(window_ms / 1000 * self.sampling_rate)
        start_index = max(0, self.total_samples - samples_per_window)
        end_index = self.total_samples
        return self.buffer[:, start_index:end_index]


class LSLClient:
    def __init__(self, maxlen=10000, stream_type="EMG"):
        print(f"[LSLClient] Looking for a stream of type '{stream_type}'...")
        streams = resolve_byprop("type", stream_type, timeout=5)
        if not streams:
            raise RuntimeError(f"No LSL stream with type '{stream_type}' found.")

        self.inlet = StreamInlet(streams[0])
        self.stream_info = self.inlet.info()
        self.n_channels = self.stream_info.channel_count()
        self.fs = self._get_sampling_rate()
        self.channel_labels, self.units = self._get_channel_metadata()

        self.buffers = [deque(maxlen=maxlen) for _ in range(self.n_channels)]
        self.lock = threading.Lock()
        self.running = True
        self.thread = threading.Thread(target=self._pull_data_loop, daemon=True)
        self.thread.start()
        print(f"[LSLClient] Connected to stream '{self.stream_info.name()}'")
        print(f"  Channels: {self.n_channels}, Sampling Rate: {self.fs} Hz")

    def _get_sampling_rate(self):
        try:
            rate = self.stream_info.nominal_srate()
            return float(rate) if rate > 0 else None
        except Exception:
            return None

    def _get_channel_metadata(self):
        try:
            ch_info = self.stream_info.desc().child("channels").child("channel")
            labels = []
            units = []
            for _ in range(self.n_channels):
                labels.append(ch_info.child_value("label") or f"Ch{_}")
                units.append(ch_info.child_value("unit") or "unknown")
                ch_info = ch_info.next_sibling()
            return labels, units
        except Exception:
            return [f"Ch{i}" for i in range(self.n_channels)], ["unknown"] * self.n_channels

    def _pull_data_loop(self):
        while self.running:
            sample, _ = self.inlet.pull_sample(timeout=0.1)
            if sample is not None:
                with self.lock:
                    for ch, val in enumerate(sample):
                        if ch < self.n_channels:
                            self.buffers[ch].append(val)

    def get_samples(self, channel: int, n_samples: int):
        with self.lock:
            buf = list(self.buffers[channel])
        if len(buf) < n_samples:
            buf = [0.0] * (n_samples - len(buf)) + buf
        return buf[-n_samples:]

    def stop(self):
        self.running = False
        self.thread.join()
        print("[LSLClient] Stopped.")


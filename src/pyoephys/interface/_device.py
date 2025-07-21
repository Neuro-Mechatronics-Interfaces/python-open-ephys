import numpy as np
import zmq
import time
import uuid
import json
from threading import Thread, Lock
from collections import deque
from ._gui_client import GUIClient
from ._gui_events import Event, Spike


class OpenEphysDevice(GUIClient):
    """
    Class for interacting with Open Ephys via ZMQ in real time.

    Responsibilities:
        - Maintains a real-time circular buffer of streamed EMG data
        - Provides time window access
        - Can record data for fixed durations

    Parameters:
        ip (str): IP address or ZMQ prefix (default 'tcp://localhost')
        data_port (int): Port number for ZMQ data stream
        heartbeat_port (int): Port number for heartbeat messages
        num_channels (int): Number of channels to collect
        sample_rate (float): Approximate sample rate in Hz (Open Ephys does not send this directly)
        buffer_duration_sec (int): Circular buffer length in seconds
        verbose (bool): Print debug information
    """
    def __init__(self, zqm_ip="tcp://localhost", http_ip="127.0.0.1", data_port=5556, heartbeat_port=5557,
                 num_channels=128, sample_rate=2000.0, buffer_duration_sec=5, verbose=False):
        super().__init__(host=http_ip)
        self.ip = zqm_ip
        self.data_port = data_port
        self.heartbeat_port = heartbeat_port
        self.num_channels = num_channels
        self.sample_rate = sample_rate
        self.buffer_duration_sec = buffer_duration_sec
        self.verbose = verbose

        self.uuid = str(uuid.uuid4())
        self.message_num = 0
        self.last_reply_time = time.time()
        self.last_heartbeat_time = time.time()
        self.socket_waits_reply = False

        # Buffers
        self.buffers = [deque(maxlen=int(sample_rate * buffer_duration_sec)) for _ in range(num_channels)]
        self.last_buffer_lens = [0] * self.num_channels
        self.circular_buffer = np.zeros((num_channels, int(sample_rate * buffer_duration_sec)), dtype=np.float32)
        self.circular_idx = 0
        self.total_samples_written = 0  # Track total samples for proper indexing
        self.buffer_lock = Lock()

        # Connection status tracking
        self.connection_lost = False
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 5

        self.streaming = False
        self.streaming_thread = None

        # --- ZMQ Setup ---
        self.context = zmq.Context()
        self.poller = zmq.Poller()
        self.heartbeat_socket = None
        self.data_socket = None
        self.initialize_sockets()

    def initialize_sockets(self):
        """Initialize the data socket with improved error handling"""
        try:
            if not self.data_socket:
                ip_string = f'{self.ip}:{self.data_port}'
                print("Initializing data socket on " + ip_string)
                self.data_socket = self.context.socket(zmq.SUB)
                self.data_socket.connect(ip_string)
                self.data_socket.setsockopt(zmq.SUBSCRIBE, b'')
                self.data_socket.setsockopt(zmq.RCVTIMEO, 1000)  # 1 second timeout
                self.poller.register(self.data_socket, zmq.POLLIN)

            if not self.heartbeat_socket:
                ip_string = f'{self.ip}:{self.heartbeat_port}'
                print("Initializing heartbeat socket on " + ip_string)
                self.heartbeat_socket = self.context.socket(zmq.REQ)
                self.heartbeat_socket.connect(ip_string)
                self.heartbeat_socket.setsockopt(zmq.RCVTIMEO, 2000)  # 2 second timeout
                self.poller.register(self.heartbeat_socket, zmq.POLLIN)

            self.connection_lost = False
            self.reconnect_attempts = 0

        except Exception as e:
            print(f"[Socket Error] Failed to initialize sockets: {e}")
            self.connection_lost = True

    def _reconnect_sockets(self):
        """Attempt to reconnect sockets after connection loss"""
        if self.reconnect_attempts >= self.max_reconnect_attempts:
            print(f"[Connection] Max reconnection attempts reached ({self.max_reconnect_attempts})")
            return False

        print(f"[Connection] Attempting reconnection ({self.reconnect_attempts + 1}/{self.max_reconnect_attempts})")

        try:
            # Close existing sockets
            if self.data_socket:
                self.data_socket.close()
                self.data_socket = None
            if self.heartbeat_socket:
                self.heartbeat_socket.close()
                self.heartbeat_socket = None

            # Wait before reconnecting
            time.sleep(1)

            # Reinitialize
            self.initialize_sockets()
            self.reconnect_attempts += 1

            return not self.connection_lost

        except Exception as e:
            print(f"[Reconnection Error] {e}")
            self.reconnect_attempts += 1
            return False

    def _send_heartbeat(self):
        """
        Sends heartbeat message to ZMQ Interface to indicate that the app is alive
        """
        if self.connection_lost:
            return

        try:
            msg = json.dumps({'application': 'OpenEphysDevice', 'uuid': self.uuid, 'type': 'heartbeat'})
            self.heartbeat_socket.send(msg.encode('utf-8'))
            self.last_heartbeat_time = time.time()
            self.socket_waits_reply = True
            if self.verbose:
                print("[Heartbeat] Sent")
        except Exception as e:
            print(f"[Heartbeat Error] {e}")
            self.connection_lost = True
            # self._reconnect_sockets()

    def _streaming_worker(self):
        while self.streaming:

            # Handle reconnection as needed
            if self.connection_lost:
                if not self._reconnect_sockets():
                    #print("[Streaming] Connection lost. Stopping streaming.")
                    #self.stop_streaming()
                    #continue
                    time.sleep(1)
                    continue

            if (time.time() - self.last_heartbeat_time) > 2:
                self._send_heartbeat()

            try:
                socks = dict(self.poller.poll(10))
                if self.data_socket in socks:
                    try:
                        msg = self.data_socket.recv_multipart(zmq.NOBLOCK)
                        if len(msg) < 2:
                            continue

                        header = json.loads(msg[1].decode('utf-8'))
                        self.message_num = header['message_num']

                        if header['type'] == 'data':
                            content = header['content']
                            ch = content['channel_num']
                            samples = np.frombuffer(msg[2], dtype=np.float32)

                            if 0 <= ch < self.num_channels:
                                with (self.buffer_lock):
                                    self.buffers[ch].extend(samples)

                                    # Write into circular_buffer
                                    n = len(samples)
                                    buf_len = self.circular_buffer.shape[1]

                                    # Get start index for channel
                                    start_idx = self.total_samples_written % buf_len
                                    end_idx = (start_idx + n) % buf_len

                                    if start_idx < end_idx:
                                        # No wraparound
                                        self.circular_buffer[ch, start_idx:end_idx] = samples[:n]
                                    else:
                                        # handle wraparound
                                        split = buf_len - start_idx
                                        self.circular_buffer[ch, start_idx:] = samples[:split]
                                        if n > split:
                                            self.circular_buffer[ch, :end_idx] = samples[split:n]

                                    if ch == 0: # Can use channel 0 as master clock
                                        self.total_samples_written += n
                                        self.circular_idx = self.total_samples_written % buf_len

                                if self.verbose:
                                    print(f"[Data] Ch {ch}: {len(samples)} samples")

                        elif header['type'] == 'event':
                            evt = Event(header['content'], msg[2] if header['data_size'] > 0 else None)
                            if self.verbose:
                                print(evt)

                        elif header['type'] == 'spike':
                            spk = Spike(header['spike'], msg[2])
                            if self.verbose:
                                print(spk)

                    except zmq.Again:
                        pass

                    except Exception as e:
                        print(f"[Data Error] {e}")

                if self.heartbeat_socket in socks and self.socket_waits_reply:
                    try:
                        _ = self.heartbeat_socket.recv()
                        self.socket_waits_reply = False
                        self.last_reply_time = time.time()
                    except zmq.Again:
                        pass
                    except Exception as e:
                        print(f"[Heartbeat Error] {e}")
                        self.connection_lost = True
                        # self._reconnect_sockets()

            except Exception as e:
                print(f"Streaming worker error: {e}")
                self.connection_lost = True
                time.sleep(0.1)

    def start_streaming(self):
        if self.streaming:
            print("Already streaming")
            return
        self.streaming = True
        self.streaming_thread = Thread(target=self._streaming_worker, daemon=True)
        self.streaming_thread.start()

    def stop_streaming(self):
        self.streaming = False
        if self.streaming_thread:
            self.streaming_thread.join()

    def get_latest_window(self, duration_ms):
        """ Get most recent window of data"""
        num_samples = int(self.sample_rate * duration_ms / 1000)
        buf_len = self.circular_buffer.shape[1]

        with self.buffer_lock:
            available_samples = min(self.total_samples_written, buf_len)
            if available_samples == 0:
                return np.zeros((self.num_channels, 0), dtype=np.float32)

            # Clamp requested samples to available
            num_samples = min(num_samples, available_samples)

            # Calculate the proper read range
            current_idx = self.total_samples_written % buf_len
            start_idx = (current_idx - num_samples) % buf_len

            if start_idx < current_idx:
                # No wraparound needed
                return self.circular_buffer[:, start_idx:current_idx].copy()
            else:
                # Handle wraparound
                return np.hstack([self.circular_buffer[:, start_idx:], self.circular_buffer[:, :current_idx] ])

    def get_latest_sample(self):
        """ Get most recent sample from all channels"""
        with self.buffer_lock:
            if self.total_samples_written == 0:
                return np.zeros(self.num_channels, dtype=np.float32)

            latest_idx = (self.total_samples_written - 1) % self.circular_buffer.shape[1]
            return self.circular_buffer[:, latest_idx].copy()

    def record(self, duration_sec=10, verbose=True):
        """
        Record EMG data for a specified duration.

        Parameters:
            duration_sec (int): Number of seconds to record.
            verbose (bool): Whether to print stream rate.

        Returns:
            np.ndarray: EMG data array of shape (channels, samples)
        """
        total_samples = int(self.sample_rate * duration_sec)
        collected_emg = np.zeros((self.num_channels, total_samples), dtype=np.float32)
        write_index = 0

        # TO-DO: Send the record command to the gui
        if verbose:
            print(f"[Recording] {duration_sec}s...")

        start_time = time.time()
        samples_collected = 0

        # Record start position to avoid collecting old data
        with self.buffer_lock:
            start_sample_count = self.total_samples_written

        while samples_collected < total_samples:
            current_time = time.time()
            elapsed = current_time - start_time

            # Calculate expected samples based on elapsed time
            expected_samples = int(elapsed * self.sample_rate)
            expected_samples = min(expected_samples, total_samples)

            if expected_samples > samples_collected:
                # Get new samples since recording started
                with self.buffer_lock:
                    available_new_samples = self.total_samples_written - start_sample_count

                if available_new_samples > samples_collected:
                    samples_to_get = min(expected_samples - samples_collected,
                                         available_new_samples - samples_collected)

                    if samples_to_get > 0:
                        # Get the specific range of new samples
                        window = self.get_latest_window(int(samples_to_get * 1000 / self.sample_rate))
                        if window.shape[1] >= samples_to_get:
                            end_idx = samples_collected + samples_to_get
                            collected_emg[:, samples_collected:end_idx] = window[:, -samples_to_get:]
                            samples_collected += samples_to_get

            #time.sleep(0.01)  # Smaller sleep for better temporal resolution

        return collected_emg

    def record_to_file(self, path, duration_sec=10):
        emg = self.record(duration_sec)
        np.savez(path, emg=emg, sample_rate=self.sample_rate)
        if self.verbose:
            print(f"[Saved] {path}")

    def get_connection_status(self):
        """Get current connection status"""
        return {
            'connected': not self.connection_lost,
            'streaming': self.streaming,
            'reconnect_attempts': self.reconnect_attempts,
            'total_samples': self.total_samples_written
        }

    def close(self):
        self.stop_streaming()
        if self.data_socket:
            self.data_socket.close()
        if self.heartbeat_socket:
            self.heartbeat_socket.close()
        self.context.term()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
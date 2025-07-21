import matplotlib

matplotlib.use('Qt5Agg')  # Use a faster interactive backend

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.animation import FuncAnimation
from scipy.signal import iirnotch, butter, lfilter, lfilter_zi
from collections import deque


class RealtimePlotter:
    def __init__(self, client, sampling_rate=2000.0, plotting_interval=1.0,
                 samples_per_fetch=50, channels_to_plot=None, apply_filters=True, downsample_factor=1):
        self.client = client
        self.sampling_rate = sampling_rate
        self.plotting_interval = plotting_interval
        self.samples_per_fetch = samples_per_fetch
        self.apply_filters = apply_filters
        self.downsample_factor = downsample_factor

        self.buffer_size = int(self.plotting_interval * self.sampling_rate)
        self.x = np.linspace(0, self.plotting_interval, self.buffer_size)

        # Define which channels to plot
        if channels_to_plot is None:
            self.channels_to_plot = [0]  # default to channel 0
        else:
            self.channels_to_plot = channels_to_plot

        self.n_channels = len(self.channels_to_plot)

        # Use deque for efficient rolling buffer updates
        self.ydata = {
            ch: deque(np.zeros(self.buffer_size, dtype=np.float32), maxlen=self.buffer_size)
            for ch in self.channels_to_plot
        }

        # Track last processed sample count for incremental updates
        self.last_processed_samples = 0

        if self.apply_filters:
            self.init_filters()
        self.init_plot()

    def init_filters(self):
        """Initialize real-time filters"""
        notch_freq = 60.0
        Q = 30.0
        self.notch_b, self.notch_a = iirnotch(notch_freq, Q, fs=self.sampling_rate)

        lowcut = 10.0
        highcut = 500.0
        nyq = 0.5 * self.sampling_rate
        self.bp_b, self.bp_a = butter(2, [lowcut / nyq, highcut / nyq], btype='band')

        # Initialize filter states for each channel
        self.notch_zi = {ch: lfilter_zi(self.notch_b, self.notch_a) for ch in self.channels_to_plot}
        self.bp_zi = {ch: lfilter_zi(self.bp_b, self.bp_a) for ch in self.channels_to_plot}

    def apply_realtime_filters(self, signal, channel):
        """ Apply notch and bandpass filters to the signal """
        if not self.apply_filters or len(signal) == 0:
            return signal

        # Apply notch filter
        filtered_signal, self.notch_zi[channel] = lfilter(
            self.notch_b, self.notch_a, signal, zi=self.notch_zi[channel]
        )

        # Apply bandpass filter
        filtered_signal, self.bp_zi[channel] = lfilter(
            self.bp_b, self.bp_a, filtered_signal, zi=self.bp_zi[channel]
        )

        return filtered_signal

    def init_plot(self):
        self.figure, self.axes = plt.subplots(
            self.n_channels, 1,
            figsize=(10, 2 * self.n_channels),
            sharex=True
        )
        if self.n_channels == 1:
            self.axes = [self.axes]

        self.hl = []
        colors = ['#d92eab', '#2ed9a3', '#d9a32e', '#2e5dd9', '#d92e2e', '#2ed95d']

        for i, (ax, ch) in enumerate(zip(self.axes, self.channels_to_plot)):
            ax.set_xlim(0, self.plotting_interval)
            ax.set_ylim(-200, 200)
            ax.set_ylabel(f"Ch {ch} (μV)", fontsize=10)
            ax.set_facecolor('#001230')
            ax.grid(True, alpha=0.3, color='white', linestyle='--', linewidth=0.5)

            # line, = ax.plot(self.x, self.ydata[ch], lw=0.5, color='#d92eab')
            # self.hl.append(line)
            # Use different colors for different channels
            color = colors[i % len(colors)]
            line, = ax.plot(
                self.x[::self.downsample_factor],  # downsampled x
                list(self.ydata[ch])[::self.downsample_factor],  # downsampled y
                lw=0.8,
                color=color,
                alpha=0.9
            )
            self.hl.append(line)

        self.axes[-1].set_xlabel("Time (s)", fontsize=10)
        self.figure.suptitle("Realtime EMG Signal", fontsize=14, fontweight='bold')
        plt.subplots_adjust(hspace=0.4, bottom=0.2)

        # Slider to control y-axis range
        axcolor = '#f0f0f0'
        axylim = plt.axes([0.1, 0.02, 0.65, 0.03], facecolor=axcolor)
        self.sylim = Slider(axylim, 'Y range (μV)', 10, 1000, valinit=200, valstep=10, valfmt='%d')

        def update_slider(val):
            for ax in self.axes:
                ax.set_ylim(-val, val)
            self.figure.canvas.draw_idle()

        self.sylim.on_changed(update_slider)

        # Connection status text
        self.status_text = self.figure.text(
            0.7, 0.02,
            "Status: Connecting...",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
            animated=True,
        )

        self.anim = FuncAnimation(
            self.figure, self.update,
            interval=100,
            blit=False,
            cache_frame_data=False,
        )

    def _init_lines(self):
        return self.hl + [self.status_text]

    def update(self, frame):
        """Update the plot with new data"""
        try:
            # Get connection status
            status = self.client.get_connection_status()
            if status['connected']:
                status_text = f"Connected | Samples: {status['total_samples']}"
                self.status_text.set_bbox(dict(boxstyle="round,pad=0.3", facecolor="green", alpha=0.7))
            else:
                status_text = f"Reconnecting...({status['reconnect_attempts']}/5)"
                self.status_text.set_bbox(dict(boxstyle="round,pad=0.3", facecolor="red", alpha=0.7))

            self.status_text.set_text(status_text)

            if not status['connected'] or status['total_samples'] == 0:
                return self.hl + [self.status_text]

            # Get latest window of data
            window_ms = min(200, int(self.plotting_interval * 1000))  # Get recent data
            window = self.client.get_latest_window(window_ms)

            if window.shape[1] == 0:
                return self.hl + [self.status_text]

            # Process each channel
            for i, ch in enumerate(self.channels_to_plot):
                if ch >= window.shape[0]:
                    continue

                new_samples = window[ch, :]
                if len(new_samples) > 0:
                    if self.apply_filters and len(new_samples) >= 5:
                        filtered_samples = self.apply_realtime_filters(new_samples, ch)
                    else:
                        filtered_samples = new_samples

                    # Update rolling buffer
                    self.ydata[ch].extend(filtered_samples)

                    # Downsample the data for plotting
                    downsampled = list(self.ydata[ch])[::self.downsample_factor]

                    # Update plot data
                    self.hl[i].set_data(self.x[::self.downsample_factor], downsampled)

            return self.hl + [self.status_text]

        except Exception as e:
            print(f"[Plot Update Error] {e}")
            self.status_text.set(text="Error: " + str(e))
            self.status_text.set_bbox(dict(boxstyle="round,pad=0.3", facecolor="red", alpha=0.7))
            return self.hl + [self.status_text]

    def toggle_filters(self):
        """Toggle filter application on/off"""
        self.apply_filters = not self.apply_filters
        if self.apply_filters:
            self.init_filters()  # Reinitialize filter states
        print(f"Filters {'enabled' if self.apply_filters else 'disabled'}")

    def save_current_view(self, filename="emg_snapshot.png"):
        """Save current plot view to file"""
        try:
            self.figure.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Plot saved as {filename}")
        except Exception as e:
            print(f"Failed to save plot: {e}")

    def run(self):
        """Start the real-time plotter"""
        if not self.client.streaming:
            self.client.start_streaming()
            auto_started = True
        else:
            auto_started = False

        try:
            plt.show()
        finally:
            if auto_started:
                self.client.stop_streaming()

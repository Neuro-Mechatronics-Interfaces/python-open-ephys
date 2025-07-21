"""
pyoephys.applications._emg_viewer

Graphical tool for interactively labeling trial events on EMG recordings.

This GUI allows researchers to:
- Load and visualize EMG signals from `.rhd` files
- Select individual channels
- Click to mark trial onset points
- Assign labels to each indexed event
- Append new recordings for multi-session review
- Export trial events to a timestamped CSV or TXT file

Clicking the signal while "Set Trial Index" is enabled will add a labeled marker.
This tool is useful for supervised training of gesture classifiers, post-hoc annotation,
or protocol validation in EMG experiments.
"""
import tkinter as tk
from tkinter import filedialog, messagebox, ttk, Scrollbar, VERTICAL
import numpy as np
import datetime
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from pyoephys.io import load_oebin_file
from pyoephys.processing import bandpass_filter, common_average_reference


class EMGViewer:
    """
    Tkinter-based application for manual EMG trial indexing.

    Attributes:
        emg_data (np.ndarray): EMG signal matrix (channels × samples)
        time_vector (np.ndarray): Time vector aligned with EMG samples
        sampling_rate (float): Sampling rate of amplifier
        current_channel (int): Channel index currently displayed
        indexing_enabled (bool): If True, allows user to click to insert marker

    """
    def __init__(self, root):
        self.root = root
        self.root.title("EMG Trial Selector")

        self.emg_data = None
        self.time_vector = None
        self.sampling_rate = None
        self.current_channel = 0
        self.indexing_enabled = False

        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        # --- Top Controls Frame ---
        control_frame = tk.Frame(root)
        control_frame.pack(side="top", fill="x", pady=5)

        tk.Button(control_frame, text="Load EMG File", command=self.load_file).pack(side="left", padx=5)
        tk.Button(control_frame, text="Set Trial Index", command=self.enable_indexing).pack(side="left", padx=5)
        tk.Button(control_frame, text="Append EMG File", command=self.append_file).pack(side="left", padx=5)

        # --- Main Frame (Canvas + Sidebar) ---
        main_frame = tk.Frame(root)
        main_frame.pack(side="top", fill="both", expand=True)

        # === Plot Area ===
        self.figure, self.ax = plt.subplots(figsize=(10, 4))
        self.canvas = FigureCanvasTkAgg(self.figure, master=main_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(side="left", fill="both", expand=True)

        # === Sidebar Frame ===
        sidebar_frame = tk.Frame(main_frame)
        sidebar_frame.pack(side="right", fill="y", padx=10)

        # --- Channel Selector ---
        tk.Label(sidebar_frame, text="Channel:").pack(anchor="w")
        self.channel_selector = ttk.Combobox(sidebar_frame, state="readonly")
        self.channel_selector.bind("<<ComboboxSelected>>", self.update_channel)
        self.channel_selector.pack(fill="x", pady=5)

        # --- Common Average Reference Toggle ---
        self.car_enabled = tk.BooleanVar()
        self.car_checkbox = tk.Checkbutton(
            sidebar_frame,
            text="Enable Common Average Reference",
            variable=self.car_enabled,
            command=self.plot_channel  # refresh plot when toggled
        )
        self.car_checkbox.pack(anchor="w", pady=5)

        # --- Bandpass Filter Toggle ---
        self.filter_enabled = tk.BooleanVar()
        self.filter_checkbox = tk.Checkbutton(
            sidebar_frame,
            text="Enable Bandpass Filter",
            variable=self.filter_enabled,
            command=self.plot_channel  # refresh plot when toggled
        )
        self.filter_checkbox.pack(anchor="w", pady=5)

        # --- Filter Settings ---
        tk.Label(sidebar_frame, text="Low Cut (Hz):").pack(anchor="w")
        self.low_cut_var = tk.DoubleVar(value=20.0)
        self.low_cut_entry = tk.Entry(sidebar_frame, textvariable=self.low_cut_var)
        self.low_cut_entry.pack(fill="x", pady=2)

        tk.Label(sidebar_frame, text="High Cut (Hz):").pack(anchor="w")
        self.high_cut_var = tk.DoubleVar(value=400.0)
        self.high_cut_entry = tk.Entry(sidebar_frame, textvariable=self.high_cut_var)
        self.high_cut_entry.pack(fill="x", pady=2)

        tk.Label(sidebar_frame, text="Filter Order:").pack(anchor="w")
        self.filter_order_var = tk.IntVar(value=2)
        self.filter_order_entry = tk.Entry(sidebar_frame, textvariable=self.filter_order_var)
        self.filter_order_entry.pack(fill="x", pady=2)

        # --- Plot Time Range ---
        tk.Label(sidebar_frame, text="Start Time (s):").pack(anchor="w")
        self.start_time_var = tk.DoubleVar(value=0.0)
        tk.Entry(sidebar_frame, textvariable=self.start_time_var).pack(fill="x", pady=2)

        tk.Label(sidebar_frame, text="End Time (s):").pack(anchor="w")
        self.end_time_var = tk.DoubleVar(value=10.0)  # Default to 10 seconds or something reasonable
        tk.Entry(sidebar_frame, textvariable=self.end_time_var).pack(fill="x", pady=2)


        # --- Label Entry Field ---
        tk.Label(sidebar_frame, text="Custom Label:").pack(anchor="w")
        self.label_entry = tk.Entry(sidebar_frame)
        self.label_entry.pack(fill="x", pady=5)
        self.label_entry.insert(0, "Label")  # Default text

        # --- Table ---
        self.table = ttk.Treeview(sidebar_frame, columns=("Sample Index", "Label"), show="headings", height=20)
        self.table.heading("Sample Index", text="Sample Index")
        self.table.heading("Label", text="Label")
        self.table.column("Sample Index", width=100)
        self.table.column("Label", width=100)
        self.table.pack(side="top", fill="y")

        scrollbar = Scrollbar(sidebar_frame, orient=VERTICAL, command=self.table.yview)
        self.table.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")

        # --- Save & Delete Buttons ---
        button_frame = tk.Frame(sidebar_frame)
        button_frame.pack(side="bottom", pady=10)
        tk.Button(button_frame, text="Save", command=self.save_table).pack(side="left", padx=5)
        tk.Button(button_frame, text="Delete", command=self.delete_selected).pack(side="left", padx=5)

        self.canvas.mpl_connect("button_press_event", self.on_click)

    def load_file(self):
        """
        Load EMG data from a .rhd file and initialize the GUI with the first channel.
        """
        #path = filedialog.askdirectory(title="Select Open Ephys Session folder")
        path = filedialog.askopenfilename(filetypes=[
            ("OEBIN Files", "*.oebin"),
            ("CSV Files", "*.csv")])
        if not path:
            return

        if path.endswith('.csv'):
            # Load CSV file. first column has timestamp data in milliseconds elapsed, the rest are EMG channels if
            # they have "EMG" in the name. The first row has only header information
            data = np.loadtxt(path, delimiter=',', skiprows=1)
            self.time_vector = data[:, 0] / 1000.0

            # Find the columns that contain "EMG" in their header
            with open(path, 'r') as f:
                header = f.readline().strip().split(',')
            emg_columns = [i for i, col in enumerate(header) if "EMG" in col]
            self.emg_data = data[:, emg_columns].T

            self.sampling_rate = 1000.0 / (self.time_vector[1] - self.time_vector[0])  # Assuming uniform sampling

        elif path.endswith('.oebin'):
            # Load OEBIN file
            #folder_path = os.path.dirname(path)
            #result = load_open_ephys_file(folder_path)
            result = load_oebin_file(path)
            self.emg_data = result["amplifier_data"]
            self.time_vector = result["t_amplifier"]
            self.sampling_rate = result["sample_rate"]

        print("Data shape:", self.emg_data.shape)
        if self.emg_data.ndim == 1:
            # If data is 1D, convert to 2D with one channel
            self.emg_data = self.emg_data[np.newaxis, :]
        elif self.emg_data.ndim != 2:
            messagebox.showerror("Error", "EMG data must be 1D or 2D array.")
            return

        print("Sampling rate:", self.sampling_rate)

        self.channel_selector['values'] = [f"Channel {i}" for i in range(self.emg_data.shape[0])]
        self.channel_selector.current(0)
        self.current_channel = 0
        self.plot_channel()

    def append_file(self):
        """
        Append EMG data from another .oebin file to the current data.
        """
        path = filedialog.askdirectory(title="Select Open Ephys Session folder")
        if not path:
            return

        #result = load_open_ephys_file(path)
        result = load_oebin_file(path)
        new_emg = result["amplifier_data"]
        new_time = result["t_amplifier"]
        new_rate = result["sample_rate"]

        if self.emg_data is None:
            # If no prior data, treat this like a fresh load
            self.emg_data = new_emg
            self.time_vector = new_time
            self.sampling_rate = new_rate
            self.channel_selector['values'] = [f"Channel {i}" for i in range(self.emg_data.shape[0])]
            self.channel_selector.current(0)
            self.current_channel = 0
            self.plot_channel()
            return

        # Sanity check: channel count and sampling rate must match
        if new_emg.shape[0] != self.emg_data.shape[0] or new_rate != self.sampling_rate:
            messagebox.showerror("Error", "Appended file must have same channel count and sampling rate.")
            return

        # Offset time vector based on last timestamp
        last_time = self.time_vector[-1]
        offset_time = new_time

        self.emg_data = np.concatenate((self.emg_data, new_emg), axis=1)
        self.time_vector = np.concatenate((self.time_vector, offset_time))
        self.plot_channel()

    def sample_index_to_timestamp(self, index):
        """
        Convert a sample index to a timestamp string.

        Parameters:
            index (int): Sample index to convert.

        Returns:
            str: Formatted timestamp string (HH:MM:SS).
        """
        seconds = index / self.sampling_rate
        return str(datetime.timedelta(seconds=int(seconds)))

    def save_table(self):
        """
        Save the trial markers to a text file with sample index and timestamp.
        """
        path = filedialog.asksaveasfilename(
            defaultextension=".event",
            filetypes=[("Event Files", "*.event")],
            title="Save Trial Markers"
        )
        if not path:
            return

        # Collect and sort table data by sample index
        rows = []
        for row in self.table.get_children():
            sample_index, label = self.table.item(row)["values"]
            sample_index = int(sample_index)
            timestamp = self.sample_index_to_timestamp(sample_index)
            rows.append((sample_index, timestamp, label))

        rows.sort(key=lambda x: x[0])  # Sort by sample index

        # Write to text file
        with open(path, "w") as f:
            f.write("Sample Index,Timestamp,Label\n")
            for sample_index, timestamp, label in rows:
                f.write(f"{sample_index},{timestamp},{label}\n")

        messagebox.showinfo("Saved", f"Trial markers saved to:\n{path}")

    def delete_selected(self):
        """
        Delete selected rows from the table.
        """
        selected = self.table.selection()
        for item in selected:
            self.table.delete(item)

    def update_channel(self, event=None):
        """
        Update the current channel based on the selection from the dropdown.
        """
        if self.emg_data is None:
            return
        self.current_channel = self.channel_selector.current()
        self.plot_channel()

    def enable_indexing(self):
        """
        Enable the indexing mode to allow trial marking on the plot.
        """
        self.indexing_enabled = True

    def on_click(self, event):
        """
        Handle mouse click events on the plot to mark trial onset points.

        Parameters:
            event (matplotlib.backend_bases.Event): The mouse event.
        """
        if not self.indexing_enabled or event.inaxes != self.ax:
            return

        time_clicked = event.xdata
        amp_clicked = event.ydata
        if time_clicked is None:
            return

        sample_index = max(0, int(time_clicked * self.sampling_rate))
        self.ax.axvline(x=time_clicked, color='blue', linestyle='--')
        self.ax.axhline(y=amp_clicked, color='red', linestyle='--')
        self.canvas.draw()

        # Insert editable row into the table
        label = self.label_entry.get()
        self.table.insert("", "end", values=(sample_index, label))
        self.indexing_enabled = False

    def plot_channel(self):
        """
        Plot the currently selected EMG channel.
        """
        self.ax.clear()

        data_to_plot = self.emg_data

        # Apply CAR if enabled
        if self.car_enabled.get():
            data_to_plot = common_average_reference(data_to_plot)

        data_to_plot = data_to_plot[self.current_channel]
        try:
            if self.filter_enabled.get():
                lowcut = self.low_cut_var.get()
                highcut = self.high_cut_var.get()
                order = self.filter_order_var.get()

                if lowcut >= highcut:
                    messagebox.showerror("Invalid Filter", "Low cut must be lower than high cut frequency.")
                    return

                data_to_plot = bandpass_filter(
                    data_to_plot[np.newaxis, :],
                    lowcut=lowcut,
                    highcut=highcut,
                    fs=self.sampling_rate,
                    order=order,
                    axis=1
                )[0]  # extract filtered 1D array

            start_time = self.start_time_var.get()
            end_time = self.end_time_var.get()

            # Validate times
            if start_time >= end_time:
                messagebox.showerror("Invalid Time Range", "Start time must be less than end time.")
                return

            # Convert times to sample indices
            start_idx = int(start_time * self.sampling_rate)
            end_idx = int(end_time * self.sampling_rate)

            # Clip indices to valid range
            start_idx = max(0, start_idx)
            end_idx = min(len(self.time_vector), end_idx)

            # Slice the data
            time_window = self.time_vector[start_idx:end_idx]
            data_window = data_to_plot[start_idx:end_idx]

            self.ax.plot(time_window, data_window, label=f"Channel {self.current_channel}")
            self.ax.set_xlabel("Time (s)")
            self.ax.set_ylabel("Amplitude (µV)")
            self.ax.set_title("EMG Signal")
            self.ax.legend()
            self.canvas.draw()

        except Exception as e:
            messagebox.showerror("Plot Error", f"Failed to plot channel: {e}")

    def on_closing(self):
        self.root.quit()


def launch_emg_viewer():
    """
    Launch the EMG trial selector GUI.
    """
    root = tk.Tk()
    app = EMGViewer(root)
    root.mainloop()

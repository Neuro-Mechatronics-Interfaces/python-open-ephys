"""
pyoephys.applications._emg_trial_selector

Graphical tool for interactively labeling trial events on EMG recordings.
Ported from python-intan for use with python-open-ephys.
"""

import tkinter as tk
from tkinter import filedialog, messagebox, ttk, Scrollbar, VERTICAL
import numpy as np
import datetime
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# IO Imports
try:
    from pyoephys.io import load_npz_file
except Exception:
    load_npz_file = None

class EMGTrialSelector:
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
        # Append functionality requires consistent file formats (e.g. RHD), disabled or simplified for now
        # tk.Button(control_frame, text="Append EMG File", command=self.append_file).pack(side="left", padx=5)

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

        # Track drawn marker artists so we can clear them
        self._marker_artists = []

        # --- Edge Extraction Controls ---
        tk.Label(sidebar_frame, text="Edge detect (selected chan)").pack(anchor="w", pady=(12, 0))

        edge_frame = tk.Frame(sidebar_frame)
        edge_frame.pack(fill="x", pady=4)

        tk.Label(edge_frame, text="Threshold").grid(row=0, column=0, sticky="w")
        self.edge_threshold_entry = tk.Entry(edge_frame, width=10)
        self.edge_threshold_entry.grid(row=0, column=1, sticky="w", padx=4)
        self.edge_threshold_entry.insert(0, "0.5")  # default threshold

        tk.Label(edge_frame, text="Type").grid(row=1, column=0, sticky="w")
        self.edge_type_combo = ttk.Combobox(edge_frame, state="readonly", width=12,
                                            values=["rising", "falling"])
        self.edge_type_combo.grid(row=1, column=1, sticky="w", padx=4)
        self.edge_type_combo.current(0)

        tk.Label(edge_frame, text="Min gap (ms)").grid(row=2, column=0, sticky="w")
        self.edge_mingap_entry = tk.Entry(edge_frame, width=10)
        self.edge_mingap_entry.grid(row=2, column=1, sticky="w", padx=4)
        self.edge_mingap_entry.insert(0, "50")  # debounce gap

        btns = tk.Frame(sidebar_frame)
        btns.pack(fill="x", pady=4)
        tk.Button(btns, text="Extract Edges", command=self.extract_edges).pack(side="left", padx=3)
        tk.Button(btns, text="Clear Indices", command=self.clear_indices).pack(side="left", padx=3)

        self.canvas.mpl_connect("button_press_event", self.on_click)

    def load_file(self):
        """
        Load EMG data from a file and initialize the GUI with the first channel.
        """
        path = filedialog.askopenfilename(filetypes=[
            ("NPZ Files", "*.npz"),
            ("CSV Files", "*.csv"),
            ("All files", "*.*")])
        if not path:
            return

        if path.endswith('.csv'):
            # Simple CSV loader
            try:
                data = np.loadtxt(path, delimiter=',') #, skiprows=1
                # Assume first col is time or simple array
                if data.ndim == 2:
                    # check if cols look like time
                    # simple heuristic: if col 0 is increasing monotonically
                    if np.all(np.diff(data[:,0]) > 0):
                        self.time_vector = data[:, 0]
                        self.emg_data = data[:, 1:].T
                    else:
                        # assume no time vector, make one
                        self.emg_data = data.T
                        self.time_vector = np.arange(data.shape[0]) / 2000.0
                else:
                    self.emg_data = data.reshape(1, -1)
                    self.time_vector = np.arange(data.shape[0]) / 2000.0
                
                # estimate FS
                if len(self.time_vector) > 1:
                     # ms or s?
                    dt = self.time_vector[1] - self.time_vector[0]
                    if dt > 0:
                        fs = 1.0 / dt
                        # if fs is tiny (e.g. 0.001), maybe time is in ms
                        if fs < 10 and dt > 1: # time likely in ms
                            self.time_vector = self.time_vector / 1000.0
                            fs = 1000.0 / dt
                        self.sampling_rate = fs
                    else:
                        self.sampling_rate = 2000.0
                else:
                    self.sampling_rate = 2000.0

            except Exception as e:
                messagebox.showerror("Error", f"Failed to load CSV: {e}")
                return

        elif path.endswith('.npz') or path.endswith('.npy'):
            # Load NPZ file. Should have 'emg_data' or 'emg', 'time_vector' or 't', and 'sampling_rate' or 'fs' keys
            try:
                data = np.load(path)
                if isinstance(data, np.ndarray):
                    # just array
                    self.emg_data = data
                    self.sampling_rate = 2000.0
                    self.time_vector = np.arange(self.emg_data.shape[1] if self.emg_data.ndim > 1 else self.emg_data.shape[0]) / self.sampling_rate
                else:
                    # NpzFile
                    # try standard keys
                    if 'emg' in data: self.emg_data = data['emg']
                    elif 'data' in data: self.emg_data = data['data']
                    elif 'emg_data' in data: self.emg_data = data['emg_data']
                    else:
                        # use first array
                        self.emg_data = data[data.files[0]]
                    
                    if 'fs' in data: self.sampling_rate = float(data['fs'])
                    elif 'sampling_rate' in data: self.sampling_rate = float(data['sampling_rate'])
                    else: self.sampling_rate = 2000.0
                    
                    if 't' in data: self.time_vector = data['t']
                    elif 'time_vector' in data: self.time_vector = data['time_vector']
                    else:
                        # construct from fs
                        ns = self.emg_data.shape[1] if self.emg_data.ndim > 1 else self.emg_data.shape[0]
                        self.time_vector = np.arange(ns) / self.sampling_rate
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load NPZ: {e}")
                return
        else:
             messagebox.showerror("Error", "Unsupported file type.")
             return

        # Ensure correct shape (channels x samples)
        if self.emg_data.ndim == 1:
            self.emg_data = self.emg_data.reshape(1, -1)
        elif self.emg_data.shape[0] > self.emg_data.shape[1]:
            # assume samples x channels -> transpose
             self.emg_data = self.emg_data.T

        self.channel_selector['values'] = [f"Channel {i}" for i in range(self.emg_data.shape[0])]
        self.channel_selector.current(0)
        self.current_channel = 0
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
            defaultextension=".txt",
            filetypes=[("Text Files", "*.txt")],
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

    def _detect_edges(self, x: np.ndarray, threshold: float, edge_type: str, min_gap_samples: int) -> np.ndarray:
        """
        Return sample indices where the signal crosses the threshold with a specified edge type.

        x: 1D array (samples)
        threshold: float
        edge_type: 'rising' or 'falling'
        min_gap_samples: minimum spacing between returned indices
        """
        if edge_type == "rising":
            crossings = np.where((x[:-1] < threshold) & (x[1:] >= threshold))[0] + 1
        elif edge_type == "falling":
            crossings = np.where((x[:-1] > threshold) & (x[1:] <= threshold))[0] + 1
        else:
            crossings = np.array([], dtype=int)

        # Debounce / min gap
        if crossings.size == 0 or min_gap_samples <= 1:
            return crossings

        keep = [crossings[0]]
        last = crossings[0]
        for ix in crossings[1:]:
            if ix - last >= min_gap_samples:
                keep.append(ix)
                last = ix
        return np.asarray(keep, dtype=int)

    def _add_marker_line(self, t_val: float, y_val: float | None = None):
        """Draw and remember marker lines so we can clear them later."""
        v = self.ax.axvline(x=t_val, color='blue', linestyle='--', alpha=0.8)
        self._marker_artists.append(v)
        if y_val is not None:
            h = self.ax.axhline(y=y_val, color='red', linestyle='--', alpha=0.4)
            self._marker_artists.append(h)

    def extract_edges(self):
        if self.emg_data is None or self.time_vector is None or self.sampling_rate is None:
            return

        try:
            thr = float(self.edge_threshold_entry.get())
        except Exception:
            messagebox.showerror("Error", "Threshold must be a number.")
            return

        # Use the combobox value for detection type:
        edge_kind = self.edge_type_combo.get()  # "rising" or "falling"

        # Use the text entry as the saved label:
        label = self.label_entry.get().strip() or edge_kind

        try:
            min_gap_ms = float(self.edge_mingap_entry.get())
        except Exception:
            messagebox.showerror("Error", "Min gap must be a number (ms).")
            return

        min_gap_samples = max(1, int((min_gap_ms / 1000.0) * float(self.sampling_rate)))

        chan = self.current_channel
        x = self.emg_data[chan].astype(float, copy=False)

        idxs = self._detect_edges(x, thr, edge_kind, min_gap_samples)
        if idxs.size == 0:
            messagebox.showinfo("Edges", "No edges found with current parameters.")
            return

        added = 0
        for idx in idxs:
            t = idx / float(self.sampling_rate)
            self.table.insert("", "end", values=(int(idx), label))
            self._add_marker_line(t_val=t, y_val=None)
            added += 1

        self.canvas.draw()
        messagebox.showinfo("Edges", f"Added {added} edge(s) labeled '{label}'.")

    def clear_indices(self):
        """Clear all rows in the table and remove drawn marker lines."""
        # Clear table
        for item in self.table.get_children():
            self.table.delete(item)
        # Remove marker artists
        try:
            for art in self._marker_artists:
                art.remove()
        except Exception:
            pass
        self._marker_artists.clear()
        self.canvas.draw()


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
        #self.ax.axvline(x=time_clicked, color='blue', linestyle='--')
        #self.ax.axhline(y=amp_clicked, color='red', linestyle='--')
        self._add_marker_line(t_val=time_clicked, y_val=amp_clicked)
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
        self.ax.plot(self.time_vector, self.emg_data[self.current_channel], label=f"Channel {self.current_channel}")
        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("Amplitude (µV)")
        self.ax.set_title("EMG Signal")
        self.ax.legend()
        self.canvas.draw()

    def on_closing(self):
        self.root.quit()

if __name__ == "__main__":
    root = tk.Tk()
    app = EMGTrialSelector(root)
    root.mainloop()

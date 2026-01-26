import numpy as np
from typing import Callable, List, Union, Dict, Optional
from scipy.signal import butter, sosfilt, sosfilt_zi
from ._realtime_filter import RealtimeFilter

def rectify(x: np.ndarray) -> np.ndarray:
    return np.abs(x)

def extract_features(signal, feature_fns=None):
    """
    Compute features for a window (C, N).
    Default: RMS.
    """
    if feature_fns is None or len(feature_fns) == 0:
        feature_fns = ["rms"]
        
    features_list = []
    
    # signal shape: (n_channels, n_samples)
    for fn in feature_fns:
        if isinstance(fn, str):
            if fn.lower() == "rms":
                # Root Mean Square
                val = np.sqrt(np.mean(signal**2, axis=1))
                features_list.append(val)
            elif fn.lower() == "mav":
                # Mean Absolute Value
                val = np.mean(np.abs(signal), axis=1)
                features_list.append(val)
            elif fn.lower() == "var":
                # Variance
                val = np.var(signal, axis=1)
                features_list.append(val)
            elif fn.lower() == "wl":
                # Waveform Length
                val = np.sum(np.abs(np.diff(signal, axis=1)), axis=1)
                features_list.append(val)
            elif fn.lower() == "zc":
                # Zero Crossings (simple threshold 0)
                # sign change
                zc = np.diff(np.signbit(signal), axis=1)
                val = np.sum(zc, axis=1)
                features_list.append(val.astype(np.float32))
        elif callable(fn):
            val = fn(signal)
            features_list.append(val)
            
    if not features_list:
        return np.array([])
        
    # Concatenate features per channel: (n_channels * n_feats,)
    # But usually we return (n_channels * n_feats) flat?
    # Or (n_channels, n_feats)?
    # The intan implementation returns flat (C*F).
    
    # Stack: (n_feats, n_channels) -> flatten -> (n_channels * n_feats)
    # Organization: CH1_F1, CH1_F2, ... CH2_F1 ... ?
    # Or CH1_F1, CH2_F1 ... ?
    # Intan spec: channels="training_order", layout="channel_major" usually means [CH1_F1...Fn, CH2...]
    
    stacked = np.stack(features_list, axis=1) # (n_channels, n_feats)
    return stacked.flatten()


class EMGPreprocessor:
    """
    Streaming-safe EMG pipeline:
      1) Band-pass + Notch (via RealtimeFilter)
      2) Rectify (|x|)
      3) Low-pass envelope (optional)
    """

    def __init__(self, fs: float = 2000.0, band: tuple[float, float] = (20.0, 498.0),
                 notch_freqs: tuple[float, ...] = (60.0,), notch_q: float = 30.0, 
                 envelope_cutoff: float | None = None,
                 envelope_order: int = 4, feature_fns: List[Union[str, Callable]] | None = None, 
                 verbose: bool = False):
        self.fs = float(fs)
        self.band = (float(band[0]), float(band[1]))
        self.notch_freqs = tuple(float(f) for f in notch_freqs)
        self.notch_q = float(notch_q)
        self.verbose = bool(verbose)
        
        self.feature_fns = feature_fns

        # front-end filter
        self.frontend: RealtimeFilter | None = None
        self._C = None

        # envelope LP stage
        self.env_cut = envelope_cutoff
        self.env_order = int(envelope_order)
        self._env_sos = None
        self._env_zi = None

    def _ensure_initialized(self, C: int):
        if self._C == C and self.frontend is not None:
            return
        self._C = int(C)

        self.frontend = RealtimeFilter(
            fs=self.fs,
            n_channels=self._C,
            bp_low=self.band[0],
            bp_high=self.band[1],
            notch_freqs=self.notch_freqs,
            notch_q=self.notch_q,
            enable_lowpass=False
        )

        if self.env_cut is not None:
            self._env_sos = butter(self.env_order, self.env_cut, btype="low", fs=self.fs, output="sos")
            base_zi = sosfilt_zi(self._env_sos)[:, None, :]
            self._env_zi = np.tile(base_zi, (1, self._C, 1)).astype(np.float32)
        else:
            self._env_sos = None
            self._env_zi = None

    def reset_states(self):
        if self.frontend:
            self.frontend.reset()
        if self._env_zi is not None:
             base_zi = sosfilt_zi(self._env_sos)[:, None, :]
             self._env_zi[:] = np.tile(base_zi, (1, self._C, 1))

    def preprocess(self, emg: np.ndarray, *, rectify: bool = True, envelope_cutoff: float | None = None) -> np.ndarray:
        """Process chunk (C, N). Returns (C, N)."""
        emg = np.asarray(emg, dtype=np.float32)
        C = emg.shape[0]
        self._ensure_initialized(C)

        # 1) Front-end
        y = self.frontend.process(emg)

        # 2) Rectify
        if rectify:
            y = np.abs(y)

        # 3) Envelope
        cut = self.env_cut if envelope_cutoff is None else envelope_cutoff
        if cut is not None:
             # Just use current state logic for simplicity (rebuild if cut changes is complex to handle statefully perfectly in lightweight port)
             # If cut matches init, use state
             if self._env_sos is not None and cut == self.env_cut:
                 for ch in range(C):
                     y[ch, :], self._env_zi[:, ch, :] = sosfilt(self._env_sos, y[ch, :], zi=self._env_zi[:, ch, :])
             else:
                 # Stateless fallback or re-init (omitted for brevity, assume const cut)
                 sos = butter(self.env_order, cut, btype="low", fs=self.fs, output="sos")
                 y = sosfilt(sos, y, axis=1) # stateless over segment
                 
        return y

    def extract_emg_features(self, emg: np.ndarray, window_ms: int = 200, step_ms: int = 50, feature_fns=None,
                             return_windows: bool = False, *, progress: bool = False, tqdm_kwargs: dict | None = None):
        """
        Sliding-window extraction.
        Returns: X (n_windows, n_features)
        """
        fs = self.fs
        w = int(window_ms * fs / 1000)
        s = int(step_ms * fs / 1000)
        n_samps = emg.shape[1]
        
        feature_fns = feature_fns or self.feature_fns

        if n_samps < w:
            return (np.zeros((0,)), np.array([], dtype=int)) if return_windows else np.zeros((0,))

        starts = np.arange(0, n_samps - w + 1, s, dtype=int)
        n_win = starts.size
        
        # Extract first to determine size
        f0 = extract_features(emg[:, starts[0]:starts[0]+w], feature_fns)
        X = np.empty((n_win, f0.size), dtype=np.float32)
        X[0] = f0
        
        iterator = range(1, n_win)
        if progress:
             try:
                 from tqdm import tqdm
                 iterator = tqdm(iterator, total=n_win-1, desc="Features", leave=False)
             except ImportError:
                 pass

        for idx in iterator:
            i = starts[idx]
            X[idx] = extract_features(emg[:, i:i+w], feature_fns)
            
        if return_windows:
            return X, starts
        return X

# Legacy support
def normalize_emg(data):
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    return (data - mean) / (std + 1e-6)

def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    from scipy.signal import butter, filtfilt
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data, axis=0)
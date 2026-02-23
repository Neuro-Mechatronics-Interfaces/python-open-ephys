"""
Signal processing, filtering, and feature extraction.
"""
from ._metrics_utils import load_metrics_data, get_metrics_file
from ._filters import (
    bandpass_filter,
    notch_filter,
    lowpass_filter,
    preprocess_emg,
)

from ._preprocessing import EMGPreprocessor

from ._features import (
    FEATURE_REGISTRY,
    mean_absolute_value,
    root_mean_square,
    zero_crossings,
    slope_sign_changes,
    waveform_length,
    extract_features,
    window_rms,
    rectify,
    common_average_reference,
    envelope_extraction,
    z_score_norm,
    compute_rolling_rms,
    downsample,
    extract_features_sliding_window,
    feature_spec_from_registry,
    compute_rms,
    calculate_rms,
    window_rms_1D,
    compute_grid_average,
    orthogonalize,
    normalize,
)

from ._transformations import (
    estimate_lag_coarse_to_fine,
    normxcorr_offset,
    align_by_lag,
    align_multichannel_by_lag,
)

from ._fill import FillStats, fix_missing_emg
from ._realtime_filter import RealtimeFilter
from ._channel_qc import ChannelQC, QCParams
from ._sync import (
    compute_landmark_movement_signal,
    compute_emg_envelope_signal,
    find_sync_offset,
    load_sync_offset,
    save_sync_offset
)
from ._temporal_filters import (
    moving_average,
    lowpass_filter as temporal_lowpass,
    savitzky_golay,
    gaussian_smooth,
    exponential_smooth,
    adaptive_smooth,
    smooth_predictions
)
from ._imu_features import aggregate_imu_features, append_imu_features
from ._data_processing import print_progress
from ._spatial import MontageMode, SpatialReference
from ._zca import ZcaParams, ZcaHandler

from ._filters import (
    notch_filter,
    bandpass_filter,
    lowpass_filter,
    #filter_emg,
    #RealtimeEMGFilter,
)
from ._realtime_filter import RealtimeFilter
from ._features import (
    rectify,
    window_rms,
    window_rms_1D,
    compute_rms,
    compute_rolling_rms,
    downsample,
    common_average_reference,
    compute_grid_average,
    z_score_norm,
    #apply_pca,
    orthogonalize,
    normalize,
    extract_features,
    extract_features_sliding_window,
)
from ._preprocessing import EMGPreprocessor
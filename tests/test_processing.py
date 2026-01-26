import pytest
import numpy as np
from pyoephys.processing import (
    bandpass_filter, 
    notch_filter, 
    lowpass_filter, 
    compute_rms, 
    calculate_rms,
    root_mean_square,
    common_average_reference
)

def test_notch_filter():
    fs = 1000
    t = np.arange(fs) / fs
    # Signal with 60 Hz noise
    f0 = 60
    sig = np.sin(2 * np.pi * 10 * t) + np.sin(2 * np.pi * f0 * t)
    sig = sig.reshape(1, -1)
    
    filtered = notch_filter(sig, fs=fs, f0=f0)
    
    # Check that 60Hz is attenuated (simple check)
    orig_power = np.sum(sig**2)
    filt_power = np.sum(filtered**2)
    assert filt_power < orig_power

def test_bandpass_filter():
    fs = 1000
    t = np.arange(fs) / fs
    # Signal with 5Hz and 100Hz components
    sig = np.sin(2 * np.pi * 5 * t) + np.sin(2 * np.pi * 100 * t) + np.sin(2 * np.pi * 400 * t)
    sig = sig.reshape(1, -1)
    
    # Bandpass 20-200Hz
    filtered = bandpass_filter(sig, lowcut=20, highcut=200, fs=fs)
    
    assert filtered.shape == sig.shape
    assert not np.array_equal(filtered, sig)

def test_rms_computations():
    # Constant signal
    data = np.ones((1, 100)) * 2
    
    # root_mean_square (flat result)
    val = root_mean_square(data[0])
    assert pytest.approx(val) == 2.0
    
    # calculate_rms (non-overlapping windows)
    rms_windowed = calculate_rms(data, window_size=10)
    assert rms_windowed.shape == (1, 10)
    assert np.all(pytest.approx(rms_windowed) == 2.0)
    
    # compute_rms (whole array)
    val2 = compute_rms(data[0])
    assert pytest.approx(val2) == 2.0

def test_common_average_reference():
    data = np.random.rand(8, 100)
    # Add common noise
    data += 10.0
    
    car_data = common_average_reference(data)
    
    assert car_data.shape == data.shape
    # Means should be close to zero after CAR
    assert np.all(np.abs(np.mean(car_data, axis=0)) < 1e-10)

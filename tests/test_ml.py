import pytest
import torch
import numpy as np
from pyoephys.ml import EMGClassifierCNNLSTM

def test_cnn_lstm_shapes():
    # Config
    batch_size = 4
    channels = 8
    input_window = 200
    n_classes = 4
    
    model = EMGClassifierCNNLSTM(
        num_classes=n_classes,
        num_channels=channels,
        input_window=input_window,
        dropout=0.0 # deterministic
    )
    
    # Fake Input: (Batch, 1, Channels, Time)
    x = torch.randn(batch_size, 1, channels, input_window)
    
    # Forward pass
    y_pred = model(x)
    
    assert y_pred.shape == (batch_size, n_classes), \
        f"Expected output shape {(batch_size, n_classes)}, got {y_pred.shape}"

def test_cnn_lstm_3d_input():
    # Test auto-unsqueeze for 3D input (Batch, Channels, Time)
    batch_size = 4
    channels = 8
    input_window = 200
    n_classes = 4
    
    model = EMGClassifierCNNLSTM(num_classes=n_classes, num_channels=channels, input_window=input_window)
    x = torch.randn(batch_size, channels, input_window)
    
    y_pred = model(x)
    assert y_pred.shape == (batch_size, n_classes)

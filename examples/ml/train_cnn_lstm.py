"""
Train CNN-LSTM Classifier Example
---------------------------------
Demonstrates how to:
1. Generate synthetic EMG data.
2. Format it for the CNN-LSTM model.
3. Train the model and evaluate performance.
"""

import numpy as np
import torch
from pyoephys.ml import EMGClassifierCNNLSTM

def generate_dummy_data(n_samples=1000, n_channels=8, n_classes=4, input_window=200):
    """
    Generates random EMG-like data for demonstration.
    X shape: (n_windows, 1, n_channels, window_size)
    y shape: (n_windows,)
    """
    X = np.random.randn(n_samples, 1, n_channels, input_window).astype(np.float32)
    y = np.random.randint(0, n_classes, size=n_samples)
    return X, y

def main():
    print("Initializing CNN-LSTM Classifier...")
    # Initialize model
    # adjust input_window to match your data segment size
    model = EMGClassifierCNNLSTM(
        num_classes=4,
        num_channels=8,
        input_window=200,
        filters=[32, 64],
        lstm_units=64,
        learning_rate=0.001
    )
    
    print("Generating dummy data...")
    X_train, y_train = generate_dummy_data(n_samples=500)
    X_test, y_test = generate_dummy_data(n_samples=100)
    
    print(f"Training data shape: {X_train.shape}")
    
    print("Starting training...")
    # Train
    model.fit(
        X_train, y_train,
        epochs=5,
        batch_size=32,
        validation_data=(X_test, y_test),
        verbose=True
    )
    
    print("\nEvaluating...")
    # Predict
    preds = model.predict(X_test)
    accuracy = np.mean(preds == y_test)
    print(f"Test Accuracy (Random Data): {accuracy:.2%}")
    
    # Save/Load
    print("\nSaving model...")
    model.save("cnn_lstm_model.pth")
    print("Model saved to cnn_lstm_model.pth")

if __name__ == "__main__":
    main()

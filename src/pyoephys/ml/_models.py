# pyoephys.ml._models.py

import torch
import torch.nn as nn
import numpy as np

class EMGRegressor(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(EMGRegressor, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(128, output_dim)
        )
        self.input_dim = input_dim
        self.output_dim = output_dim

    def forward(self, x):
        return self.model(x)


class EMGClassifier(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(EMGClassifier, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(128, output_dim)
        )
        self.input_dim = input_dim
        self.output_dim = output_dim

    def forward(self, x):
        return self.model(x)


class EMGClassifierCNNLSTM(nn.Module):
    """
    Hybrid CNN-LSTM model for spatio-temporal EMG classification.
    
    Args:
        num_classes: Number of output classes
        num_channels: Number of EMG channels (spatial dim)
        input_window: dim of time window
        filters: List of channel sizes for CNN layers
        lstm_units: Hidden size for LSTM
        dropout: Dropout rate
    """
    def __init__(self, num_classes, num_channels, input_window, filters=[32, 64], lstm_units=64, dropout=0.5, learning_rate=0.001):
        super(EMGClassifierCNNLSTM, self).__init__()
        self.num_classes = num_classes
        self.num_channels = num_channels
        self.input_window = input_window
        
        # 1. Feature Extraction (CNN)
        # Input: (Batch, 1, Channels, Time) -> (Batch, C, H, W)
        # Treating Channels as Height, Time as Width is one way, 
        # OR 1D Conv over time for each channel is another.
        # Let's do common Conv2D approach on spectrogram-like (Channels x Time)
        
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(1, filters[0], kernel_size=(1, 5), padding=(0, 2)), 
            nn.BatchNorm2d(filters[0]),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2)), # Reduce time dim by 2
            nn.Dropout(dropout),
            
            # Block 2
            nn.Conv2d(filters[0], filters[1], kernel_size=(1, 3), padding=(0, 1)),
            nn.BatchNorm2d(filters[1]),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2)), # Reduce time dim by 2 again
            nn.Dropout(dropout)
        )
        
        # Calculate Flattened Size for LSTM entry or Linear
        # Time reduced by 4 (2x MaxPool)
        t_final = input_window // 4
        # Channels (H) preserved as 1D spatial? No, Conv2d (1, K) preserves H if padding ok or kernel 1 there. 
        # Above kernel is (1, W), so H (num_channels) is preserved.
        
        # For LSTM, we want a Sequence. 
        # Shape out of CNN: (Batch, Filters, Channels, Time')
        # We can permute to (Batch, Time', Filters * Channels) to feed LSTM
        self.lstm_input_size = filters[1] * num_channels
        
        self.lstm = nn.LSTM(
            input_size=self.lstm_input_size,
            hidden_size=lstm_units,
            num_layers=1,
            batch_first=True,
            bidirectional=False # Simpler for now
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(lstm_units, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )
        
        self.lr = learning_rate
        
        # Helper for fit/predict (like sklearn)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, x):
        # x: (Batch, 1, Channels, Time) or (Batch, Channels, Time)
        if x.dim() == 3:
            x = x.unsqueeze(1)
            
        # CNN
        x = self.features(x) 
        # (Batch, Filters, Channels, Time')
        
        # Reshape for LSTM: (Batch, Time', Features)
        b, f, c, t = x.size()
        x = x.permute(0, 3, 1, 2) # (Batch, Time', Filters, Channels)
        x = x.reshape(b, t, f * c) # Flatten spatial/depth at each time step
        
        # LSTM
        lstm_out, _ = self.lstm(x)
        # Take last time step
        x = lstm_out[:, -1, :] 
        
        # Classifier
        x = self.classifier(x)
        return x

    def fit(self, X, y, epochs=10, batch_size=32, validation_data=None, verbose=True):
        """
        Simple sklearn-like fit method.
        X: numpy array (samples, 1, channels, time)
        y: numpy array (samples,) labels
        """
        self.train()
        dataset = torch.utils.data.TensorDataset(
            torch.tensor(X, dtype=torch.float32), 
            torch.tensor(y, dtype=torch.long)
        )
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        for epoch in range(epochs):
            total_loss = 0
            for batch_X, batch_y in loader:
                self.optimizer.zero_grad()
                outputs = self(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            
            if verbose:
                val_acc = ""
                if validation_data:
                    val_X, val_y = validation_data
                    val_preds = self.predict(val_X)
                    acc = np.mean(val_preds == val_y)
                    val_acc = f" | Val Acc: {acc:.2%}"
                print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(loader):.4f}{val_acc}")

    def predict(self, X):
        """Returns class indices."""
        self.eval()
        with torch.no_grad():
             X_tensor = torch.tensor(X, dtype=torch.float32)
             logits = self(X_tensor)
             return torch.argmax(logits, dim=1).numpy()
    
    def save(self, path):
        torch.save(self.state_dict(), path)
        
    def load(self, path):
        self.load_state_dict(torch.load(path))
        self.eval()


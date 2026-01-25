# pyoephys.ml._model_manager.py
import os
import pickle
import random
import json
import torch
import joblib
import logging
import numpy as np
from collections import Counter
from pyoephys.io import load_config_file
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


def setup_logger():
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')


setup_logger()


def _is_classification(y):
    return y.dtype.kind in {'i', 'u', 'O', 'S', 'U'} and len(np.unique(y)) < 1000


class ModelManager:
    def __init__(self, root_dir=None, label=None, model_cls=None, input_dim=None, output_dim=None, config=None, seed=42,
                 verbose=False):
        self.root_dir = root_dir
        self.label = label
        self.model_cls = model_cls
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.eval_metrics = {}
        #self.metadata_file = metadata_file
        self.verbose = verbose

        #if metadata_file is not None:
        #    self._load_metadata_file(metadata_file)
        #    return
        if config and isinstance(config, str):
            config = load_config_file(config)
        self.config = config or {}

        # Logging
        level = logging.DEBUG if self.config.get('verbose', False) else logging.INFO
        logging.basicConfig(format='[%(levelname)s] %(message)s', level=level)
        self.logger = logging.getLogger(self.__class__.__name__)

        # Seed for reproducibility
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        # Directories
        self.root_dir = root_dir
        self.model_dir = os.path.join(root_dir, 'model')
        os.makedirs(self.model_dir, exist_ok=True)

        # Paths
        prefix = f"{self.label}_" if label else ""
        self.scaler_path = os.path.join(self.model_dir, f"{prefix}scaler.pkl")
        self.model_path = os.path.join(self.model_dir, f"{prefix}model.pth")
        self.pca_path = os.path.join(self.model_dir, f"{prefix}pca.pkl") if self.config.get('use_pca', False) else None
        self.encoder_path = os.path.join(self.model_dir, f"{prefix}label_encoder.pkl")
        self.metadata_path = os.path.join(self.model_dir, f"{prefix}metadata.json")
        self.metrics_path = os.path.join(self.model_dir, f"{prefix}metrics.json")

        # Model initialization
        self.model_cls = model_cls
        self.model = None
        self.pca = None
        self.encoder = None
        self.scaler = None
        self.eval_metrics = {}

    # def _load_metadata_file(self, metadata_file):
    #     # --- If using metadata file ---
    #     if not os.path.exists(metadata_file):
    #         raise FileNotFoundError(f"Metadata file not found: {metadata_file}")
    #     with open(metadata_file, 'r') as f:
    #         metadata = json.load(f)
    #
    #     self.input_dim = metadata['input_dim']
    #     self.output_dim = metadata['output_dim']
    #     self.label = metadata.get('label', None)
    #     self.root_dir = os.path.dirname(metadata_file)
    #     self.model_dir = self.root_dir
    #     self.model_path = metadata['model_path']
    #     self.scaler_path = metadata['scaler_path']
    #     self.encoder_path = metadata.get('encoder_path')
    #     self.metrics_path = metadata.get('metrics_path', '')
    #     model_class_name = metadata['model_type']
    #
    #     # Dynamically instantiate model
    #     if model_class_name == "EMGClassifier":
    #         from pyoephys.ml import EMGClassifier
    #         self.model = EMGClassifier(input_dim=self.input_dim, output_dim=self.output_dim)
    #     elif model_class_name == "EMGRegressor":
    #         from pyoephys.ml import EMGRegressor
    #         self.model = EMGRegressor(input_dim=self.input_dim, output_dim=self.output_dim)
    #     else:
    #         raise ValueError(f"Unknown model type: {model_class_name}")
    #
    #     # Load scaler
    #     if os.path.exists(self.scaler_path):
    #         with open(self.scaler_path, 'rb') as f:
    #             self.scalar = pickle.load(f)
    #
    #     # Load label encoder if classification
    #     if self.encoder_path and os.path.exists(self.encoder_path):
    #         with open(self.encoder_path, 'rb') as f:
    #             self.label_encoder = joblib.load(f)
    #
    #     # Load weights
    #     self.load_weights()
    #
    #     if self.verbose:
    #         print(f"[INFO] Loaded model from metadata: {metadata_file}")
    #         print(f" - Model type: {model_class_name}")
    #         print(f" - Input dim: {self.input_dim}, Output dim: {self.output_dim}")
    #
    #     self.model_exists = True
    #     return  # Stop here if metadata was provided

    # def decode_labels(self, y_encoded):
    #     if self.label_encoder is not None:
    #         return self.label_encoder.inverse_transform(y_encoded)
    #     return y_encoded

    def _build_dataset(self, X, y, validation_data=None):
        # Optional class-imbalance warning
        if self.config.get('task', 'classification') == 'classification':
            counts = Counter(y)
            min_pct = min(counts.values()) / len(y)
            if min_pct < 0.05:
                self.logger.warning(f"Detected class imbalance (<5%): {counts}")

        # Scaling
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        joblib.dump(self.scaler, self.scaler_path)
        self.logger.info(f"Scaler saved to {self.scaler_path}")

        # PCA
        if self.config.get('use_pca', False):
            self.pca = PCA(n_components=self.config.get('pca_variance', 0.95))
            X_scaled = self.pca.fit_transform(X_scaled)
            with open(self.pca_path, 'wb') as f:
                pickle.dump(self.pca, f)
            self.logger.info(f"PCA saved to {self.pca_path}")

        # Label encoding
        is_class = self.config.get('task', 'classification') == 'classification'
        if is_class:
            self.encoder = LabelEncoder()
            y_enc = self.encoder.fit_transform(y)
            joblib.dump(self.encoder, self.encoder_path)
            self.logger.info(f"Label encoder saved to {self.encoder_path}")
        else:
            y_enc = np.array(y, dtype=np.float32)

        # Train/val split
        if validation_data:
            X_train, y_train = X_scaled, y_enc
            X_val, y_val = validation_data
            X_val = self.scaler.transform(X_val)
            if self.pca: X_val = self.pca.transform(X_val)
            if is_class: y_val = self.encoder.transform(y_val)
        else:
            test_size = self.config.get('test_size', 0.2)
            X_train, X_val, y_train, y_val = train_test_split(
                X_scaled, y_enc, test_size=test_size,
                stratify=y_enc if is_class else None,
                random_state=self.config.get('seed', 42)
            )
            self.logger.info(f"Train/Val split: {len(y_train)} / {len(y_val)} samples")

        return X_train, y_train, X_val, y_val

    def train(self, X, y, num_epochs=3000, stop_patience=5, learning_rate=1e-3, val_interval=20, validation_data=None, save_metrics=True):

        if y.ndim == 2 and y.shape[1] == 1:
            y = y.ravel()

        X_train, y_train, X_val, y_val = self._build_dataset(X, y, validation_data)
        if self.verbose:
            print(f"Training set shape: {X_train.shape}, {y_train.shape}")
            print(f"Testing set shape: {X_val.shape}, {y_val.shape}")

        # Instantiate model
        self.model = self.model_cls(
            input_dim=X_train.shape[1],
            output_dim=np.unique(y_train).shape[0] if self.encoder else y_train.shape[1]
        )

        # Training hyperarameters
        hp = self.config.get('hyperparameters', {})
        lr = hp.get('learning_rate', learning_rate)
        num_epochs = hp.get('num_epochs', num_epochs)
        stop_patience = hp.get('stop_patience', stop_patience)
        val_interval = hp.get('val_interval', val_interval)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = torch.nn.CrossEntropyLoss() if self.encoder else torch.nn.MSELoss()

        # Tensors
        X_train_t = torch.tensor(X_train, dtype=torch.float32)
        y_train_t = torch.tensor(y_train, dtype=torch.long if self.encoder else torch.float32)
        X_val_t = torch.tensor(X_val, dtype=torch.float32)
        y_val_t = torch.tensor(y_val, dtype=torch.long if self.encoder else torch.float32)


        # if _is_classification(y):
        #     self.label_encoder = LabelEncoder()
        #     y_train = self.label_encoder.fit_transform(y_train)
        #     if self.verbose:
        #         print(f"[INFO] Classes: {list(self.label_encoder.classes_)}")
        #     y_test = self.label_encoder.transform(y_test)
        #     criterion = torch.nn.CrossEntropyLoss()
        #
        #     # Save encoder to disk
        #     with open(self.encoder_path, 'wb') as f:
        #         joblib.dump(self.label_encoder, f)
        #
        # else:
        #     y_train = y_train.astype(np.float32)
        #     y_test = y_test.astype(np.float32)
        #     criterion = torch.nn.MSELoss()

        #if self.model is None:
        #    raise ValueError("Model must be set before training. Use set_model_type() to set the model class.")

        #scalar = StandardScaler()
        #X_train = scalar.fit_transform(X_train)
        #X_test = scalar.transform(X_test)

        #model = self.model
        # X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        # X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        # if _is_classification(y):
        #     y_train_tensor = torch.tensor(y_train, dtype=torch.long)
        #     y_test_tensor = torch.tensor(y_test, dtype=torch.long)
        # else:
        #     y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
        #     y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

        if self.verbose:
            print("Starting training...")

        loss_curve = []
        best_val_loss = np.inf
        epochs_no_improve = 0
        for epoch in range(num_epochs):
            self.model.train()
            optimizer.zero_grad()
            out = self.model(X_train_t)
            loss = criterion(out, y_train_t)
            loss.backward()
            optimizer.step()

            loss_curve.append(loss.item())

            if (epoch + 1) % val_interval == 0:
                self.model.eval()
                with torch.no_grad():
                    val_out = self.model(X_val_t)
                    val_loss = criterion(val_out, y_val_t).item()
                logging.info(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {loss.item():.8f} | Val Loss: {val_loss:.8f}")

                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= stop_patience:
                        logging.info("Early stopping triggered.")
                        break

        # Save model
        print(f" Training complete. Saving model to {self.model_path}")
        torch.save(self.model.state_dict(), self.model_path)
        print("Model saved")

        # with open(self.scaler_path, 'wb') as f:
        #     pickle.dump(scalar, f)
        #

        # Save metadata
        metadata = {
            "input_dim": self.model.input_dim,
            "output_dim": self.model.output_dim,
            "model_type": self.model.__class__.__name__,
            "model_path": self.model_path,
            "scaler_path": self.scaler_path,
            #"encoder_path": self.encoder_path if self.label_encoder else None,
            #"label_classes": list(self.encoder.classes_) if self.label_encoder else None,
            "metrics_path": self.metrics_path,
            "training_config": {
                "num_epochs": num_epochs,
                "stop_patience": stop_patience,
                "learning_rate": learning_rate,
                "val_interval": val_interval
            }
        }
        #metadata_path = os.path.join(self.model_dir,
        #                              f"{self.label}_metadata.json" if self.label else "model_metadata.json")
        with open(self.metadata_path, 'w') as f:
             json.dump(metadata, f, indent=2)
        self.logger.info(f"Metadata saved to {self.metadata_path}")
        #
        # if self.verbose:
        #     print(f"[INFO] Metadata saved to {metadata_path}")

        # Evaluate on validation dataset
        self.model.eval()
        with torch.no_grad():
            y_pred = self.model(X_val_t)
            if self.encoder:
                pred_labels = torch.argmax(y_pred, dim=1).numpy()
                true_labels = y_val
                self.eval_metrics = {
                    'classification_report': classification_report(true_labels, pred_labels, output_dict=True),
                }
            else:
                pred_vals = y_pred.numpy().ravel()
                #y_true_np = y_val_t.numpy()
                self.eval_metrics = {
                    'mse': float(mean_squared_error(y_val, pred_vals)),
                    'mae': float(mean_absolute_error(y_val, pred_vals)),
                    'r2': float(r2_score(y_val, pred_vals))
                }
                #    'mse': mean_squared_error(y_true_np, y_pred_np)),
                #    'mae': float(mean_absolute_error(y_true_np, y_pred_np)),
                #    'r2': float(r2_score(y_true_np, y_pred_np))
                #}

        #if save_metrics:
        with open(self.metrics_path, 'w') as f:
            json.dump(self.eval_metrics, f, indent=2)
        self.logger.info(f"Evaluation metrics saved to {self.metrics_path}")


    # def load_weights(self, model_path=None):
    #     if model_path is None and self.model_path is None:
    #         raise ValueError("Weights path must be provided or set in the ModelManager.")
    #     if model_path is None:
    #         model_path = self.model_path
    #     if not os.path.exists(model_path):
    #         raise FileNotFoundError(f"Weights file not found at {model_path}")
    #     if self.model is None:
    #         raise ValueError("Model must be set before loading weights. Use set_model_type() to set the model class.")
    #
    #     if self.verbose:
    #         print(f"Loading model weights from {model_path}")
    #
    #     state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    #     #print(f"Model state_dict keys: {list(state_dict.keys())}")
    #     self.model.load_state_dict(state_dict)
    #
    #     self.model.eval()

    # def load_scalar(self, scalar=None):
    #
    #     if scalar is not None:
    #         self.scalar = scalar
    #         if self.verbose:
    #             print("Scaler set directly.")
    #         return
    #     if not os.path.exists(self.scaler_path):
    #         raise FileNotFoundError(f"Scaler file not found: {self.scaler_path}")
    #     with open(self.scaler_path, 'rb') as f:
    #         self.scalar = pickle.load(f)
    #     if self.verbose:
    #         print(f"Scaler loaded from {self.scaler_path}")
    #         print(f"Scalar contents: {self.scalar}")
    #
    # def save_scalar(self, scalar):
    #     with open(self.scaler_path, 'wb') as f:
    #         pickle.dump(scalar, f)
    #     if self.verbose:
    #         logging.info(f"Scaler saved to {self.scaler_path}")

    # def set_model_type(self, model_class=None, input_dim=None, output_dim=None):
    #     """
    #     Set the model class and optionally input/output dimensions.
    #     If input/output dimensions are not provided, they will be taken from the model class.
    #
    #     Parameters:
    #         model_class (class): The model class to set.
    #         input_dim (int, optional): Input dimension for the model.
    #         output_dim (int, optional): Output dimension for the model.
    #
    #     """
    #     if model_class is None:
    #         if self.root_dir is None or self.label is None:
    #             raise ValueError("Model class must be provided or root_dir (and optional label) must be set.")
    #
    #         if self.root_dir and self.label:
    #             model_class = os.path.join(self.model_dir, f"{self.label}_emg_regressor.pth")
    #
    #         #raise ValueError("Model class must be provided.")
    #
    #     self.model = model_class

    # def load_model(self, model=None, weights=None, scalar=None):
    #     #if not model:
    #     #    raise ValueError("Model must be provided to load.")
    #     self.set_model_type(model)
    #     if self.verbose:
    #         logging.info(f"Model set to {self.model.__class__.__name__}")
    #
    #     if weights is not None:
    #         self.load_weights(weights)
    #     elif os.path.exists(self.model_path):
    #         self.load_weights()
    #
    #     if scalar is not None:
    #         self.load_scalar(scalar)
    #     elif os.path.exists(self.scaler_path):
    #         self.load_scalar()
    #
    #     self.model_exists = True

    # def load_model_weights(self):
    #     """
    #     Load the model and scalar from disk.
    #     If the model is not set, it will raise an error.
    #
    #     """
    #
    #     self.load_scalar()
    #     self.load_weights()
    #
    #     if self.verbose:
    #         logging.info(f"Model loaded from {self.model_path} and scalar from {self.scaler_path}")
    #     # print ut the model shape, input and output dimensions
    #     if self.verbose:
    #         logging.info(f"Model: {self.model}, Input Dim: {self.model.input_dim}, Output Dim: {self.model.output_dim}")
    #
    #     # Optionally load evaluation metrics if available
    #     if os.path.exists(self.metrics_path):
    #         with open(self.metrics_path, 'r') as f:
    #             self.eval_metrics = json.load(f)
    #
    #     #return self.model, self.scalar

    def load_model(self):
        # Load metadata
        with open(self.metadata_path, 'r') as f:
            meta = json.load(f)
        # Instantiate model
        self.model = self.model_cls(
            input_dim=meta['input_dim'],
            output_dim=meta['output_dim']
        )
        # Load weights
        self.model.load_state_dict(torch.load(self.model_path))
        self.model.eval()
        self.logger.info(f"Loaded model weights from {self.model_path}")
        # Load scaler, PCA, encoder
        self.scaler = joblib.load(self.scaler_path)
        if self.pca_path:
            with open(self.pca_path, 'rb') as f:
                self.pca = pickle.load(f)
        if os.path.exists(self.encoder_path):
            self.encoder = joblib.load(self.encoder_path)
        return self.model

    def predict(self, X):
        """
        Predict using the loaded model and scalar.
        If scalar is not yet loaded, attempt to load it from disk.

        Parameters:
            X (np.ndarray): Input data to predict.

        Returns:
            np.ndarray: Predicted values.
        """
        if self.model is None:
            #raise ValueError("Model must be loaded before prediction.")
            self.load_model()

        # Load scalar, pca, encoder if needed
        if self.scaler is None:
            self.scaler = joblib.load(self.scaler_path)
            
            #if os.path.exists(self.scaler_path):
            #    with open(self.scaler_path, 'rb') as f:
            #        self.scalar = pickle.load(f)
            #    if self.verbose:
            #        logging.info(f"Scaler loaded from {self.scaler_path}")
            #else:
            #    raise ValueError("Scaler is not loaded and could not be found.")
        X_scaled = self.scaler.transform(X)

        if self.pca:
            with open(self.pca_path, 'rb') as f:
                self.pca = pickle.load(f)
            X_scaled = self.pca.transform(X_scaled)

        # Predict
        self.model.eval()
        with torch.no_grad():
            #predictions = self.model(X_tensor).numpy()
            #X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
            #logits = self.model(X_tensor)
            out = self.model(torch.tensor(X_scaled, dtype=torch.float32))
            if self.encoder:
                idx = torch.argmax(out, axis=1).numpy()
                return self.encoder.inverse_transform(idx)
            else:
                return out.numpy().ravel()
            #    predictions = out.numpy()
            #if self.label_encoder is not None:
            #    # Classification, return string
            #    y_pred = torch.argmax(logits, axis=1).numpy()
            #    predictions = self.label_encoder.inverse_transform(y_pred)

        #return predictions

    def grid_search(self, X, y, param_grid, cv=5, scoring='accuracy'):
        # Build a sklearn Pipeline for search
        from sklearn.pipeline import Pipeline
        steps = [('scaler', StandardScaler())]
        if self.config.get('use_pca', False):
            steps.append(('pca', PCA()))
        if self.encoder:
            le = LabelEncoder()
            y = le.fit_transform(y)
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        est = RandomForestClassifier() if self.encoder else RandomForestRegressor()
        steps.append(('estimator', est))
        pipe = Pipeline(steps)

        gs = GridSearchCV(pipe, param_grid, cv=cv, scoring=scoring)
        gs.fit(X, y)
        self.logger.info(f"GridSearch best params: {gs.best_params_}")
        return gs.best_estimator_, gs.best_params_

    def cross_validate(self, X, y, k=5):
        is_class = self.config.get('task', 'classification') == 'classification'
        kf = KFold(n_splits=k, shuffle=True, random_state=self.config.get('seed', 42))
        metrics = []
        for train_idx, val_idx in kf.split(X):
            self.train(X[train_idx], y[train_idx], validation_data=(X[val_idx], y[val_idx]))
            metrics.append(self.eval_metrics)
        return metrics

    # def cross_validate(self, X, y, k=5, num_epochs=3000, early_stop_patience=5, learning_rate=1e-3, val_interval=20):
    #     """
    #     Perform k-fold cross-validation on the model.
    #
    #     Parameters:
    #         X (np.ndarray): Input features.
    #         y (np.ndarray): Target values.
    #         k (int): Number of folds for cross-validation.
    #         num_epochs (int): Number of epochs for training.
    #         early_stop_patience (int): Patience for early stopping.
    #         learning_rate (float): Learning rate for the optimizer.
    #         val_interval (int): Validation interval.
    #
    #     Returns:
    #         dict: Cross-validation metrics.
    #     """
    #
    #     kf = KFold(n_splits=k, shuffle=True, random_state=42)
    #     metrics_list = []
    #     loss_curves = []
    #     best_val_loss = float('inf')
    #     best_fold_index = -1
    #     best_model_state = None
    #     best_scalar = None
    #
    #     for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    #         print(f"\n--- Fold {fold + 1}/{k} ---")
    #         X_train, X_val = X[train_idx], X[val_idx]
    #         y_train, y_val = y[train_idx], y[val_idx]
    #
    #         scalar = StandardScaler()
    #         X_train_scaled = scalar.fit_transform(X_train)
    #         X_val_scaled = scalar.transform(X_val)
    #
    #         model = self.model.__class__(input_dim=X.shape[1],
    #                                      output_dim=np.unique(y).shape[0] if _is_classification(y) else y.shape[1])
    #         optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    #         criterion = torch.nn.MSELoss()
    #
    #         X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    #         y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    #         X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)
    #         y_val_tensor = torch.tensor(y_val, dtype=torch.float32)
    #
    #         loss_curve = []
    #         no_improve = 0
    #         best_fold_val_loss = float('inf')
    #
    #         for epoch in range(num_epochs):
    #             model.train()
    #             optimizer.zero_grad()
    #             pred = model(X_train_tensor)
    #             loss = criterion(pred, y_train_tensor)
    #             loss.backward()
    #             optimizer.step()
    #             loss_curve.append(loss.item())
    #
    #             if (epoch + 1) % val_interval == 0:
    #                 model.eval()
    #                 with torch.no_grad():
    #                     val_pred = model(X_val_tensor)
    #                     val_loss = criterion(val_pred, y_val_tensor).item()
    #                     print(f"Epoch {epoch + 1} - Train Loss: {loss.item():.6f}, Val Loss: {val_loss:.6f}")
    #
    #                 if val_loss < best_fold_val_loss:
    #                     best_fold_val_loss = val_loss
    #                     no_improve = 0
    #                 else:
    #                     no_improve += 1
    #                     if no_improve >= early_stop_patience:
    #                         print("Early stopping")
    #                         break
    #
    #         # Evaluate final model for this fold
    #         model.eval()
    #         with torch.no_grad():
    #             y_pred = model(X_val_tensor).numpy()
    #             y_true = y_val_tensor.numpy()
    #         metrics = {
    #             'fold': fold,
    #             'mse': float(mean_squared_error(y_true, y_pred)),
    #             'mae': float(mean_absolute_error(y_true, y_pred)),
    #             'r2': float(r2_score(y_true, y_pred))
    #         }
    #         metrics_list.append(metrics)
    #         loss_curves.append(loss_curve)
    #
    #         if best_fold_val_loss < best_val_loss:
    #             best_val_loss = best_fold_val_loss
    #             best_model_state = model.state_dict()
    #             best_scalar = scalar
    #             best_fold_index = fold
    #
    #     # Save best model and scalar
    #     self.model = self.model.__class__(input_dim=X.shape[1],
    #                                  output_dim=np.unique(y).shape[0] if _is_classification(y) else y.shape[1])
    #     self.model.load_state_dict(best_model_state)
    #     torch.save(self.model.state_dict(), self.model_path)
    #     with open(self.scaler_path, 'wb') as f:
    #         pickle.dump(best_scalar, f)
    #
    #     average_metrics = {
    #         'mse': float(np.mean([m['mse'] for m in metrics_list])),
    #         'mae': float(np.mean([m['mae'] for m in metrics_list])),
    #         'r2': float(np.mean([m['r2'] for m in metrics_list]))
    #     }
    #     with open(self.metrics_path.replace('.json', '_kfold.json'), 'w') as f:
    #         json.dump({'folds': metrics_list, 'average': average_metrics, 'loss_curves': loss_curves}, f, indent=2)
    #
    #     print("\nK-fold cross-validation complete. Average metrics:")
    #     print(average_metrics)
    #     print(f"\nBest model came from fold {best_fold_index + 1}")
    #     return self.model, best_scalar

    @property
    def classes(self):
        return list(self.label_encoder.classes_) if self.label_encoder else []

# pyoephys.ml._model_manager.py
import os
import time
import pickle
import random
import platform
import json
import torch
from typing import Any, Dict, Iterable, Optional, List
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


log = logging.getLogger(__name__)


def setup_logger():
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')


setup_logger()


def _jsonify(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, dict):
        return {k: _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonify(v) for v in obj]
    return obj


def _is_classification(y):
    return y.dtype.kind in {'i', 'u', 'O', 'S', 'U'} and len(np.unique(y)) < 1000


def load_training_metadata(file_path: str) -> Dict:
    """Load model metadata saved at training time.

    Accepts either:
    - a directory path (``root_dir``) — looks for ``root_dir/model/metadata.json``
    - a direct path to a ``metadata.json`` file
    """
    if file_path.endswith(".json") and os.path.isfile(file_path):
        # Caller already gave us the exact file.
        meta_path = file_path
    elif file_path.endswith(".json"):
        # Treat the directory that would contain this json as root_dir.
        meta_path = os.path.join(os.path.dirname(file_path), "model", "metadata.json")
    else:
        meta_path = os.path.join(file_path, "model", "metadata.json")
    if not os.path.isfile(meta_path):
        raise FileNotFoundError(f"Missing metadata.json at {meta_path}")
    with open(meta_path, "r") as f:
        return json.load(f)


def write_training_metadata(root_dir: str, *, window_ms: int, step_ms: int, envelope_cutoff_hz: float,
    channel_names: Optional[Iterable[str]] = None, selected_channels: Optional[Iterable[int]] = None,
    sample_rate_hz: Optional[float] = None, n_features: Optional[int] = None, feature_set: Optional[Iterable[str]] = None,
    ignore_labels: Optional[Iterable[str]] = None,
    require_complete: bool = True, required_fraction: float = 1.0, channel_wait_timeout_sec: float = 15.0) -> None:
    """
    Merge/update fields in model/metadata.json so real-time ZMQ prediction has a single source of truth.
    Non-destructive: preserves any existing keys unless overridden here.
    """
    meta_dir = os.path.join(root_dir, "model")
    os.makedirs(meta_dir, exist_ok=True)
    meta_path = os.path.join(meta_dir, "metadata.json")

    # Load existing metadata if present (read directly — path already resolved)
    if os.path.isfile(meta_path):
        with open(meta_path, "r") as _f:
            meta = json.load(_f)
    else:
        meta = {}

    # Core pipeline params (training-time truth)
    meta.update({
        "window_ms": int(window_ms),
        "step_ms": int(step_ms),
        "envelope_cutoff_hz": float(envelope_cutoff_hz),
        "require_complete": bool(require_complete),
        "required_fraction": float(required_fraction),
        "channel_wait_timeout_sec": float(channel_wait_timeout_sec),
    })

    if channel_names is not None:
        meta["channel_names"] = list(channel_names)
    if selected_channels is not None:
        meta["selected_channels"] = [int(i) for i in selected_channels]
    if sample_rate_hz is not None:
        meta["sample_rate_hz"] = float(sample_rate_hz)
    if n_features is not None:
        meta["n_features"] = int(n_features)
    if feature_set is not None:
        meta["feature_set"] = list(feature_set)
    if ignore_labels is not None:
        meta["ignore_labels"] = [str(l) for l in ignore_labels]

    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[INFO] Metadata updated at {meta_path}")



class ModelManager:
    def __init__(self, root_dir=None, label=None, model_cls=None, input_dim=None, output_dim=None, config=None, seed=42,
                 verbose=False):
        self.root_dir = root_dir
        self.label = label
        self.model_cls = model_cls
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.eval_metrics = {}
        self.verbose = verbose

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

    #@property
    #def metadata_path(self) -> str:
    #    return os.path.join(self.model_dir, "metadata.json")

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

        # Save minimal metadata so load_model() can reconstruct the architecture
        _min_meta = {
            "schema_version": "1.1.0",
            "created_by": "pyoephys.ml.ModelManager",
            "model": {"class": type(self.model).__name__},
            "input_dim": int(self.model.input_dim),
            "output_dim": int(self.model.output_dim),
        }
        os.makedirs(self.model_dir, exist_ok=True)
        with open(self.metadata_path, "w", encoding="utf-8") as _f:
            json.dump(_min_meta, _f, indent=2)
        self.logger.info(f"Minimal metadata saved to {self.metadata_path}")

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
                self.eval_metrics = {
                    'mse': float(mean_squared_error(y_val, pred_vals)),
                    'mae': float(mean_absolute_error(y_val, pred_vals)),
                    'r2': float(r2_score(y_val, pred_vals))
                }

        print("Training complete. Validation metrics:")
        print(self.eval_metrics)

        with open(self.metrics_path, 'w') as f:
            json.dump(self.eval_metrics, f, indent=2)
        self.logger.info(f"Evaluation metrics saved to {self.metrics_path}")

    def fine_tune(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        freeze_body: bool = True,
        num_epochs: int = 500,
        learning_rate: float = 5e-4,
        stop_patience: int = 10,
        val_interval: int = 10,
        validation_data=None,
        save_metrics: bool = True,
    ) -> None:
        """Retrain using a pre-loaded model as warm-start, adapting to new user data.

        This implements the "frozen-body / retrained-head" paradigm from
        MindRove-EMG's ``fine_tune_model``: the existing weights are used as
        strong initialisation so only a small amount of new data is needed.

        When *freeze_body* is ``True`` only the last linear layer's parameters
        receive gradients; all earlier layers are frozen.  This is appropriate
        when *X* is small (< a few hundred windows).  Set to ``False`` to
        fine-tune the whole network (requires more data but allows full
        adaptation).

        The model must be loaded before calling this method — either because
        :meth:`train` was called earlier in the same session, or because
        :meth:`load_model` was called explicitly.

        Parameters
        ----------
        X : ndarray, shape (N, n_features)
        y : array-like of labels
        freeze_body : bool
            When True, freeze all layers except the final linear layer.
        num_epochs : int
            Maximum training epochs for the fine-tuning pass.
        learning_rate : float
            Learning rate for Adam during fine-tuning.
        stop_patience : int
            Early-stopping patience (epochs without val improvement).
        val_interval : int
            Evaluate on validation set every *val_interval* epochs.
        validation_data : tuple (X_val, y_val) or None
            Held-out set.  When None, 20 % of *X* is held out automatically.
        save_metrics : bool
            Overwrite the ``metrics.json`` file after fine-tuning.

        Examples
        --------
        >>> manager = ModelManager(root_dir="data", model_cls=EMGClassifier)
        >>> manager.load_model()
        >>> manager.fine_tune(X_new_user, y_new_user, freeze_body=True)
        >>> manager.save_metadata(manager.build_metadata(...))
        """
        if self.model is None:
            raise RuntimeError(
                "Model is not loaded.  Call load_model() or train() first, "
                "then call fine_tune() to adapt to new data."
            )

        if y.ndim == 2 and y.shape[1] == 1:
            y = y.ravel()
        y = np.asarray(y)

        # Split manually — deliberately bypass _build_dataset() so it does NOT
        # re-fit and overwrite self.scaler / self.encoder from the new user data.
        if validation_data is not None:
            X_train, y_train = X, y
            X_val, y_val = np.asarray(validation_data[0]), np.asarray(validation_data[1])
        else:
            test_size = self.config.get("test_size", 0.2)
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=test_size,
                stratify=y,
                random_state=self.config.get("seed", 42),
            )

        # ------------------------------------------------------------------
        # Optionally freeze all layers except the last linear layer
        # ------------------------------------------------------------------
        if freeze_body:
            # Freeze all parameters first, then unfreeze the last Linear layer
            for param in self.model.parameters():
                param.requires_grad = False

            # Find the last nn.Linear and unfreeze it
            last_linear = None
            for module in self.model.modules():
                if isinstance(module, torch.nn.Linear):
                    last_linear = module
            if last_linear is not None:
                for param in last_linear.parameters():
                    param.requires_grad = True
                log.info("[fine_tune] Frozen body — adapting only last Linear layer.")
            else:
                log.warning(
                    "[fine_tune] No nn.Linear found; fine-tuning all parameters."
                )
                for param in self.model.parameters():
                    param.requires_grad = True

        # ------------------------------------------------------------------
        # Scaler: fit on new data if not yet fitted, else transform only
        # ------------------------------------------------------------------
        if self.scaler is None:
            self.scaler = StandardScaler()
            X_train = self.scaler.fit_transform(X_train)
            X_val = self.scaler.transform(X_val)
        else:
            X_train = self.scaler.transform(X_train)
            X_val = self.scaler.transform(X_val)

        # Encoder: fit on new labels while inheriting class ordering if possible
        if self.encoder is None:
            self.encoder = LabelEncoder()
            self.encoder.fit(y_train)
        else:
            # Merge existing classes and new ones (preserves old label indices)
            new_classes = np.union1d(self.encoder.classes_, np.unique(y_train))
            self.encoder.classes_ = new_classes

        y_train_enc = self.encoder.transform(y_train)
        y_val_enc = self.encoder.transform(y_val)

        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=learning_rate,
        )
        criterion = torch.nn.CrossEntropyLoss()

        X_tr_t = torch.tensor(X_train, dtype=torch.float32)
        y_tr_t = torch.tensor(y_train_enc, dtype=torch.long)
        X_va_t = torch.tensor(X_val, dtype=torch.float32)
        y_va_t = torch.tensor(y_val_enc, dtype=torch.long)

        best_val_loss = np.inf
        epochs_no_improve = 0

        for epoch in range(num_epochs):
            self.model.train()
            optimizer.zero_grad()
            out = self.model(X_tr_t)
            loss = criterion(out, y_tr_t)
            loss.backward()
            optimizer.step()

            if (epoch + 1) % val_interval == 0:
                self.model.eval()
                with torch.no_grad():
                    val_loss = criterion(self.model(X_va_t), y_va_t).item()
                logging.info(
                    f"[fine_tune] Epoch {epoch+1}/{num_epochs} | "
                    f"train={loss.item():.6f} | val={val_loss:.6f}"
                )
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= stop_patience:
                        logging.info("[fine_tune] Early stopping triggered.")
                        break

        # Re-unfreeze all parameters so subsequent calls work as expected
        for param in self.model.parameters():
            param.requires_grad = True

        # Persist adapted weights over the existing model file
        torch.save(self.model.state_dict(), self.model_path)
        joblib.dump(self.scaler, self.scaler_path)
        joblib.dump(self.encoder, self.encoder_path)

        # Evaluate
        self.model.eval()
        with torch.no_grad():
            pred_idx = torch.argmax(self.model(X_va_t), dim=1).numpy()
            pred_labels = self.encoder.inverse_transform(pred_idx)
            self.eval_metrics = {
                "classification_report": classification_report(
                    y_val, pred_labels, output_dict=True
                )
            }

        print(f"[fine_tune] Complete. Val metrics: {self.eval_metrics}")

        if save_metrics:
            os.makedirs(self.model_dir, exist_ok=True)
            with open(self.metrics_path, "w") as _f:
                json.dump(self.eval_metrics, _f, indent=2)

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

        X_scaled = self.scaler.transform(X)

        if self.pca:
            X_scaled = self.pca.transform(X_scaled)

        if X_scaled.shape[1] != self.model.input_dim:
            raise ValueError(f"Feature dim {X_scaled.shape[1]} != model.input_dim {self.model.input_dim}")

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

    def predict_proba(self, X) -> np.ndarray:
        """Return softmax class probabilities for each sample.

        Only valid for classifiers (encoder must be set).

        Parameters
        ----------
        X : np.ndarray, shape (N, n_features)

        Returns
        -------
        np.ndarray, shape (N, n_classes)
            Row-normalised softmax probabilities in class order matching
            ``self.encoder.classes_``.
        """
        if not self.encoder:
            raise RuntimeError(
                "predict_proba is only available for classifiers. "
                "This ModelManager was initialised without a label encoder."
            )

        if self.model is None:
            self.load_model()

        if self.scaler is None:
            self.scaler = joblib.load(self.scaler_path)

        X_scaled = self.scaler.transform(X)
        if self.pca:
            X_scaled = self.pca.transform(X_scaled)

        if X_scaled.shape[1] != self.model.input_dim:
            raise ValueError(
                f"Feature dim {X_scaled.shape[1]} != model.input_dim {self.model.input_dim}"
            )

        import torch.nn.functional as F  # local import to keep top-level clean

        self.model.eval()
        with torch.no_grad():
            logits = self.model(torch.tensor(X_scaled, dtype=torch.float32))
            probs = F.softmax(logits, dim=1).numpy()
        return probs
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
        """Run k-fold cross-validation and return per-fold metrics plus an aggregated summary.

        Returns
        -------
        dict with keys:
            ``folds``   – list of per-fold eval_metrics dicts
            ``summary`` – dict of metric_name → {mean, std} over all folds (scalar metrics only)
        """
        kf = KFold(n_splits=k, shuffle=True, random_state=self.config.get('seed', 42))
        fold_metrics = []
        for train_idx, val_idx in kf.split(X):
            self.train(X[train_idx], y[train_idx], validation_data=(X[val_idx], y[val_idx]))
            fold_metrics.append(self.eval_metrics)

        # For classifiers, promote accuracy out of the nested classification_report dict
        # so the summary aggregation can find it as a scalar.
        for fold in fold_metrics:
            cr = fold.get("classification_report")
            if isinstance(cr, dict) and "accuracy" in cr and "accuracy" not in fold:
                fold["accuracy"] = float(cr["accuracy"])

        # Aggregate scalar metrics (skip nested dicts like classification_report)
        summary: dict = {}
        scalar_keys = [key for key, v in fold_metrics[0].items() if isinstance(v, (int, float))]
        for key in scalar_keys:
            vals = [m[key] for m in fold_metrics if key in m]
            summary[key] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
        if self.verbose:
            print(f"[cross_validate] {k}-fold summary: {summary}")
        return {"folds": fold_metrics, "summary": summary}

    def build_metadata(self, *, sample_rate_hz: float, window_ms: int, step_ms: int, envelope_cutoff_hz: float,
            selected_channels: Optional[List[int]], channel_names: Optional[List[str]], feature_spec: Dict[str, Any],
            n_features: int, label_classes: List[str], scaler_mean: Optional[List[float]] = None,
            scaler_scale: Optional[List[float]] = None, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Build a self-describing metadata dict the prediction paths can trust.
        """
        now_iso = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())
        meta = {
            "schema_version": "1.1.0",
            "created_by": "pyoephys.ml.ModelManager",
            "system": {"platform": platform.platform()},
            "model": {
                "class": type(self.model).__name__ if getattr(self, "model",
                                                              None) is not None else self.model_cls.__name__,
                "path": os.path.relpath(self.model_path, self.root_dir),
                "label_encoder_path": os.path.relpath(self.label_encoder_path, self.root_dir) if hasattr(self,
                                                                                                         "label_encoder_path") else None,
                "scaler_path": os.path.relpath(self.scaler_path, self.root_dir) if hasattr(self,
                                                                                           "scaler_path") else None,
            },
            "input_dim": int(self.model.input_dim),
            "output_dim": int(self.model.output_dim),
            "data": {
                "sample_rate_hz": float(sample_rate_hz),
                "window_ms": int(window_ms),
                "step_ms": int(step_ms),
                "envelope_cutoff_hz": float(envelope_cutoff_hz),
                "selected_channels": selected_channels,  # indices in raw order used for training (may be None)
                "channel_names": channel_names,  # full list (raw order) if available
            },
            "features": {
                "n_features": int(n_features),
                # feature_spec describes how the feature vector was built
                # (names, per-channel vs global, order, etc.)
                "spec": feature_spec or {},
            },
            "labels": {
                "classes": list(map(str, label_classes)),
            },
            "scaler": {
                "mean": scaler_mean,
                "scale": scaler_scale,
            },
        }
        if extra:
            # attach any training-time metrics or notes
            meta["extra"] = extra
        return meta

    def save_metadata(self, meta: Dict[str, Any]) -> None:
        os.makedirs(self.model_dir, exist_ok=True)
        with open(self.metadata_path, "w", encoding="utf-8") as f:
            json.dump(_jsonify(meta), f, indent=2)
        if self.config.get("verbose"):
            print(f"[ModelManager] wrote metadata → {self.metadata_path}")

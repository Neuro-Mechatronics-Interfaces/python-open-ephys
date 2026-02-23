"""
Machine Learning models and utilities for gesture classification.
"""
from ._models import (
    EMGRegressor,
    EMGClassifier,
    EMGClassifierCNNLSTM,
)
from ._model_manager import (
    ModelManager,
    write_training_metadata,
    load_training_metadata,
)
from ._evaluation import evaluate_against_events
from ._model_router import ModelRouter, ModelRouterConfig

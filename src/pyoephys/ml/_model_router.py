"""
pyoephys.ml._model_router
==========================
Route inference to orientation-specific classification models.

Inspired by MindRove-EMG (nml/gesture_classifier/multi_model.py) and extended
to work natively with :class:`~pyoephys.ml.ModelManager`.

Use case — prosthetic / EMG control where the electrode array sits on the
forearm: muscle geometry changes with wrist rotation, so a single model
trained in "neutral" position degrades badly in pronation / supination.
:class:`ModelRouter` holds one :class:`ModelManager` per orientation and
dispatches inference to whichever model covers the current IMU reading.

Quickstart
----------
>>> configs = [
...     ModelRouterConfig(
...         name="neutral",
...         root_dir="data/neutral",
...         orientation_range=dict(roll=(-30, 30), pitch=(-30, 30)),
...     ),
...     ModelRouterConfig(
...         name="pronation",
...         root_dir="data/pronation",
...         orientation_range=dict(roll=(-90, -30), pitch=(-30, 30)),
...     ),
...     ModelRouterConfig(
...         name="supination",
...         root_dir="data/supination",
...         orientation_range=dict(roll=(30, 90), pitch=(-30, 30)),
...     ),
... ]
>>> router = ModelRouter(configs, model_cls=EMGClassifier)
>>> router.load_all()
>>> label = router.predict(features, imu_angles=dict(roll=-45.0, pitch=5.0))
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence

import numpy as np

from ._model_manager import ModelManager
from ._models import EMGClassifier

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------

@dataclass
class ModelRouterConfig:
    """Configuration for one orientation slot in a :class:`ModelRouter`.

    Parameters
    ----------
    name : str
        Human-readable label (e.g. ``"neutral"``, ``"pronation"``).
    root_dir : str
        Root directory for the :class:`ModelManager` artefacts of this slot.
    orientation_range : dict
        IMU angle bounds used to match this model.  Recognised keys are
        ``"roll"``, ``"pitch"``, and ``"yaw"`` (all optional), each mapping
        to a ``(min, max)`` tuple in degrees.  Example::

            {"roll": (-30, 30), "pitch": (-30, 30)}

    label : str
        Optional prefix for model artefact filenames (forwarded to
        :class:`ModelManager`).
    manager_kwargs : dict
        Any extra keyword arguments forwarded to :class:`ModelManager`
        (e.g. ``verbose=True``).
    """

    name: str
    root_dir: str
    orientation_range: Dict[str, tuple] = field(default_factory=dict)
    label: str = ""
    manager_kwargs: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

class ModelRouter:
    """Dispatch EMG inference to the orientation-specific model that matches
    the current IMU reading.

    Parameters
    ----------
    configs : sequence of :class:`ModelRouterConfig`
        One entry per wrist orientation / deployment context.
    model_cls : type
        PyTorch model class forwarded to every :class:`ModelManager`.
        Defaults to :class:`~pyoephys.ml.EMGClassifier`.
    default : str or None
        Name of the config to use as the fallback when no range matches.
        When ``None`` the *first* config is used.

    Examples
    --------
    Training::

        router = ModelRouter(configs, model_cls=EMGClassifier)
        for cfg in configs:
            X, y = load_data_for(cfg.name)
            router.managers[cfg.name].train(X, y)

    Inference::

        router.load_all()
        label = router.predict(features, imu_angles={"roll": -50.0})
    """

    def __init__(
        self,
        configs: Sequence[ModelRouterConfig],
        model_cls=None,
        default: str | None = None,
    ) -> None:
        if not configs:
            raise ValueError("configs must be a non-empty sequence.")

        self._model_cls = model_cls or EMGClassifier
        self.configs: Dict[str, ModelRouterConfig] = {c.name: c for c in configs}
        self.managers: Dict[str, ModelManager] = {
            c.name: ModelManager(
                root_dir=c.root_dir,
                label=c.label,
                model_cls=self._model_cls,
                **c.manager_kwargs,
            )
            for c in configs
        }
        self._default = default or configs[0].name

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load_all(self) -> None:
        """Load model weights and scalers for all orientation slots."""
        for name, mgr in self.managers.items():
            mgr.load_model()
            log.info(f"[ModelRouter] loaded model: {name!r}")

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------

    def detect_orientation(self, imu_angles: Dict[str, float]) -> str:
        """Return the config name whose ``orientation_range`` covers *imu_angles*.

        Parameters
        ----------
        imu_angles : dict
            Keys as in the config ``orientation_range`` (e.g. ``roll``,
            ``pitch``, ``yaw``), values in degrees.

        Returns
        -------
        str
            Name of the matched config, or :attr:`_default` if none matches.
        """
        for name, cfg in self.configs.items():
            if self._in_range(imu_angles, cfg.orientation_range):
                return name

        log.debug(
            f"[ModelRouter] no orientation matched for {imu_angles}; "
            f"falling back to {self._default!r}"
        )
        return self._default

    def predict(
        self,
        X: np.ndarray,
        imu_angles: Optional[Dict[str, float]] = None,
        orientation: Optional[str] = None,
    ) -> np.ndarray:
        """Predict gesture labels for feature matrix *X*.

        Parameters
        ----------
        X : ndarray, shape (N, n_features)
        imu_angles : dict, optional
            IMU readings used to auto-select the model.  Required when
            *orientation* is not given.
        orientation : str, optional
            Explicitly specify which model to use (bypasses IMU routing).

        Returns
        -------
        ndarray of predicted labels (same shape/type as
        :meth:`ModelManager.predict`).
        """
        if orientation is None:
            if imu_angles is None:
                raise ValueError(
                    "Provide either imu_angles (for automatic routing) or "
                    "orientation (to explicitly select a model)."
                )
            orientation = self.detect_orientation(imu_angles)

        if orientation not in self.managers:
            raise KeyError(
                f"No model registered for orientation {orientation!r}. "
                f"Available: {list(self.managers)}"
            )

        return self.managers[orientation].predict(X)

    def predict_proba(
        self,
        X: np.ndarray,
        imu_angles: Optional[Dict[str, float]] = None,
        orientation: Optional[str] = None,
    ) -> np.ndarray:
        """Return softmax class probabilities from the matched model.

        See :meth:`ModelManager.predict_proba` for details.
        """
        if orientation is None:
            if imu_angles is None:
                raise ValueError("Provide either imu_angles or orientation.")
            orientation = self.detect_orientation(imu_angles)

        return self.managers[orientation].predict_proba(X)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _in_range(
        angles: Dict[str, float],
        ranges: Dict[str, tuple],
    ) -> bool:
        """Return True if every angle axis in *ranges* is within its bounds."""
        for axis, (lo, hi) in ranges.items():
            val = angles.get(axis)
            if val is None:
                return False
            if not (lo <= val <= hi):
                return False
        return True

    def __repr__(self) -> str:
        names = list(self.configs)
        return f"ModelRouter(models={names}, default={self._default!r})"

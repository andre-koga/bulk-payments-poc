from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier


FEATURE_ORDER: tuple[str, ...] = (
    "payment_amount_minor",
    "subset_sum_minor",
    "abs_residual_minor",
    "subset_size",
    "competing_subset_count",
    "unique_amount_closure",
    "max_vendor_link_score",
    "min_date_delta_days",
    "desc_len",
)


@dataclass
class RankerArtifact:
    """Frozen sklearn pipeline: calibrated HGBT on tabular features."""

    model: Any
    feature_order: tuple[str, ...] = FEATURE_ORDER


def features_to_vector(features: dict[str, Any]) -> np.ndarray:
    vec = [float(features.get(k, 0.0)) for k in FEATURE_ORDER]
    # bool as 0/1
    vec[FEATURE_ORDER.index("unique_amount_closure")] = 1.0 if features.get("unique_amount_closure") else 0.0
    return np.array(vec, dtype=np.float64).reshape(1, -1)


def load_ranker(path: str | None) -> RankerArtifact | None:
    if not path:
        return None
    p = Path(path)
    if not p.is_file():
        return None
    obj = joblib.load(p)
    if isinstance(obj, RankerArtifact):
        return obj
    # allow raw sklearn model for quick tests
    return RankerArtifact(model=obj, feature_order=FEATURE_ORDER)


def predict_accept_prob(
    ranker: RankerArtifact | None,
    features: dict[str, Any],
) -> tuple[float | None, float | None]:
    if ranker is None or not features:
        return None, None
    X = features_to_vector(features)
    m = ranker.model
    if hasattr(m, "predict_proba"):
        proba = m.predict_proba(X)[0, 1]
        return float(proba), float(proba)
    if hasattr(m, "decision_function"):
        df = float(m.decision_function(X)[0])
        return None, df
    return None, None


def train_global_ranker(
    X: np.ndarray,
    y: np.ndarray,
    *,
    random_state: int = 42,
) -> RankerArtifact:
    base = HistGradientBoostingClassifier(
        max_depth=6,
        max_iter=200,
        learning_rate=0.06,
        random_state=random_state,
    )
    # sigmoid Platt-style calibration
    cal = CalibratedClassifierCV(base, method="sigmoid", cv=3)
    cal.fit(X, y)
    return RankerArtifact(model=cal, feature_order=FEATURE_ORDER)


def save_ranker(artifact: RankerArtifact, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, path)

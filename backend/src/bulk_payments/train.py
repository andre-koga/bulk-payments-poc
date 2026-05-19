from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression

from bulk_payments.ranker import FEATURE_ORDER, RankerArtifact, features_to_vector, save_ranker, train_global_ranker

# Hand-crafted rows so a fresh DB can still train a non-trivial model (global prior + noise).
_DEFAULT_SYNTHETIC: list[tuple[dict[str, Any], int]] = [
    (
        {
            "payment_amount_minor": 500_00,
            "subset_sum_minor": 500_00,
            "abs_residual_minor": 0,
            "subset_size": 3,
            "competing_subset_count": 1,
            "unique_amount_closure": True,
            "max_vendor_link_score": 0.9,
            "min_date_delta_days": 2,
            "desc_len": 4,
        },
        1,
    ),
    (
        {
            "payment_amount_minor": 500_00,
            "subset_sum_minor": 500_00,
            "abs_residual_minor": 0,
            "subset_size": 2,
            "competing_subset_count": 4,
            "unique_amount_closure": False,
            "max_vendor_link_score": 0.9,
            "min_date_delta_days": 2,
            "desc_len": 4,
        },
        0,
    ),
    (
        {
            "payment_amount_minor": 777_00,
            "subset_sum_minor": 0,
            "abs_residual_minor": 777_00,
            "subset_size": 0,
            "competing_subset_count": 0,
            "unique_amount_closure": False,
            "max_vendor_link_score": 0.1,
            "min_date_delta_days": 5,
            "desc_len": 0,
        },
        0,
    ),
    (
        {
            "payment_amount_minor": 125_00,
            "subset_sum_minor": 125_00,
            "abs_residual_minor": 0,
            "subset_size": 2,
            "competing_subset_count": 1,
            "unique_amount_closure": True,
            "max_vendor_link_score": 0.88,
            "min_date_delta_days": 1,
            "desc_len": 3,
        },
        1,
    ),
    (
        {
            "payment_amount_minor": 10_000_00,
            "subset_sum_minor": 10_000_00,
            "abs_residual_minor": 0,
            "subset_size": 20,
            "competing_subset_count": 1,
            "unique_amount_closure": True,
            "max_vendor_link_score": 0.4,
            "min_date_delta_days": 1,
            "desc_len": 2,
        },
        0,
    ),
    (
        {
            "payment_amount_minor": 10_000_00,
            "subset_sum_minor": 10_000_00,
            "abs_residual_minor": 0,
            "subset_size": 2,
            "competing_subset_count": 1,
            "unique_amount_closure": True,
            "max_vendor_link_score": 0.92,
            "min_date_delta_days": 0,
            "desc_len": 6,
        },
        1,
    ),
]


def rows_from_match_events(
    conn: sqlite3.Connection,
    *,
    tenant_id: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Load labeled rows from match_events (outcome != pending). y=1 for accepted_as_is."""
    if tenant_id:
        rows = conn.execute(
            """
            SELECT features_json, outcome FROM match_events
            WHERE tenant_id = ?
              AND outcome IN ('accepted_as_is', 'rejected', 'edited_subset', 'manual_alternative')
            """,
            (tenant_id,),
        ).fetchall()
    else:
        rows = conn.execute(
            """
            SELECT features_json, outcome FROM match_events
            WHERE outcome IN ('accepted_as_is', 'rejected', 'edited_subset', 'manual_alternative')
            """
        ).fetchall()
    if not rows:
        return np.zeros((0, len(FEATURE_ORDER))), np.zeros((0,))
    X_list: list[list[float]] = []
    y_list: list[int] = []
    for features_json, outcome in rows:
        feats: dict[str, Any] = json.loads(features_json)
        if not feats:
            continue
        y = 1 if outcome == "accepted_as_is" else 0
        v = features_to_vector(feats).ravel().tolist()
        X_list.append(v)
        y_list.append(y)
    return np.array(X_list, dtype=np.float64), np.array(y_list, dtype=np.int64)


def _merge_default_rows(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    Xd = [features_to_vector(f).ravel().tolist() for f, _ in _DEFAULT_SYNTHETIC]
    yd = [yy for _, yy in _DEFAULT_SYNTHETIC]
    if X.size == 0:
        return np.array(Xd, dtype=np.float64), np.array(yd, dtype=np.int64)
    return np.vstack([np.array(Xd, dtype=np.float64), X]), np.concatenate([np.array(yd, dtype=np.int64), y])


def train_and_save(
    conn: sqlite3.Connection,
    out_path: str | Path,
    *,
    tenant_id: str | None = None,
    random_state: int = 42,
) -> RankerArtifact | None:
    X, y = rows_from_match_events(conn, tenant_id=tenant_id)
    X, y = _merge_default_rows(X, y)
    if len(y) < 4:
        return None
    if len(np.unique(y)) < 2:
        # trivial: always predict majority
        maj = int(y.mean() >= 0.5)
        clf = DummyClassifier(strategy="constant", constant=maj)
        clf.fit(X, y)
        art = RankerArtifact(model=clf, feature_order=FEATURE_ORDER)
        save_ranker(art, out_path)
        return art
    if len(y) < 12:
        from sklearn.calibration import CalibratedClassifierCV

        base = LogisticRegression(max_iter=200, random_state=random_state)
        cal = CalibratedClassifierCV(base, method="sigmoid", cv=2)
        cal.fit(X, y)
        art = RankerArtifact(model=cal, feature_order=FEATURE_ORDER)
        save_ranker(art, out_path)
        return art
    art = train_global_ranker(X, y, random_state=random_state)
    save_ranker(art, out_path)
    return art


def train_per_tenant_models(conn: sqlite3.Connection, out_dir: str | Path) -> dict[str, str | None]:
    """Fit one calibrated model per tenant with any labeled rows; merges default synthetic priors."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tenants = [
        r[0]
        for r in conn.execute(
            "SELECT DISTINCT tenant_id FROM match_events ORDER BY tenant_id"
        ).fetchall()
    ]
    paths: dict[str, str | None] = {}
    for t in tenants:
        path = out_dir / f"ranker_{t}.joblib"
        art = train_and_save(conn, path, tenant_id=t)
        paths[t] = str(path) if art is not None else None
    return paths

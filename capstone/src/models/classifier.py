"""Classifier models — Logistic Regression and Random Forest.

Client-grouped 4-fold validation with precision@K.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from capstone.src.data.features import FEATURES8, get_feature_matrix
from capstone.src.evaluation.metrics import precision_at_k


def train_and_evaluate(
    fv: pd.DataFrame,
    n_splits: int = 4,
    base_rate: float = 0.5181,
) -> dict:
    """Train LR + RF with GroupKFold, return pooled OOF predictions."""
    X = get_feature_matrix(fv)
    y = fv["is_declining"].values
    groups = fv["client_hash_id"].values

    n = len(y)
    oof_lr = np.zeros(n, dtype=float)
    oof_rf = np.zeros(n, dtype=float)
    groups_seen = []

    gkf = GroupKFold(n_splits=n_splits)
    for fold, (trn_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        gv = set(groups[trn_idx])
        groups_seen.append(gv)

        X_trn, X_val = X.iloc[trn_idx], X.iloc[val_idx]
        y_trn, y_val = y[trn_idx], y[val_idx]

        # Logistic Regression
        lr_pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(C=1.0, max_iter=5000, random_state=0)),
        ])
        lr_pipe.fit(X_trn, y_trn)
        oof_lr[val_idx] = lr_pipe.predict_proba(X_val)[:, 1]

        # Random Forest
        rf = RandomForestClassifier(
            n_estimators=200, max_depth=8, max_features=0.5,
            random_state=0, n_jobs=-1,
        )
        rf.fit(X_trn, y_trn)
        oof_rf[val_idx] = rf.predict_proba(X_val)[:, 1]

    results = {"lr": oof_lr, "rf": oof_rf, "y_true": y, "groups": groups}
    return results


def get_feature_importances(fv: pd.DataFrame) -> pd.DataFrame:
    """Train RF on full data and return feature importances."""
    X = get_feature_matrix(fv)
    y = fv["is_declining"].values

    rf = RandomForestClassifier(
        n_estimators=200, max_depth=8, max_features=0.5,
        random_state=0, n_jobs=-1,
    )
    rf.fit(X, y)

    imp = pd.DataFrame({
        "feature": FEATURES8,
        "importance": rf.feature_importances_,
    }).sort_values("importance", ascending=False)

    return imp


def get_lr_coefficients(fv: pd.DataFrame) -> pd.DataFrame:
    """Train LR on full data and return standardized coefficients."""
    X = get_feature_matrix(fv)
    y = fv["is_declining"].values

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(C=1.0, max_iter=5000, random_state=0)),
    ])
    pipe.fit(X, y)

    coef = pd.DataFrame({
        "feature": FEATURES8,
        "coefficient": pipe.named_steps["model"].coef_[0],
    }).sort_values("coefficient", ascending=False)

    return coef

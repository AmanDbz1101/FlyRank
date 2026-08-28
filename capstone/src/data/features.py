"""Feature vector builder — the same 8 inputs as w03/w04/w05.

All features are knowable at the decision moment (2026-03-31).
Missingness is handled with fillna(0) plus has_* flags.
"""
import numpy as np
import pandas as pd

NUMERIC = [
    "log_mar_impressions",
    "ctr_mar",
    "mar_avg_position",
    "momentum_feb_to_mar_pct",
    "engagement_rate_mar",
]
FLAGS = ["has_clicks", "has_feb_data", "has_ga4"]
FEATURES8 = NUMERIC + FLAGS


def build_feature_vector(pop: pd.DataFrame) -> pd.DataFrame:
    """Add the 8 model features to a population DataFrame.

    Input: output of warehouse.load_population()
    Returns: same DataFrame with feature columns added.
    """
    fv = pop.copy()
    fv["log_mar_impressions"] = np.log1p(fv.mar_impressions)
    fv["has_clicks"] = (fv.mar_clicks > 0).astype(int)
    fv["has_feb_data"] = fv.feb_impressions.gt(0).astype(int)
    fv["has_ga4"] = (fv.mar_sessions > 0).astype(int)
    fv["engagement_rate_mar"] = np.where(
        fv.mar_sessions > 0,
        fv.mar_engaged / fv.mar_sessions * 100,
        np.nan,
    )
    fv["pos_band"] = pd.cut(
        fv.mar_avg_position,
        bins=[0, 3, 10, 20, 50, 1e9],
        labels=["<3", "3-10", "10-20", "20-50", "50+"],
    )

    fv = fv.sort_values("content_hash_id").reset_index(drop=True)
    return fv


def get_feature_matrix(fv: pd.DataFrame) -> pd.DataFrame:
    """Return the 8-column feature matrix with fillna(0).

    Suitable for direct input to sklearn models.
    """
    return fv[FEATURES8].fillna(0)

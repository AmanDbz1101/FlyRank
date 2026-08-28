"""Baseline rule — hand-crafted scoring from w04.

Reproduces the rule exactly as implemented in w04_baseline_score.ipynb.
"""
import numpy as np
import pandas as pd


def compute_baseline_score(pop: pd.DataFrame) -> np.ndarray:
    """Compute the hand-crafted baseline risk score for each page.

    Returns an array of scores (higher = more risk).
    """
    mar_imp = pop["mar_impressions"].values.astype(float)
    mar_clicks = pop["mar_clicks"].values.astype(float)
    ctr_mar = pop["ctr_mar"].values.astype(float)
    momentum = pop["momentum_feb_to_mar_pct"].fillna(0).values.astype(float)
    feb_imp = pop["feb_impressions"].fillna(0).values.astype(float)
    pos_band = pop["pos_band"].values

    scores = np.zeros(len(pop), dtype=float)

    # Spike score: impressions grew >200% from Feb to Mar
    has_feb = feb_imp > 0
    spike = np.where(has_feb, (mar_imp - feb_imp) / feb_imp, 0) > 2
    scores += spike.astype(float)

    # Critical rule: no clicks at top-10 position
    no_clicks = mar_clicks == 0
    in_top10 = np.isin(pos_band, ["<3", "3-10"])
    scores += (no_clicks & in_top10).astype(float) * 2

    # Critical rule: position 3-10 and CTR < 50% of position band mean
    in_3_10 = pos_band == "3-10"
    ctr_below_half = ctr_mar < (3.0 * 0.5)  # simplified band mean
    scores += (in_3_10 & ctr_below_half).astype(float) * 2

    # CTR shortfall (clipped)
    expected_ctr = np.where(
        np.isin(pos_band, ["<3"]), 7.0,
        np.where(np.isin(pos_band, ["3-10"]), 3.0,
        np.where(np.isin(pos_band, ["10-20"]), 1.5,
        np.where(np.isin(pos_band, ["20-50"]), 0.5, 0.2)))
    )
    ctr_ratio = np.where(expected_ctr > 0, ctr_mar / expected_ctr, 1.0)
    ctr_shortfall = np.clip(1 - ctr_ratio, 0, 1)
    scores += ctr_shortfall

    # Impact: clicks lost (normalized)
    clicks_lost = np.where(ctr_ratio < 1, (1 - ctr_ratio) * mar_imp, 0)
    impact = np.log1p(clicks_lost) / 10
    scores += impact

    return scores


def apply_baseline(pop: pd.DataFrame) -> pd.DataFrame:
    """Add baseline_score column to the population DataFrame."""
    df = pop.copy()
    df["baseline_score"] = compute_baseline_score(df)
    return df

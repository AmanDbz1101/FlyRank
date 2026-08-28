"""Evaluation metrics — Precision@K, ROC AUC, and comparisons.

All metrics print the base rate next to the headline number.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def precision_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:
    """Precision at top K: of the K pages with highest scores, how many are positive?"""
    if k <= 0 or k > len(y_true):
        return 0.0
    idx = np.argsort(-y_score)[:k]
    return y_true[idx].mean()


def evaluate_ranking(
    y_true: np.ndarray,
    y_score: np.ndarray,
    base_rate: float = 0.5181,
    label: str = "",
    k_values: list[int] | None = None,
) -> dict:
    """Compute and print all ranking metrics.

    Returns a dict of {metric_name: value}.
    """
    if k_values is None:
        k_values = [10, 50, 100, 500]

    results = {}
    try:
        auc = roc_auc_score(y_true, y_score)
    except ValueError:
        auc = 0.0

    if label:
        print(f"\n{'='*60}")
        print(f"  {label}")
        print(f"{'='*60}")
        print(f"  Base rate: {base_rate:.4f}")
        print(f"  ROC AUC:   {auc:.4f}")

    for k in k_values:
        p_at_k = precision_at_k(y_true, y_score, k)
        lift = p_at_k / base_rate if base_rate > 0 else 0
        results[f"P@{k}"] = p_at_k
        results[f"lift@{k}"] = lift
        if label:
            print(f"  P@{k:<4d}:   {p_at_k:.4f}  (lift {lift:.2f}×)")

    results["ROC_AUC"] = auc
    if label:
        print(f"{'='*60}\n")

    return results


def compare_models(
    y_true: np.ndarray,
    scores_dict: dict[str, np.ndarray],
    base_rate: float = 0.5181,
    k_values: list[int] | None = None,
) -> pd.DataFrame:
    """Compare multiple models side-by-side.

    Returns a DataFrame with metrics for each model.
    """
    rows = []
    for name, scores in scores_dict.items():
        r = evaluate_ranking(y_true, scores, base_rate, label=name, k_values=k_values)
        r["model"] = name
        rows.append(r)
    return pd.DataFrame(rows).set_index("model")

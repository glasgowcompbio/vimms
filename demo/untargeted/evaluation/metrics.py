import pandas as pd
from sklearn.metrics import adjusted_rand_score, pair_confusion_matrix


def compute_group_metrics(
    peaks: pd.DataFrame,
    compound_col: str = "compound_id",
    group_col: str = "group",
) -> dict:
    """Return alignment metrics comparing predicted groups to ground truth.

    Parameters
    ----------
    peaks : pd.DataFrame
        Table containing columns ``compound_col`` and ``group_col``.
    compound_col : str, optional
        Column giving ground truth compound identifiers.
    group_col : str, optional
        Column giving predicted alignment group labels.

    Returns
    -------
    dict
        Dictionary with ``precision``, ``recall``, ``f1`` and ``ari`` keys.
    """
    labels_true = peaks[compound_col].to_numpy()
    labels_pred = peaks[group_col].to_numpy()
    tn, fp, fn, tp = pair_confusion_matrix(labels_true, labels_pred).ravel()
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    ari = adjusted_rand_score(labels_true, labels_pred)
    return {"precision": precision, "recall": recall, "f1": f1, "ari": ari}

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
    brier_score_loss,
    matthews_corrcoef,
    confusion_matrix,
)


def _check_binary(probs, n_classes):
    return n_classes == 2 or (probs is not None and probs.shape[-1] == 2)


def compute_metrics(y_true, y_pred, y_prob=None):
    n_classes = len(set(y_true))
    results = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted"),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }

    if n_classes <= 2:
        results["auc"] = roc_auc_score(y_true, y_prob[:, 1]) if y_prob is not None else None
        results["brier"] = brier_score_loss(y_true, y_prob[:, 1]) if y_prob is not None else None
    elif y_prob is not None:
        try:
            results["macro_auc_ovr"] = roc_auc_score(y_true, y_prob, multi_class="ovr")
        except (ValueError, IndexError):
            results["macro_auc_ovr"] = None

    return results


def compute_calibration_error(y_true, y_prob, n_bins=10):
    from sklearn.calibration import calibration_curve

    if y_prob.ndim == 2 and y_prob.shape[-1] > 2:
        y_pred = y_prob.argmax(axis=1)
        y_prob = y_prob.max(axis=1)
        y_true_bin = (y_true == y_pred).astype(int)
    else:
        y_prob = y_prob[:, 1] if y_prob.ndim == 2 else y_prob
        y_true_bin = y_true

    prob_true, prob_pred = calibration_curve(y_true_bin, y_prob, n_bins=n_bins, strategy="uniform")
    ece = np.mean(np.abs(prob_true - prob_pred))
    return {"ece": float(ece)}


def aggregate_fold_metrics(all_fold_metrics):
    metrics_keys = [k for k in all_fold_metrics[0] if k != "confusion_matrix"]
    aggregated = {}
    for k in metrics_keys:
        values = [m[k] for m in all_fold_metrics if m.get(k) is not None]
        if values:
            aggregated[f"{k}_mean"] = float(np.mean(values))
            aggregated[f"{k}_std"] = float(np.std(values))
    return aggregated

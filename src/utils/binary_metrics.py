"""Metrics for binary tasks (e.g. glaucoma vs non-glaucoma).

All values are fractions in [0, 1]. ``scores`` is the predicted probability of the positive
class (label 1); ``preds`` are the hard predictions used for sensitivity and specificity.
"""

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def sensitivity_at_specificity(labels, scores, specificity):
    """Highest sensitivity over all thresholds whose specificity is at least ``specificity``."""
    fpr, tpr, _ = roc_curve(labels, scores)
    allowed = (1.0 - fpr) >= specificity - 1e-12
    # roc_curve always includes the threshold above every score (fpr = 0), so one point is allowed.
    return float(tpr[allowed].max())


def binary_point_metrics(labels, scores, preds, specificity=0.85):
    labels, scores, preds = np.asarray(labels), np.asarray(scores, dtype=float), np.asarray(preds)
    positives, negatives = labels == 1, labels == 0
    if not positives.any() or not negatives.any():
        raise ValueError("Binary metrics need both classes in the labels")
    sensitivity = float((preds[positives] == 1).mean())
    specificity_at_preds = float((preds[negatives] == 0).mean())
    return {
        "auc": float(roc_auc_score(labels, scores)),
        "balanced_accuracy": (sensitivity + specificity_at_preds) / 2,
        "sensitivity": sensitivity,
        "specificity": specificity_at_preds,
        f"sensitivity_at_specificity_{round(specificity * 100)}": sensitivity_at_specificity(labels, scores, specificity),
    }


def bootstrap_binary_metrics(labels, scores, preds, n_bootstrap=10000, confidence_level=0.95,
                             specificity=0.85, rng=None):
    """Percentile bootstrap over test images: {metric: (mean, lower, upper)}.

    Resamples that contain a single class are skipped, since AUC is undefined for them.
    """
    labels, scores, preds = np.asarray(labels), np.asarray(scores, dtype=float), np.asarray(preds)
    rng = np.random.default_rng() if rng is None else rng
    samples = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, len(labels), len(labels))
        if labels[idx].min() == labels[idx].max():
            continue
        samples.append(binary_point_metrics(labels[idx], scores[idx], preds[idx], specificity))
    if not samples:
        raise ValueError("No bootstrap resample contained both classes")
    tail = (1 - confidence_level) / 2 * 100
    return {key: (float(np.mean([s[key] for s in samples])),
                  *(float(v) for v in np.percentile([s[key] for s in samples], [tail, 100 - tail])))
            for key in samples[0]}

"""Helpers for building JSON-friendly experiment and evaluation reports."""

# Class names of the UWF lesion dataset, indexed by label.
UWF_CLASS_NAMES = ("healthy", "nevus", "melanoma", "chrpe")


def to_jsonable(value):
    """Recursively convert values into JSON-safe Python primitives."""

    if isinstance(value, dict):
        return {str(key): to_jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    return value


def _ci_triplet_to_pct(summary):
    """Convert a ``(mean, lower, upper)`` metric summary from 0..1 to percentage."""

    mean, lower, upper = summary
    return {
        "mean": mean * 100.0,
        "lower": lower * 100.0,
        "upper": upper * 100.0,
    }


def format_bootstrap_metrics(raw_metrics, class_names=UWF_CLASS_NAMES):
    """Normalize bootstrap output into explicit percentage-based JSON sections."""

    metric_key_map = {
        "accuracy": "accuracy_overall",
        "precision": "precision_macro",
        "recall": "recall_macro",
        "f1_score": "f1_macro",
    }

    bootstrap_ci_pct = {}
    for raw_key, target_key in metric_key_map.items():
        bootstrap_ci_pct[target_key] = _ci_triplet_to_pct(raw_metrics[raw_key])

    class_metrics = {}
    for class_name in class_names:
        if f"accuracy_{class_name}" not in raw_metrics:
            continue  # class absent from the test labels
        raw_name = class_name
        class_metrics[class_name] = {}
        for raw_key, target_key in metric_key_map.items():
            metric_key = target_key.replace("_overall", "").replace("_macro", "")
            class_metrics[class_name][metric_key] = _ci_triplet_to_pct(
                (
                    raw_metrics[f"{raw_key}_{raw_name}"][0],
                    raw_metrics[f"{raw_key}_{raw_name}"][1][0],
                    raw_metrics[f"{raw_key}_{raw_name}"][1][1],
                )
            )

    return {
        "bootstrap_ci_pct": to_jsonable(bootstrap_ci_pct),
        "class_one_vs_rest_bootstrap_ci_pct": to_jsonable(class_metrics),
    }


def format_test_metrics(point_metrics_pct, raw_bootstrap_metrics=None, class_names=None,
                        binary_point=None, binary_bootstrap=None):
    """Build a clear evaluation report with explicit metric conventions.

    Returned sections use these conventions:
    - ``point_metrics_pct``:
      - ``accuracy_overall``
      - ``precision_weighted``
      - ``recall_weighted``
      - ``f1_weighted``
    - ``bootstrap_ci_pct``:
      - ``accuracy_overall``
      - ``precision_macro``
      - ``recall_macro``
      - ``f1_macro``
    - ``class_one_vs_rest_bootstrap_ci_pct``:
      per-class one-vs-rest summaries for accuracy / precision / recall / f1
    - binary tasks only (``binary_point`` / ``binary_bootstrap``, fractions in, percentages out):
      ``binary_point_metrics_pct`` and ``binary_bootstrap_ci_pct`` with ``auc``,
      ``balanced_accuracy``, ``sensitivity``, ``specificity``, ``sensitivity_at_specificity_<s>``
    """

    report = {
        "point_metrics_pct": to_jsonable(point_metrics_pct),
    }
    if raw_bootstrap_metrics is not None:
        report.update(format_bootstrap_metrics(raw_bootstrap_metrics, class_names or UWF_CLASS_NAMES))
    if binary_point is not None:
        report["binary_point_metrics_pct"] = to_jsonable({k: v * 100.0 for k, v in binary_point.items()})
    if binary_bootstrap is not None:
        report["binary_bootstrap_ci_pct"] = to_jsonable(
            {k: _ci_triplet_to_pct(v) for k, v in binary_bootstrap.items()})
    return report

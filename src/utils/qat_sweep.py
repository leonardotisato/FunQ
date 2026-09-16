"""QAT artifact naming and provenance, importable without training packages."""

import hashlib
import json
import math
from pathlib import Path
import re

from utils.kd_sweep import TEACHER_SPECS, cfg_select, optional_tag


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_qat_run(architecture, resolution, weight_bits, act_bits,
                    teacher_mode=None, seed=None, experiment_tag=None):
    match = re.fullmatch(r"test_resnet(?:_(slim\d+x\d+))?", architecture)
    if not match:
        raise ValueError(f"Unsupported QAT architecture: {architecture}")
    trim = f"trim{int(resolution)}"
    bit_tag = f"{int(weight_bits)}w{int(act_bits)}a"
    variant = match.group(1)
    fp32_run_tag = f"{variant}_{trim}" if variant else trim
    suffix = ""
    if optional_tag(experiment_tag):
        if teacher_mode not in TEACHER_SPECS or seed is None:
            raise ValueError("QAT experiment requires a valid teacher_mode and seed")
        suffix = f"_{teacher_mode}_seed{int(seed)}"
    run_tag = f"{fp32_run_tag}_{bit_tag}{suffix}"
    return {"bit_tag": bit_tag, "trim_tag": trim, "variant": variant,
            "run_tag": run_tag, "fp32_run_tag": fp32_run_tag,
            "fp32_checkpoint_name": f"{architecture}_fp32_kd_{trim}_ft.pth",
            "results_dir_name": f"qat_test_resnet_{run_tag}",
            "checkpoint_name": f"test_resnet_{run_tag}_qat.pth",
            "report_name": f"qat_test_resnet_{run_tag}_report.json",
            "log_name": f"qat_test_resnet_{run_tag}_log.csv",
            "model_type": f"test_resnet_{run_tag}_qat"}


def validate_weight_loading(missing, unexpected):
    quantizer_tokens = ("tensor_quant", "scaling_impl", "int_scaling_impl", "zero_point",
                        "msb_clamp_bit_width_impl", "act_quant", "weight_quant", "bias_quant")
    model_missing = [key for key in missing if not any(token in key for token in quantizer_tokens)]
    if model_missing or unexpected:
        raise ValueError(f"Incompatible FP32 initialization: missing model keys={model_missing}; "
                         f"unexpected keys={list(unexpected)}")


def valid_metrics(report):
    """A checkpoint without a selected finite model/test result is not complete."""
    try:
        if report["epochs"] < 1 or not math.isfinite(report["best_val_loss"]):
            return False
        if not math.isfinite(report["best_val_f1"]):
            return False
        metrics = report["test_metrics"]["point_metrics_pct"]
        return all(math.isfinite(metrics[k]) and 0 <= metrics[k] <= 100
                   for k in ("accuracy_overall", "precision_weighted", "recall_weighted", "f1_weighted"))
    except (KeyError, TypeError, ValueError):
        return False


def run_provenance(cfg, teacher_spec, init_checkpoint, train_df, val_df, test_df):
    source_report = cfg_select(cfg, "fp32_source_report", None)
    data_csv = Path(cfg.data_dir) / cfg.csv_file
    split_manifest = {}
    for name, frame in (("train", train_df), ("validation", val_df), ("test", test_df)):
        columns = [c for c in ("patient_id", "image", "label") if c in frame.columns]
        records = frame[columns].to_dict(orient="records")
        payload = json.dumps(records, sort_keys=True, default=str).encode()
        split_manifest[name] = {"images": len(frame),
                                "patients": int(frame["patient_id"].nunique()),
                                "rows_sha256": hashlib.sha256(payload).hexdigest()}
    return {"teacher_mode": teacher_spec["mode"], "teacher_arch": teacher_spec["arch"],
            "teacher_checkpoint": teacher_spec["checkpoint_path"],
            "teacher_checkpoint_sha256": file_sha256(teacher_spec["checkpoint_path"]),
            "student_init_checkpoint_sha256": file_sha256(init_checkpoint),
            "experiment_tag": cfg_select(cfg, "experiment_tag", None),
            "random_seed": int(cfg.RANDOM_SEED), "split_seed": int(cfg.RANDOM_SEED),
            "qat_seed": int(cfg.RANDOM_SEED),
            "fp32_seed": int(cfg.RANDOM_SEED) if source_report else None,
            "seed_protocol": "RANDOM_SEED controls patient split and QAT randomness",
            "fp32_source_report": source_report,
            "fp32_source_report_sha256": file_sha256(source_report) if source_report else None,
            "data_csv": str(data_csv), "data_csv_sha256": file_sha256(data_csv),
            "patient_splits": split_manifest,
            "patient_split_fractions": {"train": 0.70, "validation": 0.15, "test": 0.15},
            "sweep_run_fingerprint": cfg_select(cfg, "sweep_run_fingerprint", None)}

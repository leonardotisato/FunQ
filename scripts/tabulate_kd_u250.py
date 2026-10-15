"""Tabulate the FP32 KD sweep and inventory the FINN U250 builds (figures: scripts/make_figures.py).

Run from any directory: python scripts/tabulate_kd_u250.py
Requires numpy and scipy; no training libraries or checkpoints.
Reads only the tracked summaries in results/ (fp32_kd_sweep, legacy, hw); see analysis/kd_u250/README.md.
"""

import argparse
import ast
import csv
import hashlib
import itertools
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import t


ROOT = Path(__file__).resolve().parents[1]
METRICS = ("accuracy_overall", "f1_weighted", "precision_weighted", "recall_weighted")
GROUP = ("experiment", "architecture", "resolution", "teacher")
TEACHERS = ("r18_ta", "r50_direct")
ARCH_LABELS = {"test_resnet": "TestResNet", "test_resnet_slim128x64": "TestResNet slim 128/64"}


def check(condition, message):
    if not condition:
        raise ValueError(message)


def relative(path):
    return path.resolve().relative_to(ROOT).as_posix()


def load_report(path, provenance):
    content = path.read_bytes()
    provenance.append({"path": relative(path), "sha256": hashlib.sha256(content).hexdigest()})
    return json.loads(content)


def write_csv(path, rows):
    check(bool(rows), f"No rows for {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def statistics(values):
    values = np.asarray(values, dtype=float)
    check(bool(len(values)) and np.isfinite(values).all(), "Empty or non-finite metric values")
    n = len(values)
    mean = float(values.mean())
    sd = float(values.std(ddof=1)) if n > 1 else None
    se = sd / math.sqrt(n) if n > 1 else None
    half = float(t.ppf(0.975, n - 1) * se) if n > 1 else None
    return {"n": n, "mean_pct": mean, "sd_pp": sd,
            "variance_pp2": sd**2 if sd is not None else None, "se_pp": se,
            "ci95_low_pct": mean - half if half is not None else None,
            "ci95_high_pct": mean + half if half is not None else None}


def bootstrap_rows(report, source, cohort):
    result = []
    metrics = report.get("test_metrics", {})
    for section in ("bootstrap_ci_pct", "class_one_vs_rest_bootstrap_ci_pct"):
        data = metrics.get(section, {})
        sections = {"all": data} if section == "bootstrap_ci_pct" else data
        for label, entries in sections.items():
            for metric, triplet in entries.items():
                check(all(math.isfinite(triplet[k]) for k in ("mean", "lower", "upper")), source)
                check(0 <= triplet["lower"] <= triplet["upper"] <= 100, source)
                result.append({"cohort": cohort, "source_report": source,
                               "class": label, "metric": metric,
                               "bootstrap_mean_pct": triplet["mean"],
                               "reported_ci95_low_pct": triplet["lower"],
                               "reported_ci95_high_pct": triplet["upper"]})
    return result


def collect_fp32(sweeps, provenance):
    runs, bootstrap, seen = [], [], set()
    for sweep in sweeps:
        for path in sorted(sweep.glob("*_report.json")):
            report = load_report(path, provenance)
            run = path.name.removeprefix("train_").removesuffix("_report.json")
            architecture = run.split("_trim")[0]
            row = {"experiment": report["experiment_tag"], "architecture": architecture,
                   "resolution": int(report["student_resolution"]),
                   "teacher": report["teacher_mode"], "seed": int(report["random_seed"])}
            key = tuple(row[k] for k in GROUP) + (row["seed"],)
            check(key not in seen, f"Duplicate run: {key}")
            seen.add(key)
            expected = f"{architecture}_trim{row['resolution']}_{row['teacher']}_seed{row['seed']}"
            check(run == expected, f"Filename/metadata mismatch: {path}")
            check(report["input_size"] == [1, 3, row["resolution"], row["resolution"]], str(path))
            check(row["teacher"] in TEACHERS and architecture in ARCH_LABELS, str(path))
            point = report["test_metrics"]["point_metrics_pct"]
            for metric in METRICS:
                check(math.isfinite(point[metric]) and 0 <= point[metric] <= 100, str(path))
                row[metric] = point[metric]
            check(math.isclose(point["recall_weighted"], point["accuracy_overall"], abs_tol=1e-9), str(path))
            row.update({"n_params": report["n_params"], "student_init": report["student_init"],
                        "init_checkpoint": report.get("init_checkpoint"),
                        "checkpoint": report["checkpoint"],
                        "checkpoint_exists_locally": (ROOT / report["checkpoint"]).is_file(),
                        "teacher_checkpoint": report["teacher_checkpoint"],
                        "kd_temperature": report["kd_temperature"], "kd_alpha": report["kd_alpha"],
                        "best_val_loss": report["best_val_loss"],
                        "val_f1_at_selected_epoch": report["best_val_f1"],
                        "selected_epoch_one_based": report["epochs"], "source_report": relative(path)})
            runs.append(row)
            bootstrap.extend(bootstrap_rows(report, relative(path), "fp32_sweep"))
    check(bool(runs), "No sweep reports found")
    return runs, bootstrap


def summarize(runs):
    groups = defaultdict(list)
    for row in runs:
        groups[tuple(row[k] for k in GROUP)].append(row)
    summaries = []
    for key, rows in sorted(groups.items()):
        for field in ("n_params", "student_init", "teacher_checkpoint", "kd_temperature", "kd_alpha"):
            check(len({row[field] for row in rows}) == 1, f"Mixed {field} within {key}")
        for metric in METRICS:
            summaries.append({**dict(zip(GROUP, key)), "metric": metric,
                              "seeds": " ".join(str(r["seed"]) for r in sorted(rows, key=lambda r: r["seed"])),
                              **statistics([r[metric] for r in rows])})
    paired = []
    for key in sorted({tuple(r[k] for k in GROUP[:3]) for r in runs}):
        by_teacher = {teacher: {r["seed"]: r for r in runs
                               if tuple(r[k] for k in GROUP[:3]) == key and r["teacher"] == teacher}
                      for teacher in TEACHERS}
        check(set(by_teacher[TEACHERS[0]]) == set(by_teacher[TEACHERS[1]]), f"Unpaired teachers: {key}")
        seeds = sorted(by_teacher[TEACHERS[0]])
        for metric in METRICS:
            delta = [by_teacher["r50_direct"][s][metric] - by_teacher["r18_ta"][s][metric] for s in seeds]
            stats = statistics(delta)
            paired.append({**dict(zip(GROUP[:3], key)), "metric": metric,
                           "contrast": "r50_direct minus r18_ta", "seeds": " ".join(map(str, seeds)),
                           **{k.replace("_pct", "_pp"): v for k, v in stats.items()}})
    return summaries, paired


def collect_legacy(provenance):
    rows, bootstrap = [], []
    legacy = ROOT / "results/legacy"
    paths = [*legacy.glob("qat-test-resnet/160/*/*_report.json"), *legacy.glob("qat-test-resnet-slim/*/*_report.json")]
    for path in sorted(paths, key=lambda p: p.name):
        report = load_report(path, provenance)
        rows.append({"run": path.name.removesuffix("_report.json"), "resolution": report["student_resolution"],
                     "weight_bits": report["weight_bits"], "act_bits": report["act_bits"],
                     "seed": report.get("random_seed"),
                     "student_init_checkpoint": report["student_init_checkpoint"],
                     "checkpoint": report["checkpoint"],
                     **report["test_metrics"]["point_metrics_pct"], "source_report": relative(path)})
        bootstrap.extend(bootstrap_rows(report, relative(path), "legacy_single_run_qat"))
    return rows, bootstrap


def driver_shapes(path):
    """Read only literal shapes; never import/execute a generated driver."""
    parsed = ast.parse(path.read_text(encoding="utf-8"))
    for node in parsed.body:
        if isinstance(node, ast.Assign) and any(isinstance(x, ast.Name) and x.id == "io_shape_dict" for x in node.targets):
            return {k.value: ast.literal_eval(v) for k, v in zip(node.value.keys, node.value.values)
                    if isinstance(k, ast.Constant) and k.value in ("ishape_normal", "oshape_normal")}
    raise ValueError(f"No driver shapes: {path}")


def collect_finn(provenance):
    rows = []
    with (ROOT / "results/hw/builds.csv").open(encoding="utf-8") as handle:
        bitfiles = {row["build"]: row["bitfile"] for row in csv.DictReader(handle)}
    for directory in sorted((ROOT / "results/hw/u250").glob("build_finn_*")):
        path = directory / "report/estimate_network_performance.json"
        check(path.exists(), f"Missing FINN estimate: {directory}")
        report = load_report(path, provenance)
        model, target = directory.name.removeprefix("build_finn_").split("_u250_")
        topology, bits = model.rsplit("_", 1)
        wb, ab = map(int, bits.removesuffix("a").split("w"))
        driver = directory / "driver/driver.py"
        shapes = driver_shapes(driver)
        provenance.append({"path": relative(driver), "sha256": hashlib.sha256(driver.read_bytes()).hexdigest()})
        ishape, oshape = shapes["ishape_normal"][0], shapes["oshape_normal"][0]
        resolution = ishape[1]
        check(ishape == (1, resolution, resolution, 3), f"Unexpected input: {directory}")
        if "_trim" in topology:
            check(int(topology.rsplit("_trim", 1)[1]) == resolution, f"Resolution mismatch: {directory}")
        target_fps = int(target.removesuffix("fps")) if target.endswith("fps") else None
        fps = report["estimated_throughput_fps"]
        period = report["estimated_latency_ns"] / report["critical_path_cycles"]
        check(fps > 0 and math.isclose(fps, 1e9 / period / report["max_cycles"]), str(path))
        rows.append({"build": directory.name, "topology_from_build_name": topology,
                     "input_resolution_from_driver": resolution,
                     "preprocessing_identity": "trim_named" if "_trim" in topology else "legacy_unspecified",
                     "weight_bits_from_name": wb, "act_bits_from_name": ab,
                     "target_setting": target, "target_fps_from_name": target_fps,
                     "estimated_throughput_fps": fps,
                     "estimated_latency_ms": report["estimated_latency_ns"] / 1e6,
                     "clock_period_ns_inferred_from_estimate": period,
                     "critical_path_cycles": report["critical_path_cycles"],
                     "max_cycles": report["max_cycles"], "bottleneck_node": report["max_cycles_node_name"],
                     "driver_output_shape": str(oshape),
                     "xclbin_present": bool(bitfiles.get(directory.name)),
                     "performance_scope": "analytical accelerator estimate; excludes host classifier and I/O",
                     "accuracy_join_status": "pending matched multi-seed QAT and export provenance",
                     "source_report": relative(path), "source_driver": relative(driver)})
    check(bool(rows), "No U250 builds found")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep", type=Path, action="append", help="Folder of FP32 sweep report JSONs; repeat to analyze several separately")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "analysis/kd_u250")
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    sweeps = args.sweep or [ROOT / "results/fp32_kd_sweep"]
    provenance = []
    runs, bootstrap = collect_fp32(sweeps, provenance)
    summary, paired = summarize(runs)
    legacy, legacy_bootstrap = collect_legacy(provenance)
    finn = collect_finn(provenance)
    for name, rows in (("fp32_runs.csv", runs), ("fp32_summary.csv", summary),
                       ("fp32_paired_teacher_differences.csv", paired), ("legacy_qat_runs.csv", legacy),
                       ("reported_bootstrap_intervals.csv", bootstrap + legacy_bootstrap), ("u250_estimates.csv", finn)):
        if rows:
            write_csv(out / name, rows)
    missing = []
    for experiment in sorted({r["experiment"] for r in runs}):
        subset = [r for r in runs if r["experiment"] == experiment]
        grid = itertools.product(*[sorted({r[k] for r in subset}) for k in GROUP[1:] + ("seed",)])
        actual = {tuple(r[k] for k in GROUP[1:] + ("seed",)) for r in subset}
        missing.extend([[experiment, *key] for key in grid if key not in actual])
    quality = {"fp32_runs": len(runs), "fp32_groups": len(summary) // len(METRICS),
               "missing_grid_cells": missing, "legacy_qat_reports": len(legacy), "u250_builds": len(finn),
               "missing_fp32_checkpoints": [r["checkpoint"] for r in runs if not r["checkpoint_exists_locally"]],
               "weighted_recall_equals_accuracy_all_runs": True,
               "interval_method": "mean +/- t(0.975,n-1) * sample_sd / sqrt(n); ddof=1; unweighted run mean",
               "interval_scope": "Nominal descriptive interval across split/training runs; overlapping splits limit population inference",
               "macro_point_metrics_available": False, "quantized_mean_accuracy_join_performed": False,
               "sources": provenance}
    (out / "data_quality.json").write_text(json.dumps(quality, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(out), "fp32_runs": len(runs), "fp32_groups": len(summary) // 4,
                      "legacy_qat_reports": len(legacy), "u250_builds": len(finn), "missing_grid_cells": missing}, indent=2))


if __name__ == "__main__":
    main()

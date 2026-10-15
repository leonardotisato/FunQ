"""Plot the matched FP32 -> QAT seed sweep and the U250 area/throughput trade-off.

Run from any directory: python scripts/plot_qat_sweep.py
Requires numpy, scipy and matplotlib. Reads only results/ (fp32_kd_sweep, qat_sweep, hw);
writes figures and CSVs to analysis/qat_sweep/.
"""

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from scipy.stats import t

ROOT = Path(__file__).resolve().parents[1]
PRECISIONS = ("fp32", "8w8a", "6w6a", "4w4a")
PRECISION_LABELS = {"fp32": "FP32", "8w8a": "8-bit", "6w6a": "6-bit", "4w4a": "4-bit"}
ARCHS = ("test_resnet", "test_resnet_slim128x64")
ARCH_LABELS = {"test_resnet": "TestResNet", "test_resnet_slim128x64": "TestResNet slim 128/64"}
TEACHERS = ("r18_ta", "r50_direct")
TEACHER_LABELS = {"r18_ta": "ResNet18 teacher assistant", "r50_direct": "ResNet50 direct teacher"}
RESOLUTIONS = (160, 192, 224)

# Reference categorical palette, light mode (slots 1-4), plus text/grid tokens.
SERIES = {160: "#2a78d6", 192: "#eb6834", 224: "#1baf7a"}
RESOURCE_COLORS = {"LUT": "#2a78d6", "FF": "#eb6834", "BRAM": "#1baf7a", "DSP": "#eda100"}
MARKERS = {160: "o", 192: "s", 224: "^"}
INK, INK_2, INK_3, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df", "#ffffff"

# Xilinx Alveo U250 (xcu250) device totals.
U250 = {"LUT": 1_728_000, "FF": 3_456_000, "BRAM": 2_688, "DSP": 12_288}


def check(condition, message):
    if not condition:
        raise ValueError(message)


def ci95(values):
    values = np.asarray(values, dtype=float)
    n, mean = len(values), float(values.mean())
    sd = float(values.std(ddof=1))
    half = float(t.ppf(0.975, n - 1) * sd / math.sqrt(n))
    return {"n": n, "mean": mean, "sd": sd, "ci_low": mean - half, "ci_high": mean + half}


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_runs():
    """One row per run: architecture, resolution, teacher, precision, seed, metrics."""
    runs = []
    sources = [(p, "fp32") for p in sorted((ROOT / "results/fp32_kd_sweep").glob("*_report.json"))]
    sources += [(p, None) for p in sorted((ROOT / "results/qat_sweep").glob("*_report.json"))]
    for path, precision in sources:
        report = json.loads(path.read_text(encoding="utf-8"))
        arch = "test_resnet_slim128x64" if "_slim128x64_" in path.name else "test_resnet"
        precision = precision or f"{report['weight_bits']}w{report['act_bits']}a"
        point = report["test_metrics"]["point_metrics_pct"]
        runs.append({"architecture": arch, "resolution": int(report["student_resolution"]),
                     "teacher": report["teacher_mode"], "precision": precision,
                     "seed": int(report["random_seed"]),
                     "accuracy": point["accuracy_overall"], "f1_weighted": point["f1_weighted"],
                     "source_report": path.relative_to(ROOT).as_posix()})
    keys = [(r["architecture"], r["resolution"], r["teacher"], r["precision"], r["seed"]) for r in runs]
    check(len(keys) == len(set(keys)), "Duplicate runs")
    check(len(runs) == 2 * 3 * 2 * 4 * 5, f"Expected 240 runs, found {len(runs)}")
    return runs


def summarize(runs):
    groups = defaultdict(list)
    for r in runs:
        groups[(r["architecture"], r["resolution"], r["teacher"], r["precision"])].append(r)
    summary = []
    for (arch, res, teacher, precision), rows in sorted(groups.items()):
        for metric in ("accuracy", "f1_weighted"):
            summary.append({"architecture": arch, "resolution": res, "teacher": teacher,
                            "precision": precision, "metric": metric,
                            **ci95([r[metric] for r in rows])})
    fp32 = {(r["architecture"], r["resolution"], r["teacher"], r["seed"]): r
            for r in runs if r["precision"] == "fp32"}
    paired = []
    for r in runs:
        if r["precision"] == "fp32":
            continue
        base = fp32[(r["architecture"], r["resolution"], r["teacher"], r["seed"])]
        paired.append({**{k: r[k] for k in ("architecture", "resolution", "teacher", "precision", "seed")},
                       "fp32_accuracy": base["accuracy"], "qat_accuracy": r["accuracy"],
                       "delta_accuracy_pp": r["accuracy"] - base["accuracy"],
                       "delta_f1_weighted_pp": r["f1_weighted"] - base["f1_weighted"]})
    return summary, paired


def style():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10.5,
                         "axes.edgecolor": INK_3, "axes.labelcolor": INK_2, "axes.linewidth": 0.8,
                         "xtick.color": INK_2, "ytick.color": INK_2, "text.color": INK,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "svg.fonttype": "none"})


def title(fig, text, subtitle):
    fig.text(0.06, 0.965, text, fontsize=16, weight="bold", ha="left", va="top")
    fig.text(0.06, 0.915, subtitle, fontsize=10.5, color=INK_2, ha="left", va="top")


def save(fig, out, name):
    for ext in ("png", "svg"):
        fig.savefig(out / f"{name}.{ext}", dpi=200, facecolor=SURFACE)
    plt.close(fig)


def plot_accuracy_by_precision(runs, summary, out):
    lookup = {(s["architecture"], s["resolution"], s["teacher"], s["precision"]): s
              for s in summary if s["metric"] == "accuracy"}
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.6), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.78, bottom=0.12, hspace=0.32, wspace=0.08)
    title(fig, "Test accuracy after quantization-aware training",
          "Mean of 5 seeds with 95% t interval; small dots are individual seeds. "
          "Each seed also draws a new patient split.")
    x = np.arange(len(PRECISIONS))
    offsets = {160: -0.16, 192: 0.0, 224: 0.16}
    for row, arch in enumerate(ARCHS):
        for col, teacher in enumerate(TEACHERS):
            ax = axes[row, col]
            for res in RESOLUTIONS:
                stats = [lookup[(arch, res, teacher, p)] for p in PRECISIONS]
                xs = x + offsets[res]
                means = [s["mean"] for s in stats]
                ax.plot(xs, means, color=SERIES[res], linewidth=2, zorder=3)
                ax.errorbar(xs, means, yerr=[[m - s["ci_low"] for m, s in zip(means, stats)],
                                             [s["ci_high"] - m for m, s in zip(means, stats)]],
                            fmt="none", ecolor=SERIES[res], elinewidth=1.4, capsize=3, zorder=3)
                ax.scatter(xs, means, s=46, marker=MARKERS[res], color=SERIES[res],
                           edgecolor=SURFACE, linewidth=1.5, zorder=4)
                for xi, p in zip(xs, PRECISIONS):
                    seeds = [r["accuracy"] for r in runs if (r["architecture"], r["resolution"],
                             r["teacher"], r["precision"]) == (arch, res, teacher, p)]
                    ax.scatter(np.full(len(seeds), xi + 0.05), seeds, s=9, color=SERIES[res],
                               alpha=0.35, linewidth=0, zorder=2)
            ax.set_title(f"{ARCH_LABELS[arch]}  ·  {TEACHER_LABELS[teacher]}", fontsize=11,
                         weight="bold", loc="left", color=INK)
            ax.set_ylim(40, 100)
            ax.set_yticks(range(40, 101, 10))
            ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
            ax.set_xticks(x, [PRECISION_LABELS[p] for p in PRECISIONS])
            ax.tick_params(length=0)
        axes[row, 0].set_ylabel("Test accuracy (%)")
    handles = [Line2D([], [], color=SERIES[r], marker=MARKERS[r], linewidth=2, markersize=7,
                      markeredgecolor=SURFACE) for r in RESOLUTIONS]
    fig.legend(handles, [f"{r} × {r} input" for r in RESOLUTIONS], loc="upper left",
               bbox_to_anchor=(0.055, 0.875), ncol=3, frameon=False, fontsize=10.5)
    fig.text(0.06, 0.04, "Test set: 133 images per seed. 8-bit matches FP32, 6-bit costs about 1 point, "
             "4-bit drops by about 24 points in every configuration.", fontsize=9.5, color=INK_2)
    save(fig, out, "qat_accuracy_by_precision")


def plot_paired_drop(paired, out):
    fig, ax = plt.subplots(figsize=(9, 6.2))
    fig.subplots_adjust(left=0.12, right=0.96, top=0.80, bottom=0.14)
    title(fig, "Accuracy change from FP32 to QAT",
          "Same seed, patient split, student and teacher; 60 pairs per precision\n"
          "(2 students × 3 input sizes × 2 teachers × 5 seeds).")
    rng = np.random.default_rng(0)
    precisions = PRECISIONS[1:]
    for i, p in enumerate(precisions):
        deltas = [r["delta_accuracy_pp"] for r in paired if r["precision"] == p]
        stats = ci95(deltas)
        ax.scatter(i + rng.uniform(-0.17, 0.17, len(deltas)), deltas, s=16, color="#86b6ef",
                   linewidth=0, alpha=0.9, zorder=2)
        ax.plot([i - 0.26, i + 0.26], [stats["mean"]] * 2, color="#184f95", linewidth=2.4, zorder=4)
        ax.errorbar(i, stats["mean"], yerr=[[stats["mean"] - stats["ci_low"]], [stats["ci_high"] - stats["mean"]]],
                    fmt="none", ecolor="#184f95", elinewidth=1.6, capsize=6, zorder=4)
        ax.annotate(f"{stats['mean']:+.1f} pp\n[{stats['ci_low']:+.1f}, {stats['ci_high']:+.1f}]",
                    (i + 0.3, stats["mean"]), va="center", fontsize=10, color=INK)
    ax.axhline(0, color=INK_3, linewidth=1, zorder=1)
    ax.set_xticks(range(len(precisions)), [PRECISION_LABELS[p] for p in precisions])
    ax.set_xlim(-0.5, len(precisions) - 0.2)
    ax.set_ylabel("QAT minus FP32 test accuracy (percentage points)")
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.tick_params(length=0)
    fig.text(0.06, 0.04, "Dark line = mean change with 95% t interval; dots = individual pairs.", fontsize=9.5, color=INK_2)
    save(fig, out, "qat_accuracy_change_vs_fp32")


def load_builds():
    builds = []
    for directory in sorted((ROOT / "results/hw/u250").glob("build_finn_*")):
        name = directory.name.removeprefix("build_finn_")
        model, target = name.split("_u250_")
        topology, bits = model.rsplit("_", 1)
        perf = json.loads((directory / "report/estimate_network_performance.json").read_text())
        res = json.loads((directory / "report/post_synth_resources.json").read_text())["(top)"]
        arch = "test_resnet_slim128x64" if "slim128x64" in topology else "test_resnet"
        resolution = int(topology.rsplit("_trim", 1)[1]) if "_trim" in topology else 224
        used = {"LUT": res["LUT"], "FF": res["FF"], "BRAM": res["BRAM_36K"] + res["BRAM_18K"] / 2, "DSP": res["DSP"]}
        builds.append({"build": directory.name, "architecture": arch, "resolution": resolution,
                       "precision": bits, "target": target,
                       "target_fps": int(target.removesuffix("fps")) if target.endswith("fps") else None,
                       "estimated_fps": perf["estimated_throughput_fps"],
                       **{f"{k}_used": v for k, v in used.items()},
                       **{f"{k}_pct": 100 * v / U250[k] for k, v in used.items()},
                       "legacy_untrimmed_224": "_trim" not in topology})
    check(len(builds) == 17, f"Expected 17 U250 builds, found {len(builds)}")
    return builds


def plot_u250_tradeoff(builds, summary, out):
    accuracy = defaultdict(list)
    for s in summary:
        if s["metric"] == "accuracy":
            accuracy[(s["architecture"], s["resolution"], s["precision"])].append(s["mean"])
    pairs = [b for b in builds if b["target_fps"] in (2000, 5000)]
    fig, axes = plt.subplots(1, 2, figsize=(13, 7.2), sharey=True)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.72, bottom=0.18, wspace=0.06)
    title(fig, "U250: throughput against logic area",
          "Each line joins a model's 2,000 FPS-target build (lower left) to its 5,000 FPS-target build (upper right).\n"
          "Labels give the mean QAT test accuracy of that configuration (5 seeds × 2 teachers).")
    for ax, precision in zip(axes, ("6w6a", "8w8a")):
        models = sorted({(b["architecture"], b["resolution"]) for b in pairs if b["precision"] == precision})
        for arch, res in models:
            pts = sorted([b for b in pairs if (b["architecture"], b["resolution"], b["precision"]) == (arch, res, precision)],
                         key=lambda b: b["target_fps"])
            xs, ys = [b["LUT_pct"] for b in pts], [b["estimated_fps"] for b in pts]
            filled = arch == "test_resnet"
            ax.plot(xs, ys, color=SERIES[res], linewidth=2, linestyle="-" if filled else (0, (4, 2)), zorder=2)
            ax.scatter(xs, ys, s=70, marker=MARKERS[res], zorder=3, linewidth=1.8,
                       color=SERIES[res] if filled else SURFACE, edgecolor=SERIES[res])
            acc = np.mean(accuracy[(arch, res, precision)])
            short = "Slim" if arch != "test_resnet" else "TestResNet"
            ax.annotate(f"{short} {res}  ·  {acc:.1f}%", (xs[-1], ys[-1]), xytext=(8, 0),
                        textcoords="offset points", va="center", fontsize=9.5, color=INK)
        ax.set_title(f"{PRECISION_LABELS[precision]} weights and activations ({precision})", loc="left",
                     fontsize=11, weight="bold", color=INK)
        ax.set_xlim(12, 32)
        ax.set_xticks(range(12, 33, 4))
        ax.set_ylim(0, 4300)
        ax.set_xlabel("LUTs used (% of U250)")
        ax.grid(color=GRID, linewidth=0.8, zorder=0)
        ax.tick_params(length=0)
    axes[0].set_ylabel("Estimated throughput (frames/s)")
    handles = [Line2D([], [], color=SERIES[r], marker=MARKERS[r], linewidth=2, markersize=7) for r in RESOLUTIONS]
    handles += [Line2D([], [], color=INK_2, marker="o", linewidth=2, markersize=7),
                Line2D([], [], color=INK_2, marker="o", markerfacecolor=SURFACE, linewidth=2, linestyle=(0, (4, 2)), markersize=7)]
    fig.legend(handles, [f"{r} × {r} input" for r in RESOLUTIONS] + ["TestResNet", "Slim 128/64"],
               loc="upper left", bbox_to_anchor=(0.055, 0.83), ncol=5, frameon=False, fontsize=10)
    for i, line in enumerate((
            "No model reached 5,000 FPS, so the upper point is each model's throughput ceiling. "
            "Estimates are analytical (100 MHz) and exclude host pre/post-processing.",
            "The 224 builds come from the earlier untrimmed checkpoints; throughput and area depend only on "
            "architecture, input size and bit-width.")):
        fig.text(0.06, 0.065 - 0.032 * i, line, fontsize=9.5, color=INK_2)
    save(fig, out, "u250_throughput_vs_area")


def plot_slim_curve(builds, out):
    pts = sorted([b for b in builds if b["architecture"] == "test_resnet_slim128x64"
                  and b["precision"] == "6w6a" and b["target_fps"]], key=lambda b: b["estimated_fps"])
    fig, ax = plt.subplots(figsize=(10.5, 6.4))
    fig.subplots_adjust(left=0.09, right=0.86, top=0.84, bottom=0.17)
    title(fig, "U250: area cost of throughput, slim 128/64 trim160 6-bit",
          "Same model built for six FINN throughput targets; resources are post-synthesis, as % of the U250.")
    xs = [b["estimated_fps"] for b in pts]
    for resource, color in RESOURCE_COLORS.items():
        ys = [b[f"{resource}_pct"] for b in pts]
        ax.plot(xs, ys, color=color, linewidth=2, zorder=2)
        ax.scatter(xs, ys, s=40, color=color, edgecolor=SURFACE, linewidth=1.5, zorder=3)
        ax.annotate(f"{resource}  {ys[-1]:.1f}%", (xs[-1], ys[-1]), xytext=(10, 0), textcoords="offset points",
                    va="center", fontsize=10, color=INK)
    ax.set_xscale("log")
    ax.set_xticks([100, 200, 500, 1000, 2000, 4000], ["100", "200", "500", "1,000", "2,000", "4,000"])
    ax.minorticks_off()
    ax.set_xlabel("Estimated throughput (frames/s, log scale)")
    ax.set_ylabel("Resources used (% of U250)")
    ax.set_ylim(0, 40)
    ax.grid(color=GRID, linewidth=0.8, zorder=0)
    ax.tick_params(length=0)
    base, top = pts[0], pts[-1]
    fig.text(0.06, 0.07, f"From {base['estimated_fps']:,.0f} to {top['estimated_fps']:,.0f} FPS ({top['estimated_fps'] / base['estimated_fps']:.0f}×), "
             f"LUTs grow {top['LUT_used'] / base['LUT_used']:.2f}× and DSPs {top['DSP_used'] / base['DSP_used']:.0f}×: "
             "most logic and memory is a fixed cost.", fontsize=9.5, color=INK_2)
    fig.text(0.06, 0.035, "Targets 100–2,000 FPS were met; the 5,000 FPS target reached 3,810 FPS. BRAM counts 18K blocks as half a 36K block.",
             fontsize=9.5, color=INK_2)
    save(fig, out, "u250_slim6w6a_area_vs_throughput")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "analysis/qat_sweep")
    out = parser.parse_args().output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    style()
    runs = load_runs()
    summary, paired = summarize(runs)
    builds = load_builds()
    write_csv(out / "runs.csv", runs)
    write_csv(out / "summary.csv", summary)
    write_csv(out / "paired_change_vs_fp32.csv", paired)
    write_csv(out / "u250_builds.csv", builds)
    plot_accuracy_by_precision(runs, summary, out)
    plot_paired_drop(paired, out)
    plot_u250_tradeoff(builds, summary, out)
    plot_slim_curve(builds, out)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()

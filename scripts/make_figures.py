"""Paper figures for the KD/QAT seed sweeps and the U250 builds.

Run from any directory: python scripts/make_figures.py
Requires numpy, scipy and matplotlib. Reads only results/ (fp32_kd_sweep, qat_sweep, hw/u250);
writes PDF + PNG figures and the plotted numbers (data/*.csv) to analysis/figures/.
Figures carry no titles or captions; captions live in analysis/figures/README.md.

One encoding is shared by every figure:
  colour            = input size (160 / 192 / 224 px)
  solid vs hatched  = architecture (Base vs Slim); markers: filled vs hollow
  marker shape      = bit-width (circle 6-bit, triangle 8-bit)
  dark vs light     = teacher (TA ResNet18 vs Direct ResNet50), only where teachers are split
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
from matplotlib.colors import to_rgb
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.stats import t

ROOT = Path(__file__).resolve().parents[1]

ARCHS = ("test_resnet", "test_resnet_slim128x64")
ARCH_NAMES = {"test_resnet": "Base", "test_resnet_slim128x64": "Slim"}
RESOLUTIONS = (160, 192, 224)
TEACHERS = ("r18_ta", "r50_direct")
TEACHER_NAMES = {"r18_ta": "TA (ResNet18)", "r50_direct": "Direct (ResNet50)"}
PRECISIONS = ("fp32", "8w8a", "6w6a", "4w4a")
PRECISION_NAMES = {"fp32": "FP32", "8w8a": "8-bit", "6w6a": "6-bit", "4w4a": "4-bit"}
BIT_MARKERS = {"6w6a": "o", "8w8a": "^"}

# Reference categorical palette (light mode, slots 1-3); validated all-pairs for three series.
RES_COLORS = {160: "#2a78d6", 192: "#eb6834", 224: "#1baf7a"}
INK, INK_2, GRID = "#1a1a1a", "#555555", "#e2e2e2"
HATCH = "//////"
COLUMN, TEXT_WIDTH = 3.5, 7.16  # IEEEtran conference, inches

# Xilinx Alveo U250 device totals.
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


def tint(color, amount=0.55):
    return tuple(c + (1 - c) * amount for c in to_rgb(color))


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------- data

def load_runs():
    """One row per run: architecture, resolution, teacher, precision, seed, metrics."""
    sources = [(p, "fp32") for p in sorted((ROOT / "results/fp32_kd_sweep").glob("*_report.json"))]
    sources += [(p, None) for p in sorted((ROOT / "results/qat_sweep").glob("*_report.json"))]
    runs = []
    for path, precision in sources:
        report = json.loads(path.read_text(encoding="utf-8"))
        point = report["test_metrics"]["point_metrics_pct"]
        runs.append({"architecture": "test_resnet_slim128x64" if "_slim128x64_" in path.name else "test_resnet",
                     "resolution": int(report["student_resolution"]), "teacher": report["teacher_mode"],
                     "precision": precision or f"{report['weight_bits']}w{report['act_bits']}a",
                     "seed": int(report["random_seed"]),
                     "accuracy": point["accuracy_overall"], "f1_weighted": point["f1_weighted"],
                     "source_report": path.relative_to(ROOT).as_posix()})
    keys = {(r["architecture"], r["resolution"], r["teacher"], r["precision"], r["seed"]) for r in runs}
    check(len(keys) == len(runs) == 240, f"Expected 240 unique runs, found {len(runs)}")
    return runs


def summarize(runs, by):
    groups = defaultdict(list)
    for r in runs:
        groups[tuple(r[k] for k in by)].append(r)
    return {key: {m: ci95([r[m] for r in rows]) for m in ("accuracy", "f1_weighted")}
            for key, rows in groups.items()}


def paired_changes(runs):
    fp32 = {(r["architecture"], r["resolution"], r["teacher"], r["seed"]): r for r in runs if r["precision"] == "fp32"}
    return [{**{k: r[k] for k in ("architecture", "resolution", "teacher", "precision", "seed")},
             "delta_accuracy_pp": r["accuracy"] - fp32[(r["architecture"], r["resolution"], r["teacher"], r["seed"])]["accuracy"],
             "delta_f1_weighted_pp": r["f1_weighted"] - fp32[(r["architecture"], r["resolution"], r["teacher"], r["seed"])]["f1_weighted"]}
            for r in runs if r["precision"] != "fp32"]


def load_builds():
    builds = []
    for directory in sorted((ROOT / "results/hw/u250").glob("build_finn_*")):
        model, target = directory.name.removeprefix("build_finn_").split("_u250_")
        topology, bits = model.rsplit("_", 1)
        perf = json.loads((directory / "report/estimate_network_performance.json").read_text())
        res = json.loads((directory / "report/post_synth_resources.json").read_text())["(top)"]
        used = {"LUT": res["LUT"], "FF": res["FF"], "BRAM": res["BRAM_36K"] + res["BRAM_18K"] / 2, "DSP": res["DSP"]}
        builds.append({"build": directory.name,
                       "architecture": "test_resnet_slim128x64" if "slim128x64" in topology else "test_resnet",
                       "resolution": int(topology.rsplit("_trim", 1)[1]) if "_trim" in topology else 224,
                       "precision": bits, "target": target, "throughput_fps": perf["estimated_throughput_fps"],
                       **{f"{k}_used": v for k, v in used.items()},
                       **{f"{k}_pct": 100 * v / U250[k] for k, v in used.items()}})
    check(len(builds) == 17, f"Expected 17 U250 builds, found {len(builds)}")
    return builds


# ---------------------------------------------------------------- style

def style():
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Liberation Serif", "Nimbus Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix", "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5, "legend.fontsize": 7, "axes.linewidth": 0.6, "axes.edgecolor": INK_2,
        "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK, "text.color": INK,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "axes.spines.top": False, "axes.spines.right": False, "hatch.linewidth": 0.6,
        "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 300, "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02, "legend.frameon": False, "legend.handlelength": 1.6,
        "legend.columnspacing": 1.2, "legend.handletextpad": 0.5})


def save(fig, out, name):
    for ext in ("pdf", "png"):
        fig.savefig(out / f"{name}.{ext}")
    plt.close(fig)


def bar(ax, x, stats, width, res, arch, light=False, interval=True, value=True):
    color = tint(RES_COLORS[res]) if light else RES_COLORS[res]
    if arch == "test_resnet":
        ax.bar(x, stats["mean"], width, color=color, edgecolor=color, linewidth=0.6, zorder=2)
    else:
        ax.bar(x, stats["mean"], width, facecolor="white", edgecolor=color, hatch=HATCH, linewidth=0.8, zorder=2)
    if interval:
        ax.errorbar(x, stats["mean"], yerr=[[stats["mean"] - stats["ci_low"]], [stats["ci_high"] - stats["mean"]]],
                    fmt="none", ecolor=INK, elinewidth=0.7, capsize=1.6, capthick=0.7, zorder=3)
    if value:
        top = stats["ci_high"] if interval else stats["mean"]
        ax.annotate(f"{stats['mean']:.1f}", (x, top), xytext=(0, 1.5), textcoords="offset points",
                    ha="center", va="bottom", fontsize=5.5, color=INK)


def legend_handles(teachers=False, bits=False, markers=False):
    handles, labels = [], []
    for res in RESOLUTIONS:
        handles.append(Patch(facecolor=RES_COLORS[res], edgecolor=RES_COLORS[res]))
        labels.append(f"{res} px")
    if markers:
        handles += [Line2D([], [], marker="s", linestyle="", color="#777777", markersize=5),
                    Line2D([], [], marker="s", linestyle="", markerfacecolor="white", markeredgecolor="#777777", markersize=5)]
    else:
        handles += [Patch(facecolor="#777777", edgecolor="#777777"),
                    Patch(facecolor="white", edgecolor="#777777", hatch=HATCH)]
    labels += ["Base", "Slim"]
    if teachers:
        handles += [tuple(Patch(facecolor=RES_COLORS[r]) for r in RESOLUTIONS),
                    tuple(Patch(facecolor=tint(RES_COLORS[r])) for r in RESOLUTIONS)]
        labels += [TEACHER_NAMES[t] for t in TEACHERS]
    if bits:
        handles += [Line2D([], [], marker=BIT_MARKERS[b], linestyle="", color="#777777", markersize=5) for b in ("8w8a", "6w6a")]
        labels += ["8-bit", "6-bit"]
    return handles, labels


def finish_axes(ax, ylabel, step=None):
    ax.set_ylabel(ylabel)
    if step:
        low, high = ax.get_ylim()
        ax.set_yticks(np.arange(low, high + 1e-9, step))
    ax.grid(axis="y", color=GRID, linewidth=0.5, zorder=0)
    ax.tick_params(axis="x", length=0)


# ---------------------------------------------------------------- figures

def fig_fp32_teachers(runs, out, metric, label):
    """FP32 KD sweep: every student/input pair, both teachers."""
    stats = summarize([r for r in runs if r["precision"] == "fp32"], ("architecture", "resolution", "teacher"))
    fig, ax = plt.subplots(figsize=(TEXT_WIDTH, 2.1))
    width, centers = 0.36, []
    for i, (arch, res) in enumerate((a, r) for a in ARCHS for r in RESOLUTIONS):
        center = i + (0.35 if arch != ARCHS[0] else 0)
        centers.append(center)
        for j, teacher in enumerate(TEACHERS):
            bar(ax, center + (j - 0.5) * width, stats[(arch, res, teacher)][metric], width * 0.92, res, arch, light=j == 1)
    ax.set_xticks(centers, [f"{ARCH_NAMES[a]} {r}" for a in ARCHS for r in RESOLUTIONS])
    ax.set_ylim(60, 100)
    finish_axes(ax, label, step=5)
    handles, labels = legend_handles(teachers=True)
    ax.legend(handles, labels, ncol=7, loc="lower center", bbox_to_anchor=(0.5, 1.0),
              handler_map={tuple: HandlerTuple(ndivide=None, pad=0)})
    save(fig, out, f"fp32_teachers_{metric}")


def fig_qat_precision(runs, out, metric, label, precisions, ylim, name):
    """Accuracy for each precision, teachers pooled (5 seeds x 2 teachers)."""
    stats = summarize(runs, ("architecture", "resolution", "precision"))
    fig, ax = plt.subplots(figsize=(TEXT_WIDTH, 2.1))
    width = 0.13
    configs = [(a, r) for a in ARCHS for r in RESOLUTIONS]
    for g, precision in enumerate(precisions):
        for k, (arch, res) in enumerate(configs):
            x = g + (k - 2.5) * width + (0.03 if k >= 3 else -0.03)
            bar(ax, x, stats[(arch, res, precision)][metric], width * 0.9, res, arch)
    ax.set_xticks(range(len(precisions)), [PRECISION_NAMES[p] for p in precisions])
    ax.set_ylim(*ylim)
    finish_axes(ax, label, step=5)
    handles, labels = legend_handles()
    ax.legend(handles, labels, ncol=5, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    save(fig, out, f"{name}_{metric}")


def max_throughput_builds(builds):
    peaks = [b for b in builds if b["target"] == "5000fps"]
    check(len(peaks) == 6, "Expected six 5000fps builds")
    order = {("6w6a", "test_resnet"): 0, ("6w6a", "test_resnet_slim128x64"): 1,
             ("8w8a", "test_resnet"): 2, ("8w8a", "test_resnet_slim128x64"): 3}
    return sorted(peaks, key=lambda b: (order[(b["precision"], b["architecture"])], b["resolution"]))


def fig_u250_throughput(builds, out):
    """Maximum throughput reached by each configuration on the U250."""
    peaks = max_throughput_builds(builds)
    fig, ax = plt.subplots(figsize=(COLUMN, 2.0))
    xs, ticks, gap = [], [], 0
    for i, b in enumerate(peaks):
        if i and b["precision"] != peaks[i - 1]["precision"]:
            gap += 0.6
        x = i + gap
        xs.append(x)
        bar(ax, x, {"mean": b["throughput_fps"]}, 0.7, b["resolution"], b["architecture"], interval=False, value=False)
        ax.text(x, b["throughput_fps"] + 60, f"{b['throughput_fps']:,.0f}", ha="center", va="bottom", fontsize=6.5)
    for precision in ("6w6a", "8w8a"):
        group = [x for x, b in zip(xs, peaks) if b["precision"] == precision]
        ticks.append((np.mean(group), PRECISION_NAMES[precision]))
    ax.set_xticks([p for p, _ in ticks], [n for _, n in ticks])
    ax.set_ylim(0, 4400)
    finish_axes(ax, "Throughput (FPS)")
    handles, labels = legend_handles()
    ax.legend(handles, labels, ncol=5, loc="lower center", bbox_to_anchor=(0.5, 1.0), columnspacing=0.8)
    save(fig, out, "u250_max_throughput")


def fig_u250_accuracy_vs_throughput(builds, runs, out):
    """QAT accuracy (teachers pooled) against the maximum U250 throughput of the same configuration."""
    stats = summarize(runs, ("architecture", "resolution", "precision"))
    fig, ax = plt.subplots(figsize=(COLUMN, 2.3))
    rows = []
    for b in max_throughput_builds(builds):
        s = stats[(b["architecture"], b["resolution"], b["precision"])]["accuracy"]
        color = RES_COLORS[b["resolution"]]
        base = b["architecture"] == "test_resnet"
        ax.scatter(b["throughput_fps"], s["mean"], marker=BIT_MARKERS[b["precision"]], s=34, zorder=3,
                   facecolor=color if base else "white", edgecolor=color, linewidth=1.1)
        rows.append({"build": b["build"], "architecture": b["architecture"], "resolution": b["resolution"],
                     "precision": b["precision"], "throughput_fps": b["throughput_fps"], **s})
    ax.set_xlim(1500, 4200)
    ax.set_ylim(76, 86)
    ax.set_yticks(range(76, 87, 2))
    ax.set_xlabel("Throughput (FPS)")
    ax.set_ylabel("Test accuracy (%)")
    ax.grid(color=GRID, linewidth=0.5, zorder=0)
    handles, labels = legend_handles(bits=True, markers=True)
    ax.legend(handles, labels, ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.0), columnspacing=0.9)
    save(fig, out, "u250_accuracy_vs_throughput")
    return rows


def fig_u250_slim_area(builds, out):
    """Resource use of Slim 160 6-bit across throughput targets."""
    pts = sorted([b for b in builds if b["architecture"] == "test_resnet_slim128x64" and b["precision"] == "6w6a"
                  and b["target"].endswith("fps")], key=lambda b: b["throughput_fps"])
    styles = {"DSP": ("#e87ba4", "D", "-"), "LUT": ("#008300", "o", "-"),
              "BRAM": ("#6250d6", "s", "-"), "FF": ("#e34948", "^", "-")}
    fig, ax = plt.subplots(figsize=(COLUMN, 2.1))
    xs = [b["throughput_fps"] for b in pts]
    for resource, (color, marker, line) in styles.items():
        ys = [b[f"{resource}_pct"] for b in pts]
        ax.plot(xs, ys, color=color, linestyle=line, linewidth=1.2, marker=marker, markersize=3.5,
                markerfacecolor=color, markeredgewidth=0)
        ax.annotate(resource, (xs[-1], ys[-1]), xytext=(4, 0), textcoords="offset points", va="center", fontsize=7)
    ax.set_xscale("log")
    ax.set_xticks([100, 200, 500, 1000, 2000, 4000], ["100", "200", "500", "1k", "2k", "4k"])
    ax.minorticks_off()
    ax.set_xlim(90, 5200)
    ax.set_ylim(0, 36)
    ax.set_xlabel("Throughput (FPS)")
    ax.set_ylabel("U250 resources used (%)")
    ax.grid(color=GRID, linewidth=0.5, zorder=0)
    save(fig, out, "u250_slim160_6bit_area_vs_throughput")
    return [{k: b[k] for k in ("build", "throughput_fps", "LUT_pct", "FF_pct", "BRAM_pct", "DSP_pct")} for b in pts]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "analysis/figures")
    out = parser.parse_args().output_dir.resolve()
    (out / "data").mkdir(parents=True, exist_ok=True)
    style()
    runs, builds = load_runs(), load_builds()
    paired = paired_changes(runs)
    for metric, label in (("accuracy", "Test accuracy (%)"), ("f1_weighted", "Weighted F1 (%)")):
        fig_fp32_teachers(runs, out, metric, label)
        fig_qat_precision(runs, out, metric, label, PRECISIONS[:3], (70, 90), "qat_precision")
        fig_qat_precision(runs, out, metric, label, PRECISIONS, (50, 100), "qat_precision_with_4bit")
    fig_u250_throughput(builds, out)
    acc_fps = fig_u250_accuracy_vs_throughput(builds, runs, out)
    area = fig_u250_slim_area(builds, out)

    summary = []
    for by, scope in ((("architecture", "resolution", "teacher", "precision"), "per_teacher"),
                      (("architecture", "resolution", "precision"), "teachers_pooled")):
        for key, metrics in sorted(summarize(runs, by).items()):
            for metric, s in metrics.items():
                summary.append({"scope": scope, **dict(zip(by, key)), "metric": metric, **s})
    write_csv(out / "data/runs.csv", runs)
    write_csv(out / "data/summary.csv", summary)
    write_csv(out / "data/paired_change_vs_fp32.csv", paired)
    write_csv(out / "data/u250_builds.csv", builds)
    write_csv(out / "data/u250_accuracy_vs_throughput.csv", acc_fps)
    write_csv(out / "data/u250_slim160_6bit_area.csv", area)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()

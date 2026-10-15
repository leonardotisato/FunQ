# KD sweep and U250 results

## Outputs

Analyzed `results/fp32_kd_sweep` (sweep `kd_teacher_sweep_20260703_121750`): **60 FP32 runs**, covering two student architectures, input sizes 160/192/224, two teachers, and seeds 42–46. All 12 combinations contain all five seeds. Analyzed all **17 builds** in `results/hw/u250` (report summaries copied from `build/u250` on the training VM).

Figures built from these tables are in `analysis/figures/` (`scripts/make_figures.py`).

| File | Contents |
| --- | --- |
| `fp32_runs.csv` | 60 individual runs, point metrics, initialization, checkpoint paths and provenance |
| `fp32_summary.csv` | 48 summaries: four metrics × 12 configurations; mean, sample SD, variance, standard error, 95% t interval, sample size and seeds |
| `fp32_paired_teacher_differences.csv` | ResNet50-direct minus ResNet18-assistant differences, paired by seed within each architecture/resolution |
| `reported_bootstrap_intervals.csv` | Original per-run test bootstrap summaries; retained separately from the across-run analysis |
| `u250_estimates.csv` | 17 analytical FPS estimates, latency, bottleneck, inferred clock, driver shapes, target settings and source paths |
| `legacy_qat_runs.csv` | Six older single-run QAT reports outside the seed sweep; seed metadata is absent |
| `data_quality.json` | Coverage checks, missing checkpoint paths, calculation conventions, and SHA-256 hashes of the source reports/drivers |

## FP32 accuracy

Values below are **mean ± sample standard deviation**, in percent / percentage points. Every cell contains five runs. The chart whiskers instead show the nominal 95% t interval of the mean.

| Student | Input | ResNet18 teacher assistant | ResNet50 direct teacher |
| --- | --- | ---: | ---: |
| TestResNet | 160 × 160 | 82.70 ± 2.91 | 80.53 ± 4.29 |
| TestResNet | 192 × 192 | 81.48 ± 1.90 | 83.30 ± 3.23 |
| TestResNet | 224 × 224 | 82.22 ± 3.72 | 84.82 ± 2.78 |
| Slim 128/64 | 160 × 160 | 81.92 ± 2.43 | 81.30 ± 2.51 |
| Slim 128/64 | 192 × 192 | 82.84 ± 2.37 | 81.00 ± 3.38 |
| Slim 128/64 | 224 × 224 | 82.39 ± 2.32 | 83.44 ± 3.22 |

The highest observed mean is TestResNet 224 with ResNet50 direct: 84.82%, with a nominal interval [81.37%, 88.28%]. Teacher ordering changes across configurations. These descriptive rankings do not establish that the top configuration is statistically superior, and test rankings should not become the checkpoint-selection rule for subsequent experiments.

The slim student has 217,876 FP32 parameters versus 373,076 for TestResNet. In this sweep, slim models start from their matching trained TestResNet checkpoint; TestResNet starts from ImageNet. Architecture comparisons therefore include different initialization and training histories.

## Metric and interval definitions

1. Use the actual held-out point estimates in `test_metrics.point_metrics_pct`, evaluated after restoring the checkpoint selected by minimum validation loss. The report's `best_val_f1` is the F1 at that selected epoch, not necessarily the maximum F1 over training. The field `epochs` records the selected epoch plus one, not necessarily the total training duration.
2. Compute the arithmetic mean over the five runs. This gives equal weight to each run; it is not pooled accuracy over all predictions. SD uses `ddof=1`; variance is in squared percentage points.
3. Whiskers use `mean ± t(0.975, n−1) × SD / sqrt(n)`. For five runs the multiplier is approximately 2.776. SD describes run-to-run spread; the t interval describes uncertainty in a run mean under its usual assumptions. These are different quantities.
4. `prepare_dataframes()` in `src/utils/dataset.py` uses `RANDOM_SEED` for both patient-level train/test and train/validation splitting. The training scripts also use that seed for training randomness. The observed spread combines **split variation and training variation**. Repeated holdouts share the same underlying dataset and may overlap, so the t intervals are nominal/descriptive, not guaranteed population-generalization confidence intervals. No significance claims are made.
5. The source reports' bootstrap intervals resample test examples within one trained model. They answer a different question; their bounds are neither averaged nor used as across-run error bars. The bootstrap code resamples images rather than patient clusters, which matters if multiple test images belong to one patient.
6. Weighted recall equals overall accuracy for these single-label multiclass results, and equality was verified in all 60 runs. Plotting it again adds no information. Weighted precision and weighted F1 are separately available.
7. Macro precision/recall/F1 and per-class metrics are stored only as bootstrap means and bounds. Those bootstrap means are not exact original-test point estimates. They are preserved in the bootstrap CSV, without mislabeling them as exact macro/per-class scores or mixing them with weighted metrics. The reports do not contain prediction probabilities for recovering AUROC.
8. Both teachers are fixed checkpoints across seeds. Their actual training-patient membership is not recorded in these sweep reports. Audit the teachers' own training and validation splits before treating all student results as clean held-out performance: a teacher trained on one split may already have seen patients placed in another seed's student test split. The available reports do not establish whether that happened.

## U250 interpretation and future joins

FPS comes directly from `results/hw/u250/<build>/report/estimate_network_performance.json` → `estimated_throughput_fps`. The `2000fps` and `5000fps` suffixes are requested targets, not measured or achieved throughput. This is the analytical performance-report type described in the [FINN documentation](https://finn-dev.readthedocs.io/en/stable/command_line.html#generated-outputs).

All 17 estimates imply a 10 ns / 100 MHz clock, verified by reconciling latency and critical-path cycles, then checking `FPS = 1e9 / (clock_period_ns × max_cycles)`. All 17 builds have an `.xclbin` on the VM (listed with SHA-256 in `results/hw/builds.csv`); its presence does not establish board throughput, timing closure, or numerical correctness.

Examples:

| Build family | Target FPS | Estimated FPS |
| --- | ---: | ---: |
| Slim 128/64 trim160, 6w6a or 8w8a | 5,000 | 3,810.10 |
| TestResNet trim160, 6w6a | 5,000 | 3,546.23 |
| TestResNet trim192, 6w6a | 5,000 | 2,656.89 |
| Legacy TestResNet 224, 6w6a or 8w8a | 5,000 | 1,957.79 |
| TestResNet trim160, 6w6a, naive folding | — | 3.39 |

Generated driver output shapes are `(1, 1, 1, 96)` for TestResNet and `(1, 1, 1, 64)` for slim. They are feature vectors. The board-evaluation code applies the classifier tail on the host (`src/run_test_resnet_board_accuracy.py`). When post-route timing is met, the FINN figure matches the accelerator throughput measured on the board, so the figures report it as throughput. End-to-end FPS also includes preprocessing, data transfers and the host tail.

The four legacy `test_resnet_*` builds accept 224 × 224 inputs, verified from their drivers. They do not carry the `trim224` identity, so input size alone is insufficient to join them to the new trimmed-input sweep. Weight/activation widths and target settings in this inventory are explicitly taken from folder names; exact source checkpoint/export identity still needs verification.

No accuracy–FPS scatter is generated. FP32 means cannot represent quantized model accuracy. The six older QAT reports are single runs, lack seed metadata, and reference older initializations. They provide useful pilot evidence, but not the requested multi-seed QAT means. They are also not automatically linked to the compiled weights merely because names look compatible.

For the eventual scatter, use one point per model configuration and folding configuration, with a vertical QAT accuracy interval. Retain architecture, input preprocessing, input resolution, weight/activation widths, teacher, split protocol, checkpoint/export identity, clock and folding settings. Multiple folding configurations for the same verified numerical model share its accuracy estimate; they are not independent accuracy observations. Re-run inexpensive FINN estimates on new exports if graph structure, optimized widths or hardware configuration changes.

## Recommended quantized sweep initialization

### Continue the existing experiment with matched checkpoints

For each `(architecture, resolution, teacher, seed)`:

1. Retrieve that exact FP32 checkpoint from the HPC run. All 60 paths referenced by the sweep reports are currently absent locally.
2. Instantiate a fresh quantized copy with matching dimensions and copy the trained floating-point weights and BatchNorm state. Reinitialize/calibrate the quantizers using that run's training split only; do not reuse optimizer state or scales from another bit width. Loading a matching FP32 state into a quantized model and reinitializing quantizer parameters is supported by [Brevitas](https://xilinx.github.io/brevitas/v0.13.0/settings.html).
3. Start **8w8a, 6w6a and 4w4a independently from the same corresponding FP32 checkpoint**. For example, the three quantized descendants of `test_resnet_fp32_kd_trim160_r18_ta_seed43_ft.pth` all retain seed43's split and ResNet18 teacher. Slim QAT starts from the matching **slim FP32** checkpoint, which already incorporates the slimming/fine-tuning stage.
4. Keep the teacher, preprocessing, patient split, KD settings, calibration subset and checkpoint-selection rule consistent across bit widths. Select checkpoints using validation data and evaluate the held-out test once after selection. Record the paired change in accuracy/F1 relative to the source FP32 checkpoint.
5. Keep every seed rather than initializing every run from the best-performing seed. A shared best checkpoint would change the experiment to QAT variability conditional on one selected starting model. Averaging FP32 accuracy does not produce an initialization checkpoint, and averaging independently trained weights is not the intended operation.

This protocol estimates the performance and variation of the full FP32→QAT pipeline across the existing split/seed runs. Audit teacher split compatibility first. Never warm-start from a checkpoint trained under one split and then silently change the patient split: that can put previously seen patients into the new test set.

### If the goal is training-seed variance on a fixed test set

Separate `split_seed` (or, preferably, saved patient split manifests) from `train_seed` and `qat_seed`. Choose a fixed held-out patient split that was never used by the teachers or student initialization. Retrain the FP32 repetitions on that split, then quantize each corresponding checkpoint. Existing checkpoints from other splits cannot simply be reclassified as clean fixed-split baselines.

To isolate QAT randomness alone, hold both the FP32 checkpoint and patient split fixed and vary only `qat_seed`. To separate FP32-initialization variation from QAT variation, use multiple QAT seeds per FP32 checkpoint and a hierarchical analysis; do not count those dependent descendants as independent FP32 runs.

### Practical run count

- Complete continuation: 2 architectures × 3 resolutions × 2 teachers × 5 seeds × 3 bit widths = **180 QAT runs**.
- A first stage aligned with explicitly trimmed existing hardware configurations: TestResNet160 6w6a, TestResNet192 6w6a, Slim160 6w6a, Slim160 8w8a × 2 teachers × 5 seeds = **40 QAT runs**. This can populate part of the design-space plot once model/export compatibility is verified.
- The older 4w4a runs both report 63.16% accuracy, versus approximately 80–81% for older 6/8-bit runs. Treat 4-bit as a separate tuning concern. Progressive 8→6→4 fine-tuning can be an additional, explicitly labeled experiment; it changes the initialization protocol and should not silently replace direct FP32→4-bit runs.

The matched sweep is now implemented by `scripts/run_qat_kd_teacher_sweep.py`; see `docs/qat_sweep.md`. Both QAT trainers accept the selected teacher and use unique experiment/teacher/seed artifact names. The reports preserve checkpoint and dataset fingerprints, resolved configuration, split hashes and explicit seed semantics. Per-example predictions, exact macro F1, balanced accuracy and per-class point recall remain useful future reporting additions. Export/build paths must preserve the new run identity when the trained checkpoints are compiled.

## Reproduce and validation

From the repository root:

```text
python scripts/tabulate_kd_u250.py
```

The script requires NumPy and SciPy, reads reports and drivers without importing training code, and writes only the analysis outputs. `--sweep` can select another folder of FP32 report JSONs; multiple supplied sweeps are summarized separately. The default is `results/fp32_kd_sweep`. Everything it reads is tracked in git, so it runs on a checkout without the VM.

Validation performed: all 48 summaries independently recomputed from the source JSONs with Python's standard-library statistics; all exported FPS values matched their source fields exactly; full seed coverage, metric ranges, metadata consistency and recall/accuracy identity checked. Source reports and drivers are fingerprinted for reproducibility. The QAT protocol is a recommendation; training-script changes and new training runs are outside this plotting pass.

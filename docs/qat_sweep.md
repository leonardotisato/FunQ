# Matched FP32 → QAT sweep

`scripts/run_qat_kd_teacher_sweep.py` reads the completed FP32 sweep reports and launches one QAT process at a time. Every QAT run uses the FP32 checkpoint with the same:

- student architecture and slim widths;
- input resolution;
- teacher;
- seed and patient split.

Each precision starts independently from FP32. A 6-bit run never starts from the 8-bit result, and a 4-bit run never starts from the 6-bit result.

## Required files

The default paths are taken from each FP32 JSON report. For `kd_teacher_sweep_20260703_121750`, the expected layout is:

```text
runs/kd_teacher_sweep_20260703_121750/<run>/<report>.json
models/kd_teacher_sweep_20260703_121750/<matching FP32 checkpoints>.pth
models/resnet18_from_resnet50_fp32_kd.pth
models/resnet50_fp32_kd.pth
Data/fundus_data_final.csv
```

The reports are present in this checkout, but the 60 matching FP32 checkpoints and the ResNet50 teacher checkpoint are not. Copy them back from the HPC run before starting. If the checkpoints live elsewhere, pass `--fp32-models-dir` and `--teacher-models-dir`; filenames must remain unchanged.

Run the launcher in the same Python environment used for training, with PyTorch, Hydra, Brevitas and the project dependencies installed.

## Recommended first run: U250 subset

Preview all commands without importing training packages or requiring checkpoints:

```bash
python3 scripts/run_qat_kd_teacher_sweep.py \
  --source-sweep runs/kd_teacher_sweep_20260703_121750 \
  --preset u250 \
  --dry-run
```

The U250 preset creates 40 runs:

| Student | Input | Precision |
| --- | --- | --- |
| TestResNet | 160 | 6w6a |
| TestResNet | 192 | 6w6a |
| Slim 128/64 | 160 | 6w6a |
| Slim 128/64 | 160 | 8w8a |

Each row is evaluated with both teachers and all five seeds.

After copying the checkpoints, validate the entire queue before allocating a GPU:

```bash
python3 scripts/run_qat_kd_teacher_sweep.py \
  --source-sweep runs/kd_teacher_sweep_20260703_121750 \
  --preset u250 \
  --check
```

Start the sequential sweep:

```bash
python3 scripts/run_qat_kd_teacher_sweep.py \
  --source-sweep runs/kd_teacher_sweep_20260703_121750 \
  --preset u250
```

Use the HPC scheduler or terminal multiplexer normally used for long jobs. The launcher itself runs one training process at a time and stops at the first failure.

## Full sweep

This schedules 180 runs: two architectures × three resolutions × two teachers × five seeds × three precisions.

```bash
python3 scripts/run_qat_kd_teacher_sweep.py \
  --source-sweep runs/kd_teacher_sweep_20260703_121750 \
  --preset full
```

The full preset defaults to `8w8a 6w6a 4w4a`. Filters can produce smaller queues:

```bash
python3 scripts/run_qat_kd_teacher_sweep.py \
  --source-sweep runs/kd_teacher_sweep_20260703_121750 \
  --preset full \
  --bits 8 6 \
  --teachers r18_ta \
  --seeds 42 43 \
  --resolutions 160 \
  --check
```

`--batch-size 16` is the default. Changing it changes the run identity, so use a new `--experiment-tag` when comparing batch sizes or other protocols.

## Outputs and resuming

The default output tag is `qat_<source-sweep-folder>`, for example:

```text
models/qat_kd_teacher_sweep_20260703_121750/
runs/qat_kd_teacher_sweep_20260703_121750/
logs/qat_kd_teacher_sweep_20260703_121750/
```

Names include architecture, resolution, precision, teacher and seed, so runs cannot overwrite one another. Each result folder contains the training CSV, resolved Hydra configuration, report, split fingerprints and invocation manifest.

Re-running the same command skips a run only when its report, QAT checkpoint, source FP32 checkpoint, teacher checkpoint, dataset and configuration fingerprints all agree. An interrupted or invalid run stops preflight. After inspecting its log, retrain matching partial runs with:

```bash
python3 scripts/run_qat_kd_teacher_sweep.py \
  --source-sweep runs/kd_teacher_sweep_20260703_121750 \
  --preset u250 \
  --restart-incomplete
```

The launcher refuses outputs from a different invocation under the same experiment tag. Choose a new tag instead:

```bash
python3 scripts/run_qat_kd_teacher_sweep.py \
  --source-sweep runs/kd_teacher_sweep_20260703_121750 \
  --preset u250 \
  --experiment-tag qat_u250_batch32 \
  --batch-size 32
```

A sweep lock prevents two launchers from writing the same experiment concurrently. If a machine dies, confirm no sweep process is running before removing a stale `sweep.lock` file.

## Safeguards

- All selected checkpoints, reports, teachers and the dataset are checked before the first run.
- Model-weight mismatches stop before calibration rather than leaving random layers.
- The selected teacher and FP32 source are saved with content hashes.
- Training must produce a finite validation-selected model and finite test metrics.
- The final QAT checkpoint is hashed; corrupt or replaced artifacts are not treated as complete.
- Inputs and training code are rechecked before every queued run and after every completed process.
- The current seed continues to control both the patient split and training randomness, matching the FP32 sweep. Historical teacher split compatibility still needs a separate audit because it is not encoded in the old reports.

Run the offline orchestration tests with:

```bash
python3 -m unittest discover -s tests -p 'test_qat_sweep.py' -v
```

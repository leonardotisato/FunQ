# Fundus Image Classification on FPGA

4-class fundus-image classification with a small quantized ResNet (`test_resnet` and its
slim 128/64 variant), trained with knowledge distillation, quantized with Brevitas QAT and
deployed with FINN on Alveo U250 and Ultra96-v2-G.

## Layout

| Path | Contents |
| --- | --- |
| `src/` | Training, QAT, export and board-evaluation scripts; `src/finn_build/` holds the FINN build flow |
| `scripts/` | Seed-sweep launchers and the analysis/plot script |
| `config/config.yaml` | Hydra config shared by all training scripts |
| `run.sh` / `run_finn.sh` | Launchers for the training container and the FINN container |
| `results/` | Tracked, small summaries — the source of truth for reported numbers |
| `results/fp32_kd_sweep/` | 60 FP32 KD runs: 2 students × input 160/192/224 × 2 teachers × seeds 42–46 |
| `results/qat_sweep/` | 180 QAT runs: the FP32 grid × 4w4a/6w6a/8w8a |
| `results/hw/` | Per-build FINN reports for U250 and Ultra96; `builds.csv` lists every bitstream |
| `results/legacy/` | Earlier single-seed experiments (teachers, first QAT ladders, Ultra96 notes) |
| `analysis/` | Tables and figures generated from `results/` |
| `report/` | LaTeX report |
| `docs/` | How-tos: `qat_sweep.md`, `finn_builds.md` |
| `tests/` | Unit tests for the sweep helpers (`python -m unittest discover -s tests`) |
| `CHANGELOG.md` | Experiment diary |

## Training VM (dwarf5, `~/EyeGuide`)

Large artifacts live only on the VM and are gitignored:

| Path | Contents |
| --- | --- |
| `Data/` | Images, ground truth, `fundus_data_final.csv`, board test payload |
| `models/` | Checkpoints and ONNX exports; `models/archive/` holds pre-sweep models |
| `runs/` | Raw per-run outputs (reports, training curves, confusion matrices) |
| `logs/` | Per-run stdout logs; `logs/hydra/` holds Hydra run folders |
| `build/` | Full FINN build folders with bitstreams (`u250/`, `ultra96/`) |
| `venv/` | Python environment for training (`source venv/bin/activate`) |
| `experiments-archive/` | Code of abandoned model families (MobileNetV1, ResNet18, custom nets), kept for reference |

After a sweep, copy its `*_report.json` files into `results/` and regenerate the analysis:

```bash
python scripts/plot_kd_u250_results.py   # writes analysis/kd_u250/
```

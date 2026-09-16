"""KD-QAT fine-tuning for the slim test_resnet experiment.

Default artifact:
  models/test_resnet_slim128x64_trim160_6w6a_qat.pth
"""

import csv
import json
import math
import os
import sys
import warnings

warnings.filterwarnings("ignore")

import hydra
import torch
import torch.optim as optim
from brevitas.graph.calibrate import calibration_mode
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from qat_test_resnet_trim import (  # noqa: E402
    BN_FREEZE_EPOCH,
    CALIB_BATCHES,
    KD_ALPHA,
    KD_TEMPERATURE,
    PATIENCE,
    QAT_EPOCHS,
    QAT_LR,
    QAT_WEIGHT_DECAY,
    DualResTrimDataset,
    freeze_bn,
    qat_train_one_epoch,
    qat_validate,
    teacher_test_transform,
    teacher_train_transform,
)
from utils.dataset import FundusClsDataset, prepare_dataframes, trim_fundus_black_border  # noqa: E402
from utils.kd_sweep import build_teacher_model, resolve_experiment_paths, resolve_teacher_spec  # noqa: E402
from utils.qat_sweep import file_sha256, resolve_qat_run, run_provenance, valid_metrics, validate_weight_loading  # noqa: E402
from utils.quant_test_resnet_slim import (  # noqa: E402
    QuantTestResNetSlim,
    load_test_resnet_slim_quant_weights,
    model_tag,
)
from utils.seed import set_seeds  # noqa: E402
from utils.test_resnet_slim import (  # noqa: E402
    DEFAULT_LAYER3_OUT,
    DEFAULT_LAYER4_OUT,
    slim_variant_tag,
)
from utils.training import test  # noqa: E402
from utils.transforms import make_strong_train_transform, make_test_transform  # noqa: E402


DEFAULT_STUDENT_RESOLUTION = 160
DEFAULT_QAT_ARTIFACT = "test_resnet_slim128x64_trim160_6w6a_qat.pth"
QAT_SWEEP_PROTOCOL_VERSION = 1


def resolve_slim_qat_run(
    student_resolution: int,
    weight_bits: int,
    act_bits: int,
    layer3_out: int = DEFAULT_LAYER3_OUT,
    layer4_out: int = DEFAULT_LAYER4_OUT,
    teacher_mode=None,
    seed=None,
    experiment_tag=None,
):
    variant = slim_variant_tag(layer3_out, layer4_out)
    return resolve_qat_run(f"test_resnet_{variant}", student_resolution, weight_bits, act_bits,
                           teacher_mode, seed, experiment_tag)


@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    set_seeds(cfg.RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    weight_bits = int(OmegaConf.select(cfg, "weight_bits", default=6))
    act_bits = int(OmegaConf.select(cfg, "act_bits", default=6))
    student_resolution = int(
        OmegaConf.select(cfg, "student_resolution", default=DEFAULT_STUDENT_RESOLUTION)
    )
    layer3_out = int(OmegaConf.select(cfg, "slim_layer3_out", default=DEFAULT_LAYER3_OUT))
    layer4_out = int(OmegaConf.select(cfg, "slim_layer4_out", default=DEFAULT_LAYER4_OUT))
    student_test_transform = make_test_transform(student_resolution)
    student_train_transform = make_strong_train_transform(student_resolution)
    paths = resolve_experiment_paths(cfg)
    teacher_spec = resolve_teacher_spec(cfg, paths)

    run_cfg = resolve_slim_qat_run(
        student_resolution=student_resolution,
        weight_bits=weight_bits,
        act_bits=act_bits,
        layer3_out=layer3_out,
        layer4_out=layer4_out,
        teacher_mode=teacher_spec["mode"],
        seed=int(cfg.RANDOM_SEED),
        experiment_tag=paths.experiment_tag,
    )

    teacher_path = teacher_spec["checkpoint_path"]
    student_init_checkpoint = OmegaConf.select(cfg, "warm_start_checkpoint", default=None)
    if paths.experiment_tag and not student_init_checkpoint:
        raise ValueError("A QAT sweep requires an explicit matching warm_start_checkpoint")
    student_init_checkpoint = student_init_checkpoint or os.path.join(cfg.models_dir, run_cfg["fp32_checkpoint_name"])
    for required in (teacher_path, student_init_checkpoint):
        if not os.path.isfile(required):
            raise FileNotFoundError(f"Required checkpoint not found: {required}")

    results_dir = os.path.join(paths.results_root, run_cfg["results_dir_name"])
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(paths.models_dir, exist_ok=True)
    OmegaConf.save(cfg, os.path.join(results_dir, "resolved_config.yaml"), resolve=True)

    train_df, val_df, test_df = prepare_dataframes(cfg)
    provenance = run_provenance(cfg, teacher_spec, student_init_checkpoint, train_df, val_df, test_df)

    train_dataset = DualResTrimDataset(
        train_df,
        student_transform=student_train_transform,
        teacher_transform=teacher_train_transform,
    )
    val_dataset = DualResTrimDataset(
        val_df,
        student_transform=student_test_transform,
        teacher_transform=teacher_test_transform,
    )
    test_dataset = FundusClsDataset(
        test_df,
        train=False,
        transform=student_test_transform,
        preprocess=trim_fundus_black_border,
    )
    calib_dataset = FundusClsDataset(
        train_df,
        train=False,
        transform=student_test_transform,
        preprocess=trim_fundus_black_border,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=4, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=4, pin_memory=True
    )
    calib_loader = DataLoader(
        calib_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=4, pin_memory=True
    )

    teacher = build_teacher_model(teacher_spec["arch"], nr_classes=cfg.nr_classes)
    teacher.load_state_dict(torch.load(teacher_path, map_location="cpu"))
    teacher.to(device)
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False

    print("\n" + "=" * 50)
    print(f"Creating QuantTestResNetSlim [{run_cfg['bit_tag']}] for {run_cfg['run_tag']}")
    print("=" * 50)
    student = QuantTestResNetSlim(
        nr_classes=cfg.nr_classes,
        weight_bit_width=weight_bits,
        act_bit_width=act_bits,
        layer3_out=layer3_out,
        layer4_out=layer4_out,
    )
    n_params = sum(p.numel() for p in student.parameters())
    print(f"Student parameters: {n_params:,}")

    print(f"\nLoading slim student init weights from: {student_init_checkpoint}")
    missing, unexpected = load_test_resnet_slim_quant_weights(student, student_init_checkpoint)
    validate_weight_loading(missing, unexpected)
    print("Weight loading OK: all model weights loaded.")

    student.to(device)

    print("\n" + "=" * 50)
    print(f"Calibrating quantizer scales ({CALIB_BATCHES} batches) ...")
    print("=" * 50)
    student.eval()
    with calibration_mode(student):
        for batch_idx, (inputs, _) in enumerate(calib_loader):
            if batch_idx >= CALIB_BATCHES:
                break
            with torch.no_grad():
                student(inputs.to(device))
            if (batch_idx + 1) % 25 == 0:
                print(f"  Calibration batch {batch_idx + 1}/{CALIB_BATCHES}")
    print("Calibration complete.")

    print("\n" + "=" * 50)
    print(
        "QAT fine-tuning: "
        f"{QAT_EPOCHS} epochs, LR={QAT_LR}, patience={PATIENCE}, BN freeze={BN_FREEZE_EPOCH}"
    )
    print(
        f"Configuration: slim {run_cfg['variant']}, trim black border -> "
        f"{student_resolution} student"
    )
    print("=" * 50)

    optimizer = optim.Adam(student.parameters(), lr=QAT_LR, weight_decay=QAT_WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=QAT_EPOCHS)

    best_val_loss = float("inf")
    best_val_f1 = -1.0
    best_state = None
    patience_counter = 0
    best_epoch = -1

    logname = os.path.join(results_dir, run_cfg["log_name"])
    with open(logname, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch", "train_loss", "train_acc", "train_f1", "val_loss",
            "val_acc", "val_f1", "val_prec", "val_rec", "best_epoch",
        ])

    for epoch in tqdm(range(QAT_EPOCHS), desc=f"KD-QAT {run_cfg['run_tag']}"):
        train_loss, train_acc, train_f1 = qat_train_one_epoch(
            student, teacher, train_loader, optimizer, device, epoch
        )
        val_loss, val_acc, val_f1, val_prec, val_rec = qat_validate(
            student, teacher, val_loader, KD_TEMPERATURE, KD_ALPHA, device
        )
        scheduler.step()

        if epoch >= BN_FREEZE_EPOCH:
            freeze_bn(student)

        print(
            f"\nEpoch {epoch}: train_loss={train_loss:.4f} train_f1={train_f1:.2f} | "
            f"val_loss={val_loss:.4f} val_f1={val_f1:.2f} val_acc={val_acc:.2f}"
        )

        with open(logname, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch, train_loss, train_acc, train_f1, val_loss, val_acc,
                val_f1, val_prec, val_rec, best_epoch,
            ])

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_f1 = val_f1
            best_state = {k: v.cpu().clone() for k, v in student.state_dict().items()}
            best_epoch = epoch
            patience_counter = 0
            print(f"  -> New best val loss: {val_loss:.4f} (val F1: {val_f1:.2f}%, epoch {epoch})")
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            print(f"Early stopping at epoch {epoch} (patience={PATIENCE})")
            break

    if best_state is None or not math.isfinite(best_val_loss):
        raise RuntimeError("QAT did not produce a finite validation-selected model")
    if best_state is not None:
        student.load_state_dict(best_state)
        print(
            f"\nRestored best model from epoch {best_epoch} "
            f"(val loss: {best_val_loss:.4f}, val F1: {best_val_f1:.2f}%)"
        )
    student.to(device)

    qat_ckpt_path = os.path.join(paths.models_dir, run_cfg["checkpoint_name"])
    torch.save(student.state_dict(), qat_ckpt_path)
    print(f"QAT checkpoint saved -> {qat_ckpt_path}")

    print("\n" + "=" * 50)
    print("Evaluating on test set ...")
    print("=" * 50)
    test_metrics = test(
        model=student,
        test_loader=test_loader,
        device=device,
        model_type=run_cfg["model_type"],
        bootstrap=True,
        savedir=results_dir,
    )
    print(f"QAT test metrics: {test_metrics}")

    report = {
        "weight_bits": weight_bits,
        "act_bits": act_bits,
        "variant": run_cfg["variant"],
        "layer3_out": layer3_out,
        "layer4_out": layer4_out,
        "n_params": n_params,
        "epochs": best_epoch + 1,
        "best_val_f1": round(best_val_f1, 4),
        "best_val_loss": round(best_val_loss, 4),
        "checkpoint": qat_ckpt_path,
        "checkpoint_sha256": file_sha256(qat_ckpt_path),
        "student_init_checkpoint": student_init_checkpoint,
        "student_init_mode": "fp32_checkpoint",
        "teacher": f"{teacher_spec['checkpoint_name']} (512x512 full-image strong train / test eval)",
        "student_resolution": student_resolution,
        "teacher_resolution": 512,
        "input_size": [1, 3, student_resolution, student_resolution],
        "kd_temperature": KD_TEMPERATURE,
        "kd_alpha": KD_ALPHA,
        "test_metrics": test_metrics,
        "qat_hyperparameters": {"lr": QAT_LR, "max_epochs": QAT_EPOCHS,
                                "weight_decay": QAT_WEIGHT_DECAY, "patience": PATIENCE,
                                "calibration_batches": CALIB_BATCHES, "bn_freeze_epoch": BN_FREEZE_EPOCH},
        **provenance,
    }
    if not valid_metrics(report):
        raise RuntimeError("QAT evaluation produced invalid metrics; completion report was not saved")
    report_path = os.path.join(results_dir, run_cfg["report_name"])
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved -> {report_path}")


if __name__ == "__main__":
    main()

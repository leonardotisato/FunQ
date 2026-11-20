"""Fine-tune the slim FP32 test_resnet experiment.

Default artifact:
  models/test_resnet_slim128x64_fp32_kd_trim160_ft.pth
"""

import csv
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")

import hydra
import timm
import torch
import torch.optim as optim
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from train_test_resnet_trim import (  # noqa: E402
    EPOCHS,
    KD_ALPHA,
    KD_TEMPERATURE,
    LR,
    MODEL_NAME,
    PATIENCE,
    WEIGHT_DECAY,
    DualResTrimDataset,
    teacher_test_transform,
    teacher_train_transform,
    train_one_epoch,
    validate,
)
from utils.dataset import FundusClsDataset, prepare_dataframes, trim_fundus_black_border  # noqa: E402
from utils.kd_sweep import (  # noqa: E402
    build_teacher_model,
    resolve_experiment_paths,
    resolve_slim_fp32_run,
    resolve_teacher_spec,
    resolve_trim_fp32_run,
)
from utils.seed import set_seeds  # noqa: E402
from utils.test_resnet_slim import (  # noqa: E402
    DEFAULT_LAYER3_OUT,
    DEFAULT_LAYER4_OUT,
    TestResNetSlim,
    load_test_resnet_slim_weights,
)
from utils.training import test  # noqa: E402
from utils.transforms import make_strong_train_transform, make_test_transform  # noqa: E402


DEFAULT_STUDENT_RESOLUTION = 160
DEFAULT_FP32_ARTIFACT = "test_resnet_slim128x64_fp32_kd_trim160_ft.pth"


def resolve_student_init_state(cfg, run_cfg, student_resolution, paths, teacher_spec):
    warm_start = OmegaConf.select(cfg, "warm_start_checkpoint", default=None)
    if warm_start is not None:
        if not os.path.exists(warm_start):
            raise FileNotFoundError(f"warm_start_checkpoint not found: {warm_start}")
        return torch.load(warm_start, map_location="cpu"), warm_start, "warm_start_checkpoint"

    if paths.experiment_tag is not None:
        matching_trim_run = resolve_trim_fp32_run(
            student_resolution=student_resolution,
            teacher_mode=teacher_spec["mode"],
            seed=int(cfg.RANDOM_SEED),
            experiment_tag=paths.experiment_tag,
        )
        matching_trim_checkpoint = os.path.join(
            paths.models_dir,
            matching_trim_run["checkpoint_name"],
        )
        if os.path.exists(matching_trim_checkpoint):
            return (
                torch.load(matching_trim_checkpoint, map_location="cpu"),
                matching_trim_checkpoint,
                "matching_trim_fp32_checkpoint",
            )

    trim_checkpoint = os.path.join(
        cfg.models_dir,
        f"test_resnet_fp32_kd_{run_cfg['trim_tag']}_ft.pth",
    )
    if os.path.exists(trim_checkpoint):
        return torch.load(trim_checkpoint, map_location="cpu"), trim_checkpoint, "trim_fp32_checkpoint"

    print("No trim FP32 checkpoint found; falling back to timm ImageNet pretrained init.")
    base_model = timm.create_model(
        MODEL_NAME,
        pretrained=True,
        num_classes=cfg.nr_classes,
    )
    return base_model.state_dict(), MODEL_NAME, "timm_imagenet"


@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    set_seeds(cfg.RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    student_resolution = int(
        OmegaConf.select(cfg, "student_resolution", default=DEFAULT_STUDENT_RESOLUTION)
    )
    layer3_out = int(OmegaConf.select(cfg, "slim_layer3_out", default=DEFAULT_LAYER3_OUT))
    layer4_out = int(OmegaConf.select(cfg, "slim_layer4_out", default=DEFAULT_LAYER4_OUT))
    paths = resolve_experiment_paths(cfg)
    teacher_spec = resolve_teacher_spec(cfg, paths)
    student_test_transform = make_test_transform(student_resolution)
    student_train_transform = make_strong_train_transform(student_resolution)
    run_cfg = resolve_slim_fp32_run(
        student_resolution=student_resolution,
        layer3_out=layer3_out,
        layer4_out=layer4_out,
        teacher_mode=teacher_spec["mode"],
        seed=int(cfg.RANDOM_SEED),
        experiment_tag=paths.experiment_tag,
    )

    results_dir = os.path.join(paths.results_root, run_cfg["results_dir_name"])
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(paths.models_dir, exist_ok=True)

    train_df, val_df, test_df = prepare_dataframes(cfg)

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

    train_loader = DataLoader(
        train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=4, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=4, pin_memory=True
    )

    teacher_path = teacher_spec["checkpoint_path"]
    if not os.path.exists(teacher_path):
        print(f"[ERROR] Teacher checkpoint not found: {teacher_path}")
        return

    print(f"Teacher mode: {teacher_spec['mode']} ({teacher_spec['description']})")
    print(f"Loading teacher from: {teacher_path}")
    teacher = build_teacher_model(teacher_spec["arch"], cfg.nr_classes)
    teacher.load_state_dict(torch.load(teacher_path, map_location="cpu"))
    teacher.to(device)
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False

    print("\n" + "=" * 50)
    print(f"Creating FP32 TestResNetSlim {run_cfg['variant']}")
    print("=" * 50)
    model = TestResNetSlim(
        nr_classes=cfg.nr_classes,
        layer3_out=layer3_out,
        layer4_out=layer4_out,
    )
    init_state, init_source, init_mode = resolve_student_init_state(
        cfg, run_cfg, student_resolution, paths, teacher_spec
    )
    _, _, load_report = load_test_resnet_slim_weights(model, init_state)
    print(f"Init source: {init_source} ({init_mode})")
    print(
        "Weight transfer: "
        f"exact={len(load_report['exact'])}, sliced={len(load_report['sliced'])}, "
        f"missing={len(load_report['missing'])}, skipped={len(load_report['skipped'])}"
    )

    model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Student parameters: {n_params:,}")
    print(f"Training: {EPOCHS} epochs, LR={LR}, patience={PATIENCE}")
    print(
        f"Configuration: trim black border -> {student_resolution} student, "
        f"channels layer3={layer3_out}, layer4={layer4_out}, "
        f"teacher={teacher_spec['mode']}."
    )

    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

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

    for epoch in tqdm(range(EPOCHS), desc=f"FP32 KD {run_cfg['run_tag']} Epochs"):
        train_loss, train_acc, train_f1 = train_one_epoch(
            model, teacher, train_loader, optimizer, device
        )
        val_loss, val_acc, val_f1, val_prec, val_rec = validate(
            model, teacher, val_loader, KD_TEMPERATURE, KD_ALPHA, device
        )
        scheduler.step()

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
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            patience_counter = 0
            print(f"  -> New best val loss: {val_loss:.4f} (val F1: {val_f1:.2f}%, epoch {epoch})")
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            print(f"Early stopping at epoch {epoch} (patience={PATIENCE})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
        print(
            f"\nRestored best model from epoch {best_epoch} "
            f"(val loss: {best_val_loss:.4f}, val F1: {best_val_f1:.2f}%)"
        )
    model.to(device)

    ckpt_path = os.path.join(paths.models_dir, run_cfg["checkpoint_name"])
    torch.save(model.state_dict(), ckpt_path)
    print(f"Checkpoint saved -> {ckpt_path}")

    print("\n" + "=" * 50)
    print("Evaluating on test set ...")
    print("=" * 50)
    test_metrics = test(
        model=model,
        test_loader=test_loader,
        device=device,
        model_type=run_cfg["model_type"],
        bootstrap=True,
        savedir=results_dir,
        class_names=OmegaConf.select(cfg, "class_names", default=None),
    )
    print(f"Test metrics: {test_metrics}")

    report = {
        "model": "TestResNetSlim",
        "variant": run_cfg["variant"],
        "layer3_out": layer3_out,
        "layer4_out": layer4_out,
        "n_params": n_params,
        "epochs": best_epoch + 1,
        "best_val_f1": round(best_val_f1, 4),
        "best_val_loss": round(best_val_loss, 4),
        "checkpoint": ckpt_path,
        "student_init": init_mode,
        "init_checkpoint": init_source,
        "weight_transfer": load_report,
        "teacher_mode": teacher_spec["mode"],
        "teacher_arch": teacher_spec["arch"],
        "teacher_checkpoint": teacher_path,
        "teacher": f"{teacher_spec['checkpoint_name']} (512x512 full-image strong train / test eval)",
        "dataset": OmegaConf.select(cfg, "dataset_name", default="uwf"),
        "experiment_tag": paths.experiment_tag,
        "random_seed": int(cfg.RANDOM_SEED),
        "student_resolution": student_resolution,
        "teacher_resolution": 512,
        "input_size": [1, 3, student_resolution, student_resolution],
        "kd_temperature": KD_TEMPERATURE,
        "kd_alpha": KD_ALPHA,
        "test_metrics": test_metrics,
    }
    report_path = os.path.join(results_dir, run_cfg["report_name"])
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved -> {report_path}")


if __name__ == "__main__":
    main()

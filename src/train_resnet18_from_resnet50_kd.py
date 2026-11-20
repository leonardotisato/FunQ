"""
Train a new ResNet18 student by distilling the existing ResNet50 KD model.

This script mirrors the augmentation family and KD weighting used to produce
`resnet50_fp32_kd.pth` in `main.py`, while reducing the student architecture
back to ResNet18.
"""

import csv
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(__file__))

from utils.dataset import FundusClsDataset, PairedFundusFullDataset, prepare_dataframes
from utils.losses import kd_loss
from utils.model import ResNet18Classifier, ResNet50Classifier
from utils.seed import set_seeds
from utils.training import test, train_knowledge_distillation
from utils.transforms import make_strong_train_transform, make_test_transform
from utils.visualization import save_metrics_and_plot


train_transform_class = make_strong_train_transform(512)
test_transform_class = make_test_transform(512)


TEACHER_CHECKPOINT = "resnet50_fp32_kd.pth"
STUDENT_CHECKPOINT = "resnet18_from_resnet50_fp32_kd.pth"
SUMMARY_JSON = "test_summary_resnet18_from_resnet50_kd.json"
SUMMARY_CSV = "summary_metrics_resnet18_from_resnet50_kd.csv"
REPORT_JSON = "train_resnet18_from_resnet50_kd_report.json"
SOFT_TARGET_LOSS_WEIGHT = 0.75
CE_LOSS_WEIGHT = 0.25


def convert_dict(d):
    new_dict = {}
    for key, value in d.items():
        if isinstance(key, np.integer):
            key = int(key)

        if isinstance(value, np.integer):
            value = int(value)
        elif isinstance(value, np.floating):
            value = float(value)
        elif isinstance(value, tuple):
            value = tuple(
                float(v) if isinstance(v, (np.floating, float)) else int(v) if isinstance(v, np.integer) else v
                for v in value
            )
        elif isinstance(value, dict):
            value = convert_dict(value)

        new_dict[key] = value

    return new_dict


@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    set_seeds(cfg.RANDOM_SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    os.makedirs(cfg.results_dir, exist_ok=True)
    os.makedirs(cfg.models_dir, exist_ok=True)

    train_df, val_df, test_df = prepare_dataframes(cfg)

    train_dataset_kd = PairedFundusFullDataset(
        train_df,
        teacher_transform=train_transform_class,
        student_transform=train_transform_class,
        teacher_train=True,
        student_train=True,
    )
    val_dataset_kd = PairedFundusFullDataset(
        val_df,
        teacher_transform=test_transform_class,
        student_transform=test_transform_class,
        teacher_train=False,
        student_train=False,
    )
    test_dataset = FundusClsDataset(
        test_df,
        train=False,
        transform=test_transform_class,
    )

    train_loader = DataLoader(
        train_dataset_kd,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset_kd,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # UWF: the ResNet50 distilled from the lesion-centred model; REFUGE: a supervised ResNet50.
    teacher_checkpoint = OmegaConf.select(cfg, "ta_teacher_checkpoint", default=TEACHER_CHECKPOINT)
    teacher_path = os.path.join(cfg.models_dir, teacher_checkpoint)
    if not os.path.exists(teacher_path):
        print(f"[ERROR] Teacher checkpoint not found: {teacher_path}")
        return

    teacher = ResNet50Classifier(
        nr_classes=cfg.nr_classes,
        dropout=0.5,
        pretrained=False,
    )
    teacher.load_state_dict(torch.load(teacher_path, map_location="cpu"))
    teacher.to(device)
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False

    student = ResNet18Classifier(
        nr_classes=cfg.nr_classes,
        dropout=0.5,
        pretrained=True,
    )

    print("\n================ TRAIN RESNET18 FROM RESNET50 KD =================")
    print(f"Teacher checkpoint: {teacher_path}")
    print("Augmentation family: full-image 512 strong train / test-time eval")
    print(
        f"KD weights: soft={SOFT_TARGET_LOSS_WEIGHT}, ce={CE_LOSS_WEIGHT}, "
        f"T={cfg.temperature}"
    )

    exp_name = "resnet18_from_resnet50_kd"
    student, metrics_skd, plots_skd = train_knowledge_distillation(
        teacher=teacher,
        student=student,
        train_loader=train_loader,
        valid_loader=val_loader,
        epochs=cfg.n_epochs,
        learning_rate=cfg.student_lr,
        weight_decay=cfg.weight_decay,
        T=cfg.temperature,
        kd_loss=kd_loss,
        soft_target_loss_weight=SOFT_TARGET_LOSS_WEIGHT,
        ce_loss_weight=CE_LOSS_WEIGHT,
        device=device,
        exp_name=exp_name,
        log_dir=cfg.results_dir,
    )

    save_metrics_and_plot(
        name=exp_name,
        train_losses=plots_skd["train_losses"],
        val_losses=plots_skd["val_losses"],
        val_f1s=plots_skd["val_f1s"],
        results_dir=cfg.results_dir,
    )

    student_ckpt_path = os.path.join(cfg.models_dir, STUDENT_CHECKPOINT)
    torch.save(student.state_dict(), student_ckpt_path)
    print(f"Student checkpoint saved -> {student_ckpt_path}")

    summary_csv_path = os.path.join(cfg.results_dir, SUMMARY_CSV)
    with open(summary_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "accuracy", "f1", "precision", "recall"])
        writer.writerow(
            [
                "resnet18_from_resnet50_kd",
                metrics_skd["accuracy"],
                metrics_skd["f1"],
                metrics_skd["precision"],
                metrics_skd["recall"],
            ]
        )

    print("\n================ TESTING RESNET18 FROM RESNET50 KD =================")
    test_metrics = test(
        model=student,
        test_loader=test_loader,
        device=device,
        model_type="resnet18_from_resnet50_kd",
        bootstrap=True,
        savedir=cfg.results_dir,
        class_names=OmegaConf.select(cfg, "class_names", default=None),
    )
    print(f"Test metrics: {test_metrics}")

    test_summary_path = os.path.join(cfg.results_dir, SUMMARY_JSON)
    with open(test_summary_path, "w") as fp:
        json.dump(convert_dict(test_metrics), fp)

    report = {
        "dataset": OmegaConf.select(cfg, "dataset_name", default="uwf"),
        "random_seed": int(cfg.RANDOM_SEED),
        "teacher_checkpoint": teacher_checkpoint,
        "student_checkpoint": STUDENT_CHECKPOINT,
        "soft_target_loss_weight": SOFT_TARGET_LOSS_WEIGHT,
        "ce_loss_weight": CE_LOSS_WEIGHT,
        "temperature": cfg.temperature,
        "train_dataset_type": "PairedFundusFullDataset",
        "val_dataset_type": "PairedFundusFullDataset",
        "test_dataset_type": "FundusClsDataset",
        "train_transform": "utils.transforms.make_strong_train_transform(512)",
        "eval_transform": "utils.transforms.make_test_transform(512)",
        "student_architecture": "ResNet18Classifier",
        "teacher_architecture": "ResNet50Classifier",
    }
    report_path = os.path.join(cfg.results_dir, REPORT_JSON)
    with open(report_path, "w") as fp:
        json.dump(report, fp, indent=2)
    print(f"Report saved -> {report_path}")


if __name__ == "__main__":
    main()

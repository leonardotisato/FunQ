"""
Fine-tune a trimmed-input FP32 test_resnet student from ImageNet weights.

This keeps the teacher on its established 512x512 full-image domain and changes
only the student input path:

- trim the non-informative black border with a small safety pad
- resize student inputs to ``++student_resolution`` (default: 192)
- initialize the student from ImageNet pretrained timm weights
"""

import csv
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")

import hydra
import numpy as np
import timm
import torch
import torch.nn.functional as F
import torch.optim as optim
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as tv_transforms
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from utils.dataset import prepare_dataframes, resolve_split_seed, safe_pil_read, trim_fundus_black_border
from utils.generals import progress_bar
from utils.kd_sweep import (
    build_teacher_model,
    resolve_experiment_paths,
    resolve_teacher_spec,
    resolve_trim_fp32_run,
)
from utils.seed import set_seeds
from utils.training import test
from utils.transforms import make_strong_train_transform, make_test_transform


teacher_train_transform = make_strong_train_transform(512)
teacher_test_transform = make_test_transform(512)


LR = 1e-4
EPOCHS = 200
WEIGHT_DECAY = 1e-4
PATIENCE = 50
MODEL_NAME = "test_resnet.r160_in1k"
KD_TEMPERATURE = 3.0
KD_ALPHA = 0.25


class DualResTrimDataset(Dataset):
    """Return (student_image, teacher_image, label) for a trimmed-input student."""

    def __init__(self, data_csv, student_transform, teacher_transform):
        self.data_csv = data_csv
        self.student_transform = student_transform
        self.teacher_transform = teacher_transform

    def __len__(self):
        return len(self.data_csv)

    def __getitem__(self, idx):
        label = self.data_csv.iloc[idx]["label"]
        img_path = str(self.data_csv.iloc[idx]["image"]).strip()
        img = safe_pil_read(img_path)
        img_np = np.array(img)

        img_student = trim_fundus_black_border(img_np)

        aug_s = self.student_transform(image=img_student)
        img_s = tv_transforms.ToTensor()(np.float32(aug_s["image"]))

        aug_t = self.teacher_transform(image=img_np)
        img_t = tv_transforms.ToTensor()(np.float32(aug_t["image"]))

        label = torch.tensor(label, dtype=torch.long)
        return img_s, img_t, label


def kd_loss(student_logits, teacher_logits, labels, temperature, alpha):
    ce = F.cross_entropy(student_logits, labels)
    student_soft = F.log_softmax(student_logits / temperature, dim=1)
    teacher_soft = F.softmax(teacher_logits / temperature, dim=1)
    kl = F.kl_div(student_soft, teacher_soft, reduction="batchmean") * (temperature**2)
    return alpha * ce + (1 - alpha) * kl


def train_one_epoch(student, teacher, train_loader, optimizer, device):
    student.train()
    teacher.eval()

    running_loss = 0.0
    correct = 0
    total = 0
    all_labels = []
    all_preds = []

    for batch_idx, (inputs_s, inputs_t, labels) in enumerate(train_loader):
        inputs_s = inputs_s.to(device)
        inputs_t = inputs_t.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()

        student_out = student(inputs_s)
        with torch.no_grad():
            teacher_out = teacher(inputs_t)

        loss = kd_loss(student_out, teacher_out, labels, KD_TEMPERATURE, KD_ALPHA)
        loss.backward()
        clip_grad_norm_(student.parameters(), 5.0)
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(student_out.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(predicted.cpu().numpy())

        progress_bar(
            batch_idx,
            len(train_loader),
            "Train Loss: %.3f | Acc: %.3f%% (%d/%d)"
            % (running_loss / (batch_idx + 1), 100.0 * correct / total, correct, total),
        )

    avg_loss = running_loss / len(train_loader)
    acc = 100.0 * correct / total
    f1 = f1_score(all_labels, all_preds, average="weighted") * 100.0
    return avg_loss, acc, f1


def validate(student, teacher, val_loader, temperature, alpha, device):
    student.eval()
    teacher.eval()
    val_loss_sum = 0.0
    correct = 0
    total = 0
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for batch_idx, (inputs_s, inputs_t, labels) in enumerate(val_loader):
            inputs_s = inputs_s.to(device)
            inputs_t = inputs_t.to(device)
            labels = labels.to(device)

            student_logits = student(inputs_s)
            teacher_logits = teacher(inputs_t)

            loss = kd_loss(student_logits, teacher_logits, labels, temperature, alpha)
            val_loss_sum += loss.item()

            _, predicted = torch.max(student_logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

            progress_bar(
                batch_idx,
                len(val_loader),
                "Val Loss: %.3f | Acc: %.3f%% (%d/%d)"
                % (val_loss_sum / (batch_idx + 1), 100.0 * correct / total, correct, total),
            )

    avg_loss = val_loss_sum / len(val_loader)
    acc = 100.0 * correct / total
    f1 = f1_score(all_labels, all_preds, average="weighted") * 100.0
    prec = precision_score(all_labels, all_preds, average="weighted") * 100.0
    rec = recall_score(all_labels, all_preds, average="weighted") * 100.0
    return avg_loss, acc, f1, prec, rec


@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    set_seeds(cfg.RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    student_resolution = int(OmegaConf.select(cfg, "student_resolution", default=192))
    paths = resolve_experiment_paths(cfg)
    teacher_spec = resolve_teacher_spec(cfg, paths)
    student_test_transform = make_test_transform(student_resolution)
    student_train_transform = make_strong_train_transform(student_resolution)
    run_cfg = resolve_trim_fp32_run(
        student_resolution=student_resolution,
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

    from utils.dataset import FundusClsDataset

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

    print(f"\n{'=' * 50}")
    print(f"Loading {MODEL_NAME} and fine-tuning from ImageNet pretrained timm weights")
    print(f"{'=' * 50}")
    model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=cfg.nr_classes)
    model.to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Student parameters: {n_params:,}")
    print(f"Training: {EPOCHS} epochs, LR={LR}, patience={PATIENCE}")
    print(
        f"Configuration: trim black border -> {student_resolution} student, "
        f"{teacher_spec['mode']} KD teacher remains full-image 512."
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
        writer.writerow(
            [
                "epoch",
                "train_loss",
                "train_acc",
                "train_f1",
                "val_loss",
                "val_acc",
                "val_f1",
                "val_prec",
                "val_rec",
                "best_epoch",
            ]
        )

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
            writer.writerow(
                [
                    epoch,
                    train_loss,
                    train_acc,
                    train_f1,
                    val_loss,
                    val_acc,
                    val_f1,
                    val_prec,
                    val_rec,
                    best_epoch,
                ]
            )
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

    print(f"\n{'=' * 50}")
    print("Evaluating on test set ...")
    print(f"{'=' * 50}")
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
        "model": MODEL_NAME,
        "n_params": n_params,
        "epochs": best_epoch + 1,
        "best_val_f1": round(best_val_f1, 4),
        "best_val_loss": round(best_val_loss, 4),
        "checkpoint": ckpt_path,
        "student_init": "imagenet",
        "init_checkpoint": None,
        "teacher_mode": teacher_spec["mode"],
        "teacher_arch": teacher_spec["arch"],
        "teacher_checkpoint": teacher_path,
        "teacher": f"{teacher_spec['checkpoint_name']} (512x512 full-image strong train / test eval)",
        "dataset": OmegaConf.select(cfg, "dataset_name", default="uwf"),
        "experiment_tag": paths.experiment_tag,
        "random_seed": int(cfg.RANDOM_SEED),
        "split_seed": resolve_split_seed(cfg),
        "student_resolution": student_resolution,
        "teacher_resolution": 512,
        "student_preprocess": {
            "name": "trim_fundus_black_border",
            "threshold": 8,
            "pad_ratio": 0.01,
            "min_pad_px": 4,
        },
        "input_size": [1, 3, student_resolution, student_resolution],
        "kd_temperature": KD_TEMPERATURE,
        "kd_alpha": KD_ALPHA,
        "use_weighted_loss": False,
        "test_metrics": test_metrics,
    }
    report_path = os.path.join(results_dir, run_cfg["report_name"])
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved -> {report_path}")


if __name__ == "__main__":
    main()

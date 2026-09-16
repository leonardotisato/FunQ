"""
KD-QAT fine-tuning for trimmed-input test_resnet experiments.

This keeps the canonical teacher domain unchanged:
- teacher: ResNet18 assistant by default, or `++teacher_mode=r50_direct`
- teacher domain: full-image 512, strong-train / clean-eval

But changes the student path to a trimmed-input domain:
- student train domain: trim black border -> ``++student_resolution``, strong train transform
- student eval domain: trim black border -> ``++student_resolution``, clean eval transform
- student init: matching FP32 trim checkpoint (ImageNet-init)

Typical runs:
    python src/qat_test_resnet_trim.py ++student_resolution=192 ++weight_bits=8 ++act_bits=8
    python src/qat_test_resnet_trim.py ++student_resolution=160 ++weight_bits=6 ++act_bits=6
"""

import csv
import json
import math
import os
import sys
import warnings

warnings.filterwarnings("ignore")

import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from brevitas.graph.calibrate import calibration_mode
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as tv_transforms
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from utils.dataset import FundusClsDataset, prepare_dataframes, safe_pil_read, trim_fundus_black_border
from utils.generals import progress_bar
from utils.kd_sweep import build_teacher_model, resolve_experiment_paths, resolve_teacher_spec
from utils.qat_sweep import file_sha256, resolve_qat_run, run_provenance, valid_metrics, validate_weight_loading
from utils.quant_test_resnet import QuantTestResNet, load_test_resnet_weights, model_tag
from utils.seed import set_seeds
from utils.training import test
from utils.transforms import make_strong_train_transform, make_test_transform

teacher_test_transform = make_test_transform(512)
teacher_train_transform = make_strong_train_transform(512)


KD_TEMPERATURE = 3.0
KD_ALPHA = 0.25

QAT_LR = 1e-5
QAT_EPOCHS = 200
QAT_WEIGHT_DECAY = 1e-4
CALIB_BATCHES = 100
BN_FREEZE_EPOCH = 5
PATIENCE = 50
QAT_SWEEP_PROTOCOL_VERSION = 1


def resolve_trim_qat_run(student_resolution: int, weight_bits: int, act_bits: int,
                         teacher_mode=None, seed=None, experiment_tag=None):
    """Resolve stable naming for one trimmed-input QAT run."""

    return resolve_qat_run("test_resnet", student_resolution, weight_bits, act_bits,
                           teacher_mode, seed, experiment_tag)


class DualResTrimDataset(Dataset):
    """Return (student_image, teacher_image, label) for trimmed-input KD/QAT."""

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


def freeze_bn(model):
    """Set all BatchNorm layers to eval mode to freeze running stats during QAT."""
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
            module.eval()


def qat_train_one_epoch(student, teacher, train_loader, optimizer, device, epoch):
    student.train()
    teacher.eval()
    if epoch >= BN_FREEZE_EPOCH:
        freeze_bn(student)

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


def qat_validate(student, teacher, val_loader, temperature, alpha, device):
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

    weight_bits = int(OmegaConf.select(cfg, "weight_bits", default=8))
    act_bits = int(OmegaConf.select(cfg, "act_bits", default=8))
    student_resolution = int(OmegaConf.select(cfg, "student_resolution", default=192))
    student_test_transform = make_test_transform(student_resolution)
    student_train_transform = make_strong_train_transform(student_resolution)
    tag = model_tag(weight_bits, act_bits)
    paths = resolve_experiment_paths(cfg)
    teacher_spec = resolve_teacher_spec(cfg, paths)

    run_cfg = resolve_trim_qat_run(
        student_resolution=student_resolution,
        weight_bits=weight_bits,
        act_bits=act_bits,
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

    print(f"Loading teacher from: {teacher_path}")
    teacher = build_teacher_model(teacher_spec["arch"], nr_classes=cfg.nr_classes)
    teacher.load_state_dict(torch.load(teacher_path, map_location="cpu"))
    teacher.to(device)
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False
    print("Teacher loaded and frozen (512x512 full-image, strong-train/test-eval domain).")

    print("\n" + "=" * 50)
    print(f"Creating QuantTestResNet [{tag}] for {run_cfg['run_tag']}")
    print("=" * 50)
    student = QuantTestResNet(
        nr_classes=cfg.nr_classes,
        weight_bit_width=weight_bits,
        act_bit_width=act_bits,
    )
    n_params = sum(p.numel() for p in student.parameters())
    print(f"Student parameters: {n_params:,}")

    print(f"\nLoading FP32 student weights from: {student_init_checkpoint}")
    missing, unexpected = load_test_resnet_weights(student, student_init_checkpoint)
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
        f"Configuration: KD-QAT + trim black border -> {student_resolution} student "
        f"+ {teacher_spec['arch']} teacher on 512 strong/eval-test transforms"
    )
    print(f"Warm start: {student_init_checkpoint}")
    print(f"KD: temperature={KD_TEMPERATURE}, alpha={KD_ALPHA}")
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

    for epoch in tqdm(range(QAT_EPOCHS), desc=f"KD-QAT {run_cfg['run_tag']}"):
        train_loss, train_acc, train_f1 = qat_train_one_epoch(
            student, teacher, train_loader, optimizer, device, epoch
        )
        val_loss, val_acc, val_f1, val_prec, val_rec = qat_validate(
            student, teacher, val_loader, KD_TEMPERATURE, KD_ALPHA, device
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
        "student_preprocess": {
            "name": "trim_fundus_black_border",
            "threshold": 8,
            "pad_ratio": 0.01,
            "min_pad_px": 4,
        },
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

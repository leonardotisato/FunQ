import os
import csv
import yaml
import warnings
import torch
from torch.utils.data import DataLoader
import hydra
from omegaconf import DictConfig
warnings.filterwarnings("ignore")
from utils.seed import set_seeds
from utils.visualization import save_metrics_and_plot 
from utils.losses import kd_loss
from utils.training import train, train_knowledge_distillation, test
from utils.model import ResNet50Classifier
from utils.dataset import FundusClsDatasetZoom, PairedFundusDataset, prepare_dataframes
from utils.transforms import make_strong_train_transform, make_test_transform
from utils.generals import getOutFileName
import numpy as np
import json

train_transform_class = make_strong_train_transform(512)
test_transform_class = make_test_transform(512)
# =====================================================
def convert_dict(d):
    new_dict = {} 
    for key, value in d.items():
        if isinstance(key, np.int64):
            key = int(key)

        if isinstance(value, np.int64):
            value = np.float32(value)
        
        if isinstance(value, tuple):
            value = tuple(convert_dict({i: v})[i] if isinstance(v, (dict, tuple)) else (float(v) if isinstance(v, np.float64) else int(v)) for i, v in enumerate(value))
    
        elif isinstance(value, float):
            value = round(value, 6)

        elif isinstance(value, dict):
            value = convert_dict(value)

        elif isinstance(value, list):
            value = [convert_dict({idx: item}) for idx, item in enumerate(value)]  # Ricorsiva per ogni elemento

        new_dict[key] = value
    
    return new_dict

@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg: DictConfig):
    # ------------------ CONFIG & SEED ------------------
    set_seeds(cfg.RANDOM_SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    os.makedirs(cfg.results_dir, exist_ok=True)
    os.makedirs(cfg.logs_dir, exist_ok=True)
    os.makedirs(cfg.models_dir, exist_ok=True)

    exp_name = getOutFileName()

    print(f"-------> Results will be saved in: {cfg.results_dir}")

    # ------------------ DATASET & DATALOADER ------------------

    train_df, val_df, test_df = prepare_dataframes(cfg)

    train_transform = train_transform_class
    val_transform   = test_transform_class
    test_transform  = test_transform_class

    train_dataset_teacher = FundusClsDatasetZoom(
        csv_file=train_df,
        transform=train_transform,
        dilation_percentage = cfg.dilation_percentage,
        is_training = True,
        random_seed = cfg.RANDOM_SEED
    )

    val_dataset_teacher = FundusClsDatasetZoom(
        csv_file=val_df,
        transform=val_transform,
        dilation_percentage = cfg.dilation_percentage,
        is_training = False,
        random_seed = cfg.RANDOM_SEED
    )

    test_dataset_teacher = FundusClsDatasetZoom(
        csv_file=test_df,
        transform=test_transform,
        dilation_percentage = cfg.dilation_percentage,
        is_training = False,
        random_seed = cfg.RANDOM_SEED
    )

    train_dataset_kd = PairedFundusDataset(
        df=train_df,
        teacher_transform=train_transform,
        student_transform=train_transform,
        dilation_percentage=cfg.dilation_percentage, 
        student_train=True,
        teacher_train=True,
    )

    val_dataset_kd = PairedFundusDataset(
        df=val_df,
        teacher_transform=val_transform,
        student_transform=val_transform,
        dilation_percentage=cfg.dilation_percentage,
        student_train=False,
        teacher_train=False,
    )

    test_dataset_kd = PairedFundusDataset(
        df=test_df,
        teacher_transform=test_transform,
        student_transform=test_transform,
        dilation_percentage=cfg.dilation_percentage,
        student_train=False,
        teacher_train=False,
    )
    train_loader_teacher = DataLoader(
        train_dataset_teacher,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_loader_teacher = DataLoader(
        val_dataset_teacher,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    train_loader_kd = DataLoader(
        train_dataset_kd,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_loader_kd = DataLoader(
        val_dataset_kd,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    test_loader_teacher = DataLoader(
        test_dataset_teacher,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    test_loader_kd = DataLoader(
        test_dataset_kd,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    

    # ------------------ MODELS ------------------
    teacher = ResNet50Classifier(
        nr_classes=cfg.nr_classes,
        dropout=0.5,
        pretrained=True,
    )

    student_kd = ResNet50Classifier(
        nr_classes=cfg.nr_classes,
        dropout=0.5,
        pretrained=True,
    )

    # ------------------ Teacher ------------------
    print("\n================ TRAIN TEACHER =================")
    teacher, metrics_t, plots_t = train(
        model=teacher,
        train_loader=train_loader_teacher,
        valid_loader=val_loader_teacher,
        epochs=cfg.n_epochs,
        learning_rate=cfg.teacher_lr,
        weight_decay=cfg.weight_decay,
        device=device,
        exp_name=exp_name,
        model_type="teacher",
        grad_clip_value=5.0,
    )

    save_metrics_and_plot(
        name="teacher",
        train_losses=plots_t["train_losses"],
        val_losses=plots_t["val_losses"],
        val_f1s=plots_t["val_f1s"],
        results_dir=cfg.results_dir,
    )
    # ------------------ Student supervised by teacher ------------------
    print("\n================ TRAIN MODEL WITH KNOWLEDGE DISTILLATION =================")
    student_kd, metrics_skd, plots_skd = train_knowledge_distillation(
        teacher=teacher,
        student=student_kd,
        train_loader=train_loader_kd,
        valid_loader=val_loader_kd,
        epochs=cfg.n_epochs,
        learning_rate=cfg.student_lr,
        weight_decay=cfg.weight_decay,
        T=cfg.temperature,
        kd_loss=kd_loss,
        soft_target_loss_weight=cfg.soft_w,
        ce_loss_weight=cfg.ce_w,
        device=device,
        exp_name=exp_name
    )

    save_metrics_and_plot(
        name="student_kd",
        train_losses=plots_skd["train_losses"],
        val_losses=plots_skd["val_losses"],
        val_f1s=plots_skd["val_f1s"],
        results_dir=cfg.results_dir,
    )

    # ------------------ SAVE MODELS ------------------
    print("\n================ SAVING MODELS =================")
    torch.save(teacher.state_dict(), os.path.join(cfg.models_dir, "resnet50_fp32_teacher.pth"))
    torch.save(student_kd.state_dict(), os.path.join(cfg.models_dir, "resnet50_fp32_kd.pth"))
    print(f"Models saved in: {cfg.models_dir}")

    # ------------------ SAVE METRICS ------------------
    summary_metrics = {
        "teacher": metrics_t,
        "student_kd": metrics_skd,
        "config": {
            "nr_classes": cfg.nr_classes,
            "n_epochs": cfg.n_epochs,
            "batch_size": cfg.batch_size,
            "teacher_lr": cfg.teacher_lr,
            "student_lr": cfg.student_lr,
            "weight_decay": cfg.weight_decay,
            "temperature": cfg.temperature,
            "loss": cfg.loss,
        },
    }

    summary_path_yaml = os.path.join(cfg.results_dir, "summary_metrics.yaml")
    with open(summary_path_yaml, "w") as f:
        yaml.safe_dump(summary_metrics, f)

    summary_path_csv = os.path.join(cfg.results_dir, "summary_metrics.csv")
    with open(summary_path_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "accuracy", "f1", "precision", "recall"])
        for name, m in [("teacher", metrics_t), ("student_kd", metrics_skd)]:
            writer.writerow([name, m["accuracy"], m["f1"], m["precision"], m["recall"]])

    print(
        f"\nMetrics saved in:\n- {summary_path_yaml}\n- {summary_path_csv}"
    )

    # ---- Inference ---- #
    print("\n================ TESTING MODELS =================")
    print("\n---- Teacher ----")
    metrics_teacher = test(
        model=teacher,
        test_loader=test_loader_teacher,
        device=device,
        model_type="teacher",
        bootstrap = True,
        savedir=cfg.results_dir
    )
    print("\n---- Student with KD ----")
    metrics_kd = test(
        model=student_kd,
        test_loader=test_loader_kd,
        device=device,
        model_type="student_kd",
        bootstrap = True,
        savedir=cfg.results_dir
    )
    print("\n================ TESTING COMPLETED =================")
    # save test summaries in JSON
    test_summary_path = os.path.join(cfg.results_dir, "test_summary_teacher.json")
    metrics_teacher = convert_dict(metrics_teacher)
    with open(test_summary_path, 'w') as fp:
        json.dump(metrics_teacher, fp)
    test_summary_path = os.path.join(cfg.results_dir, "test_summary_kd.json")
    metrics_kd = convert_dict(metrics_kd)
    with open(test_summary_path, 'w') as fp:
        json.dump(metrics_kd, fp)

if __name__ == "__main__":
    main()

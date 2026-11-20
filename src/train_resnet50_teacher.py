"""
Train a supervised full-image ResNet50 teacher (512x512, ImageNet initialisation).

This is the teacher for datasets without lesion masks (REFUGE). On the UWF lesion dataset the
ResNet50 teacher is instead distilled from a lesion-centred model (``main.py``). The teacher
assistant then distils this model (``train_resnet18_from_resnet50_kd.py``, with
``ta_teacher_checkpoint`` set to the checkpoint written here).

    python src/train_resnet50_teacher.py --config-name refuge
"""

import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(__file__))
from utils.dataset import FundusClsDataset, prepare_dataframes
from utils.model import ResNet50Classifier
from utils.seed import set_seeds
from utils.training import test, train
from utils.transforms import make_strong_train_transform, make_test_transform

RESOLUTION = 512
CHECKPOINT = "resnet50_fp32.pth"
REPORT_JSON = "train_resnet50_teacher_report.json"


@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    set_seeds(cfg.RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    os.makedirs(cfg.models_dir, exist_ok=True)
    os.makedirs(cfg.results_dir, exist_ok=True)

    train_df, val_df, test_df = prepare_dataframes(cfg)
    loaders = {}
    for name, frame, transform, shuffle in (
        ("train", train_df, make_strong_train_transform(RESOLUTION), True),
        ("val", val_df, make_test_transform(RESOLUTION), False),
        ("test", test_df, make_test_transform(RESOLUTION), False),
    ):
        dataset = FundusClsDataset(frame, train=name == "train", transform=transform)
        loaders[name] = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=shuffle,
                                   num_workers=4, pin_memory=True)
    print(f"Images: train {len(train_df)}, val {len(val_df)}, test {len(test_df)}")

    model = ResNet50Classifier(nr_classes=cfg.nr_classes, dropout=0.5, pretrained=True)
    model, val_metrics, _ = train(
        model=model,
        train_loader=loaders["train"],
        valid_loader=loaders["val"],
        epochs=cfg.n_epochs,
        learning_rate=cfg.teacher_lr,
        weight_decay=cfg.weight_decay,
        device=device,
        exp_name="resnet50_teacher",
        model_type="teacher",
        log_dir=cfg.results_dir,
    )
    checkpoint_path = os.path.join(cfg.models_dir, CHECKPOINT)
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Checkpoint saved -> {checkpoint_path}")

    test_metrics = test(
        model=model,
        test_loader=loaders["test"],
        device=device,
        model_type="teacher",
        bootstrap=True,
        savedir=cfg.results_dir,
        class_names=OmegaConf.select(cfg, "class_names", default=None),
    )
    report = {
        "dataset": OmegaConf.select(cfg, "dataset_name", default="uwf"),
        "random_seed": int(cfg.RANDOM_SEED),
        "architecture": "ResNet50Classifier",
        "init": "imagenet",
        "resolution": RESOLUTION,
        "train_transform": f"utils.transforms.make_strong_train_transform({RESOLUTION})",
        "eval_transform": f"utils.transforms.make_test_transform({RESOLUTION})",
        "learning_rate": cfg.teacher_lr,
        "weight_decay": cfg.weight_decay,
        "max_epochs": cfg.n_epochs,
        "checkpoint": checkpoint_path,
        "best_val_metrics": val_metrics,
        "test_metrics": test_metrics,
    }
    report_path = os.path.join(cfg.results_dir, REPORT_JSON)
    with open(report_path, "w") as fp:
        json.dump(report, fp, indent=2)
    print(f"Report saved -> {report_path}")


if __name__ == "__main__":
    main()

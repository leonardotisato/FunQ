"""Build Data/refuge/refuge.csv from the extracted REFUGE release (1200 images, official split).

    python scripts/prepare_refuge.py

Source (figshare 26049574, Refuge.rar), extracted to Data/refuge/raw/extracted/Refuge:
  train  Training400/{Glaucoma,Non-Glaucoma}/*.jpg, label from the folder,
         masks Annotation-Training400/Disc_Cup_Masks/{Glaucoma,Non-Glaucoma}/*.bmp
  val    REFUGE-Validation400/REFUGE-Validation400/V*.jpg, label from Fovea_locations.xlsx,
         masks REFUGE-Validation400-GT/REFUGE-Validation400-GT/Disc_Cup_Masks/V*.bmp
  test   Test400/T*.jpg, label from Glaucoma_label_and_Fovea_location.xlsx,
         masks REFUGE-Test-GT/Disc_Cup_Masks/{G,N}/T*.bmp

REFUGE masks are 0 = optic cup, 128 = disc rim, 255 = background. The region-of-interest mask written
here is the whole optic disc (cup included) as 255, background 0, the convention of the UWF lesion masks.
Columns: image, mask, label (1 = glaucoma), split (train / val / test), camera. Paths are relative to
the repository root, like Data/fundus_data_final.csv.
"""

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
import openpyxl

ROOT = Path(__file__).resolve().parents[1]
CAMERAS = {"train": "Zeiss Visucam 500", "val": "Canon CR-2", "test": "Canon CR-2"}


def xlsx_labels(path, label_column):
    rows = list(openpyxl.load_workbook(path, read_only=True).worksheets[0].iter_rows(values_only=True))
    header = rows[0]
    name_i, label_i = header.index("ImgName"), header.index(label_column)
    return {r[name_i]: int(r[label_i]) for r in rows[1:] if r[name_i]}


def collect(source):
    """[(split, image path, mask path, label)] for the 1200 images, labels checked against each other."""
    items = []
    for folder, label in (("Glaucoma", 1), ("Non-Glaucoma", 0)):
        for image in sorted((source / "Training400" / folder).glob("*.jpg")):
            mask = source / "Annotation-Training400" / "Disc_Cup_Masks" / folder / f"{image.stem}.bmp"
            items.append(("train", image, mask, label))

    val_dir = source / "REFUGE-Validation400-GT" / "REFUGE-Validation400-GT"
    val_labels = xlsx_labels(val_dir / "Fovea_locations.xlsx", "Glaucoma Label")
    for image in sorted((source / "REFUGE-Validation400" / "REFUGE-Validation400").glob("*.jpg")):
        items.append(("val", image, val_dir / "Disc_Cup_Masks" / f"{image.stem}.bmp", val_labels[image.name]))

    test_gt = source / "REFUGE-Test-GT"
    test_labels = xlsx_labels(test_gt / "Glaucoma_label_and_Fovea_location.xlsx", "Label(Glaucoma=1)")
    for image in sorted((source / "Test400").glob("*.jpg")):
        label = test_labels[image.name]
        mask = test_gt / "Disc_Cup_Masks" / ("G" if label else "N") / f"{image.stem}.bmp"
        if not mask.exists():  # the mask folder must agree with the spreadsheet label
            raise ValueError(f"No {'G' if label else 'N'} mask for {image.name}: label mismatch")
        items.append(("test", image, mask, label))
    return items


def disc_mask(mask_path, image_shape):
    raw = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if raw is None:
        raise ValueError(f"Cannot read mask {mask_path}")
    if raw.shape != image_shape[:2]:
        raise ValueError(f"Mask {mask_path} is {raw.shape}, image is {image_shape[:2]}")
    disc = np.where(raw < 255, 255, 0).astype(np.uint8)
    if not disc.any():
        raise ValueError(f"Empty optic disc in {mask_path}")
    return disc


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, default=ROOT / "Data/refuge/raw/extracted/Refuge")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "Data/refuge")
    args = parser.parse_args()

    items = collect(args.source)
    counts = {s: sum(1 for i in items if i[0] == s) for s in ("train", "val", "test")}
    if counts != {"train": 400, "val": 400, "test": 400}:
        raise ValueError(f"Expected 400 images per split, found {counts}")

    rows = []
    for split, image, mask, label in items:
        shape = cv2.imread(str(image)).shape
        out = args.out_dir / "disc_masks" / split / f"{image.stem}.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out), disc_mask(mask, shape))
        rows.append({"image": _rel(image), "mask": _rel(out), "label": label, "split": split,
                     "camera": CAMERAS[split]})

    csv_path = args.out_dir / "refuge.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for split in ("train", "val", "test"):
        labels = [r["label"] for r in rows if r["split"] == split]
        print(f"{split}: {len(labels)} images, {sum(labels)} glaucoma")
    print(f"Wrote {csv_path}")


def _rel(path):
    path = Path(path).resolve()
    return path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else path.as_posix()


if __name__ == "__main__":
    main()

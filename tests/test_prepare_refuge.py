"""REFUGE data preparation on a miniature copy of the release layout."""

import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import openpyxl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import prepare_refuge  # noqa: E402


def _image(path, shape=(20, 30)):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.full((*shape, 3), 100, np.uint8))


def _mask(path, shape=(20, 30)):
    path.parent.mkdir(parents=True, exist_ok=True)
    mask = np.full(shape, 255, np.uint8)
    mask[5:10, 10:20] = 128  # disc rim
    mask[6:9, 12:16] = 0     # cup
    cv2.imwrite(str(path), mask)


def _xlsx(path, label_column, labels):
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(["ID", "ImgName", label_column, "Fovea_X", "Fovea_Y"])
    for i, (name, label) in enumerate(labels.items(), 1):
        ws.append([i, name, label, 1.0, 1.0])
    path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(path)


def _tree(src, test_mask_folder_for_t2="N"):
    _image(src / "Training400/Glaucoma/g0001.jpg")
    _mask(src / "Annotation-Training400/Disc_Cup_Masks/Glaucoma/g0001.bmp")
    _image(src / "Training400/Non-Glaucoma/n0001.jpg")
    _mask(src / "Annotation-Training400/Disc_Cup_Masks/Non-Glaucoma/n0001.bmp")
    gt = src / "REFUGE-Validation400-GT/REFUGE-Validation400-GT"
    for name in ("V0001", "V0002"):
        _image(src / f"REFUGE-Validation400/REFUGE-Validation400/{name}.jpg")
        _mask(gt / f"Disc_Cup_Masks/{name}.bmp")
    _xlsx(gt / "Fovea_locations.xlsx", "Glaucoma Label", {"V0001.jpg": 1, "V0002.jpg": 0})
    _image(src / "Test400/T0001.jpg")
    _mask(src / "REFUGE-Test-GT/Disc_Cup_Masks/G/T0001.bmp")
    _image(src / "Test400/T0002.jpg")
    _mask(src / f"REFUGE-Test-GT/Disc_Cup_Masks/{test_mask_folder_for_t2}/T0002.bmp")
    _xlsx(src / "REFUGE-Test-GT/Glaucoma_label_and_Fovea_location.xlsx", "Label(Glaucoma=1)",
          {"T0001.jpg": 1, "T0002.jpg": 0})


class PrepareRefugeTests(unittest.TestCase):
    def test_labels_and_splits(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp)
            _tree(src)
            items = prepare_refuge.collect(src)
        got = {(split, image.name, label) for split, image, _, label in items}
        self.assertEqual(got, {("train", "g0001.jpg", 1), ("train", "n0001.jpg", 0),
                               ("val", "V0001.jpg", 1), ("val", "V0002.jpg", 0),
                               ("test", "T0001.jpg", 1), ("test", "T0002.jpg", 0)})

    def test_test_mask_folder_must_agree_with_the_label(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp)
            _tree(src, test_mask_folder_for_t2="G")  # T0002 is labelled 0 but its mask is in G
            with self.assertRaises(ValueError):
                prepare_refuge.collect(src)

    def test_disc_mask_is_cup_plus_rim(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "m.bmp"
            _mask(path)
            disc = prepare_refuge.disc_mask(path, (20, 30, 3))
        self.assertEqual(set(np.unique(disc)), {0, 255})
        self.assertEqual(int((disc == 255).sum()), 5 * 10)  # rim and cup, not background
        with self.assertRaises(ValueError):
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "m.bmp"
                _mask(path)
                prepare_refuge.disc_mask(path, (21, 30, 3))  # size mismatch


if __name__ == "__main__":
    unittest.main()

"""Second-dataset support: binary metrics, official splits, class-aware reports, REFUGE config."""

import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from utils.binary_metrics import (  # noqa: E402
    binary_point_metrics,
    bootstrap_binary_metrics,
    sensitivity_at_specificity,
)
from utils.reporting import UWF_CLASS_NAMES, format_test_metrics  # noqa: E402


class BinaryMetricTests(unittest.TestCase):
    def setUp(self):
        # 6 negatives, 4 positives; scores rank all positives above all but one negative.
        self.labels = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1])
        self.scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.9, 0.6, 0.7, 0.8, 0.95])
        self.preds = (self.scores >= 0.5).astype(int)

    def test_point_metrics(self):
        m = binary_point_metrics(self.labels, self.scores, self.preds, specificity=0.85)
        self.assertAlmostEqual(m["auc"], 21 / 24)  # 21 of 24 positive/negative pairs ranked correctly
        self.assertAlmostEqual(m["sensitivity"], 1.0)
        self.assertAlmostEqual(m["specificity"], 4 / 6)  # 0.5 and 0.9 are false positives
        self.assertAlmostEqual(m["balanced_accuracy"], (1.0 + 4 / 6) / 2)
        self.assertIn("sensitivity_at_specificity_85", m)

    def test_sensitivity_at_specificity_uses_the_best_threshold_meeting_the_target(self):
        # Specificity >= 5/6 allows one false positive (0.9): then every positive is above threshold 0.6.
        self.assertAlmostEqual(sensitivity_at_specificity(self.labels, self.scores, 5 / 6), 1.0)
        # Specificity 1.0 allows no false positive: only the 0.95 positive is above 0.9.
        self.assertAlmostEqual(sensitivity_at_specificity(self.labels, self.scores, 1.0), 0.25)

    def test_bootstrap_returns_mean_and_interval_and_skips_one_class_resamples(self):
        rng = np.random.default_rng(0)
        out = bootstrap_binary_metrics(self.labels, self.scores, self.preds, n_bootstrap=300, rng=rng)
        for key in ("auc", "balanced_accuracy", "sensitivity", "specificity", "sensitivity_at_specificity_85"):
            mean, low, high = out[key]
            self.assertLessEqual(low, mean)
            self.assertLessEqual(mean, high)
            self.assertTrue(0.0 <= low and high <= 1.0)

    def test_point_metrics_require_both_classes(self):
        with self.assertRaises(ValueError):
            binary_point_metrics(np.zeros(4, int), np.zeros(4), np.zeros(4, int))


class ReportTests(unittest.TestCase):
    def _raw(self, names):
        triplet = (0.5, 0.4, 0.6)
        raw = {k: triplet for k in ("accuracy", "precision", "recall", "f1_score")}
        for name in names:
            for k in ("accuracy", "precision", "recall", "f1_score"):
                raw[f"{k}_{name}"] = (0.5, (0.4, 0.6))
        return raw

    def test_uwf_report_keeps_its_four_classes(self):
        report = format_test_metrics({"accuracy_overall": 80.0}, self._raw(UWF_CLASS_NAMES))
        self.assertEqual(set(report["class_one_vs_rest_bootstrap_ci_pct"]), {"healthy", "nevus", "melanoma", "chrpe"})
        self.assertNotIn("binary_point_metrics_pct", report)

    def test_binary_report_uses_given_names_and_adds_binary_sections(self):
        names = ("non_glaucoma", "glaucoma")
        binary_point = {"auc": 0.9, "balanced_accuracy": 0.8}
        binary_boot = {"auc": (0.9, 0.85, 0.95)}
        report = format_test_metrics({"accuracy_overall": 80.0}, self._raw(names), class_names=names,
                                     binary_point=binary_point, binary_bootstrap=binary_boot)
        self.assertEqual(set(report["class_one_vs_rest_bootstrap_ci_pct"]), set(names))
        self.assertAlmostEqual(report["binary_point_metrics_pct"]["auc"], 90.0)
        self.assertAlmostEqual(report["binary_bootstrap_ci_pct"]["auc"]["lower"], 85.0)


class OfficialSplitTests(unittest.TestCase):
    def test_split_column_is_used_as_is_and_ignores_the_seed(self):
        from utils.dataset import prepare_dataframes

        rows = []
        for split, n in (("train", 6), ("val", 2), ("test", 2)):
            for i in range(n):
                rows.append({"image": f"{split}_{i}.jpg", "label": i % 2, "split": split})
        with tempfile.TemporaryDirectory() as tmp:
            pd.DataFrame(rows).to_csv(os.path.join(tmp, "data.csv"), index=False)
            frames = []
            for seed in (42, 43):
                cfg = {"data_dir": tmp, "csv_file": "data.csv", "RANDOM_SEED": seed}
                frames.append(prepare_dataframes(_Cfg(cfg)))
        (train, val, test), (train2, _, test2) = frames
        self.assertEqual((len(train), len(val), len(test)), (6, 2, 2))
        self.assertEqual(list(test["image"]), ["test_0.jpg", "test_1.jpg"])
        self.assertTrue(train.equals(train2) and test.equals(test2))
        self.assertIn("patient_id", train.columns)

    def test_unknown_split_value_is_rejected(self):
        from utils.dataset import prepare_dataframes

        with tempfile.TemporaryDirectory() as tmp:
            pd.DataFrame([{"image": "a.jpg", "label": 0, "split": "holdout"}]).to_csv(
                os.path.join(tmp, "data.csv"), index=False)
            with self.assertRaises(ValueError):
                prepare_dataframes(_Cfg({"data_dir": tmp, "csv_file": "data.csv", "RANDOM_SEED": 42}))


class SplitSeedTests(unittest.TestCase):
    """split_seed fixes the patient split while RANDOM_SEED still changes training."""

    def _csv(self, tmp):
        rows = [{"image": f"PT{p:03d}_{i}.jpg", "patient": f"PT{p:03d}_{i}", "label": p % 4}
                for p in range(80) for i in range(2)]
        pd.DataFrame(rows).to_csv(os.path.join(tmp, "data.csv"), index=False)

    def _test_patients(self, tmp, **cfg):
        from utils.dataset import prepare_dataframes

        _, _, test = prepare_dataframes(_Cfg({"data_dir": tmp, "csv_file": "data.csv", **cfg}))
        return set(test["patient_id"])

    def test_split_seed_fixes_the_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._csv(tmp)
            fixed = [self._test_patients(tmp, RANDOM_SEED=s, split_seed=42) for s in (42, 43, 44)]
            self.assertTrue(fixed[0] == fixed[1] == fixed[2])
            self.assertEqual(fixed[0], self._test_patients(tmp, RANDOM_SEED=42))

    def test_without_split_seed_the_split_follows_the_seed(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._csv(tmp)
            self.assertNotEqual(self._test_patients(tmp, RANDOM_SEED=42), self._test_patients(tmp, RANDOM_SEED=43))


class RoiCropTests(unittest.TestCase):
    """With roi_all_classes, label-0 images are cropped around their mask too (e.g. the optic disc)."""

    def _row(self, tmp, label):
        import cv2

        image = np.full((200, 300, 3), 90, np.uint8)
        mask = np.zeros((200, 300), np.uint8)
        mask[20:60, 30:90] = 255  # region of interest far from the image centre
        # The crop slices [min:max] over the mask's pixel coordinates, so it is one pixel short.
        cv2.imwrite(os.path.join(tmp, "img.png"), image)
        cv2.imwrite(os.path.join(tmp, "mask.png"), mask)
        return pd.DataFrame([{"image": os.path.join(tmp, "img.png"), "mask": os.path.join(tmp, "mask.png"),
                              "label": label}])

    def _crop_shape(self, frame, roi_all_classes):
        from utils.dataset import BaseLesionZoomDataset

        ds = BaseLesionZoomDataset(frame, dilation_percentage=0.0, roi_all_classes=roi_all_classes)
        image, mask, mask_original, _ = ds._load_image_and_mask(0)
        return ds._crop_around_lesion(image, mask, mask_original)[0].shape[:2]

    def test_label_zero_uses_its_mask_only_with_the_flag(self):
        with tempfile.TemporaryDirectory() as tmp:
            frame = self._row(tmp, label=0)
            self.assertEqual(self._crop_shape(frame, roi_all_classes=True), (39, 59))
            self.assertNotEqual(self._crop_shape(frame, roi_all_classes=False), (39, 59))  # UWF: centre zoom

    def test_positive_label_is_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            frame = self._row(tmp, label=1)
            self.assertEqual(self._crop_shape(frame, roi_all_classes=False), (39, 59))
            self.assertEqual(self._crop_shape(frame, roi_all_classes=True), (39, 59))


class ConfigTests(unittest.TestCase):
    def test_uwf_default_and_refuge_config(self):
        from hydra import compose, initialize_config_dir

        with initialize_config_dir(config_dir=str(ROOT / "config"), version_base=None):
            uwf = compose(config_name="config")
            refuge = compose(config_name="refuge")
        self.assertEqual((uwf.dataset_name, uwf.nr_classes, uwf.csv_file), ("uwf", 4, "fundus_data_final.csv"))
        self.assertEqual(list(uwf.class_names), list(UWF_CLASS_NAMES))
        self.assertEqual((refuge.dataset_name, refuge.nr_classes), ("refuge", 2))
        self.assertEqual(list(refuge.class_names), ["non_glaucoma", "glaucoma"])
        # REFUGE models never land in the UWF folders, teachers included.
        for key in ("models_dir", "teacher_models_dir", "results_dir"):
            self.assertIn("refuge", str(refuge[key]))
        self.assertEqual(refuge.n_epochs, uwf.n_epochs)  # everything else is inherited
        self.assertTrue(refuge.roi_mask_all_classes)
        self.assertFalse(uwf.get("roi_mask_all_classes", False))


class _Cfg(dict):
    """Attribute access like an OmegaConf config; missing keys read as None, like OmegaConf.select."""

    def __getattr__(self, key):
        if key.startswith("__"):
            raise AttributeError(key)
        return self.get(key)


if __name__ == "__main__":
    unittest.main()

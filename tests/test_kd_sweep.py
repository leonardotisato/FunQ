import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from utils.kd_sweep import (  # noqa: E402
    resolve_experiment_paths,
    resolve_slim_fp32_run,
    resolve_teacher_spec,
    resolve_trim_fp32_run,
)


class KDSweepNamingTests(unittest.TestCase):
    def test_trim_legacy_names_are_unchanged(self):
        run = resolve_trim_fp32_run(160)

        self.assertEqual(run["run_tag"], "trim160")
        self.assertEqual(run["checkpoint_name"], "test_resnet_fp32_kd_trim160_ft.pth")
        self.assertEqual(run["results_dir_name"], "test_resnet_trim160")

    def test_trim_sweep_names_include_teacher_and_seed(self):
        run = resolve_trim_fp32_run(
            192,
            teacher_mode="r50_direct",
            seed=44,
            experiment_tag="kd_teacher_sweep",
        )

        self.assertEqual(run["run_tag"], "trim192_r50_direct_seed44")
        self.assertEqual(
            run["checkpoint_name"],
            "test_resnet_fp32_kd_trim192_r50_direct_seed44_ft.pth",
        )
        self.assertEqual(run["results_dir_name"], "test_resnet_trim192_r50_direct_seed44")

    def test_slim_sweep_names_include_variant_teacher_and_seed(self):
        run = resolve_slim_fp32_run(
            student_resolution=160,
            layer3_out=128,
            layer4_out=64,
            teacher_mode="r18_ta",
            seed=42,
            experiment_tag="kd_teacher_sweep",
        )

        self.assertEqual(run["run_tag"], "slim128x64_trim160_r18_ta_seed42")
        self.assertEqual(
            run["checkpoint_name"],
            "test_resnet_slim128x64_fp32_kd_trim160_r18_ta_seed42_ft.pth",
        )
        self.assertEqual(
            run["results_dir_name"],
            "test_resnet_slim128x64_trim160_r18_ta_seed42",
        )

    def test_experiment_paths_keep_teacher_models_in_base_folder(self):
        cfg = {
            "models_dir": "models",
            "results_dir": "runs",
            "experiment_tag": "kd_teacher_sweep",
        }

        paths = resolve_experiment_paths(cfg)

        self.assertEqual(paths.models_dir, str(Path("models") / "kd_teacher_sweep"))
        self.assertEqual(
            paths.results_root,
            str(Path("runs") / "kd_teacher_sweep"),
        )
        self.assertEqual(paths.teacher_models_dir, "models")

    def test_teacher_spec_can_select_r50_direct(self):
        cfg = {
            "models_dir": "models",
            "teacher_mode": "r50_direct",
        }

        teacher = resolve_teacher_spec(cfg)

        self.assertEqual(teacher["mode"], "r50_direct")
        self.assertEqual(teacher["arch"], "resnet50")
        self.assertEqual(teacher["checkpoint_name"], "resnet50_fp32_kd.pth")
        self.assertEqual(
            teacher["checkpoint_path"],
            str(Path("models") / "resnet50_fp32_kd.pth"),
        )

    def test_teacher_checkpoint_override_with_path_is_not_rebased(self):
        teacher = resolve_teacher_spec(
            {
                "models_dir": "models",
                "teacher_mode": "r18_ta",
                "teacher_checkpoint": str(Path("custom") / "teacher.pth"),
            }
        )

        self.assertEqual(teacher["checkpoint_path"], str(Path("custom") / "teacher.pth"))
        self.assertEqual(teacher["checkpoint_name"], "teacher.pth")

    def test_invalid_teacher_mode_fails_fast(self):
        with self.assertRaises(ValueError):
            resolve_teacher_spec({"teacher_mode": "unknown"})


if __name__ == "__main__":
    unittest.main()

"""Offline checks for QAT orchestration; no training packages required."""

import importlib.util
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))


def load_launcher():
    path = REPO / "scripts/run_qat_kd_teacher_sweep.py"
    spec = importlib.util.spec_from_file_location("qat_launcher", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class SweepTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "runs/fp32_example"
        self.source.mkdir(parents=True)
        (self.root / "Data").mkdir()
        (self.root / "Data/fundus_data_final.csv").write_text("patient,image,label\np1,x.png,0\n")
        for architecture in ("test_resnet", "test_resnet_slim128x64"):
            for resolution in (160, 192, 224):
                for teacher in ("r18_ta", "r50_direct"):
                    for seed in range(42, 47):
                        self.add_source(architecture, resolution, teacher, seed)

    def add_source(self, architecture, resolution, teacher, seed):
        suffix = f"trim{resolution}_{teacher}_seed{seed}"
        run = self.source / f"{architecture}_{suffix}"
        run.mkdir()
        checkpoint = f"models/fp32_example/{architecture}_fp32_kd_{suffix}_ft.pth"
        report = {"experiment_tag": "fp32_example", "student_resolution": resolution,
                  "teacher_mode": teacher, "teacher_arch": "resnet18" if teacher == "r18_ta" else "resnet50",
                  "random_seed": seed, "checkpoint": checkpoint,
                  "teacher_checkpoint": f"models/{teacher}.pth", "kd_temperature": 3.0, "kd_alpha": 0.25,
                  "input_size": [1, 3, resolution, resolution],
                  "test_metrics": {"point_metrics_pct": {"accuracy_overall": 80.0}}}
        if "slim" in architecture:
            report.update({"layer3_out": 128, "layer4_out": 64})
        (run / f"train_{run.name}_report.json").write_text(json.dumps(report))
        for value in (checkpoint, report["teacher_checkpoint"]):
            path = self.root / value
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"test weights")

    def launcher(self):
        self.assertTrue((REPO / "scripts/run_qat_kd_teacher_sweep.py").exists(), "QAT sweep launcher is missing")
        return load_launcher()

    def plan(self, **kwargs):
        return self.launcher().plan_runs(self.source, root=self.root, **kwargs)

    def materialize_inputs(self, runs):
        for run in runs:
            for key in ("fp32_checkpoint", "teacher_checkpoint"):
                path = Path(run[key])
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"test weights")

    def complete(self, run, **changes):
        checkpoint = Path(run["output_checkpoint"])
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(b"QAT weights")
        report = {"sweep_run_fingerprint": run["fingerprint"], "checkpoint": str(checkpoint),
                  "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                  "student_init_checkpoint_sha256": run["fp32_sha256"],
                  "teacher_checkpoint_sha256": run["teacher_sha256"],
                  "fp32_source_report_sha256": run["source_sha256"], "data_csv_sha256": run["data_sha256"],
                  "weight_bits": run["bits"], "act_bits": run["bits"],
                  "teacher_mode": run["teacher"], "random_seed": run["seed"],
                  "student_resolution": run["resolution"],
                  "epochs": 1, "best_val_loss": 2.0, "best_val_f1": 80.0,
                  "test_metrics": {"point_metrics_pct": {
                      "accuracy_overall": 80., "precision_weighted": 80.,
                      "recall_weighted": 80., "f1_weighted": 80.}}}
        report.update(changes)
        path = Path(run["output_report"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report))

    def test_full_sweep_pairs_each_seed_with_three_independent_precisions(self):
        runs = self.plan()
        self.assertEqual(len(runs), 180)
        self.assertEqual(len({r["output_report"] for r in runs}), 180)
        for run in runs:
            self.assertIn(f"_{run['teacher']}_seed{run['seed']}_ft.pth", run["fp32_checkpoint"])
            self.assertIn(run["architecture"] + "_fp32_kd_", run["fp32_checkpoint"])
            siblings = [r for r in runs if r["source_report"] == run["source_report"]]
            self.assertEqual({r["bits"] for r in siblings}, {4, 6, 8})
            self.assertEqual(len({r["fp32_checkpoint"] for r in siblings}), 1)

    def test_u250_subset_and_filters(self):
        runs = self.plan(preset="u250")
        self.assertEqual(len(runs), 40)
        self.assertEqual({(r["architecture"], r["resolution"], r["bits"]) for r in runs}, {
            ("test_resnet", 160, 6), ("test_resnet", 192, 6),
            ("test_resnet_slim128x64", 160, 6), ("test_resnet_slim128x64", 160, 8)})
        self.assertEqual(len(self.plan(seeds=[42], teachers=["r50_direct"], resolutions=[160], bits=[6])), 2)
        with self.assertRaisesRegex(ValueError, "No .*runs"):
            self.plan(seeds=[99])

    def test_mismatched_report_and_duplicate_identity_are_rejected(self):
        path = next(self.source.glob("*/*report.json"))
        data = json.loads(path.read_text())
        data["random_seed"] = 99
        path.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, "metadata|identity|match"):
            self.plan()
        data["random_seed"] = 42
        path.write_text(json.dumps(data))
        path.with_name("duplicate_report.json").write_text(path.read_text())
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            self.plan()

    def test_preflight_checks_all_inputs_before_creating_outputs(self):
        launcher = self.launcher()
        runs = self.plan(seeds=[42], bits=[6])
        Path(runs[-1]["fp32_checkpoint"]).unlink()
        with self.assertRaises(FileNotFoundError):
            launcher.preflight(runs)
        self.assertFalse(Path(runs[0]["output_report"]).parent.exists())
        self.materialize_inputs(runs)
        self.assertEqual(launcher.preflight(runs), runs)

    def test_completed_run_requires_matching_identity_and_valid_metrics(self):
        launcher = self.launcher()
        runs = self.plan(seeds=[42], bits=[6])
        self.materialize_inputs(runs)
        self.complete(runs[0])
        self.assertEqual(len(launcher.preflight(runs)), len(runs) - 1)
        self.complete(runs[0], sweep_run_fingerprint="different")
        with self.assertRaisesRegex(ValueError, "conflict|different|identity"):
            launcher.preflight(runs, restart_incomplete=True)
        self.complete(runs[0], test_metrics={})
        with self.assertRaisesRegex(ValueError, "Incomplete|incomplete"):
            launcher.preflight(runs)
        self.assertEqual(len(launcher.preflight(runs, restart_incomplete=True)), len(runs))

    def test_partial_checkpoint_is_not_silently_overwritten(self):
        runs = self.plan(seeds=[42], bits=[6])
        self.materialize_inputs(runs)
        checkpoint = Path(runs[0]["output_checkpoint"])
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(b"partial")
        with self.assertRaisesRegex(ValueError, "Incomplete|incomplete"):
            self.launcher().preflight(runs)
        self.assertEqual(len(self.launcher().preflight(runs, restart_incomplete=True)), len(runs))

    def test_changed_batch_size_cannot_reuse_completed_output(self):
        runs = self.plan(seeds=[42], bits=[6])
        self.materialize_inputs(runs)
        self.complete(runs[0])
        changed = self.plan(seeds=[42], bits=[6], batch_size=32)
        with self.assertRaisesRegex(ValueError, "conflict|different|identity"):
            self.launcher().preflight(changed)

    def test_dry_run_needs_no_training_packages_or_checkpoints(self):
        self.launcher()
        for path in (self.root / "models").rglob("*.pth"):
            path.unlink()
        result = subprocess.run([sys.executable, str(REPO / "scripts/run_qat_kd_teacher_sweep.py"),
                                 "--source-sweep", str(self.source), "--dry-run", "--preset", "u250"],
                                capture_output=True, text=True, cwd=self.root)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("40 runs", result.stdout)
        self.assertIn("warm_start_checkpoint", result.stdout)
        self.assertIn("teacher_mode", result.stdout)

    def test_stale_trainers_are_rejected_before_sweep(self):
        launcher = self.launcher()
        code_root = self.root / "code"
        (code_root / "src").mkdir(parents=True)
        (code_root / "src/qat_test_resnet_trim.py").write_text(
            "QAT_SWEEP_PROTOCOL_VERSION = 1\n", encoding="utf-8"
        )
        (code_root / "src/qat_test_resnet_slim.py").write_text(
            "# legacy trainer without sweep support\n", encoding="utf-8"
        )
        with self.assertRaisesRegex(ValueError, "Incompatible QAT trainer"):
            launcher.validate_trainer_protocol(code_root)

        (code_root / "src/qat_test_resnet_slim.py").write_text(
            "QAT_SWEEP_PROTOCOL_VERSION = 1\n", encoding="utf-8"
        )
        launcher.validate_trainer_protocol(code_root)

    def test_replaced_weights_cannot_reuse_completed_output(self):
        runs = self.plan(seeds=[42], bits=[6])
        self.complete(runs[0])
        Path(runs[0]["fp32_checkpoint"]).write_bytes(b"replacement weights")
        changed = self.plan(seeds=[42], bits=[6])
        with self.assertRaisesRegex(ValueError, "conflict|different|identity"):
            self.launcher().preflight(changed)

    def test_changed_training_code_is_rejected(self):
        runs = self.plan(seeds=[42], bits=[6])
        first_code_file = next(iter(runs[0]["code_sha256"]))
        runs[0]["code_sha256"][first_code_file] = "changed"
        with self.assertRaisesRegex(ValueError, "code changed"):
            self.launcher().preflight(runs)

    def test_no_best_model_is_not_complete(self):
        runs = self.plan(seeds=[42], bits=[6])
        self.complete(runs[0], epochs=0, best_val_loss=float("inf"))
        self.assertFalse(self.launcher().is_complete(runs[0]))

    def test_completion_checks_actual_input_and_output_fingerprints(self):
        runs = self.plan(seeds=[42], bits=[6])
        self.complete(runs[0], student_init_checkpoint_sha256="changed")
        self.assertFalse(self.launcher().is_complete(runs[0]))
        self.complete(runs[0])
        Path(runs[0]["output_checkpoint"]).write_bytes(b"corrupted nonempty checkpoint")
        self.assertFalse(self.launcher().is_complete(runs[0]))

    def test_execution_stops_on_zero_exit_without_completion_and_releases_lock(self):
        launcher = self.launcher()
        runs = self.plan(seeds=[42], teachers=["r18_ta"], resolutions=[160], bits=[6])
        for run in runs:
            run["command"] = [sys.executable, "-c", "print('simulated incomplete trainer')"]
        with self.assertRaisesRegex(RuntimeError, "incomplete"):
            launcher.execute(runs, root=self.root)
        self.assertTrue(Path(runs[0]["run_manifest"]).exists())
        self.assertFalse(Path(runs[1]["run_manifest"]).exists())
        self.assertFalse(Path(runs[0]["output_report"]).parent.parent.joinpath("sweep.lock").exists())

    def test_execution_failure_does_not_leave_an_old_completion_marker(self):
        launcher = self.launcher()
        runs = self.plan(seeds=[42], teachers=["r18_ta"], resolutions=[160], bits=[6])[:1]
        self.complete(runs[0], test_metrics={})
        runs[0]["command"] = [sys.executable, "-c", "raise SystemExit(7)"]
        with self.assertRaisesRegex(RuntimeError, "failed"):
            launcher.execute(runs, root=self.root, restart_incomplete=True)
        self.assertFalse(Path(runs[0]["output_report"]).exists())

    def test_execution_honors_existing_lock(self):
        runs = self.plan(seeds=[42], teachers=["r18_ta"], resolutions=[160], bits=[6])[:1]
        lock = Path(runs[0]["output_report"]).parent.parent / "sweep.lock"
        lock.parent.mkdir(parents=True)
        lock.write_text("another process")
        with self.assertRaisesRegex(RuntimeError, "lock exists"):
            self.launcher().execute(runs, root=self.root)
        self.assertEqual(lock.read_text(), "another process")

    def test_weight_mismatches_fail_before_calibration(self):
        self.assertTrue((REPO / "src/utils/qat_sweep.py").exists(), "QAT helpers are missing")
        from utils.qat_sweep import validate_weight_loading
        validate_weight_loading(["act1.act_quant.tensor_quant.scaling_impl.value"], [])
        with self.assertRaises(ValueError):
            validate_weight_loading(["layer4.0.conv1.weight"], [])
        with self.assertRaises(ValueError):
            validate_weight_loading([], ["unexpected.weight"])

    def test_legacy_and_sweep_naming(self):
        self.assertTrue((REPO / "src/utils/qat_sweep.py").exists(), "QAT naming helper is missing")
        from utils.qat_sweep import resolve_qat_run
        legacy = resolve_qat_run("test_resnet", 160, 6, 6)
        self.assertEqual(legacy["checkpoint_name"], "test_resnet_trim160_6w6a_qat.pth")
        slim = resolve_qat_run("test_resnet_slim128x64", 160, 8, 8, "r50_direct", 43, "qat_example")
        self.assertEqual(slim["checkpoint_name"], "test_resnet_slim128x64_trim160_8w8a_r50_direct_seed43_qat.pth")
        self.assertEqual(slim["fp32_checkpoint_name"], "test_resnet_slim128x64_fp32_kd_trim160_ft.pth")


if __name__ == "__main__":
    unittest.main()

"""Run matched FP32 -> QAT experiments. See docs/qat_sweep.md for HPC usage."""

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from utils.kd_sweep import TEACHER_SPECS
from utils.qat_sweep import file_sha256, resolve_qat_run, valid_metrics


QAT_SWEEP_PROTOCOL_VERSION = 1


def validate_trainer_protocol(root=ROOT):
    """Reject stale QAT trainers before an expensive sweep starts."""

    for relative_path in ("src/qat_test_resnet_trim.py", "src/qat_test_resnet_slim.py"):
        path = Path(root) / relative_path
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (OSError, SyntaxError) as error:
            raise ValueError(f"Cannot validate QAT trainer protocol: {path}: {error}") from error
        version = None
        for node in tree.body:
            if (isinstance(node, ast.Assign) and len(node.targets) == 1 and
                    isinstance(node.targets[0], ast.Name) and
                    node.targets[0].id == "QAT_SWEEP_PROTOCOL_VERSION" and
                    isinstance(node.value, ast.Constant)):
                version = node.value.value
                break
        if version != QAT_SWEEP_PROTOCOL_VERSION:
            raise ValueError(
                f"Incompatible QAT trainer: {path}. Expected sweep protocol "
                f"{QAT_SWEEP_PROTOCOL_VERSION}, found {version!r}. "
                "Sync the launcher and both QAT trainers."
            )


def plan_runs(source_sweep, *, root=ROOT, preset="full", bits=(8, 6, 4), seeds=None,
              teachers=None, resolutions=None, experiment_tag=None, fp32_models_dir=None,
              teacher_models_dir=None, batch_size=16, data_dir=None, csv_file="fundus_data_final.csv"):
    root, source = Path(root).resolve(), Path(source_sweep).resolve()
    validate_trainer_protocol(ROOT)
    experiment = experiment_tag or f"qat_{source.name}"
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", experiment):
        raise ValueError("experiment-tag must be a single folder name (letters, digits, _, - or .)")
    if not source.is_dir():
        raise FileNotFoundError(f"Source sweep not found: {source}")
    bits = tuple(dict.fromkeys(bits))
    if not bits or any(b not in (4, 6, 8) for b in bits) or batch_size < 1:
        raise ValueError("Choose positive batch size and bit widths from 4, 6, 8")
    if preset not in ("full", "u250"):
        raise ValueError(f"Unknown preset: {preset}")
    if teachers and any(t not in TEACHER_SPECS for t in teachers):
        raise ValueError("Unknown teacher filter")
    data_csv = (Path(data_dir) if data_dir else root / "Data") / csv_file
    data_csv = data_csv.resolve()
    hash_cache = {}

    def fingerprint(path):
        path = Path(path)
        if path not in hash_cache:
            hash_cache[path] = file_sha256(path) if path.is_file() else None
        return hash_cache[path]

    # These files define the actual training/naming/data protocol.
    code_files = ["src/qat_test_resnet_trim.py", "src/qat_test_resnet_slim.py", "src/utils/qat_sweep.py",
                  "src/utils/kd_sweep.py", "src/utils/dataset.py", "src/utils/training.py",
                  "src/utils/metrics.py", "src/utils/transforms.py", "src/utils/seed.py",
                  "src/utils/model.py", "src/utils/quant_test_resnet.py", "src/utils/quant_test_resnet_slim.py",
                  "config/config.yaml", "scripts/run_qat_kd_teacher_sweep.py"]
    code_hashes = {name: fingerprint(ROOT / name) for name in code_files}
    runs, seen = [], set()
    for path in sorted(source.glob("*/*_report.json")):
        report = json.loads(path.read_text(encoding="utf-8"))
        match = re.fullmatch(r"(test_resnet(?:_slim(\d+)x(\d+))?)_trim(\d+)_(r18_ta|r50_direct)_seed(\d+)", path.parent.name)
        if not match:
            raise ValueError(f"Unsupported source run identity: {path.parent.name}")
        arch, layer3, layer4, resolution, teacher, seed = match.groups()
        resolution, seed = int(resolution), int(seed)
        identity = (arch, resolution, teacher, seed)
        if identity in seen:
            raise ValueError(f"Duplicate source identity: {identity}")
        seen.add(identity)
        if (report["student_resolution"] != resolution or report["random_seed"] != seed or
                report["teacher_mode"] != teacher or report["experiment_tag"] != source.name or
                report["teacher_arch"] != TEACHER_SPECS[teacher]["arch"] or
                report["input_size"] != [1, 3, resolution, resolution]):
            raise ValueError(f"Source folder/metadata mismatch: {path}")
        if layer3 and (report["layer3_out"] != int(layer3) or report["layer4_out"] != int(layer4)):
            raise ValueError(f"Slim width metadata mismatch: {path}")
        if report["kd_temperature"] != 3.0 or report["kd_alpha"] != 0.25:
            raise ValueError(f"Source KD settings do not match this QAT protocol: {path}")
        expected = f"{arch}_fp32_kd_trim{resolution}_{teacher}_seed{seed}_ft.pth"
        if Path(report["checkpoint"]).name != expected:
            raise ValueError(f"FP32 checkpoint identity mismatch: {path}")
        if (seeds is not None and seed not in seeds or teachers is not None and teacher not in teachers or
                resolutions is not None and resolution not in resolutions):
            continue
        fp32_path = ((Path(fp32_models_dir) / expected) if fp32_models_dir else root / report["checkpoint"]).resolve()
        teacher_path = ((Path(teacher_models_dir) / Path(report["teacher_checkpoint"]).name)
                        if teacher_models_dir else root / report["teacher_checkpoint"]).resolve()
        for bit in bits:
            if preset == "u250" and (arch, resolution, bit) not in {
                    ("test_resnet", 160, 6), ("test_resnet", 192, 6),
                    ("test_resnet_slim128x64", 160, 6), ("test_resnet_slim128x64", 160, 8)}:
                continue
            names = resolve_qat_run(arch, resolution, bit, bit, teacher, seed, experiment)
            results_dir = root / "results_vgg16" / experiment / names["results_dir_name"]
            output_checkpoint = root / "models" / experiment / names["checkpoint_name"]
            if output_checkpoint == fp32_path:
                raise ValueError("QAT output must not replace the FP32 checkpoint")
            run = {"run_id": names["run_tag"], "architecture": arch, "resolution": resolution,
                   "teacher": teacher, "seed": seed, "bits": bit, "experiment_tag": experiment,
                   "source_report": str(path), "fp32_checkpoint": str(fp32_path),
                   "teacher_checkpoint": str(teacher_path), "data_csv": str(data_csv),
                   "source_sha256": fingerprint(path), "fp32_sha256": fingerprint(fp32_path),
                   "teacher_sha256": fingerprint(teacher_path), "data_sha256": fingerprint(data_csv),
                   "code_sha256": code_hashes, "batch_size": batch_size,
                   "output_checkpoint": str(output_checkpoint),
                   "output_report": str(results_dir / names["report_name"]),
                   "run_manifest": str(results_dir / "run_manifest.json"),
                   "log_file": str(root / "logs" / experiment / f"{names['run_tag']}.log")}
            run["fingerprint"] = hashlib.sha256(json.dumps(run, sort_keys=True).encode()).hexdigest()
            overrides = {"RANDOM_SEED": seed, "student_resolution": resolution,
                         "teacher_mode": teacher, "teacher_checkpoint": str(teacher_path),
                         "experiment_tag": experiment, "warm_start_checkpoint": str(fp32_path),
                         "fp32_source_report": str(path), "sweep_run_fingerprint": run["fingerprint"],
                         "weight_bits": bit, "act_bits": bit, "batch_size": batch_size,
                         "models_dir": str(root / "models"), "results_dir": str(root / "results_vgg16"),
                         "data_dir": str(data_csv.parent), "csv_file": data_csv.name}
            if layer3:
                overrides.update(slim_layer3_out=int(layer3), slim_layer4_out=int(layer4))
            script = root / "src" / ("qat_test_resnet_slim.py" if layer3 else "qat_test_resnet_trim.py")
            run["command"] = [sys.executable, "-u", str(script)] + [
                f"++{key}={json.dumps(value)}" for key, value in overrides.items()]
            runs.append(run)
    if not runs:
        raise ValueError("No QAT runs match the selected reports and filters")
    return runs


def read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def is_complete(run):
    report = read_json(run["output_report"])
    checkpoint = Path(run["output_checkpoint"])
    if not isinstance(report, dict) or not checkpoint.is_file() or checkpoint.stat().st_size == 0:
        return False
    expected = {"sweep_run_fingerprint": run["fingerprint"], "weight_bits": run["bits"],
                "act_bits": run["bits"], "teacher_mode": run["teacher"],
                "random_seed": run["seed"], "student_resolution": run["resolution"],
                "student_init_checkpoint_sha256": run["fp32_sha256"],
                "teacher_checkpoint_sha256": run["teacher_sha256"],
                "fp32_source_report_sha256": run["source_sha256"], "data_csv_sha256": run["data_sha256"]}
    return (all(report.get(k) == v for k, v in expected.items()) and
            Path(report.get("checkpoint", "")).resolve() == checkpoint.resolve() and valid_metrics(report) and
            report.get("checkpoint_sha256") == file_sha256(checkpoint))


def preflight(runs, restart_incomplete=False):
    required = {r[k] for r in runs for k in ("fp32_checkpoint", "teacher_checkpoint", "data_csv", "source_report")}
    missing = sorted(p for p in required if not Path(p).is_file() or Path(p).stat().st_size == 0)
    if missing:
        raise FileNotFoundError(f"Missing or empty required files ({len(missing)}):\n" + "\n".join(missing))
    current_hashes = {p: file_sha256(p) for p in required}
    planned_code = {path: digest for run in runs for path, digest in run["code_sha256"].items()}
    current_code = {path: file_sha256(ROOT / path) if (ROOT / path).is_file() else None
                    for path in planned_code}
    pending = []
    for run in runs:
        for relative_path, planned_hash in run["code_sha256"].items():
            if current_code[relative_path] != planned_hash:
                raise ValueError(f"Training code changed since planning; rerun the launcher: {ROOT / relative_path}")
        for path_key, hash_key in (("fp32_checkpoint", "fp32_sha256"), ("teacher_checkpoint", "teacher_sha256"),
                                   ("data_csv", "data_sha256"), ("source_report", "source_sha256")):
            if current_hashes[run[path_key]] != run[hash_key]:
                raise ValueError(f"Input changed since planning; rerun the launcher: {run[path_key]}")
        for key, field in (("output_report", "sweep_run_fingerprint"), ("run_manifest", "fingerprint")):
            stored = read_json(run[key])
            if isinstance(stored, dict) and stored.get(field) != run["fingerprint"]:
                raise ValueError(f"Existing output identity conflict: {run[key]}; use a different --experiment-tag")
        if is_complete(run):
            continue
        if not restart_incomplete and any(Path(run[k]).exists() for k in ("output_report", "output_checkpoint", "run_manifest", "log_file")):
            raise ValueError(f"Incomplete run {run['run_id']}; use --restart-incomplete to retrain it from FP32")
        pending.append(run)
    return pending


def execute(runs, *, root=ROOT, restart_incomplete=False):
    pending = preflight(runs, restart_incomplete)
    print(f"{len(runs) - len(pending)} completed runs skipped; {len(pending)} runs pending.", flush=True)
    if not pending:
        return
    lock_path = Path(runs[0]["output_report"]).parent.parent / "sweep.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        lock = lock_path.open("x")
    except FileExistsError:
        raise RuntimeError(f"Sweep lock exists: {lock_path}. Check for an active job before removing a stale lock.") from None
    try:
        with lock:
            lock.write(f"pid={os.getpid()}\n")
        # Recheck after locking so two simultaneous invocations cannot overwrite a completed run.
        pending = preflight(runs, restart_incomplete)
        for index, run in enumerate(pending, 1):
            # The queue may run for days: detect changed inputs before each new job.
            if not preflight([run], restart_incomplete):
                continue
            report_path = Path(run["output_report"])
            report_path.parent.mkdir(parents=True, exist_ok=True)
            Path(run["log_file"]).parent.mkdir(parents=True, exist_ok=True)
            # Invalidate an interrupted run's old completion marker before retraining.
            if report_path.exists():
                report_path.unlink()
            Path(run["run_manifest"]).write_text(json.dumps(run, indent=2) + "\n", encoding="utf-8")
            print(f"[{index}/{len(pending)}] {run['run_id']}\nLog: {run['log_file']}", flush=True)
            with open(run["log_file"], "a", encoding="utf-8") as log:
                log.write("\nCommand: " + shlex.join(run["command"]) + "\n")
                log.flush()
                result = subprocess.run(run["command"], cwd=root, stdout=log, stderr=subprocess.STDOUT)
            if result.returncode != 0:
                raise RuntimeError(f"QAT run failed or produced incomplete results: {run['run_id']}. See {run['log_file']}")
            # Re-hash inputs/code and validate the final report/checkpoint after the process exits.
            if preflight([run], restart_incomplete=True):
                raise RuntimeError(f"QAT run produced incomplete results: {run['run_id']}. See {run['log_file']}")
    finally:
        lock_path.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-sweep", type=Path, required=True, help="Folder containing FP32 run reports")
    parser.add_argument("--preset", choices=("full", "u250"), default="full")
    parser.add_argument("--bits", type=int, nargs="+", default=[8, 6, 4])
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--teachers", choices=tuple(TEACHER_SPECS), nargs="+")
    parser.add_argument("--resolutions", type=int, nargs="+")
    parser.add_argument("--experiment-tag")
    parser.add_argument("--fp32-models-dir", type=Path, help="Relocated FP32 checkpoint folder; preserve filenames")
    parser.add_argument("--teacher-models-dir", type=Path, help="Relocated teacher checkpoint folder")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--csv-file", default="fundus_data_final.csv")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="Print commands; no checkpoints or training packages required")
    mode.add_argument("--check", action="store_true", help="Validate every selected input/output without starting training")
    parser.add_argument("--restart-incomplete", action="store_true", help="Retrain matching partial runs from FP32; completed runs still skipped")
    parser.add_argument("--manifest", type=Path, help="Optionally save the complete plan as JSON")
    args = parser.parse_args()
    try:
        options = {k: v for k, v in vars(args).items() if k not in ("dry_run", "check", "restart_incomplete", "manifest")}
        runs = plan_runs(**options)
        print(f"QAT sweep: {len(runs)} runs | preset={args.preset} | experiment={runs[0]['experiment_tag']}", flush=True)
        if args.manifest:
            args.manifest.parent.mkdir(parents=True, exist_ok=True)
            args.manifest.write_text(json.dumps(runs, indent=2) + "\n", encoding="utf-8")
        if args.dry_run:
            for run in runs:
                print(shlex.join(run["command"]))
            return 0
        if args.check:
            pending = preflight(runs, args.restart_incomplete)
            print(f"Preflight passed: {len(pending)} pending; {len(runs) - len(pending)} complete.")
            return 0
        execute(runs, restart_incomplete=args.restart_incomplete)
        print("QAT sweep complete.")
        return 0
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

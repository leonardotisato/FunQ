"""
FINN Dataflow Build for QAT test_resnet -> Alveo U250 artifact.

Defaults to the current trim-160 fit experiment (`6w6a`), while still allowing
explicit `--onnx` overrides such as `test_resnet_trim192_6w6a`.

Usage (estimates only):
    python src/finn_build/build_test_resnet.py --estimates-only

Usage (stop after a specific step for Netron inspection):
    python src/finn_build/build_test_resnet.py --stop-after step_test_resnet_streamline

Must run inside the FINN Docker container.
"""

import argparse
import glob
import getpass
import os
import shutil
import sys
import time
from pathlib import Path

parser = argparse.ArgumentParser(
    description="FINN dataflow build for QAT test_resnet -> Alveo U250."
)
parser.add_argument("--onnx", default="models/test_resnet_trim160_6w6a.onnx")
parser.add_argument("--output-dir", default=None,
                    help="Output directory. If omitted, derived from --onnx and --board.")
parser.add_argument("--estimates-only", action="store_true")
parser.add_argument("--stop-after", default=None,
                    help="Stop after this step name (for incremental debugging).")
parser.add_argument("--start-from", default=None,
                    help="Resume from this step using the previous intermediate checkpoint in --output-dir.")
parser.add_argument("--synth-clk-ns", type=float, default=10.0)
parser.add_argument("--target-fps", type=int, default=1)
parser.add_argument("--folding-config", default=None,
                    help="Manual folding config JSON. If None, uses target_fps auto-folding.")
parser.add_argument("--manual-fifo-depths", action="store_true",
                    help="Use FIFO depths from --folding-config instead of auto FIFO sizing.")
parser.add_argument("--board", default="U250",
                    help="Target board (default: U250)")
args = parser.parse_args()

if args.manual_fifo_depths and args.folding_config is None:
    print("\n[ERROR] --manual-fifo-depths requires --folding-config.")
    sys.exit(1)

try:
    import finn.util.basic as finn_basic

    from finn.builder.build_dataflow_config import (
        DataflowBuildConfig,
        DataflowOutputType,
        ShellFlowType,
        VitisOptStrategyCfg,
    )
    from finn.builder.build_dataflow import build_dataflow_cfg
    from finn.util.basic import pynq_part_map, alveo_part_map

    from custom_steps_test_resnet import (
        step_fundus_attach_preproc,
        step_test_resnet_fix_multithreshold_ties,
        step_test_resnet_streamline,
        step_test_resnet_lower,
        step_test_resnet_to_hw,
        step_test_resnet_apply_bram_relief_config,
    )
except ImportError as e:
    print(f"\n[ERROR] Import failed: {e}\n"
          "Are you inside the FINN Docker container?")
    sys.exit(1)

if not os.path.exists(args.onnx):
    print(f"\n[ERROR] ONNX file not found: {args.onnx}")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Build steps
# ---------------------------------------------------------------------------
estimate_steps = [
    "step_qonnx_to_finn",
    step_test_resnet_fix_multithreshold_ties,
    step_fundus_attach_preproc,
    "step_tidy_up",
    step_test_resnet_streamline,
    step_test_resnet_lower,
    step_test_resnet_to_hw,
    "step_create_dataflow_partition",
    "step_specialize_layers",
    "step_minimize_bit_width",
    "step_generate_estimate_reports",
]

def step_names(steps):
    return [s if isinstance(s, str) else s.__name__ for s in steps]


def resolve_display_steps(steps, start_from=None, stop_after=None):
    names = step_names(steps)
    start_idx = 0
    stop_idx = len(steps) - 1

    if start_from is not None:
        if start_from not in names:
            print(f"\n[ERROR] --start-from '{start_from}' not found in step list.")
            print(f"  Valid step names: {names}")
            sys.exit(1)
        start_idx = names.index(start_from)

    if stop_after is not None:
        if stop_after not in names:
            print(f"\n[ERROR] --stop-after '{stop_after}' not found in step list.")
            print(f"  Valid step names: {names}")
            sys.exit(1)
        stop_idx = names.index(stop_after)

    if start_idx > stop_idx:
        print(f"\n[ERROR] --start-from '{start_from}' comes after --stop-after '{stop_after}'.")
        sys.exit(1)

    return steps[start_idx : stop_idx + 1]


# Insert folding step before minimize_bit_width
minimize_idx = estimate_steps.index("step_minimize_bit_width")
if args.folding_config:
    estimate_steps.insert(minimize_idx, "step_apply_folding_config")
else:
    estimate_steps.insert(minimize_idx, "step_target_fps_parallelization")

# ---------------------------------------------------------------------------
# Board config
# ---------------------------------------------------------------------------
U250_INSTALLED_PLATFORM = "xilinx_u250_gen3x16_xdma_4_1_202210_1"
U250_FINN_DEFAULT_PLATFORM = "xilinx_u250_gen3x16_xdma_2_1_202010_1"
BOARD_MVAU_WWIDTH_MAX = {
    "U250": 10000,
}
BOARD_VITIS_OPT_STRATEGY = {
    "U250": VitisOptStrategyCfg.PERFORMANCE_BEST,
}


def patch_u250_vitis_platform():
    """Patch FINN's U250 Vitis platform alias without touching HLS set_part."""
    patched = []
    for name, value in vars(finn_basic).items():
        if isinstance(value, dict) and value.get("U250") == U250_FINN_DEFAULT_PLATFORM:
            value["U250"] = U250_INSTALLED_PLATFORM
            patched.append(name)
    return patched


def _vitis_link_projects():
    # FINN_BUILD_DIR is set by run-docker.sh; per-build values keep parallel builds apart.
    tmp_root = Path(os.environ.get("FINN_BUILD_DIR", f"/tmp/finn_dev_{getpass.getuser()}"))
    return {Path(p) for p in glob.glob(str(tmp_root / "vitis_link_proj_*"))}


def _copy_vitis_reports(output_dir, before_projects, build_start_time):
    """Copy useful Vitis-Alveo link reports from /tmp into the build report dir."""
    after_projects = _vitis_link_projects()
    candidates = list(after_projects - before_projects)
    if not candidates:
        candidates = [
            p for p in after_projects
            if p.stat().st_mtime >= build_start_time - 60
        ]
    if not candidates:
        print("  Vitis report copy: no vitis_link_proj_* directory found")
        return

    vitis_proj = max(candidates, key=lambda p: p.stat().st_mtime)
    impl_dir = vitis_proj / "_x/link/vivado/vpl/prj/prj.runs/impl_1"
    if not impl_dir.is_dir():
        print(f"  Vitis report copy: implementation report dir not found under {vitis_proj}")
        return

    report_dir = Path(output_dir) / "report" / "vitis_alveo"
    report_dir.mkdir(parents=True, exist_ok=True)

    report_names = [
        "full_util_synthed.rpt",
        "full_util_placed.rpt",
        "full_util_routed.rpt",
        "kernel_util_synthed.rpt",
        "kernel_util_placed.rpt",
        "kernel_util_routed.rpt",
        "slr_util_placed.rpt",
        "slr_util_routed.rpt",
        "hw_bb_locked_timing_summary_init.rpt",
        "hw_bb_locked_timing_summary_placed.rpt",
        "hw_bb_locked_timing_summary_routed.rpt",
        "dr_timing_summary.rpt",
    ]

    copied = []
    for name in report_names:
        src = impl_dir / name
        if src.is_file():
            shutil.copy2(src, report_dir / name)
            copied.append(name)

    for log_name in ["v++_a.log", "run_vitis_link.sh", "config.txt"]:
        src = vitis_proj / log_name
        if src.is_file():
            shutil.copy2(src, report_dir / log_name)
            copied.append(log_name)

    manifest = report_dir / "source.txt"
    manifest.write_text(
        "Vitis-Alveo reports copied from:\n"
        f"{vitis_proj}\n\n"
        "Key files:\n"
        "- full_util_routed.rpt: full linked Alveo design, including shell/platform\n"
        "- kernel_util_routed.rpt: user kernel region\n"
        "- slr_util_routed.rpt: resource distribution by SLR\n"
        "- hw_bb_locked_timing_summary_routed.rpt: routed timing summary\n"
        "- dr_timing_summary.rpt: dynamic region timing summary\n",
        encoding="utf-8",
    )
    print(f"  Vitis reports copied to {report_dir} ({len(copied)} files)")


board = args.board
supported_boards = {"Ultra96", "U250"}
if board not in supported_boards:
    print(f"\n[ERROR] Unsupported board for this test_resnet build: {board}")
    print(f"  Supported boards: {sorted(supported_boards)}")
    sys.exit(1)

patched_platform_maps = patch_u250_vitis_platform() if board == "U250" else []
mvau_wwidth_max = BOARD_MVAU_WWIDTH_MAX.get(board, 36)
vitis_opt_strategy = BOARD_VITIS_OPT_STRATEGY.get(board, VitisOptStrategyCfg.DEFAULT)

if board in pynq_part_map:
    shell_flow_type = ShellFlowType.VIVADO_ZYNQ
elif board in alveo_part_map:
    shell_flow_type = ShellFlowType.VITIS_ALVEO
else:
    print(f"\n[ERROR] Unknown board: {board}")
    print(f"  Valid Zynq boards: {list(pynq_part_map.keys())}")
    print(f"  Valid Alveo boards: {list(alveo_part_map.keys())}")
    sys.exit(1)

if args.output_dir is None:
    model_stem = os.path.splitext(os.path.basename(args.onnx))[0]
    board_tag = board.lower().replace("-", "_")
    args.output_dir = f"./build_finn_{model_stem}_{board_tag}"


hardware_steps = [
    "step_hw_codegen",
    "step_hw_ipgen",
    "step_set_fifo_depths",
    step_test_resnet_apply_bram_relief_config,
    "step_create_stitched_ip",
    "step_synthesize_bitfile",
    "step_make_pynq_driver",
    "step_deployment_package",
]

full_steps = estimate_steps + hardware_steps

# ---------------------------------------------------------------------------
# Select steps based on mode
# ---------------------------------------------------------------------------
if args.estimates_only:
    selected_steps = estimate_steps
    mode_label = "Estimate-only"
else:
    selected_steps = full_steps
    mode_label = "Full bitstream"

display_steps = resolve_display_steps(selected_steps, args.start_from, args.stop_after)
if args.start_from or args.stop_after:
    start_lbl = args.start_from or step_names(selected_steps)[0]
    stop_lbl = args.stop_after or step_names(selected_steps)[-1]
    mode_label = f"Incremental ({start_lbl} -> {stop_lbl})"


# ---------------------------------------------------------------------------
# Print config
# ---------------------------------------------------------------------------
print(f"\n{'=' * 60}")
print(f"  FINN Dataflow Build - test_resnet")
print(f"{'=' * 60}")
print(f"  Mode:    {mode_label}")
print(f"  Steps:   {len(selected_steps)}")
print(f"  Model:   {args.onnx}")
part = pynq_part_map.get(board) or alveo_part_map.get(board, "N/A")
print(f"  Board:   {board} ({part})")
print(f"  Flow:    {shell_flow_type}")
if patched_platform_maps:
    print(f"  Platform:{U250_INSTALLED_PLATFORM} (patched {patched_platform_maps})")
if shell_flow_type == ShellFlowType.VITIS_ALVEO:
    print(f"  Vitis:   optimization strategy {vitis_opt_strategy.value}")
print(f"  Clock:   {args.synth_clk_ns} ns ({1000/args.synth_clk_ns:.0f} MHz)")
print(f"  Target:  {args.target_fps} FPS")
print(f"  MVAU W:  max weight stream width {mvau_wwidth_max}")
print("  Folding: two-pass relaxation False")
if args.folding_config:
    print(f"  Folding: {args.folding_config}")
else:
    print(f"  Folding: auto (target_fps_parallelization)")
if args.manual_fifo_depths:
    print("  FIFO:    manual depths from folding config")
else:
    print("  FIFO:    auto sizing (largefifo_rtlsim)")
print(f"  Output:  {args.output_dir}")
print(f"  Steps:   {' -> '.join(step_names(display_steps))}")
print(f"{'=' * 60}\n")


# ---------------------------------------------------------------------------
# Build config
# ---------------------------------------------------------------------------
if args.estimates_only or args.stop_after:
    generate_outputs = [DataflowOutputType.ESTIMATE_REPORTS]
else:
    generate_outputs = [
        DataflowOutputType.ESTIMATE_REPORTS,
        DataflowOutputType.BITFILE,
        DataflowOutputType.PYNQ_DRIVER,
        DataflowOutputType.DEPLOYMENT_PACKAGE,
    ]
cfg = DataflowBuildConfig(
    steps=selected_steps,
    start_step=args.start_from,
    stop_step=args.stop_after,
    output_dir=args.output_dir,
    synth_clk_period_ns=args.synth_clk_ns,
    board=board,
    shell_flow_type=shell_flow_type,
    target_fps=args.target_fps,
    mvau_wwidth_max=mvau_wwidth_max,
    folding_two_pass_relaxation=False,
    vitis_opt_strategy=vitis_opt_strategy,
    folding_config_file=args.folding_config,
    auto_fifo_depths=not args.manual_fifo_depths,
    generate_outputs=generate_outputs,
    save_intermediate_models=True,
    split_large_fifos=True,
    default_swg_exception=True,
)

vitis_projects_before = _vitis_link_projects()
build_start_time = time.time()
build_dataflow_cfg(args.onnx, cfg)

if shell_flow_type == ShellFlowType.VITIS_ALVEO and not args.estimates_only and not args.stop_after:
    _copy_vitis_reports(args.output_dir, vitis_projects_before, build_start_time)

print(f"\n{'=' * 60}")
print(f"  Build complete!")
print(f"{'=' * 60}")
print(f"\nOutputs:        {args.output_dir}")
print(f"Intermediates:  {args.output_dir}/intermediate_models/")
print(f"\nInspect intermediate ONNX files in Netron for debugging.")

# FINN builds

Builds run on the dwarf5 VM inside the FINN container:

```bash
bash run_finn.sh                     # finds ~/finn (or $HPPS_FINN_ROOT) and opens a shell at the repo root
python3 src/finn_build/build_test_resnet.py --board U250 \
  --onnx models/<model>.onnx --output-dir build_finn_<model>_u250_<target> --target-fps <fps>
```

Finished builds are moved to `build/u250/` or `build/ultra96/` on the VM (gitignored).
Each build's report JSONs, `final_hw_config.json`, `time_per_step.json` and generated
`driver/driver.py` are copied to `results/hw/<board>/<build>/`; `results/hw/builds.csv`
lists every bitstream with its SHA-256.

## FINN checkout

`~/finn` on the VM is upstream FINN at `39f0c9a6` with one local change: in
`src/finn/transformation/fpgadataflow/create_stitched_ip.py` the stitched-IP synthesis runs
with `launch_runs synth_1 -jobs 1` instead of one job per CPU. Re-apply it after updating FINN.

## U250 build commands

Folder names encode `<model>_<bits>_u250_<target>`. The target is the `--target-fps` passed to
FINN, not a measured throughput.

| Build folder suffix | ONNX | Extra flags |
| --- | --- | --- |
| `test_resnet_{6w6a,8w8a}_u250_{2000,5000}fps` | `test_resnet_{6w6a,8w8a}.onnx` (legacy 224 input) | |
| `test_resnet_trim160_6w6a_u250_{2000,5000}fps` | `test_resnet_trim160_6w6a.onnx` | |
| `test_resnet_trim160_6w6a_u250_naive` | `test_resnet_trim160_6w6a.onnx` | not recorded |
| `test_resnet_trim192_6w6a_u250_{2000,5000}fps` | `test_resnet_trim192_6w6a.onnx` | |
| `test_resnet_slim128x64_trim160_6w6a_u250_{100,200,500,1000,2000,5000}fps` | `test_resnet_slim128x64_trim160_6w6a.onnx` | |
| `test_resnet_slim128x64_trim160_8w8a_u250_{2000,5000}fps` | `test_resnet_slim128x64_trim160_8w8a.onnx` | |

A rerun of the slim 6w6a 5000fps build and a `--synth-clk-ns 7.5` attempt were deleted: the first
was a replica, and in the second Vitis scaled the 133 MHz kernel clock back to 102.6 MHz.

The ONNX files used for these builds are archived on the VM in `models/archive/archived-for-sweep/`.

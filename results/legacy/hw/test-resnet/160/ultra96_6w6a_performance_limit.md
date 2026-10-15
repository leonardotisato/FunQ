# Ultra96 Performance Limit: test_resnet trim160 6w6a

## Summary

| Field | Value |
|---|---|
| Model | `test_resnet_trim160_6w6a` |
| Board | Ultra96 |
| FPGA part | `xczu3eg-sbva484-1-e` |
| FINN requested clock period | 5 ns |
| Routed/exported PL clock | 5.333 ns / 187.512 MHz |
| Measured board clock | 187.498125 MHz |
| Measured throughput | 6.35 images/s |
| Bottleneck node | `MVAU_rtl_2` |

FINN was configured with a 5 ns clock target, but the exported design metadata and routed timing report use a 5.333 ns PL clock, corresponding to about 187.5 MHz rather than 200 MHz. 

## Board Throughput

The generated FINN `driver.py` defaults to 100 MHz, so the measurement was taken by instantiating `FINNExampleOverlay` directly with `fclk_mhz=187.5`.

| Batch size | Runtime (ms) | Throughput (images/s) | fclk (MHz) |
|---:|---:|---:|---:|
| 16 | 2530.9126 | 6.3218 | 187.498125 |
| 128 | 20147.6309 | 6.3531 | 187.498125 |
| 256 | 40281.0788 | 6.3553 | 187.498125 |

## Timing Evidence

| Build | Routed clock | WNS (ns) | TNS (ns) | Failing setup endpoints | Result |
|---|---:|---:|---:|---:|---|
| 5 ns requested | 5.333 ns / 187.512 MHz | 0.296 | 0.000 | 0 | Pass |
| 4 ns requested | same folding/resources | -0.751 | -30.341 | 123 | Fail |

The 4 ns experiment did not change the folding shape or resource footprint, but it failed timing.

## Resource Evidence

| Resource | Used | Available | Utilization |
|---|---:|---:|---:|
| LUT | 57,058 | 70,560 | 80.86% |
| LUTRAM | 14,160 | - | - |
| SRL | 8,456 | - | - |
| LUTRAM + SRL | 22,616 | 28,800 | 78.53% |
| BRAM36 | 178 | - | - |
| BRAM18 | 75 | - | - |
| BRAM 36K-equivalent tiles | 215.5 | 216 | 99.77% |
| DSP | 15 | 360 | 4.17% |

BRAM is essentially exhausted and LUTRAM/SRL packing is close to the practical placement limit.

## Higher FPS Attempt

Increasing `target_fps` to the next folding point changes the bottleneck node, `MVAU_rtl_2`, by increasing its parallelism setting SIMD to 2. For all other nodes both SIMD and PE are set to 1. That configuration failed during placement with a LUTRAM/SRL packing error.

## Conclusion

For this network and FINN flow on Ultra96, the practical validated point is the 5 ns-requested build exported at 187.5 MHz, with measured throughput of about 6.35 images/s. Further clock tightening fails timing, and further folding for higher target FPS fails placement.

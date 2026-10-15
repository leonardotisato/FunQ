# Ultra96 Attempt: test_resnet trim192 6w6a

## Summary

| Field | Value |
|---|---|
| Model | `models/archive/test_resnet_192/test_resnet_trim192_6w6a.onnx` |
| Board | Ultra96 |
| Status | Failed during Vivado implementation DRC, before placement |
| Failure type | Resource overutilization |

## Resources

| Resource class | Used / Required | Available | Excess |
|---|---:|---:|---:|
| LUT as Memory | 29,336 | 28,800 | 536 |
| Block RAM Tile | 216 | 216 | 0 |
| RAMB18/RAMB36/FIFO compatible sites | 440 | 432 | 8 |

Conclusion: trim192 is closer than 224, but still does not fit Ultra96 with the
current FINN build settings. Both LUTRAM and BRAM/FIFO resources are effectively
exhausted, so moving BRAM FIFOs to LUTRAM is not enough.

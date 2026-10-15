# Ultra96 Attempt: test_resnet 224x224 6w6a

## Summary

| Field | Value |
|---|---|
| Model | `models/archive/test_resnet_224/test_resnet_6w6a.onnx` |
| Board | Ultra96 |
| Status | Failed during Vivado implementation DRC, before placement |
| Failure type | Resource overutilization |

## Resources

| Resource class | Required | Available | Excess |
|---|---:|---:|---:|
| LUT as Memory | 30,209 | 28,800 | 1,409 |
| RAMB18/RAMB36/FIFO compatible sites | 440 | 432 | 8 |

Conclusion: the non-trimmed 224x224 6w6a model does not fit Ultra96 with the
current FINN build settings.

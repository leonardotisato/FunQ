# 4w4a recovery notes

| Init source | LR | Weighted F1 | Accuracy | Best val F1 | Status |
|---|---:|---:|---:|---:|---|
| `fp32` | `1e-5` | `52.45%` | `57.14%` | `54.56%` | failed |
| `fp32` | `5e-5` | `63.12%` | `68.42%` | `66.08%` | failed |
| `fp32` | `1e-4` | `58.50%` | `62.41%` | `61.25%` | failed |
| `6w6a` | `1e-5` | `50.08%` | `55.64%` | `50.01%` | failed |
| `6w6a` | `5e-5` | `57.73%` | `63.16%` | `60.06%` | failed |
| **`6w6a`** | **`1e-4`** | **`65.79%`** | **`68.42%`** | **`62.84%`** | **failed** |
| `8w8a` | `1e-5` | `55.70%` | `60.15%` | `51.77%` | failed |
| `8w8a` | `5e-5` | `66.64%` | `69.92%` | `67.05%` | failed |
| `8w8a` | `1e-4` | `66.66%` | `70.68%` | `68.13%` | failed |

## FP32 initialization

Source checkpoint:
- `models/test_resnet_fp32_kd.pth`

### LR = 1e-5

- Result: original canonical `4w4a` KD-QAT baseline
- Gate A: failed
- Best epoch: `178`
- Best validation loss: `3.4506`
- Best validation weighted F1: `54.56%`
- Test accuracy: `57.14%`
- Test weighted precision: `52.86%`
- Test weighted recall: `57.14%`
- Test weighted F1: `52.45%`
- Test macro F1: `43.00%`

Class observations:
- `healthy`: still strong
  - precision `61.92%`
  - recall `93.60%`
  - F1 `74.37%`
- `nevus`: moderate
  - precision `57.10%`
  - recall `54.45%`
  - F1 `55.46%`
- `melanoma`: weak
  - precision `41.24%`
  - recall `26.99%`
  - F1 `32.12%`
- `chrpe`: detected only marginally
  - precision `31.75%`
  - recall `6.27%`
  - F1 `10.05%`

### LR = 5e-5

- Result: failed recovery attempt
- Gate A: failed
- Restored best epoch: `198`
- Best validation loss: `3.0346`
- Best validation weighted F1: `66.08%`
- Test accuracy: `68.42%`
- Test weighted precision: `62.07%`
- Test weighted recall: `68.42%`
- Test weighted F1: `63.12%`

Class observations:
- `healthy`: strong
  - precision `69.09%`
  - recall `100.00%`
  - F1 `81.59%`
- `nevus`: strong relative to the other FP32-initialized runs
  - precision `67.36%`
  - recall `74.97%`
  - F1 `70.74%`
- `melanoma`: improved but still below the desired level
  - precision `78.58%`
  - recall `42.35%`
  - F1 `54.40%`
- `chrpe`: collapsed completely
  - precision `0.00%`
  - recall `0.00%`
  - F1 `0.00%`

### LR = 1e-4

- Result: failed recovery attempt
- Gate A: failed
- Restored best epoch: `110`
- Best validation loss: `3.3338`
- Best validation weighted F1: `61.25%`
- Test accuracy: `62.41%`
- Test weighted precision: `60.62%`
- Test weighted recall: `62.41%`
- Test weighted F1: `58.50%`

Class observations:
- `healthy`: strong
  - precision `63.51%`
  - recall `100.00%`
  - F1 `77.54%`
- `nevus`: moderate
  - precision `64.15%`
  - recall `56.81%`
  - F1 `59.97%`
- `melanoma`: improved over low-LR FP32, but still below the desired level
  - precision `69.33%`
  - recall `34.76%`
  - F1 `45.64%`
- `chrpe`: still weak
  - precision `28.63%`
  - recall `12.49%`
  - F1 `16.76%`

## 6w6a warm start

Source checkpoint:
- `models/test_resnet_6w6a_qat.pth`

### LR = 1e-5

- Result: failed recovery attempt
- Gate A: failed
- Restored best epoch: `95`
- Best validation loss: `3.8015`
- Best validation weighted F1: `50.01%`
- Test accuracy: `55.64%`
- Test weighted precision: `48.82%`
- Test weighted recall: `55.64%`
- Test weighted F1: `50.08%`

Class observations:
- `healthy`: still relatively strong
  - precision `59.73%`
  - recall `97.89%`
  - F1 `74.04%`
- `nevus`: moderate-to-weak
  - precision `58.09%`
  - recall `41.01%`
  - F1 `47.75%`
- `melanoma`: weak
  - precision `43.54%`
  - recall `38.42%`
  - F1 `40.32%`
- `chrpe`: collapsed completely
  - precision `0.00%`
  - recall `0.00%`
  - F1 `0.00%`

### LR = 5e-5

- Result: failed recovery attempt
- Gate A: failed
- Restored best epoch: `167`
- Best validation loss: `3.3447`
- Best validation weighted F1: `60.06%`
- Test accuracy: `63.16%`
- Test weighted precision: `59.77%`
- Test weighted recall: `63.16%`
- Test weighted F1: `57.73%`

Class observations:
- `healthy`: strong
  - precision `70.14%`
  - recall `100.00%`
  - F1 `82.32%`
- `nevus`: moderate
  - precision `57.97%`
  - recall `65.84%`
  - F1 `61.41%`
- `melanoma`: weak
  - precision `49.99%`
  - recall `27.01%`
  - F1 `34.50%`
- `chrpe`: still almost absent
  - precision `43.29%`
  - recall `6.23%`
  - F1 `10.52%`

### LR = 1e-4

- Result: failed recovery attempt
- Gate A: failed
- Restored best epoch: `124`
- Best validation loss: `3.0696`
- Best validation weighted F1: `62.84%`
- Test accuracy: `68.42%`
- Test weighted precision: `67.87%`
- Test weighted recall: `68.42%`
- Test weighted F1: `65.79%`

Class observations:
- `healthy`: strong
  - precision `70.18%`
  - recall `100.00%`
  - F1 `82.34%`
- `nevus`: moderate
  - precision `65.07%`
  - recall `63.68%`
  - F1 `64.10%`
- `melanoma`: improved but still below target
  - precision `75.01%`
  - recall `46.04%`
  - F1 `56.48%`
- `chrpe`: improved but still weak
  - precision `57.23%`
  - recall `25.15%`
  - F1 `33.94%`

## 8w8a warm start

Source checkpoint:
- `models/test_resnet_8w8a_qat.pth`

### LR = 1e-5

- Result: failed recovery attempt
- Gate A: failed
- Restored best epoch: `112`
- Best validation loss: `3.5577`
- Best validation weighted F1: `51.77%`
- Test accuracy: `60.15%`
- Test weighted precision: `56.57%`
- Test weighted recall: `60.15%`
- Test weighted F1: `55.70%`

Class observations:
- `healthy`: strong
  - precision `65.26%`
  - recall `95.80%`
  - F1 `77.48%`
- `nevus`: moderate
  - precision `58.80%`
  - recall `61.44%`
  - F1 `59.82%`
- `melanoma`: weak
  - precision `54.69%`
  - recall `23.20%`
  - F1 `31.99%`
- `chrpe`: still weak
  - precision `28.54%`
  - recall `12.49%`
  - F1 `16.75%`

### LR = 5e-5

- Result: failed recovery attempt
- Gate A: failed
- Restored best epoch: `158`
- Best validation loss: `2.8088`
- Best validation weighted F1: `67.05%`
- Test accuracy: `69.92%`
- Test weighted precision: `66.05%`
- Test weighted recall: `69.92%`
- Test weighted F1: `66.64%`

Class observations:
- `healthy`: strong
  - precision `75.84%`
  - recall `93.66%`
  - F1 `83.68%`
- `nevus`: strong relative to the other 4w4a runs
  - precision `65.36%`
  - recall `77.17%`
  - F1 `70.56%`
- `melanoma`: improved and usable, though still below the desired level
  - precision `77.72%`
  - recall `53.80%`
  - F1 `63.04%`
- `chrpe`: still the main failure mode
  - precision `19.78%`
  - recall `6.23%`
  - F1 `9.07%`

### LR = 1e-4

- Result: failed recovery attempt
- Gate A: failed
- Restored best epoch: `185`
- Best validation loss: `3.0144`
- Best validation weighted F1: `68.13%`
- Test accuracy: `70.68%`
- Test weighted precision: `66.90%`
- Test weighted recall: `70.68%`
- Test weighted F1: `66.66%`

Class observations:
- `healthy`: strong
  - precision `74.97%`
  - recall `95.75%`
  - F1 `83.97%`
- `nevus`: strong relative to the other 4w4a runs
  - precision `67.24%`
  - recall `79.45%`
  - F1 `72.63%`
- `melanoma`: improved and usable, though still below the desired level
  - precision `72.18%`
  - recall `49.99%`
  - F1 `58.51%`
- `chrpe`: still the dominant failure mode
  - precision `31.75%`
  - recall `6.23%`
  - F1 `10.00%`
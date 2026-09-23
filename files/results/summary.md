### Test ROC-AUC vs. training-set size

| Model | 1% (1,440) | 10% (14,400) | 20% (28,800) | 30% (43,200) | 100% (144,000) |
|---|---:|---:|---:|---:|---:|
| CapsNet | 0.876 ± 0.004 | 0.939 ± 0.004 | 0.966 ± 0.001 | 0.971 ± 0.001 | 0.976 ± 0.001 |
| Matched CNN | 0.872 ± 0.009 | **0.961 ± 0.001** | **0.979 ± 0.000** | **0.982 ± 0.000** | **0.984 ± 0.000** |
| Simple CNN | **0.912 ± 0.006** | 0.951 ± 0.001 | 0.952 ± 0.002 | 0.952 ± 0.001 | 0.954 ± 0.001 |

### All test metrics at 100% of the training data

| Model | ROC-AUC | Accuracy | Sensitivity | Specificity | F1 |
|---|---:|---:|---:|---:|---:|
| CapsNet | 0.976 ± 0.001 | 0.923 ± 0.002 | 0.914 ± 0.009 | 0.932 ± 0.006 | 0.922 ± 0.002 |
| Matched CNN | 0.984 ± 0.000 | 0.939 ± 0.001 | 0.940 ± 0.004 | 0.938 ± 0.005 | 0.939 ± 0.001 |
| Simple CNN | 0.954 ± 0.001 | 0.885 ± 0.001 | 0.880 ± 0.011 | 0.890 ± 0.009 | 0.884 ± 0.002 |

### Model size and compute (RTX 5080, bf16 autocast, batch 128)

| Model | Parameters (inference) | + decoder (training only) | Training steps/s | Inference images/s |
|---|---:|---:|---:|---:|
| CapsNet | 3,388,736 | 324,979 | 123 | 78,976 |
| Matched CNN | 3,389,057 | – | 183 | 98,115 |
| Simple CNN | 60,545 | – | 237 | 119,581 |

### CapsNet ablations (test ROC-AUC)

| Variant | 1% | 10% | 100% |
|---|---:|---:|---:|
| CapsNet (3 routing iterations + decoder) | 0.876 ± 0.004 | 0.939 ± 0.004 | 0.976 ± 0.001 |
| CapsNet without routing (1 iteration) | 0.876 ± 0.001 | 0.938 ± 0.002 | 0.976 ± 0.000 |
| CapsNet without reconstruction decoder | 0.875 ± 0.009 | 0.936 ± 0.005 | 0.973 ± 0.002 |
| Matched CNN | 0.872 ± 0.009 | 0.961 ± 0.001 | 0.984 ± 0.000 |

### Robustness: test ROC-AUC under the strongest shift (models trained on 100%)

| Model | Clean | rotation (45) | zoom (1.5) | blur (2.5) | stain (0.2) |
|---|---:|---:|---:|---:|---:|
| CapsNet | 0.976 ± 0.001 | 0.958 ± 0.001 | 0.840 ± 0.007 | 0.812 ± 0.032 | 0.793 ± 0.008 |
| Matched CNN | 0.984 ± 0.000 | 0.968 ± 0.001 | 0.893 ± 0.007 | 0.821 ± 0.010 | 0.758 ± 0.008 |
| Simple CNN | 0.954 ± 0.001 | 0.938 ± 0.005 | 0.877 ± 0.006 | 0.709 ± 0.025 | 0.863 ± 0.019 |

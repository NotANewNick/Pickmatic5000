| run | split | GPU | class acc | body MSE | FID | copy | complete | steps/img | rule steps | rule error | check wrong | check missed | tok/step | seq. layers/img | token-layers/img | exit depth | bg depth | stroke depth | s/img |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| diffusion_s1 | valid | NVIDIA GeForce RTX 2080 Ti (cuda) | 0.877 | 0.173 | 124.033 | 2.002 | 1.000 | 1.000 | 0.000 | - | - | - | 560.000 | - | - | - | - | - | 0.003 |
| diffusion_s2 | valid | NVIDIA GeForce RTX 2080 Ti (cuda) | 0.975 | 0.084 | 20.897 | 0.981 | 1.000 | 2.000 | 0.000 | - | - | - | 280.000 | - | - | - | - | - | 0.002 |
| diffusion_s3 | valid | NVIDIA GeForce RTX 2080 Ti (cuda) | 0.974 | 0.083 | 14.567 | 0.978 | 1.000 | 3.000 | 0.000 | - | - | - | 186.667 | - | - | - | - | - | 0.003 |
| diffusion_s4 | valid | NVIDIA GeForce RTX 2080 Ti (cuda) | 0.981 | 0.083 | 11.829 | 0.992 | 1.000 | 4.000 | 0.000 | - | - | - | 140.000 | - | - | - | - | - | 0.003 |
| diffusion_s8 | valid | NVIDIA GeForce RTX 2080 Ti (cuda) | 0.989 | 0.083 | 8.190 | 0.999 | 1.000 | 8.000 | 0.000 | - | - | - | 70.000 | - | - | - | - | - | 0.006 |
| diffusion_s10 | valid | NVIDIA GeForce RTX 2080 Ti (cuda) | 0.991 | 0.084 | 7.502 | 1.005 | 1.000 | 10.000 | 0.000 | - | - | - | 56.000 | - | - | - | - | - | 0.008 |
| diffusion_s20 | valid | NVIDIA GeForce RTX 2080 Ti (cuda) | 0.990 | 0.085 | 5.953 | 1.024 | 1.000 | 20.000 | 0.000 | - | - | - | 28.000 | - | - | - | - | - | 0.014 |

Hardware-independent: class acc, body MSE, steps/img, tok/step, token-layers/img, exit depths.
Wall time (s/img) is only comparable between runs on the same GPU.

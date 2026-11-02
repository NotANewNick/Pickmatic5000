| run | split | GPU | class acc | body MSE | FID | copy | complete | steps/img | rule steps | rule error | check wrong | check missed | tok/step | seq. layers/img | token-layers/img | exit depth | bg depth | stroke depth | s/img |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| diffusion_s1 | test | NVIDIA GeForce RTX 2080 Ti (cuda) | 0.880 | 0.171 | 128.846 | 2.005 | 1.000 | 1.000 | 0.000 | - | - | - | 560.000 | - | - | - | - | - | 0.002 |
| diffusion_s10 | test | NVIDIA GeForce RTX 2080 Ti (cuda) | 0.985 | 0.078 | 6.313 | 0.981 | 1.000 | 10.000 | 0.000 | - | - | - | 56.000 | - | - | - | - | - | 0.007 |
| diffusion_s2 | test | NVIDIA GeForce RTX 2080 Ti (cuda) | 0.973 | 0.079 | 20.869 | 0.977 | 1.000 | 2.000 | 0.000 | - | - | - | 280.000 | - | - | - | - | - | 0.002 |
| diffusion_s20 | test | NVIDIA GeForce RTX 2080 Ti (cuda) | 0.981 | 0.079 | 5.203 | 0.996 | 1.000 | 20.000 | 0.000 | - | - | - | 28.000 | - | - | - | - | - | 0.013 |
| diffusion_s3 | test | NVIDIA GeForce RTX 2080 Ti (cuda) | 0.977 | 0.078 | 13.606 | 0.965 | 1.000 | 3.000 | 0.000 | - | - | - | 186.667 | - | - | - | - | - | 0.003 |
| diffusion_s4 | test | NVIDIA GeForce RTX 2080 Ti (cuda) | 0.981 | 0.077 | 10.656 | 0.971 | 1.000 | 4.000 | 0.000 | - | - | - | 140.000 | - | - | - | - | - | 0.004 |
| diffusion_s8 | test | NVIDIA GeForce RTX 2080 Ti (cuda) | 0.983 | 0.078 | 6.926 | 0.974 | 1.000 | 8.000 | 0.000 | - | - | - | 70.000 | - | - | - | - | - | 0.006 |

Hardware-independent: class acc, body MSE, steps/img, tok/step, token-layers/img, exit depths.
Wall time (s/img) is only comparable between runs on the same GPU.

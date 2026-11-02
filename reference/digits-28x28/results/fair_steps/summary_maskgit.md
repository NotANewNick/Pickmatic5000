| run | split | GPU | class acc | body MSE | FID | copy | complete | steps/img | rule steps | rule error | check wrong | check missed | tok/step | seq. layers/img | token-layers/img | exit depth | bg depth | stroke depth | s/img |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| maskgit_s1 | test | NVIDIA GeForce RTX 2080 Ti (cuda) | 0.976 | 0.078 | 36.370 | 1.272 | 1.000 | 1.000 | 0.000 | - | - | - | 140.000 | 8.000 | 1,576 | - | - | - | 0.002 |
| maskgit_s2 | test | NVIDIA GeForce RTX 2080 Ti (cuda) | 0.985 | 0.073 | 10.937 | 1.078 | 1.000 | 2.000 | 0.000 | - | - | - | 70.000 | 16.000 | 3,152 | - | - | - | 0.003 |
| maskgit_s3 | test | NVIDIA GeForce RTX 2080 Ti (cuda) | 0.983 | 0.072 | 8.314 | 1.024 | 1.000 | 3.000 | 0.000 | - | - | - | 46.667 | 24.000 | 4,728 | - | - | - | 0.004 |
| maskgit_s4 | test | NVIDIA GeForce RTX 2080 Ti (cuda) | 0.983 | 0.072 | 8.047 | 0.998 | 1.000 | 4.000 | 0.000 | - | - | - | 35.000 | 32.000 | 6,304 | - | - | - | 0.004 |

Hardware-independent: class acc, body MSE, steps/img, tok/step, token-layers/img, exit depths.
Wall time (s/img) is only comparable between runs on the same GPU.

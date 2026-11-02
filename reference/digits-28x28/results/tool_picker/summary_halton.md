| run | split | GPU | class acc | body MSE | FID | copy | complete | steps/img | rule steps | rule error | check wrong | check missed | tok/step | seq. layers/img | token-layers/img | exit depth | bg depth | stroke depth | s/img |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| maskgit_halton_s1 | valid | NVIDIA GeForce RTX 2080 Ti (cuda) | 0.980 | 0.080 | 38.973 | 1.272 | 1.000 | 1.000 | 0.000 | - | - | - | 140.000 | 8.000 | 1,576 | - | - | - | 0.002 |
| maskgit_halton_s2 | valid | NVIDIA GeForce RTX 2080 Ti (cuda) | 0.980 | 0.079 | 11.593 | 1.164 | 1.000 | 2.000 | 0.000 | - | - | - | 70.000 | 16.000 | 3,152 | - | - | - | 0.004 |
| maskgit_halton_s3 | valid | NVIDIA GeForce RTX 2080 Ti (cuda) | 0.987 | 0.079 | 4.763 | 1.088 | 1.000 | 3.000 | 0.000 | - | - | - | 46.667 | 24.000 | 4,728 | - | - | - | 0.005 |
| maskgit_halton_s4 | valid | NVIDIA GeForce RTX 2080 Ti (cuda) | 0.979 | 0.080 | 4.079 | 1.078 | 1.000 | 4.000 | 0.000 | - | - | - | 35.000 | 32.000 | 6,304 | - | - | - | 0.006 |
| maskgit_halton_s6 | valid | NVIDIA GeForce RTX 2080 Ti (cuda) | 0.989 | 0.079 | 4.077 | 1.051 | 1.000 | 6.000 | 0.000 | - | - | - | 23.333 | 48.000 | 9,456 | - | - | - | 0.008 |
| maskgit_halton_s8 | valid | NVIDIA GeForce RTX 2080 Ti (cuda) | 0.982 | 0.080 | 3.609 | 1.041 | 1.000 | 8.000 | 0.000 | - | - | - | 17.500 | 64.000 | 12,608 | - | - | - | 0.011 |
| maskgit_s4 | valid | NVIDIA GeForce RTX 2080 Ti (cuda) | 0.986 | 0.075 | 7.457 | 1.003 | 1.000 | 4.000 | 0.000 | - | - | - | 35.000 | 32.000 | 6,304 | - | - | - | 0.003 |
| maskgit_s8 | valid | NVIDIA GeForce RTX 2080 Ti (cuda) | 0.985 | 0.076 | 6.971 | 0.982 | 1.000 | 8.000 | 0.000 | - | - | - | 17.500 | 64.000 | 12,608 | - | - | - | 0.006 |

Hardware-independent: class acc, body MSE, steps/img, tok/step, token-layers/img, exit depths.
Wall time (s/img) is only comparable between runs on the same GPU.

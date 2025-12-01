# Question 1.1 Results

## (a) i: Net1 Results

| Conv Type | Validation Mean (%) | Validation Std (%) | Test Mean (%) | Test Std (%) |
|-----------|---------------------|--------------------|--------------|--------------|
| valid     |            100.00 |           0.0000 |        0.09 |      0.1136 |
| replicate |             98.30 |           0.4025 |       93.11 |      2.1116 |
| reflect   |            100.00 |           0.0000 |        0.00 |      0.0000 |
| circular  |             98.94 |           1.8975 |       81.30 |      3.5228 |
| sconv     |             98.88 |           0.2135 |        6.53 |      0.3551 |
| fconv     |             87.60 |           1.0964 |       88.75 |      0.4153 |

## (c) i: Net2 Results

| Conv Type | Validation Mean (%) | Validation Std (%) | Test Mean (%) | Test Std (%) |
|-----------|---------------------|--------------------|--------------|--------------|
| valid     |            100.00 |           0.0000 |        0.00 |      0.0000 |
| replicate |            100.00 |           0.0000 |        0.00 |      0.0000 |
| reflect   |            100.00 |           0.0000 |        0.00 |      0.0000 |
| circular  |            100.00 |           0.0000 |        0.00 |      0.0000 |
| sconv     |            100.00 |           0.0000 |        0.00 |      0.0000 |
| fconv     |            100.00 |           0.0000 |        0.00 |      0.0000 |

## (c) ii: Comparison of Test Accuracy (Net2 vs Net1)

| Conv Type | Net1 Test (%) | Net2 Test (%) | Change (%) | Direction |
|-----------|---------------|---------------|------------|----------|
| valid     |         0.09 |         0.00 |     -0.09 | Decreased |
| replicate |        93.11 |         0.00 |    -93.11 | Decreased |
| reflect   |         0.00 |         0.00 |     +0.00 | No change |
| circular  |        81.30 |         0.00 |    -81.30 | Decreased |
| sconv     |         6.53 |         0.00 |     -6.53 | Decreased |
| fconv     |        88.75 |         0.00 |    -88.75 | Decreased |

# Tiny-Imagenet-binarized-training
(Do not consider the accuracy)

Modified from brevitas CNV example.

https://github.com/Xilinx/brevitas/blob/master/src/brevitas_examples/bnn_pynq/models/CNV.py

Usage:

After download and unzip the Tiny imagenet dataset from http://cs231n.stanford.edu/tiny-imagenet-200.zip 
run

```bash
python trans.py
```
Which will rearrange the dict of the Tiny imagenet dataset and make it follow the reading requirement for the prtorch.

After that run

```bash
python traing_smaller.py
```

Which will finish the training process.


## Sub-project 3 verification

### Usage

1. copy the checkpoint file to `./checkpoint` folder
   - format: `ckpt.t7.<sess>_<seed>.pth`
   - example: `ckpt.t7.tiny_imagenet_smaller_11111.pth`
2. change the `SESS` and `SEED` in `run.sh`
3. run `run.sh` with default `python3`
   - if you want to inference only, comment `# CIFAR-10 Train` and `# tiny-imagenet Train` in `run.sh`

### Change Faulty Bit

1. change the `faulty_bit` at [line 138 in `CIFARDataset.py`](https://github.com/stevenokm/Tiny-Imagenet-binarized-training/blob/eef60de8a1a13704f959d3a205b8ea8fce82ef3a/CIFARDataset.py#L138)
2. change the `neighbor_offset` at [line 140 in `CIFARDataset.py`](https://github.com/stevenokm/Tiny-Imagenet-binarized-training/blob/eef60de8a1a13704f959d3a205b8ea8fce82ef3a/CIFARDataset.py#L140)
   - e.g. if the `faulty_bit` is 6 and want be fixed with 5th bit, then change `neighbor_offset` to -1

### Result @ CIFAR-10

Checkpoint file: [ckpt.t7.cifar10_smaller_11111.pth](https://drive.google.com/file/d/17BNowbr6Ljx9_62C9qLp2Ts2vIH02xPh/view?usp=sharing)

|          Status: Rand          | Accuracy (%) |
| :----------------------------: | :----------: |
|            Baseline            |    84.150    |
|          Faulty (6th)          |    34.261    |
| Repaired by neighbor bit (5th) |    52.230    |
|     Repaired by MSB (7th)      |    60.100    |

|          Status: Rand          | Accuracy (%) |
| :----------------------------: | :----------: |
|            Baseline            |    84.150    |
|          Faulty (5th)          |    66.464    |
| Repaired by neighbor bit (4th) |    71.390    |
|     Repaired by MSB (7th)      |    75.350    |

## Sub-project 2 verification

### Result @ CIFAR-10

Checkpoint file: 
- Baseline: [ckpt.t7.cifar10_smaller_11111.pth](https://drive.google.com/file/d/17BNowbr6Ljx9_62C9qLp2Ts2vIH02xPh/view?usp=sharing)
- Weight Standardization (WS): [ckpt.t7.cifar10_smaller_wsconv_11111.pth]()

Note: 10 times average accuracy.

| Noise \ Accuracy (%) | Baseline |  +WS   |
| :------------------: | :------: | :----: |
|  $\sigma_{val} = 0$  |  84.150  | 83.520 |
| $\sigma_{val} = 0.2$ |  80.567  | 80.700 |
| $\sigma_{val} = 0.4$ |  76.594  | 76.888 |
| $\sigma_{val} = 0.6$ |  68.495  | 68.857 |
| $\sigma_{val} = 0.8$ |  47.551  | 49.249 |
| $\sigma_{val} = 1.0$ |  18.243  | 25.752 |

## Sub-project 2 & Sub-project 3 Integration

### Result @ CIFAR-10

|  Status: Rand \ Accuracy (%)   | w/o WS | w/ WS  | $\delta$ |
| :----------------------------: | :----: | :----: | -------: |
|            Baseline            | 84.150 | 83.520 |   -0.630 |
|          Faulty (6th)          | 34.261 | 35.897 |    1.636 |
| Repaired by neighbor bit (5th) | 52.230 | 50.840 |   -1.390 |
| Repaired by neighbor bit (4th) | 45.060 | 43.930 |   -1.130 |
| Repaired by neighbor bit (3th) | 40.350 | 40.130 |   -0.220 |
| Repaired by neighbor bit (2th) | 36.870 | 37.740 |    0.870 |
| Repaired by neighbor bit (1th) | 34.800 | 37.200 |    2.400 |
| Repaired by neighbor bit (0th) | 35.750 | 36.010 |    0.260 |
|     Repaired by MSB (7th)      | 60.100 | 59.030 |   -1.070 |

|  Status: Rand \ Accuracy (%)   | w/o WS | w/ WS  | $\delta$ |
| :----------------------------: | :----: | :----: | -------: |
|            Baseline            | 84.150 | 83.520 |   -0.630 |
|          Faulty (5th)          | 66.464 | 65.692 |   -0.772 |
| Repaired by neighbor bit (6th) | 75.740 | 74.090 |   -1.650 |
| Repaired by neighbor bit (4th) | 71.390 | 70.310 |   -1.080 |
| Repaired by neighbor bit (3th) | 68.810 | 67.880 |   -0.930 |
| Repaired by neighbor bit (2th) | 66.540 | 66.350 |   -0.190 |
| Repaired by neighbor bit (1th) | 66.080 | 65.100 |   -0.980 |
| Repaired by neighbor bit (0th) | 65.810 | 65.800 |   -0.010 |
|     Repaired by MSB (7th)      | 75.350 | 74.220 |   -1.130 |

|  Status: Rand \ Accuracy (%)   | w/o WS | w/ WS  | $\delta$ |
| :----------------------------: | :----: | :----: | -------: |
|            Baseline            | 84.150 | 83.520 |   -0.630 |
|          Faulty (4th)          | 79.669 | 79.114 |   -0.555 |
| Repaired by neighbor bit (6th) | 81.960 | 81.270 |   -0.690 |
| Repaired by neighbor bit (5th) | 81.290 | 81.000 |   -0.290 |
| Repaired by neighbor bit (3th) | 80.100 | 79.860 |   -0.240 |
| Repaired by neighbor bit (2th) | 80.180 | 79.260 |   -0.920 |
| Repaired by neighbor bit (1th) | 79.890 | 78.740 |   -1.150 |
| Repaired by neighbor bit (0th) | 80.040 | 79.250 |   -0.790 |
|     Repaired by MSB (7th)      | 81.490 | 81.350 |   -0.140 |

|  Status: Rand \ Accuracy (%)   | w/o WS | w/ WS  | $\delta$ |
| :----------------------------: | :----: | :----: | -------: |
|            Baseline            | 84.150 | 83.520 |   -0.630 |
|          Faulty (3th)          | 83.213 | 82.577 |   -0.636 |
| Repaired by neighbor bit (6th) | 83.530 | 82.870 |   -0.660 |
| Repaired by neighbor bit (5th) | 83.550 | 83.310 |   -0.240 |
| Repaired by neighbor bit (4th) | 83.560 | 83.050 |   -0.510 |
| Repaired by neighbor bit (2th) | 83.570 | 82.670 |   -0.900 |
| Repaired by neighbor bit (1th) | 83.310 | 82.770 |   -0.540 |
| Repaired by neighbor bit (0th) | 83.520 | 82.220 |   -1.300 |
|     Repaired by MSB (7th)      | 83.100 | 82.630 |   -0.470 |

|  Status: Rand \ Accuracy (%)   | w/o WS | w/ WS  | $\delta$ |
| :----------------------------: | :----: | :----: | -------: |
|            Baseline            | 84.150 | 83.520 |   -0.630 |
|          Faulty (2th)          | 83.782 | 83.220 |   -0.562 |
| Repaired by neighbor bit (6th) | 83.760 | 83.030 |   -0.730 |
| Repaired by neighbor bit (5th) | 83.960 | 83.530 |   -0.430 |
| Repaired by neighbor bit (4th) | 83.720 | 83.190 |   -0.530 |
| Repaired by neighbor bit (3th) | 83.770 | 83.240 |   -0.530 |
| Repaired by neighbor bit (1th) | 83.840 | 83.300 |   -0.540 |
| Repaired by neighbor bit (0th) | 83.810 | 83.450 |   -0.360 |
|     Repaired by MSB (7th)      | 84.070 | 83.710 |   -0.360 |
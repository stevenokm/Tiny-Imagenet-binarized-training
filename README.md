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
|          Faulty (6th)          |    34.830    |
| Repaired by neighbor bit (5th) |    52.230    |
|     Repaired by MSB (7th)      |    60.100    |

|          Status: Rand          | Accuracy (%) |
| :----------------------------: | :----------: |
|            Baseline            |    84.150    |
|          Faulty (5th)          |    66.630    |
| Repaired by neighbor bit (4th) |    71.390    |
|     Repaired by MSB (7th)      |    75.350    |

## Sub-project 2 verification

### Result @ CIFAR-10

Checkpoint file: 
- Baseline: [ckpt.t7.cifar10_smaller_11111.pth](https://drive.google.com/file/d/17BNowbr6Ljx9_62C9qLp2Ts2vIH02xPh/view?usp=sharing)
- Weight Standardization (WS): [ckpt.t7.cifar10_smaller_wsconv_11111.pth]()

| Noise \ Accuracy (%) | Baseline |   WS   |
| :------------------: | :------: | :----: |
|  $\sigma_{val} = 0$  |  84.150  | 52.350 |
| $\sigma_{val} = 0.2$ |  80.270  | 51.370 |
| $\sigma_{val} = 0.4$ |  77.270  | 47.940 |
| $\sigma_{val} = 0.6$ |  69.420  | 42.620 |
| $\sigma_{val} = 0.8$ |  52.830  | 36.390 |
| $\sigma_{val} = 1.0$ |  23.700  | 26.650 |

# Description

This project is a simplified version of David Page's amazing blog post [How to Train Your ResNet 8: Bag of Tricks](https://myrtle.ai/learn/how-to-train-your-resnet-8-bag-of-tricks/), where a modified ResNet is trained to reach 94% accuracy in 26 seconds on a V100 GPU.

**Update:** Also check out https://github.com/tysam-code/hlb-CIFAR10 for even faster training!

# Usage

```bash
git clone https://github.com/99991/cifar10-fast-simple.git
cd cifar10-fast-simple
python3 train.py
```

# Example output

* Timing results using an A100 GPU only including training and excluding preprocessing and evaluation. The first run still includes some PyTorch/CuDNN initialization work and takes 15.49 sec.

```
epoch    batch    train time [sec]    validation accuracy
    1       97                1.43                 0.1557
    2      194                2.86                 0.7767
    3      291                4.29                 0.8756
    4      388                5.73                 0.8975
    5      485                7.16                 0.9118
    6      582                8.59                 0.9204
    7      679               10.02                 0.9294
    8      776               11.45                 0.9373
    9      873               12.88                 0.9401
   10      970               14.32                 0.9427

84 of 100 runs >= 94.0 % accuracy
Min  accuracy: 0.9379000000000001
Max  accuracy: 0.9438000000000001
Mean accuracy: 0.9409949999999995 +- 0.0012262442660416419
```

### Epoch vs validation accuracy

![epoch vs validation accuracy](https://raw.githubusercontent.com/99991/cifar10-fast-simple/main/doc/a100_epoch_vs_validation_error.png)

* Timing results using a P100 GPU.

```
Preprocessing: 3.03 seconds

epoch    batch    train time [sec]    validation accuracy
    1       97               10.07                 0.2460
    2      194               18.60                 0.7690
    3      291               27.13                 0.8754
    4      388               35.65                 0.8985
    5      485               44.18                 0.9107
    6      582               52.70                 0.9195
    7      679               61.23                 0.9272
    8      776               69.75                 0.9337
    9      873               78.28                 0.9397
   10      970               86.81                 0.9428
```

Train time does not include preprocessing, evaluating validation accuracy or importing the pytorch library.

The total time, i.e. what `time python3 train.py` would report, was 42.125 and 103.699 seconds respectively.

* Timing results on a V100 GPU ([thanks to @ZipengFeng](https://github.com/99991/cifar10-fast-simple/issues/1#issuecomment-1057876448))

```
Preprocessing: 4.78 seconds

epoch    batch    train time [sec]    validation accuracy
    1       97                4.24                 0.2051
    2      194                7.09                 0.7661
    3      291                9.93                 0.8749
    4      388               12.78                 0.8982
    5      485               15.62                 0.9139
    6      582               18.48                 0.9237
    7      679               21.33                 0.9301
    8      776               24.18                 0.9348
    9      873               27.04                 0.9396
   10      970               29.90                 0.9422
```

* Timing results on an RTX 3060 Laptop GPU (6 GB VRAM)

```
Files already downloaded and verified
Preprocessing: 4.67 seconds

epoch    batch    train time [sec]    validation accuracy
    1       97               10.50                 0.2578
    2      194               19.47                 0.7549
    3      291               28.21                 0.8737
    4      388               36.97                 0.9013
    5      485               45.72                 0.9127
    6      582               54.62                 0.9213
    7      679               63.39                 0.9286
    8      776               72.17                 0.9348
    9      873               80.95                 0.9395
   10      970               89.74                 0.9412
```

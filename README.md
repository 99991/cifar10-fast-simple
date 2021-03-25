# Description

This project is a simplified version of David Page's amazing blog post [How to Train Your ResNet 8: Bag of Tricks](https://myrtle.ai/learn/how-to-train-your-resnet-8-bag-of-tricks/), where a modified ResNet is trained to reach 94% accuracy in 26 seconds on a V100 GPU.

# Usage

```bash
git clone https://github.com/99991/cifar10-fast-simple.git
cd cifar10-fast-simple
python3 train.py
```

# Example output

* Timing results using an A100 GPU.

```
Preprocessing: 3.05 seconds

epoch    batch    train time [sec]    validation accuracy
    1       97                4.37                 0.2109
    2      194                7.77                 0.7620
    3      291               11.16                 0.8764
    4      388               14.54                 0.8979
    5      485               17.93                 0.9098
    6      582               21.32                 0.9177
    7      679               24.71                 0.9280
    8      776               28.09                 0.9332
    9      873               31.48                 0.9395
   10      970               34.86                 0.9430
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

The total time includes:

* Model creation and preprocessing (3 seconds)
* Evaluating validation accuracy (9 seconds)

Train time does not include preprocessing, evaluating validation accuracy or importing pytorch.

The total time, i.e. what `time python3 train.py` would report, was 42.125 and 103.699 seconds respectively.

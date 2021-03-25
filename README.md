# cifar10-fast-simple

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
epoch    batch    total time [sec]    validation accuracy 
    1       97                8.50                 0.2109
    2      194               12.09                 0.7620
    3      291               15.75                 0.8764
    4      388               19.41                 0.8979
    5      485               23.07                 0.9098
    6      582               26.74                 0.9177
    7      679               30.40                 0.9280
    8      776               34.06                 0.9332
    9      873               37.72                 0.9395
   10      970               41.38                 0.9430
```

* Timing results using a P100 GPU, which does not support half precision floating point.

```
epoch    batch    total time [sec]    validation accuracy 
    1       97               14.71                 0.2460
    2      194               24.06                 0.7690
    3      291               33.85                 0.8754
    4      388               43.63                 0.8985
    5      485               53.42                 0.9107
    6      582               63.21                 0.9195
    7      679               72.99                 0.9272
    8      776               82.78                 0.9337
    9      873               92.56                 0.9397
   10      970              102.35                 0.9428
```

The total time includes:

* Model creation and preprocessing (3 seconds)
* Evaluating validation accuracy (9 seconds)

The time to import torch (1.59 seconds), has not been included.

The time including everything, i.e. what you would get with `time python3 train.py`, was 103.94 seconds.

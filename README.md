## Models and Utilities to Train 121 UCI Datasets

![version](https://img.shields.io/badge/version-v1.0-green)
![python](https://img.shields.io/badge/python-3.8.3-blue)
![pytorch](https://img.shields.io/badge/pytorch-1.6.0-brightgreen)
![singa](https://img.shields.io/badge/singa-3.1.0-orange)

This repository provides **Models** and **Utilities** for evaluating models on [121 UCI Datasets](https://jmlr.org/papers/volume15/delgado14a/delgado14a.pdf).
There are 121 multi-class real-world classification tasks, whose features are all numerical featuers.

### Model supported

-   [ARM-Net]((https://dl.acm.org/doi/10.1145/3448016.3457321))
-   Feed-forward Networks

### How to Download Datasets

```
bash data/download.sh
```

### How to Run Models

```
python train_uci.py --model armnet --dataset adult --batch_size 256 --epoch 100 --exp_name armnet
```


### Contact
To ask questions or report issues, you can directly drop us an [email](mailto:shaofeng@comp.nus.edu.sg).
## Models and Utilities to Train 121 UCI Datasets

![version](https://img.shields.io/badge/version-v1.0-green)
![python](https://img.shields.io/badge/python-3.8.3-blue)
![pytorch](https://img.shields.io/badge/pytorch-1.6.0-brightgreen)
![singa](https://img.shields.io/badge/singa-3.1.0-orange)

This repository provides **Models** and **Utilities** for evaluating models on [121 UCI Datasets](https://jmlr.org/papers/volume15/delgado14a/delgado14a.pdf).
There are 121 multi-class real-world classification tasks, whose features are *all numerical features*.

### Model supported (updating)

| Model |  Code | Reference |
|-------|-----|-----------|
| Self-Normalizing Neural Networks | SNN, [snn.py](https://github.com/nusdbsystem/ARM-Net/blob/uci/models/snn.py) | [G Klambauer, et al. Self-Normalizing Neural Networks, 2017.](https://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf) |
| AFN | AFN/AFN+, [afn.py](https://github.com/nusdbsystem/ARM-Net/blob/uci/models/afn.py) | [W Cheng, et al. Adaptive Factorization Network: Learning Adaptive-Order Feature Interactions, 2020.](https://arxiv.org/abs/1909.03276) |
| ARM-Net | ARM-Net/ARM-Net+, [armnet.py](https://github.com/nusdbsystem/ARM-Net/blob/uci/models/armnet.py) | [S Cai, et al. ARM-Net: Adaptive Relation Modeling Network for Structured Data, 2021.](https://dl.acm.org/doi/10.1145/3448016.3457321) |

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
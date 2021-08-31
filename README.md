## Models and Utilities to Train 121 UCI Datasets

![version](https://img.shields.io/badge/version-v3.0-green)
![python](https://img.shields.io/badge/python-3.8.3-blue)
![pytorch](https://img.shields.io/badge/pytorch-1.6.0-brightgreen)
![singa](https://img.shields.io/badge/singa-3.1.0-orange)

This repository provides **Models** and **Utilities** for evaluating models on [121 UCI Datasets](https://jmlr.org/papers/volume15/delgado14a/delgado14a.pdf).
There are 121 multi-class real-world classification tasks, whose features are ***all numerical features***.

### Model supported (updating)

| Model |  Code | Reference |
|-------|-----|-----------|
| Self-Normalizing Neural Networks | SNN, [snn.py](https://github.com/nusdbsystem/ARM-Net/blob/uci_ray/models/snn.py) | [G Klambauer, et al. Self-Normalizing Neural Networks, 2017.](https://arxiv.org/pdf/1706.02515.pdf) |
| Perceiver IO | Perceiver-IO, [perceiverio.py](https://github.com/nusdbsystem/ARM-Net/blob/uci_ray/models/perceiverio.py) | [A. Jaegle, et al. Perceiver IO: A General Architecture for Structured Inputs & Outputs, 2021.](https://arxiv.org/pdf/2107.14795.pdf) |
| ARM-Net | ARM-Net/ARM-Net+, [armnet.py](https://github.com/nusdbsystem/ARM-Net/blob/uci_ray/models/armnet.py) | [S Cai, et al. ARM-Net: Adaptive Relation Modeling Network for Structured Data, 2021.](https://dl.acm.org/doi/10.1145/3448016.3457321) |

### How to Download Datasets

```sh
bash data/download.sh
```

### How to Run Models

```sh
python train.py --model snn --dataset adult --exp_name snn_adult

python train.py --model armnet --dataset bank --exp_name armnet_bank
```


### Contact
To ask questions or report issues, you can directly drop us an [email](mailto:shaofeng@comp.nus.edu.sg).
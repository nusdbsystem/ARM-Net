## ARM-Net: Adaptive Relation Modeling Network for Structured Data

![version](https://img.shields.io/badge/version-v1.0-green)
![python](https://img.shields.io/badge/python-3.8.3-blue)
![pytorch](https://img.shields.io/badge/pytorch-1.6.0-brightgreen)
![singa](https://img.shields.io/badge/singa-3.1.0-orange)

This repository contains our PyTorch implementation of [ARM-Net: Adaptive Relation Modeling Network for Structured Data](https://dl.acm.org/doi/10.1145/3448016.3457321).
We also provide the implementation of relevant baseline models for structured (tabular) data learning.

Our lightweight framework for structured data analytics implemented in [Singa](https://singa.incubator.apache.org/) can be found on our project [site](https://www.comp.nus.edu.sg/~dbsystem/armnet/).

<img src="https://user-images.githubusercontent.com/14588544/123823881-2659a980-d930-11eb-918e-dc3bfa83ad97.png" width="820" />


### Benchmark Dataset

#### Large Real-world Dataset (reported in the [ARM-Net paper](https://dl.acm.org/doi/10.1145/3448016.3457321))

* [Frappe - App Recommendation](https://www.baltrunas.info/research-menu/frappe)
* [MovieLens - Movie Recommendation](https://grouplens.org/datasets/movielens)
* [Avazu - Click-Through Rate Prediction](https://www.kaggle.com/c/avazu-ctr-prediction)
* [Criteo - Display Advertising Challenge](https://labs.criteo.com/2014/02/kaggle-display-advertising-challenge-dataset/)
* [Diabetes130 - Diabetes Readmission Prediction](https://archive.ics.uci.edu/ml/datasets/diabetes+130-us+hospitals+for+years+1999-2008)

#### Small to Medium Tabular Datasets ([121 UCI datasets](https://jmlr.org/papers/volume15/delgado14a/delgado14a.pdf))

> We also evaluated prior arts, latest models and our ARM-Net on the [UCI datasets](https://jmlr.org/papers/volume15/delgado14a/delgado14a.pdf). These datasets are ***multi-class*** real-world classification tasks, whose features are ***all converted into numerical features*** following [common practice](https://arxiv.org/pdf/2107.14795.pdf).
> 
> **Models** and **Utilities** for evaluating models on [121 UCI Datasets](https://jmlr.org/papers/volume15/delgado14a/delgado14a.pdf) are included in this [repository](https://github.com/nusdbsystem/ARM-Net/tree/uci).

### Baseline Model

| Model |  Code | Reference |
|-------|-----|-----------|
| Logistic Regression | LR, [lr.py](https://github.com/nusdbsystem/ARM-Net/blob/main/models/lr.py) | |
| Factorization Machine | FM, [fm.py](https://github.com/nusdbsystem/ARM-Net/blob/main/models/fm.py) | [S Rendle, Factorization Machines, 2010.](https://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf) |
| Attentional Factorization Machine | AFM, [afm.py](https://github.com/nusdbsystem/ARM-Net/blob/main/models/afm.py) | [J Xiao, et al. Attentional Factorization Machines: Learning the Weight of Feature Interactions via Attention Networks, 2017.](https://arxiv.org/abs/1708.04617) |
| Higher-Order Factorization Machines | HOFM, [hofm.py](https://github.com/nusdbsystem/ARM-Net/blob/main/models/hofm.py) | [ M Blondel, et al. Higher-Order Factorization Machines, 2016.](https://dl.acm.org/doi/10.5555/3157382.3157473) |
| Deep Neural Network | DNN, [dnn.py](https://github.com/nusdbsystem/ARM-Net/blob/main/models/dnn.py) | |
| Graph Convolutional Networks | GCN, [gcn.py](https://github.com/nusdbsystem/ARM-Net/blob/main/models/gcn.py) | [T Kipf, et al. Semi-Supervised Classification with Graph Convolutional Networks, 2016.](https://arxiv.org/abs/1609.02907)|
| Graph Convolutional Networks | GAT, [gat.py](https://github.com/nusdbsystem/ARM-Net/blob/main/models/gat.py) | [P Veličković, et al. Graph Attention Networks, 2017.](https://arxiv.org/abs/1710.10903)|
| Wide&Deep | Wide&Deep, [wd.py](https://github.com/nusdbsystem/ARM-Net/blob/main/models/wd.py) | [HT Cheng, et al. Wide & Deep Learning for Recommender Systems, 2016.](https://arxiv.org/abs/1606.07792) |
| Product Neural Network | IPNN/KPNN, [pnn.py](https://github.com/nusdbsystem/ARM-Net/blob/main/models/pnn.py) | [Y Qu, et al. Product-based Neural Networks for User Response Prediction, 2016.](https://arxiv.org/abs/1611.00144) |
| Neural Factorization Machine | NFM, [nfm.py](https://github.com/nusdbsystem/ARM-Net/blob/main/models/nfm.py) | [X He and TS Chua, Neural Factorization Machines for Sparse Predictive Analytics, 2017.](https://arxiv.org/abs/1708.05027) |
| DeepFM | DeepFM, [dfm.py](https://github.com/nusdbsystem/ARM-Net/blob/main/models/dfm.py) | [H Guo, et al. DeepFM: A Factorization-Machine based Neural Network for CTR Prediction, 2017.](https://arxiv.org/abs/1703.04247) |
| Deep & Cross Network | DCN/DCN+, [dcn.py](https://github.com/nusdbsystem/ARM-Net/blob/main/models/dcn.py) | [R Wang, et al. Deep & Cross Network for Ad Click Predictions, 2017.](https://arxiv.org/abs/1708.05123) |
| Gated Linear Unit | SA_GLU, [sa_glu.py](https://github.com/nusdbsystem/ARM-Net/blob/main/models/sa_glu.py) | [Y N. Dauphin, et al. Language Modeling with Gated Convolutional Networks, 2017](https://arxiv.org/abs/1612.08083) |
| xDeepFM | CIN/xDeepFM, [xdfm.py](https://github.com/nusdbsystem/ARM-Net/blob/main/models/xdfm.py) | [J Lian, et al. xDeepFM: Combining Explicit and Implicit Feature Interactions for Recommender Systems, 2018.](https://arxiv.org/abs/1803.05170) |
| Context-Aware Self-Attention Network | GC_ARM, [gc_arm.py](https://github.com/nusdbsystem/ARM-Net/blob/main/models/gc_arm.py) | [B Yang, et al. Context-Aware Self-Attention Networks, 2019](https://arxiv.org/abs/1902.05766) |
| AFN | AFN/AFN+, [afn.py](https://github.com/nusdbsystem/ARM-Net/blob/main/models/afn.py) | [W Cheng, et al. Adaptive Factorization Network: Learning Adaptive-Order Feature Interactions, 2020.](https://arxiv.org/abs/1909.03276) |
| ARM-Net | ARM-Net/ARM-Net+, [armnet.py](https://github.com/nusdbsystem/ARM-Net/blob/main/models/armnet.py) | [S Cai, et al. ARM-Net: Adaptive Relation Modeling Network for Structured Data, 2021.](https://dl.acm.org/doi/10.1145/3448016.3457321) |


### Citation

If you use our code in your research, please cite:
```
@inproceedings{DBLP:conf/sigmod/CaiZ0JOZ21,
  author    = {Shaofeng Cai and
               Kaiping Zheng and
               Gang Chen and
               H. V. Jagadish and
               Beng Chin Ooi and
               Meihui Zhang},
  title     = {ARM-Net: Adaptive Relation Modeling Network for Structured Data},
  booktitle = {{SIGMOD} '21: International Conference on Management of Data, Virtual
               Event, China, June 20-25, 2021},
  pages     = {207--220},
  publisher = {{ACM}},
  year      = {2021},
}
```

### Contact
To ask questions or report issues, you can directly drop us an [email](mailto:shaofeng@comp.nus.edu.sg).


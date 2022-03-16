## ARM-Net for End-to-end Log-based Anomaly Detection
![version](https://img.shields.io/badge/version-v4-green)
![python](https://img.shields.io/badge/python-3.9.7-blue)
![pytorch](https://img.shields.io/badge/pytorch-1.10.0-brightgreen)
![singa](https://img.shields.io/badge/singa-3.1.0-orange)

This repository contains our implementation of state-of-the-art *anomaly detection* models and [ARM-Net](https://dl.acm.org/doi/10.1145/3448016.3457321) 
for **Log-based Anomaly Detecion**.
Log-based anomaly detection aims to discover abnormal system behaviors (*binary classification*) by analyzing log sequences that are generated routinely by the system at runtime.

### [Data Format](https://github.com/nusdbsystem/ARM-Net/blob/log/data_loader.py)

Each log is a message in unstructued data format (raw text), which can be parsed into structured data format of a number of key information *fields*, e.g., date, pid, level, event ID and etc.

<img src="https://user-images.githubusercontent.com/14588544/158519066-bf5f0a77-2507-4235-ae78-b4521ffdb906.png" width="500" />

##### [Feature Extraction](https://github.com/nusdbsystem/ARM-Net/blob/log/data_loader.py)

1. Sampling method (sample a log sequence for anomaly detection)
   * window-based: extract features for detection from **a sliding window** over the log sequence
   * session-based: extract features for detection from **the whole log sequence**
2. Features
   * sequential: a sequence of log **eventID**s
   * quantitative: the **event count vector** of the sequence
   * semantic: a sequence of the **semantic embedding** of log eventIDs
   * tabular: a sequence of logs in **tabular data** format (time, date, pid, ...)

##### Benchmark Datasets

* HDFS
* BGL

### [Baseline Models](https://github.com/nusdbsystem/ARM-Net/tree/log/models)

| Model               |    Sampling    |         Feature          | Code                                                                                           | Reference                                                                        |
|---------------------|:--------------:|:------------------------:|------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------|
| Logistic Regression |    Session     |       quantitative       | LR, [lr.py](https://github.com/nusdbsystem/ARM-Net/blob/log/models/lr.py)                      |                                                                                  |
| Deep Neural Network |    Session     |       quantitative       | DNN, [dnn.py](https://github.com/nusdbsystem/ARM-Net/blob/log/models/dnn.py)                   |                                                                                  |
| DeepLog             |     Window     |        sequential        | DeepLog, [deeplog.py](https://github.com/nusdbsystem/ARM-Net/blob/log/models/deeplog.py)       | [**CCS'17**] [DeepLog](https://www.cs.utah.edu/~lifeifei/papers/deeplog.pdf)     |
| LogAnomaly             |     Window     | sequential, quantitative | LogAnomaly, [loganomaly.py](https://github.com/nusdbsystem/ARM-Net/blob/log/models/loganomaly.py) | [**IJCAI'19**] [LogAnomaly](https://www.ijcai.org/Proceedings/2019/658) |
| RobustLog             |    Session     |         semantic         | RobustLog, [robustlog.py](https://github.com/nusdbsystem/ARM-Net/blob/log/models/robustlog.py)       | [**FSE'19**] [RobustLog](https://dl.acm.org/doi/10.1145/3338906.3338931)      |
| ARM-Net             | Window/Session |         tabular          | ARM-Net, [armnet.py](https://github.com/nusdbsystem/ARM-Net/blob/log/models/armnet.py) | [**SIGMOD-21**][ ARM-Net](https://dl.acm.org/doi/10.1145/3448016.3457321)        |


##### Requirement

* python>=3.9.7
* PyTorch>=1.10


### Citation

If you use our code in your research, please cite:
```
S. Cai, K. Zheng, G. Chen, H.V. Jagadish, B.C. Ooi, M. Zhang. ARM-Net: Adaptive Relation Modeling Network for Structured Data. ACM International Conference on Management of Data (SIGMOD), 2021
```

### Contact
To ask questions or report issues, you can drop us an [email](mailto:shaofeng@comp.nus.edu.sg).




## Log Parsing and Embedding
![python](https://img.shields.io/badge/python-3.9.7-blue)

This repository contains our implementation of log parsing and embedding techniques.
As a basic phase of log-based anomaly detection, log parsing aims to convert raw unstructured log messages into structured logs.
Before feeding structured logs into machine learning and deep learning models to detect anomalous log messages,
log embedding first converts structured logs into numerical feature vectors.

### Log Parsing

With log parsing techniques, each unstructured log message is parsed into two parts, namely message header and log event.
The message header comprises a fixed number of key information fields, e.g., time, date, pid, level, and component for the dataset HDFS.
The log event consists of a constant log event template and multiple variable parameters.
Parameters in a log event are then replaced by <*>.

<img src="https://user-images.githubusercontent.com/14588544/158519066-bf5f0a77-2507-4235-ae78-b4521ffdb906.png" width="500" />

In this repository, we utilize a standard technique [Drain](https://jiemingzhu.github.io/pub/pjhe_icws2017.pdf) to perform log parsing.
Drain achieves higher parsing accuracy and is more efficient for publicly accessible datasets.

### Log Embedding

For each field of key information, we use numerical values from 0 to index its corresponding categorical values.
For the field of log event templates, we implement the [FastText](https://fasttext.cc/) model to obtain an embedding vector for each word,
and then adopt the TF-IDF strategy to assign a weight to each word.
The final semantic vector of the log event template is obtained by a weighted average of these word embedding vectors.

#### Benchmark Datasets
| Dataset | Fields in Log Message                                                               | #Features |
|---------|:-----------------------------------------------------------------------------------:|-----------|
| HDFS    | hour, minute, second, pid, level, component, log event template                     | 28002     |
| BGL     | hour, minute, second, millisecond, node, type, component, level, log event template | 71363     |

#### Requirements
* python>=3.9.7
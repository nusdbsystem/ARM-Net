## Models and Utilities to Train 121 UCI Datasets

![version](https://img.shields.io/badge/version-v3.0-green)
![python](https://img.shields.io/badge/python-3.8.3-blue)
![pytorch](https://img.shields.io/badge/pytorch-1.6.0-brightgreen)
![singa](https://img.shields.io/badge/singa-3.1.0-orange)

This repository provides **Models** and **Utilities** for evaluating models on [121 UCI Datasets](https://jmlr.org/papers/volume15/delgado14a/delgado14a.pdf).
There are 121 ***multi-class*** real-world classification tasks, whose features are ***all converted into numerical features*** following [common practice](https://arxiv.org/pdf/2107.14795.pdf).

### Model supported (updating)

| Model |  Code | Reference |
|-------|-----|-----------|
| Logistic Regression | LR, [lr.py](https://github.com/nusdbsystem/ARM-Net/blob/uci/models/lr.py) | |
| Factorization Machine | FM, [fm.py](https://github.com/nusdbsystem/ARM-Net/blob/uci/models/fm.py) | [S Rendle, Factorization Machines, 2010.](https://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf)
| Deep Neural Network | DNN, [dnn.py](https://github.com/nusdbsystem/ARM-Net/blob/uci/models/dnn.py) | |
| Self-Normalizing Neural Networks | SNN, [snn.py](https://github.com/nusdbsystem/ARM-Net/blob/uci/models/snn.py) | [G Klambauer, et al. Self-Normalizing Neural Networks, 2017.](https://arxiv.org/pdf/1706.02515.pdf) |
| Perceiver IO | Perceiver-IO, [perceiverio.py](https://github.com/nusdbsystem/ARM-Net/blob/uci/models/perceiverio.py) | [A. Jaegle, et al. Perceiver IO: A General Architecture for Structured Inputs & Outputs, 2021.](https://arxiv.org/pdf/2107.14795.pdf) |
| ARM-Net | ARM-Net/ARM-Net+, [armnet.py](https://github.com/nusdbsystem/ARM-Net/blob/uci/models/armnet.py) | [S Cai, et al. ARM-Net: Adaptive Relation Modeling Network for Structured Data, 2021.](https://dl.acm.org/doi/10.1145/3448016.3457321) |

### Main Results (evaluated on first 36/121 datasets, updating)
| Model |  Rank(Best_Cnt)  | abalone|  acute-inflammation|  acute-nephritis|  adult|  annealing|  arrhythmia|  audiology-std|  balance-scale|  balloons|  bank|  blood|  breast-cancer|  breast-cancer-wisc|  breast-cancer-wisc-diag|  breast-cancer-wisc-prog|  breast-tissue|  car|  cardiotocography-10clases|  cardiotocography-3clases|  chess-krvk|  chess-krvkp|  congressional-voting|  conn-bench-sonar-mines-rocks|  conn-bench-vowel-deterding|  connect-4|  contrac|  credit-approval|  cylinder-bands|  dermatology|  echocardiogram|  ecoli|  energy-y1|  energy-y2|  fertility|  flags|  glass|
|:-----------:|:-----------:|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|
| `n_samples` | - | 4177|  120|  120|  48842|  898|  452|  196|  625|  16|  4521|  748|  286|  699|  569|  198|  106|  1728|  2126|  2126|  28056|  3196|  435|  208|  990|  67557|  1473|  690|  512|  366|  131|  336|  768|  768|  100|  194|  214|
| `n_features` | - | 9|  7|  7|  15|  32|  263|  60|  5|  5|  17|  5|  10|  10|  31|  34|  10|  7|  22|  22|  7|  37|  17|  61|  12|  43|  10|  16|  36|  35|  11|  8|  9|  9|  10|  29|  10|
| LR | 6-th (0/36) | 0.6293/0.0080|  0.9833/0.0211|  0.9533/0.0552|  0.8423/0.0008|  0.1280/0.0172|  0.5442/0.0184|  0.7040/0.0480|  0.8718/0.0310|  0.7250/0.0935|  0.8904/0.0023|  0.7610/0.0043|  0.6923/0.0171|  0.9490/0.0090|  0.9641/0.0103|  0.6626/0.0656|  0.5283/0.1371|  0.8032/0.0052|  0.7595/0.0118|  0.8798/0.0120|  0.2743/0.0009|  0.9438/0.0035|  0.5705/0.0328|  0.7385/0.0186|  0.7121/0.0088|  0.7547/0.0004|  0.4829/0.0383|  0.8557/0.0119|  0.6305/0.0647|  0.9399/0.0313|  0.7600/0.0605|  0.7988/0.0510|  0.8391/0.0123|  0.8448/0.0297|  0.5800/0.1066|  0.4206/0.0365|  0.5290/0.0281|
| FM | 5-th (3/36) | 0.6329/0.0067|  0.9767/0.0389|  0.8700/0.0945|  0.8443/0.0005|  0.1960/0.1493|  0.5283/0.0211|  0.4880/0.0588|  `0.9224/0.0087`|  0.5750/0.1275|  0.8882/0.0028|  0.7647/0.0000|  0.6909/0.0604|  0.9599/0.0048|  `0.9697/0.0048`|  0.6626/0.0849|  0.5094/0.0818|  0.8882/0.0097|  0.7616/0.0161|  0.8903/0.0172|  0.3127/0.0035|  0.9796/0.0038|  0.5705/0.0306|  `0.9502/0.0087`|  0.9502/0.0087|  0.8264/0.0005|  0.4524/0.0140|  0.8638/0.0093|  0.7016/0.0250|  0.9202/0.0350|  0.7846/0.0600|  0.7595/0.0680|  0.8823/0.0086|  0.8604/0.0283|  0.7720/0.0688|  0.3423/0.0200|  0.5907/0.0361|
| DNN | 4-th (6/36) |0.6560/0.0051|  0.9900/0.0200|  0.9500/0.0316|  0.8519/0.0015|  0.4420/0.2346|  0.6442/0.0114|  0.6880/0.0466|  0.8987/0.0048|  0.5500/0.2318|  0.8900/0.0035|  0.7583/0.0050|  0.7147/0.0082|  0.9633/0.0033|  0.9648/0.0107|  0.7091/0.0475|  0.5849/0.0396|  0.9442/0.0034|  0.7797/0.0121|  0.9178/0.0031|  0.6842/0.0147|  0.9775/0.0032|  0.5834/0.0147|  0.7481/0.0377|  `0.9745/0.0063`|  0.8501/0.0023|  0.5084/0.0158|  0.8417/0.0187|  `0.7359/0.0386`|  `0.9639/0.0101`|  0.7846/0.0337|  `0.8524/0.0166`|  0.8688/0.0107|  `0.8865/0.0094`|  0.8320/0.0722|  `0.4969/0.0272`|  0.5850/0.0316|
| SNN | 3rd (6/36) |0.6457/0.0043|  0.9567/0.0389|  0.9000/0.0548|  0.8489/0.0009|  0.2280/0.2671|  0.5841/0.0410|  `0.7200/0.0253`|  0.9058/0.0240|  0.7250/0.1225|  0.8885/0.0019|  0.8885/0.0019|  0.7105/0.0105|  `0.9656/0.0041`|  0.9690/0.0112|  0.6727/0.0903|  `0.6000/0.0690`|  `0.9632/0.0066`|  `0.8008/0.0125`|  0.9029/0.0086|  0.6796/0.0141|  0.9726/0.0061|  0.5779/0.0209|  0.7135/0.0300|  0.9693/0.0100|  0.8491/0.0013|  0.5106/0.0098|  `0.8719/0.0121`|  0.7000/0.0163|  0.9388/0.0269|  0.7877/0.0439|  0.8179/0.035|  0.8714/0.0142|  0.8854/0.0154|  0.7600/0.1180|  0.4804/0.0231|  0.5738/0.0602|
| Perceiver-IO | 2nd (6/36) |0.6381/0.0143|  `1.0000/0.0000`|  0.9367/0.0531|  0.8521/0.0011|  `0.7600/0.0000`|  0.5602/0.0053|  0.0080/0.0160|  0.8821/0.0166|  `0.7750/0.0500`|  0.8850/0.0000|  0.7620/0.0000|  0.7063/0.0088|  0.9352/0.0313|  0.9556/0.0142|  `0.7596/0.0118`|  0.3208/0.0597|  0.9326/0.0120|  0.5325/0.0861|  0.7817/0.0035|  0.6834/0.0151|  0.8106/0.0895|  `0.6129/0.0000`|  0.5635/0.0817|  0.6732/0.0521|  0.7538/0.0000|  0.4457/0.0122|  0.7745/0.1075|  0.6133/0.0078|  0.4295/0.0754|  0.7662/0.0834|  0.6440/0.0239|  0.8417/0.0295|  0.8807/0.0325|  `0.8560/0.0480`|  0.3010/0.0247|  0.4093/0.0415|
| ARM-Net | 1st (15/36) |`0.6603/0.0034`|  0.9767/0.0389|  `0.9600/0.0800`|  `0.8562/0.0011`|  0.1500/0.1131|  `0.6487/0.0214`|  0.5520/0.0299|  0.9135/0.0070|  0.7500/0.0791|  `0.8922/0.0012`|  `0.8922/0.0012`|  `0.7203/0.0193`|  0.9530/0.0118|  0.9521/0.0186|  0.6828/0.0485|  0.5170/0.0638|  0.9463/0.0086|  0.7868/0.0054|  `0.9146/0.0051`|  `0.6982/0.0109`|  `0.9826/0.0040`|  0.5760/0.0193|  0.7712/0.0335|  0.9675/0.0115|  `0.8672/0.0028`|  `0.5228/0.0119`|  0.8620/0.0187|  0.7133/0.0305|  0.9497/0.0181|  `0.8338/0.0406`|  0.8214/0.0279|  `0.8844/0.0048`|  0.8750/0.0304|  0.8240/0.0528|  0.4330/0.0526|  `0.6150/0.0232`|

#### Dataset Preprocessing
> ##### All the datasets are preprocessed via [data_loader.py](https://github.com/nusdbsystem/ARM-Net/blob/uci/data_loader.py)
>
> 1. All features are converted into numerical features of `zero mean` and `unit variance` following [conventions](https://arxiv.org/pdf/2107.14795.pdf).
> 2. Each dataset is split into a fixed train and test set with 1:1 ratio following [convetions](https://arxiv.org/pdf/2107.14795.pdf).
> 

#### Evaluation Pipeline

> ##### All the results are obtained using [train.py](https://github.com/nusdbsystem/ARM-Net/blob/uci/train.py), the training pipeline is summarized as follows:
>
> 1. For each dataset, the fixed train set is split into [4:1 train/valid set](https://github.com/nusdbsystem/ARM-Net/blob/uci/train.py#L99), where hyper-parameters of each model are grid searched.
> 2. Each model is configured with a hyper-parameter search space, e.g., [ARM-Net CONFIG](https://github.com/nusdbsystem/ARM-Net/blob/uci/models/armnet.py#L37), with around 100~200 configs.
> 3. The config of hyper-params with the [highest valid set accuracy](https://github.com/nusdbsystem/ARM-Net/blob/uci/train.py#L153) is used for reporting the final results on the test set.
> 4. Reporting (`mean/std` of accuracy) by repeating the above steps with [5 independent runs](https://github.com/nusdbsystem/ARM-Net/blob/uci/train.py#L133).

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
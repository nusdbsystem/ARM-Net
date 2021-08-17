import torch
import sklearn


def cuda(v):
    if torch.cuda.is_available(): return v.cuda()
    return v


def uci_validation_set(X, y, split_perc=0.2):
    return sklearn.model_selection.train_test_split(X, y, test_size=split_perc, random_state=0)


def nsample_bsz_mapping(nsample):
    if nsample < 1000:
        bsz = 16
    elif nsample < 5000:
        bsz = 64
    elif nsample < 10000:
        bsz = 128
    else:
        bsz = 256
    return bsz
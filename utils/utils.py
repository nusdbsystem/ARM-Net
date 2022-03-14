import os
import torch
import time
import math
import logging
import sys
import shutil
from typing import Tuple
import random
import numpy as np
from torch import FloatTensor, LongTensor, Tensor


# setup logger
def logger(log_dir, need_time=True, need_stdout=False):
    log = logging.getLogger(__name__)
    log.setLevel(logging.DEBUG)
    fh = logging.FileHandler(log_dir)
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter(fmt='%(asctime)s %(message)s', datefmt='%m/%d/%Y-%I:%M:%S')
    if need_stdout:
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.DEBUG)
        log.addHandler(ch)
    if need_time:
        fh.setFormatter(formatter)
        if need_stdout:
            ch.setFormatter(formatter)
    log.addHandler(fh)
    return log


# detach and del logger
def remove_logger(logger):
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)
    del logger


def timeSince(since=None, s=None):
    if s is None:
        s = int(time.time() - since)
    m = math.floor(s / 60)
    s %= 60
    h = math.floor(m / 60)
    m %= 60
    return '%dh %dm %ds' %(h, m, s)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def roc_auc_compute_fn(y_pred, y_target):
    """ IGNITE.CONTRIB.METRICS.ROC_AUC """
    try:
        from sklearn.metrics import roc_auc_score
    except ImportError:
        raise RuntimeError("This contrib module requires sklearn to be installed.")

    if y_pred.requires_grad:
        y_pred = y_pred.detach()

    if y_target.is_cuda:
        y_target = y_target.cpu()
    if y_pred.is_cuda:
        y_pred = y_pred.cpu()

    y_true = y_target.numpy()
    y_pred = y_pred.numpy()
    try:
        return roc_auc_score(y_true, y_pred)
    except ValueError:
        print('ValueError: Only one class present in y_true. ROC AUC score is not defined in that case.')
        return 0.


def accuracy(y_pred, y_target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = y_target.size(0)

    _, pred = y_pred.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(y_target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size).item())

    return res


def is_in_topk(y_pred: FloatTensor, y_target: LongTensor, topk: int) -> LongTensor:
    """
    :param y_pred:      [bsz, nclass], FloatTensor
    :param y_target:    [bsz], LongTensor
    :param topk:        top_k accuracy
    :return:            [bsz], LongTensor, whether targets lie in top_k predictions
    """
    assert topk <= y_pred.size(1), f'top_k {topk} larger than nclass {y_pred.size(1)} !'
    _, pred = y_pred.topk(topk, 1, True, True)                  # bsz*topk
    pred = pred.t()                                             # topk*bsz
    correct = pred.eq(y_target.view(1, -1).expand_as(pred))     # topk_bsz
    return correct.sum(dim=0)                                   # bsz


def correct_to_acc(is_correct: LongTensor) -> float:
    """ convert prediction correctness into accuracy """
    return is_correct.sum().float().item() * 100. / is_correct.size(0)


def is_log_seq_anomaly(event_acc: LongTensor, nsamples: LongTensor) -> LongTensor:
    '''
    :param event_acc:   [N], LongTensor
    :param nsamples:    [bsz], LongTensor
    :return:            [bsz], LongTensor, whether log seq is anomaly
    '''
    pos, bsz = 0, nsamples.size(0)
    log_acc = torch.ones(bsz).long()                        # bsz
    for log_idx in range(bsz):
        log_pred = event_acc[pos: pos+nsamples[log_idx]]    # n_i
        if torch.all(log_pred):
            log_acc[log_idx] = 0
        pos += nsamples[log_idx].item()
    return log_acc


def f1_score(y_pred: Tensor, y_target: Tensor, epsilon: float = 1e-7) -> Tuple[float, float, float]:
    '''
    :param y_pred:      prediction label, ndim 1 or 2 (label or logits)
    :param y_target:    true label, ndim==1
    :param epsilon:     for numerical stability
    :return:            precision, recall and f1_score
    For Binary Clasification, can work with gpu tensors
    Reference: https://gist.github.com/SuperShinyEyes/dcc68a08ff8b615442e3bc6a9b55a354
    '''
    assert y_target.ndim == 1 and (y_pred.ndim == 1 or y_pred.ndim == 2)
    if y_pred.ndim == 2: y_pred = y_pred.argmax(dim=1)

    tp = (y_target * y_pred).sum().to(torch.float32)
    # tn = ((1 - y_target) * (1 - y_pred)).sum().to(torch.float32)
    fp = ((1 - y_target) * y_pred).sum().to(torch.float32)
    fn = (y_target * (1 - y_pred)).sum().to(torch.float32)

    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)

    f1 = 2 * (precision * recall) / (precision + recall + epsilon)
    return precision.item(), recall.item(), f1.item()


def load_checkpoint(args):
    try:
        return torch.load(args.resume)
    except RuntimeError:
        raise RuntimeError(f"Fail to load checkpoint at {args.resume}")


def save_checkpoint(ckpt, is_best, file_dir, file_name='model.ckpt'):
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
    ckpt_name = "{0}{1}".format(file_dir, file_name)
    torch.save(ckpt, ckpt_name)
    if is_best: shutil.copyfile(ckpt_name, "{0}{1}".format(file_dir, 'best_'+file_name))


def seed_everything(seed=2022):
    ''' [reference] https://gist.github.com/KirillVladimirov/005ec7f762293d2321385580d3dbe335 '''
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

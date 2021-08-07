import os
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import numpy as np
import sklearn.model_selection
from sklearn.metrics import roc_auc_score, f1_score
import random
from models import create_model, get_hyperparams
from utils  import logger
import argparse
from utils.uci_utils import *
from data_loader import uci_loader

torch.manual_seed(2021)
np.random.seed(2021)

def get_args():
    parser = argparse.ArgumentParser(description='UCI Training')
    parser.add_argument('--exp_name', default='test', type=str, help='exp name used to store log & checkpoint')
    parser.add_argument('--model', default='ffn', type=str, help='model type')
    parser.add_argument('--data_dir', default='./data/uci', type=str, help='dataset dir path')
    parser.add_argument('--dataset', default='abalone', type=str, help='dataset name')
    parser.add_argument('--metric', default='acc', type=str, help='evaluation metric')
    parser.add_argument('--epoch', default=100, type=int, metavar='N', help='training epoch')
    parser.add_argument('--batch_size', default=64, type=int, metavar='N', help='mini-batch size')
    parser.add_argument('--hyper_val_cnt', default=5, type=int, metavar='N', help='test cnt per best hyperparam set')
    parser.add_argument('--moving_avg_num', default=5, type=int, help='number of epochs for perf moving average')
    args = parser.parse_args()
    return args

args = get_args()
log_path = f'log/{args.exp_name}/stdout.log'
if not os.path.isdir(f'log/{args.exp_name}'): os.makedirs(f'log/{args.exp_name}', exist_ok=True)
plogger = logger(log_path, True, True).info
plogger(vars(args))

def score_key(score):
    # Note: sort in a descending way
    param, epochs, score = score
    return (score, -epochs)

def evaluate(model, val_loader, metric='acc'):
    model.eval()
    correct, total = 0, 0
    all_labels, all_preds = [], []
    for batch, labels in val_loader:
        batch = cuda(autograd.Variable(batch))
        labels = cuda(labels)
        outputs = model(batch)
        _, predictions = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predictions == labels).sum()
        all_labels += list(labels)
        all_preds += list(predictions)
    if metric == 'acc':
        return correct.item() / float(total)
    if metric == 'auc':
        try: return roc_auc_score(all_labels, all_preds)
        except: return 0
    if metric == 'f1':
        return f1_score(all_labels, all_preds)

def fit(params, train_loader, val_loader, test_loader, epochs=100, metric='acc'):
    # run once for finding best params, run val_cnt times only for the final test [each epoch times]
    if val_loader is None and test_loader is not None:
        n_test = args.hyper_val_cnt
    else:
        n_test = 1

    all_combinations, all_test_accs = [], []
    plogger(f'start training params {params} ...')
    for test_idx in range(n_test):
        model = cuda(create_model(params, plogger))
        criterion = cuda(nn.CrossEntropyLoss())
        optimizer = optim.SGD(model.parameters(), lr=params['lr'], weight_decay=params.get('l2', 0.0))

        for epoch in range(epochs):
            model.train()
            for batch, labels in train_loader:
                if batch.size(0) == 1: continue     # armnet bn batch_size>1
                batch = cuda(autograd.Variable(batch))
                labels = cuda(autograd.Variable(labels))
                optimizer.zero_grad()
                outputs = model(batch)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            accuracy = evaluate(model, val_loader or test_loader, metric)
            all_combinations.append((params, epoch+1, accuracy))
            plogger(f'[Test {test_idx+1}/{n_test}]\tEpoch:\t{epoch:>3d}/{args.epoch}\tacc:\t{accuracy:.4f}')

        if test_loader is not None:
            test_acc = evaluate(model, test_loader, metric)
            all_test_accs.append(test_acc)

    return all_combinations, all_test_accs

def get_performance(path, params, batch_size=args.batch_size, metric=args.metric):
    train_loader_small, val_loader, test_loader = uci_loader(path, batch_size, valid_perc=0.15)
    train_loader_big, _, _ = uci_loader(path, batch_size, valid_perc=0.)
    params['nclass'], params['nfeat'] = [train_loader_big.nclass], [train_loader_big.nfeat]
    params['model'] = [args.model]

    all_combinations = []
    all_params = list(sklearn.model_selection.ParameterGrid(params))
    # random.shuffle(all_params)
    for params in all_params:
        comb_list = fit(params, train_loader_small, val_loader, None,
                        metric=metric, epochs=args.epoch)[0]
        # smooth learning curve; comb_list item format: (params, epoch+1, accuracy)
        new_comb_list = []
        last_perfs = [0.0] * args.moving_avg_num
        for params, epochs, val_score in comb_list:
            last_perfs.pop(0)
            last_perfs.append(val_score)
            new_comb_list.append((params, epochs, np.mean(last_perfs)))
        all_combinations += new_comb_list
        max_perf_tuple = max(new_comb_list, key=score_key)
        plogger(f'==>>params:\t{params}\tbest_acc @{max_perf_tuple[1]}-th epoch:\t{max_perf_tuple[2]:.4f}')

    best_params, best_epochs, val_score = max(all_combinations, key=score_key)
    return (best_params, best_epochs, val_score,
        fit(best_params, train_loader_big, None, test_loader, best_epochs, metric=metric)[1]
    )


if __name__ == '__main__':

    data_path = f'{args.data_dir}/{args.dataset}'
    params = get_hyperparams(args.model)
    best_params, best_epochs, val_score, test_scores = get_performance(path=data_path, params=params)
    plogger(f'===>>>>best_params:\t{best_params}\n\tbest_epoch:\t\t{best_epochs:<3d}'
         f'\n\tval_acc:\t{val_score:.4f}\n\ttest_acc:\t{np.mean(test_scores):.4f}/{np.std(test_scores):.4f}')

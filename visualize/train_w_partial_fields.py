import os
import time
import argparse
from typing import Tuple
import sys
sys.path.append('../')

import torch
from torch import nn
from torch import optim

from data_loader import log_loader, get_vocab_size
from utils.utils import logger, remove_logger, AverageMeter, timeSince, seed_everything
from utils.utils import f1_score, is_in_topk, is_log_seq_anomaly, correct_to_acc, obtain_all_field_index
from models.utils import create_model, update_default_config
from optimizer.irm import IRM
from optimizer.randomizer import Randomizer


def get_args():
    parser = argparse.ArgumentParser(description='Log-Based Anomaly Detection')
    parser.add_argument('--exp_name', default='test', type=str, help='exp name for log & checkpoint')
    # 1. model config
    parser.add_argument('--model', default='tabularlog', type=str, help='model name')
    parser.add_argument('--nstep', type=int, default=10, help='number of log events per sequence')
    parser.add_argument('--tabular_cap', type=int, default=100, help='feature cap for tabular data, e.g., pid in HDFS')
    parser.add_argument('--dropout', default=0.0, type=float, help='dropout rate')
    ## 1.1 tabular
    parser.add_argument('--nemb', type=int, default=20, help='tabular embedding size')
    parser.add_argument('--nhid', type=int, default=64, help='number of hidden units')
    # 1.1.1 armnet
    parser.add_argument('--alpha', default=1.7, type=float, help='entmax alpha to control sparsity in ARM-Module')
    parser.add_argument('--nquery', type=int, default=8, help='number of output query vectors for each step')
    ## 1.2 log sequence
    parser.add_argument('--nhead', type=int, default=8, help='attention head per layer for log seq encoder')
    parser.add_argument('--nlayer', type=int, default=2, help='number of layers for log seq encoder')
    parser.add_argument('--dim_feedforward', type=int, default=256, help='FFN dimension for log seq encoder')
    ## 1.3 predictor
    parser.add_argument('--mlp_nlayer', type=int, default=2, help='number of layers for MLP predictor')
    parser.add_argument('--mlp_nhid', type=int, default=256, help='FFN dimension for MLP predictor')
    # 2 optimizer
    parser.add_argument('--epoch', type=int, default=100, help='number of maximum epochs')
    parser.add_argument('--patience', type=int, default=1, help='number of epochs for early stopping training')
    parser.add_argument('--bsz', type=int, default=256, help='batch size')
    parser.add_argument('--lr', default=3e-4, type=float, help='learning rate, default 3e-4')
    parser.add_argument('--eval_freq', type=int, default=1500, help='max number of training batches per epoch')
    parser.add_argument('--nenv', type=int, default=1, help='number of training environments')
    parser.add_argument('--rand_type', type=int, default=1, help='data type random type, see Randomizer.data_generator')
    # 2.0 env-generalization
    parser.add_argument('--lambda_p', default=3e-2, type=float, help='lambda for IRM/Reptile penalty, default 3e-2')
    # 2.1 irm
    parser.add_argument("--irm", action="store_true", default=False, help="whether to use irm for DG")
    # 3. dataset
    parser.add_argument("--session_based", action="store_true", default=False, help="to use only session features")
    parser.add_argument('--feature_code', type=int, default=4, help='1~15, default quantitative, binary code for'
        '[sequential, quantitative, semantic, tabular] <-> [0/1][0/1][0/1][0/1] see data_loader.decode_feature_code')
    parser.add_argument('--shuffle', type=int, default=1, help='0~3, default no shuffle, binary code for'
        '[whole dataset, valid+test set] <-> [0/1][0/1] see data_loader.decode_shuffle_code/log_loader')
    parser.add_argument("--only_normal", action="store_true", default=False, help="only train using normal log seq")
    parser.add_argument('--dataset', type=str, default='hdfs', help='dataset name for data_loader')
    parser.add_argument('--data_path', type=str, default='../data/Drain_result/HDFS.log_all.log', help='path')
    parser.add_argument('--test_perc', default=0.5, type=float, help='train/test data split perc among all data')
    parser.add_argument('--valid_perc', default=0.2, type=float, help='valid data split over test set')
    parser.add_argument('--nworker', default=0, type=int, help='number of data loading workers')
    parser.add_argument('--session_len', type=int, default=100, help='number of logs per session (for bgl dataset)')
    parser.add_argument('--step_size', type=int, default=20, help='number of logs to skip for the next session (bgl)')
    # 4. evaluation metric
    parser.add_argument('--topk', default=10, type=int, help='number of top candidate events for anomaly detection')
    # 5. log & checkpoint
    parser.add_argument('--log_dir', type=str, default='./log/', help='path to store log')
    parser.add_argument('--report_freq', type=int, default=50, help='report frequency')
    parser.add_argument('--seed', type=int, default=2022, help='seed for reproducibility')
    parser.add_argument('--repeat', type=int, default=1, help='number of repeats with seeds [seed, seed+repeat)')
    # 6. train with partial fields
    parser.add_argument('--nfield', type=int, default=1, help='number of fields (exclude eventID) used in modeling')

    args = parser.parse_args()
    # update args default arguments
    update_default_config(args)
    return args


def main():
    global args, best_valid_f1, start_time, vocab_sizes, field_idx, plogger
    # create model
    model = create_model(args, plogger, vocab_sizes[field_idx])
    plogger.info(vars(args))
    # optimizer
    opt_metric = nn.CrossEntropyLoss(reduction='none').cuda()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # gradient clipping
    for p in model.parameters():
        p.register_hook(lambda grad: torch.clamp(grad, -1., 1.))

    patience_cnt = 0
    for epoch in range(args.epoch):
        plogger.info(f'Epoch [{epoch:3d}/{args.epoch:3d}]')

        # train and eval
        run(epoch, model, train_loaders, opt_metric, plogger, optimizer=optimizer)
        valid_precision, valid_recall, valid_f1 = run(epoch, model, [valid_loader], opt_metric, plogger, namespace='val')
        test_precidion, test_recall, test_f1 = run(epoch, model, [test_loader], opt_metric, plogger, namespace='test')

        # record best f1 and save checkpoint
        if valid_f1 > best_valid_f1:
            patience_cnt = 0
            best_valid_f1, best_test_f1, best_precision, best_recall = valid_f1, test_f1, test_precidion, test_recall
            plogger.info(f'best test: valid {valid_f1:.4f}, test {test_f1:.4f}')
        else:
            patience_cnt += 1
            plogger.info(f'valid {valid_f1:.4f}, test {test_f1:.4f}')
            plogger.info(f'Early stopped, {patience_cnt}-th best f1 at epoch {epoch-1}')
        if patience_cnt >= args.patience:
            plogger.info(f'Final best valid f1 {best_valid_f1:.4f}, with test precision {best_precision:.4f} '
                         f'recall {best_recall:.4f} f1 {best_test_f1:.4f}')
            break

    plogger.info(f'Total running time: {timeSince(since=start_time)}')


#  train one epoch of train/val/test
def run(epoch, model, data_loaders, opt_metric, plogger, optimizer=None, namespace='train') \
        -> Tuple[float, float, float]:
    if namespace == 'train': model.train()
    else: model.eval()

    time_avg, loss_avg, accuracy_avg = AverageMeter(), AverageMeter(), AverageMeter()
    timestamp, precision, recall, f1, all_pred, all_label = time.time(), 0., 0., 0., [], []
    if args.irm: dummy_w = IRM.dummy_w.cuda()                                       # 1

    loader = Randomizer.data_generator(data_loaders, args.rand_type, max_nbatch=args.eval_freq)
    for env_idx, batch_idx, batch in loader:
        features, pred_label = batch['features'], batch['pred_label']
        if namespace == 'train':
            pred = model(features)                                                  # bsz*2 or nwindow*nevent
            if args.irm:
                # inject dummy_w to the loss computation
                losses = opt_metric(pred*dummy_w, pred_label)                       # bsz or nwindow
                # compute penalty
                penalty = IRM.compute_penalty(losses, dummy_w)                      # 0
            else:
                losses = opt_metric(pred, pred_label)                               # bsz or nwindow

            optimizer.zero_grad()
            loss = losses.mean()                                                    # 0
            if args.irm: loss += args.lambda_p*penalty                              # 0
            loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                pred = model(features)                                              # bsz*2 or nwindow*nevent
                loss = opt_metric(pred, pred_label).mean()                          # 0

        loss_avg.update(loss.item(), batch['pred_label'].size(0))
        if args.session_based or namespace == 'train':
            correct = is_in_topk(pred, pred_label, topk=1)                          # bsz or nwindow
        else:
            correct = is_in_topk(pred, pred_label, topk=args.topk)                  # nwindow
            pred = is_log_seq_anomaly(correct, batch['nsamples'])                   # bsz (converted from nwindow)
            pred_label = batch['label']                                             # bsz

        accuracy_avg.update(correct_to_acc(correct), batch['pred_label'].size(0))
        if pred.ndim == 2:                                                          # reduce memory usage - window-based
            pred = pred.argmax(dim=1)                                               # nwindow*nevent -> nwindow
        all_pred.append(pred.cpu())                                                 # bsz or nwindow
        all_label.append(pred_label.cpu())                                          # bsz or nwindow

        time_avg.update(time.time() - timestamp); timestamp = time.time()
        if batch_idx % args.report_freq == 0:
            plogger.info(f'Env-{env_idx} Epoch [{epoch:3d}/{args.epoch}][{batch_idx:3d}/{len(data_loaders[env_idx])}] '
                         f'{time_avg.val:.3f} ({time_avg.avg:.3f}) Acc {accuracy_avg.val:.3f} ({accuracy_avg.avg:.3f}) '
                         f'Loss {loss_avg.val:.4f} ({loss_avg.avg:.4f})')

    precision, recall, f1 = f1_score(torch.cat(all_pred), torch.cat(all_label))
    plogger.info(f'{namespace}\tTime {timeSince(s=time_avg.sum):>12s}  Accuracy {accuracy_avg.avg:.4f}  '
                 f'Precision {precision:.4f}  Recall {recall:.4f}  F1-score {f1:.4f}  Loss {loss_avg.avg:8.4f}\n')
    if namespace == 'test':
        global last_f1, last_loss, current_f1, current_loss
        last_f1, last_loss, current_f1, current_loss = current_f1, current_loss, f1, loss_avg.avg
    return precision, recall, f1


# initialize global variables, load dataset
args = get_args()
vocab_sizes = get_vocab_size(args.dataset, args.tabular_cap)
# prepare all indexes given nfield (always use eventID i.e., the last fields)
field_idx_set = obtain_all_field_index(vocab_sizes.size(0) - 1, args.nfield)
f1_avg, loss_avg, last_f1, last_loss, current_f1, current_loss = AverageMeter(), AverageMeter(), 0., 0., 0., 0.
args.exp_name = f'{args.exp_name}_{args.seed}'
for idx, field_idx in enumerate(field_idx_set):
    field_idx = list(field_idx) + [len(vocab_sizes)-1]

    train_loaders, valid_loader, test_loader = log_loader(args.data_path, args.nstep, vocab_sizes, args.session_based,
      args.feature_code, args.shuffle, args.only_normal, args.valid_perc, args.test_perc, args.nenv,
      args.session_len, args.step_size, args.bsz, args.nworker, field_idx)

    start_time, best_valid_f1 = time.time(), -100.
    for seed in range(args.seed, args.seed+args.repeat):
        seed_everything(seed)
        if not os.path.isdir(f'{args.log_dir}{args.exp_name}'):
            os.makedirs(f'{args.log_dir}{args.exp_name}', exist_ok=True)
        plogger = logger(f'{args.log_dir}{args.exp_name}/stdout.log', True, True)
        plogger.info(f'field_idx {idx}/{len(field_idx_set)}: {field_idx}')
        plogger.info(f'original vocab: {vocab_sizes}, field idx: {field_idx}')
        plogger.info(f'current partial vocab: {vocab_sizes[field_idx]}')

        main()
        start_time, best_valid_f1 = time.time(), -100.

        f1_avg.update(last_f1)
        loss_avg.update(last_loss)
        plogger.info(f'Current average f1 {f1_avg.avg: 8.4f}\tloss {loss_avg.avg: 8.4f}\n')
        remove_logger(plogger)
import os
import time
import argparse
from typing import Tuple

import torch
from torch import nn
import torch.backends.cudnn as cudnn
from torch import optim

from data_loader import log_loader, get_vocab_size
from utils.utils import logger, remove_logger, AverageMeter, timeSince
from utils.utils import f1_score, is_in_topk, is_log_seq_anomaly
from models.utils import create_model

def get_args():
    parser = argparse.ArgumentParser(description='Log-Based Anomaly Detection')
    parser.add_argument('--exp_name', default='test', type=str, help='exp name for log & checkpoint')
    # model config
    parser.add_argument('--model', default='armnet', type=str, help='model name')
    parser.add_argument('--nstep', type=int, default=10, help='number of log events per sequence')
    parser.add_argument('--tabular_cap', type=int, default=1000, help='feature cap for tabular data, e.g., pid in HDFS')
    ## tabular
    parser.add_argument('--nemb', type=int, default=20, help='tabular embedding size')
    parser.add_argument('--alpha', default=1.7, type=float, help='entmax alpha to control sparsity in ARM-Module')
    parser.add_argument('--nhid', type=int, default=64, help='# (exponential neurons) cross features in ARM-Module')
    parser.add_argument('--nquery', type=int, default=8, help='number of output query vectors for each step')
    ## log sequence
    parser.add_argument('--nhead', type=int, default=8, help='attention head per layer for log seq encoder')
    parser.add_argument('--num_layers', type=int, default=6, help='number of layers for log seq encoder')
    parser.add_argument('--dim_feedforward', type=int, default=256, help='FFN dimension for log seq encoder')
    parser.add_argument('--dropout', default=0.1, type=float, help='dropout rate for text encoder and predictor')
    ## predictor
    parser.add_argument('--predictor_nlayer', type=int, default=2, help='number of layers for predictor')
    parser.add_argument('--d_predictor', type=int, default=256, help='FFN dimension for predictor')
    # optimizer
    parser.add_argument('--epoch', type=int, default=100, help='number of maximum epochs')
    parser.add_argument('--patience', type=int, default=1, help='number of epochs for early stopping training')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size')
    parser.add_argument('--lr', default=0.0003, type=float, help='learning rate, default 3e-4')
    parser.add_argument('--eval_freq', type=int, default=10000, help='max number of batches to train per epoch')
    # dataset
    parser.add_argument('--dataset', type=str, default='hdfs', help='dataset name for data_loader')
    parser.add_argument('--data_path', type=str, default='./data/Drain_result/HDFS.log_encode_shuffle.log', help='dataset path')
    # parser.add_argument('--data_path', type=str, default='./data/Drain_result/small_data.log', help='dataset path')
    parser.add_argument('--valid_perc', default=0.2, type=float, help='validation set percentage')
    parser.add_argument('--workers', default=4, type=int, help='number of data loading workers')
    # evaluation metric
    parser.add_argument('--topk', default=10, type=float, help='number of top candidate events for anomaly detection')
    # log & checkpoint
    parser.add_argument('--log_dir', type=str, default='./log/', help='path to dataset')
    parser.add_argument('--report_freq', type=int, default=50, help='report frequency')
    parser.add_argument('--seed', type=int, default=2021, help='seed for reproducibility')
    parser.add_argument('--repeat', type=int, default=1, help='number of repeats with seeds [seed, seed+repeat)')
    args = parser.parse_args()
    return args


def main():
    global args, best_valid_f1, start_time, vocab_sizes
    plogger = logger(f'{args.log_dir}{args.exp_name}/stdout.log', True, True)
    plogger.info(vars(args))

    # create model
    model = create_model(args, plogger, vocab_sizes)
    # optimizer
    opt_metric = nn.CrossEntropyLoss().cuda()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    # gradient clipping
    for p in model.parameters():
        p.register_hook(lambda grad: torch.clamp(grad, -1., 1.))
    cudnn.benchmark = True

    patience_cnt = 0
    for epoch in range(args.epoch):
        plogger.info(f'Epoch [{epoch:3d}/{args.epoch:3d}]')

        # train and eval
        run(epoch, model, train_loader, opt_metric, plogger, optimizer=optimizer)
        valid_precision, valid_recall, valid_f1 = run(epoch, model, val_loader, opt_metric, plogger, namespace='val')
        test_precidion, test_recall, test_f1 = run(epoch, model, test_loader, opt_metric, plogger, namespace='test')

        # record best f1 and save checkpoint
        if valid_f1 >= best_valid_f1:
            best_valid_f1 = valid_f1
            best_test_f1 = test_f1
            plogger.info(f'best test: valid {valid_f1:.4f}, test {test_f1:.4f}')
        else:
            patience_cnt += 1
            plogger.info(f'valid {valid_f1:.4f}, test {test_f1:.4f}')
            plogger.info(f'Early stopped, {patience_cnt}-th best f1 at epoch {epoch-1}')
        if patience_cnt >= args.patience:
            plogger.info(f'Final best valid f1 {best_valid_f1:.4f}, with test f1 {best_test_f1:.4f}')
            break

    plogger.info(f'Total running time: {timeSince(since=start_time)}')
    remove_logger(plogger)


#  train one epoch of train/val/test
def run(epoch, model, data_loader, opt_metric, plogger, optimizer=None, namespace='train') \
        -> Tuple[float, float, float]:
    if namespace == 'train': model.train()
    else: model.eval()

    time_avg, loss_avg, event_acc_avg = AverageMeter(), AverageMeter(), AverageMeter()
    precision_avg, recall_avg, f1_avg = AverageMeter(), AverageMeter(), AverageMeter()

    timestamp = time.time()
    for idx, batch in enumerate(data_loader):
        tabular = batch['tabular'].cuda(non_blocking=True)                      # N*nstep*nfield
        eventID_y = batch['eventID_y'].cuda(non_blocking=True)                  # N
        nsamples = batch['nsamples']                                            # bsz
        log_seq_y = batch['log_seq_y']                                          # bsz

        if namespace == 'train':
            pred_eventID_y = model(tabular)                                     # N*noutput
            loss = opt_metric(pred_eventID_y, eventID_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                pred_eventID_y = model(tabular)
                loss = opt_metric(pred_eventID_y, eventID_y)

        global args

        event_acc = is_in_topk(pred_eventID_y, eventID_y, topk=1)               # N
        # calculate event prediction accuracy, record in event_acc_avg
        batch_acc = event_acc.sum().float().item() * 100. / event_acc.size(0)
        # valid/test, evaluate log seq prediction precision/recall/f1
        if namespace != 'train':
            log_pred = is_log_seq_anomaly(event_acc, nsamples)                  # N
            precision, recall, f1 = f1_score(log_pred, log_seq_y)
            precision_avg.update(precision, eventID_y.size(0))
            recall_avg.update(recall, eventID_y.size(0))
            f1_avg.update(f1, eventID_y.size(0))

        time_avg.update(time.time() - timestamp)
        loss_avg.update(loss.item(), eventID_y.size(0))
        event_acc_avg.update(batch_acc, event_acc.size(0))
        timestamp = time.time()
        if idx % args.report_freq == 0:
            plogger.info(f'Epoch [{epoch:3d}/{args.epoch:3d}][{idx:3d}/{len(data_loader):3d}]\t'
                         f'{time_avg.val:.3f} ({time_avg.avg:.3f})\t'
                         f'A {event_acc_avg.val:.3f} ({event_acc_avg.avg:.3f})\t'
                         f'P {precision_avg.val:4f} ({precision_avg.avg:4f})\t'
                         f'R {recall_avg.val:4f} ({recall_avg.avg:4f})\t'
                         f'F1 {f1_avg.val:4f} ({f1_avg.avg:4f})\t'
                         f'Loss {loss_avg.val:8.4f} ({loss_avg.avg:8.4f})')

        # stop training current epoch for evaluation
        if idx >= args.eval_freq: break

    plogger.info(f'{namespace}\tTime {timeSince(s=time_avg.sum):>12s}\t'
                 f'Precision {precision_avg.avg:4f}\tRecall {recall_avg.avg:4f}\t'
                 f'F1-score {f1_avg.avg:4f}\tLoss {loss_avg.avg:8.4f}')
    return precision_avg.avg, recall_avg.avg, f1_avg.avg


# initialize global variables, load dataset
args = get_args()
vocab_sizes = get_vocab_size(args.dataset, args.tabular_cap)
train_loader, val_loader, test_loader = log_loader(args.data_path, args.nstep, vocab_sizes,
                                                   args.batch_size, args.valid_perc, args.workers)
start_time, best_valid_f1, base_exp_name = time.time(), 0., args.exp_name
for args.seed in range(args.seed, args.seed+args.repeat):
    torch.manual_seed(args.seed)
    args.exp_name = f'{base_exp_name}_{args.seed}'
    if not os.path.isdir(f'{args.log_dir}{args.exp_name}'):
        os.makedirs(f'{args.log_dir}{args.exp_name}', exist_ok=True)
    main()
    start_time, best_valid_f1 = time.time(), 0.

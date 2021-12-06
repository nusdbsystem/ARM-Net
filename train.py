import os
import time
import argparse
from typing import Tuple

import torch
from torch import nn
import torch.backends.cudnn as cudnn
from torch import optim

from data_loader import log_loader, get_vocab_size
from models.log_seq_encoder import LogSeqEncoder
from utils.utils import logger, remove_logger, AverageMeter, timeSince
from utils.utils import f1_score, F1_Loss

def get_args():
    parser = argparse.ArgumentParser(description='ARMOR framework')
    parser.add_argument('--exp_name', default='test', type=str, help='exp name for log & checkpoint')
    # model config
    parser.add_argument('--nstep', type=int, default=10, help='number of log events per sequence')
    parser.add_argument('--nfield', type=int, default=7, help='the number of fields (tabular+text)')
    parser.add_argument('--tabular_cap', type=int, default=1000, help='feature cap for tabular data')
    parser.add_argument('--text_cap', type=int, default=5000, help='feature cap for text')
    ## tabular
    parser.add_argument('--nemb', type=int, default=32, help='tabular embedding size')
    parser.add_argument('--alpha', default=1.7, type=float, help='entmax alpha to control sparsity in ARM-Module')
    parser.add_argument('--nhid', type=int, default=256, help='number of cross features in ARM-Module')
    parser.add_argument('--d_hid', type=int, default=512, help='inner Query/Key dimension in ARM-Module')
    ## text
    parser.add_argument('--max_seq_len', type=int, default=500, help='max text sequence length')
    parser.add_argument('--d_model', type=int, default=128, help='text token embedding size')
    parser.add_argument('--nhead', type=int, default=8, help='attention head per layer')
    parser.add_argument('--num_layers', type=int, default=6, help='number of layers for text encoder')
    parser.add_argument('--dim_feedforward', type=int, default=2048, help='FFN dimension for Text Encoder')
    parser.add_argument('--dropout', default=0.1, type=float, help='dropout rate for text encoder and predictor')
    ## tcn
    parser.add_argument('--tcn_layers', type=int, default=3, help='number of TCN layers for tabular & text')
    ## predictor
    parser.add_argument('--predictor_layers', type=int, default=3, help='number of layers for predictor')
    parser.add_argument('--d_predictor', type=int, default=512, help='FFN dimension for predictor')

    # optimizer
    parser.add_argument('--epoch', type=int, default=100, help='number of maximum epochs')
    parser.add_argument('--patience', type=int, default=1, help='number of epochs for stopping training')
    parser.add_argument('--batch_size', type=int, default=512, help='batch size')
    parser.add_argument('--lr', default=0.0003, type=float, help='learning rate, default 3e-4')
    parser.add_argument('--eval_freq', type=int, default=10000, help='max number of batches to train per epoch')
    # dataset
    parser.add_argument('--dataset', type=str, default='hdfs', help='dataset name for data_loader')
    parser.add_argument('--data_path', type=str, default='./data/HDFS_1/small_data.log', help='path to dataset')
    parser.add_argument('--valid_perc', default=0.1, type=float, help='validation set percentage')
    parser.add_argument('--test_perc', default=0.1, type=float, help='test set percentage')
    parser.add_argument('--workers', default=4, type=int, help='number of data loading workers')
    # log & checkpoint
    parser.add_argument('--log_dir', type=str, default='./log/', help='path to dataset')
    parser.add_argument('--tensorboard_dir', default='./tensorboard/', type=str, metavar='PATH',
                        help='path to tensorboard')
    parser.add_argument('--report_freq', type=int, default=50, help='report frequency')
    parser.add_argument('--seed', type=int, default=2021, help='seed for reproducibility')
    parser.add_argument('--repeat', type=int, default=1, help='number of repeats with seeds [seed, seed+repeat)')
    args = parser.parse_args()
    return args

def main():
    global args, best_f1, start_time, ret_vocab_sizes
    plogger = logger(f'{args.log_dir}{args.exp_name}/stdout.log', True, True)
    plogger.info(vars(args))

    model = LogSeqEncoder(args.nstep, args.nfield-1, ret_vocab_sizes[0], args.nemb, args.alpha, args.nhid, args.d_hid,
                          args.d_model, ret_vocab_sizes[1], ret_vocab_sizes[1]-1, args.nhead, args.num_layers,
                          args.dim_feedforward, args.dropout, args.tcn_layers, args.predictor_layers, args.d_predictor)
    model = torch.nn.DataParallel(model).cuda()
    plogger.info(f'model parameters: {sum([p.data.nelement() for p in model.parameters()])}')

    # optimizer
    opt_metric = F1_Loss()
    if torch.cuda.is_available(): opt_metric = opt_metric.cuda()
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
        precision, recall, f1 = run(epoch, model, val_loader, opt_metric, plogger, namespace='val')
        precision_test, recall_test, f1_test = run(epoch, model, test_loader, opt_metric, plogger, namespace='test')

        # record best aue and save checkpoint
        if f1 >= best_f1:
            best_f1 = f1
            report_f1 = f1_test
            plogger.info(f'best test: valid {f1:.4f}, test {f1_test:.4f}')
        else:
            patience_cnt += 1
            plogger.info(f'valid {f1:.4f}, test {f1_test:.4f}')
            plogger.info(f'Early stopped, {patience_cnt}-th best auc at epoch {epoch-1}')
        if patience_cnt >= args.patience:
            plogger.info(f'Final best valid auc {best_f1:.4f}, with test auc {report_f1:.4f}')
            break

    plogger.info(f'Total running time: {timeSince(since=start_time)}')
    remove_logger(plogger)

#  train one epoch of train/val/test
def run(epoch, model, data_loader, opt_metric, plogger, optimizer=None, namespace='train') -> Tuple[int, int, int]:
    if optimizer: model.train()
    else: model.eval()

    time_avg = AverageMeter()
    loss_avg, precision_avg = AverageMeter(), AverageMeter()
    recall_avg, f1_avg = AverageMeter(), AverageMeter()

    timestamp = time.time()
    for idx, batch in enumerate(data_loader):
        target = batch['y'].cuda(non_blocking=True).long()
        tabular = batch['tabular'].cuda(non_blocking=True)
        text = batch['text'].cuda(non_blocking=True)

        if optimizer:
            y = model(tabular, text)
            loss = opt_metric(y, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                y = model(tabular, text)
                loss = opt_metric(y, target)

        precision, recall, f1 = f1_score(target, y)
        loss_avg.update(loss.item(), target.size(0))
        precision_avg.update(precision, target.size(0))
        recall_avg.update(recall, target.size(0))
        f1_avg.update(f1, target.size(0))

        time_avg.update(time.time() - timestamp)
        timestamp = time.time()
        if idx % args.report_freq == 0:
            plogger.info(f'Epoch [{epoch:3d}/{args.epoch:3d}][{idx:3d}/{len(data_loader):3d}]\t'
                         f'{time_avg.val:.3f} ({time_avg.avg:.3f})\t'
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
vocab_sizes = get_vocab_size(args.dataset, args.tabular_cap, args.text_cap)
train_loader, val_loader, \
test_loader, ret_vocab_sizes = log_loader(args.data_path, args.nstep, vocab_sizes,
                                      args.max_seq_len, args.batch_size, args.valid_perc,
                                      args.test_perc, args.workers)
start_time, best_f1, base_exp_name = time.time(), 0., args.exp_name
for args.seed in range(args.seed, args.seed+args.repeat):
    torch.manual_seed(args.seed)
    args.exp_name = f'{base_exp_name}_{args.seed}'
    if not os.path.isdir(f'log/{args.exp_name}'): os.makedirs(f'log/{args.exp_name}', exist_ok=True)
    main()
    start_time, best_f1 = time.time(), 0.

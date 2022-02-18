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
from utils.utils import f1_score, is_in_topk, is_log_seq_anomaly, correct_to_acc
from models.utils import create_model
from optimizer.irm import IRM
from optimizer.randomizer import Randomizer
from optimizer.reptile import Reptile


def get_args():
    parser = argparse.ArgumentParser(description='Log-Based Anomaly Detection')
    parser.add_argument('--exp_name', default='test', type=str, help='exp name for log & checkpoint')
    # 1. model config
    parser.add_argument('--model', default='armnet', type=str, help='model name')
    parser.add_argument('--nstep', type=int, default=10, help='number of log events per sequence')
    parser.add_argument('--tabular_cap', type=int, default=1000, help='feature cap for tabular data, e.g., pid in HDFS')
    parser.add_argument('--dropout', default=0.0, type=float, help='dropout rate')
    ## 1.1 tabular
    parser.add_argument('--nemb', type=int, default=20, help='tabular embedding size')
    parser.add_argument('--alpha', default=1.7, type=float, help='entmax alpha to control sparsity in ARM-Module')
    parser.add_argument('--nhid', type=int, default=64, help='# (exponential neurons) cross features in ARM-Module')
    parser.add_argument('--nquery', type=int, default=8, help='number of output query vectors for each step')
    ## 1.2 log sequence
    parser.add_argument('--nhead', type=int, default=8, help='attention head per layer for log seq encoder')
    parser.add_argument('--nlayer', type=int, default=6, help='number of layers for log seq encoder')
    parser.add_argument('--dim_feedforward', type=int, default=256, help='FFN dimension for log seq encoder')
    ## 1.3 predictor
    parser.add_argument('--mlp_nlayer', type=int, default=2, help='number of layers for MLP predictor')
    parser.add_argument('--mlp_nhid', type=int, default=256, help='FFN dimension for MLP predictor')
    # 2 optimizer
    parser.add_argument('--epoch', type=int, default=100, help='number of maximum epochs')
    parser.add_argument('--patience', type=int, default=1, help='number of epochs for early stopping training')
    parser.add_argument('--bsz', type=int, default=256, help='batch size')
    parser.add_argument('--lr', default=0.0003, type=float, help='learning rate, default 3e-4')
    parser.add_argument('--eval_freq', type=int, default=10000, help='max number of batches to train per epoch')
    parser.add_argument('--nenv', type=int, default=1, help='number of training environments')
    parser.add_argument('--rand_type', type=int, default=0, help='data type random type')
    # 2.1 irm
    parser.add_argument("--irm", action="store_true", default=False, help="whether to use irm for DG")
    parser.add_argument('--lambda_p', default=3e-2, type=float, help='lambda for IRM penalty, default 3e-2')
    # 2.2 meta-learning-reptile
    parser.add_argument("--reptile", action="store_true", default=False, help="whether to use reptile for DG")
    parser.add_argument('--outer_iter', type=int, default=100, help='number of outer training iterations per epoch')
    parser.add_argument('--inner_iter', type=int, default=3, help='number of inner training iterations')
    parser.add_argument('--inner_lr', default=0.0003, type=float, help='inner training learning rate, default 3e-4')
    # 3. dataset
    parser.add_argument("--session_based", action="store_true", default=False, help="to use only session features")
    parser.add_argument("--shuffle", action="store_true", default=False, help="shuffle the whole dataset before split")
    parser.add_argument('--dataset', type=str, default='hdfs', help='dataset name for data_loader')
    parser.add_argument('--data_path', type=str, default='./data/Drain_result/HDFS.log_all.log', help='path')
    parser.add_argument('--valid_perc', default=0.2, type=float, help='train/valid data split among train set')
    parser.add_argument('--test_perc', default=0.5, type=float, help='train/test data split perc among all data')
    parser.add_argument('--workers', default=4, type=int, help='number of data loading workers')
    # 4. evaluation metric
    parser.add_argument('--topk', default=10, type=int, help='number of top candidate events for anomaly detection')
    # 5. log & checkpoint
    parser.add_argument('--log_dir', type=str, default='./log/', help='path to dataset')
    parser.add_argument('--report_freq', type=int, default=50, help='report frequency')
    parser.add_argument('--seed', type=int, default=2021, help='seed for reproducibility')
    parser.add_argument('--repeat', type=int, default=5, help='number of repeats with seeds [seed, seed+repeat)')
    args = parser.parse_args()
    return args


def main():
    global args, best_valid_f1, start_time, vocab_sizes
    plogger = logger(f'{args.log_dir}{args.exp_name}/stdout.log', True, True)
    plogger.info(vars(args))

    # create model
    model = create_model(args, plogger, vocab_sizes)
    # optimizer
    opt_metric = nn.CrossEntropyLoss(reduction='none').cuda()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # gradient clipping
    for p in model.parameters():
        p.register_hook(lambda grad: torch.clamp(grad, -1., 1.))
    cudnn.benchmark = True

    patience_cnt = 0
    for epoch in range(args.epoch):
        plogger.info(f'Epoch [{epoch:3d}/{args.epoch:3d}]')

        # train and eval - leave-one-domain (env) for validation
        if args.reptile:
            meta_train(epoch, model, train_loaders, opt_metric, plogger, optimizer)
        else:
            run(epoch, model, train_loaders, opt_metric, plogger, optimizer=optimizer)
        valid_precision, valid_recall, valid_f1 = run(epoch, model, [valid_loader], opt_metric, plogger, namespace='val')
        test_precidion, test_recall, test_f1 = run(epoch, model, [test_loader], opt_metric, plogger, namespace='test')

        # record best f1 and save checkpoint
        if valid_f1 > best_valid_f1:
            patience_cnt = 0
            best_valid_f1, best_test_f1 = valid_f1, test_f1
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


def meta_train(epoch, model, data_loaders, opt_metric, plogger, optimizer):
    model.train()
    model_tilde = Reptile.copy_model(model)
    inner_optimizer = optim.Adam(model_tilde.parameters(), lr=args.inner_lr)
    time_avg, loss_avg, timestamp = AverageMeter(), AverageMeter(), time.time()
    for idx in range(args.outer_iter):
        loss = Reptile.train_steps(args.inner_iter, model_tilde, Randomizer.select_generator(data_loaders),
                                   opt_metric, inner_optimizer, True)
        loss_avg.update(loss)

        Reptile.update_grad(model, model_tilde)
        optimizer.step()
        optimizer.zero_grad()

        time_avg.update(time.time() - timestamp); timestamp = time.time()
        if idx % args.report_freq == 0:
            plogger.info(f'Meta-Train Epoch [{epoch:3d}/{args.epoch:3d}]'
                         f'[{idx:3d}/{args.outer_iter:3d}] {time_avg.val:.3f} ({time_avg.avg:.3f}) '
                         f'Loss {loss_avg.val:.4f} ({loss_avg.avg:.4f})')


#  train one epoch of train/val/test
def run(epoch, model, data_loaders, opt_metric, plogger, optimizer=None, namespace='train') \
        -> Tuple[float, float, float]:
    if namespace == 'train': model.train()
    else: model.eval()

    time_avg, loss_avg, accuracy_avg = AverageMeter(), AverageMeter(), AverageMeter()
    timestamp, precision, recall, f1, all_pred, all_target = time.time(), 0., 0., 0., [], []
    if args.irm: dummy_w = IRM.dummy_w.cuda()                                       # 1

    loader = Randomizer.data_generator(data_loaders, args.rand_type, max_nbatch=args.eval_freq)
    for env_idx, batch_idx, batch in loader:
        if args.session_based:
            event_count = batch['event_count'].cuda(non_blocking=True)              # bsz*nevent
            log_seq_y = batch['log_seq_y'].cuda(non_blocking=True)                  # bsz

            if namespace == 'train':
                log_pred = model(event_count)                                       # bsz*2
                if args.irm:
                    # inject dummy_w to the loss computation
                    losses = opt_metric(log_pred*dummy_w, log_seq_y)                # bsz
                    # compute penalty
                    penalty = IRM.compute_penalty(losses, dummy_w)                  # 0
                else:
                    losses = opt_metric(log_pred, log_seq_y)                        # bsz

                optimizer.zero_grad()
                if args.irm: loss = losses.mean() + args.lambda_p*penalty           # 0
                else: loss = losses.mean()                                          # 0
                loss.backward()
                optimizer.step()
            else:
                with torch.no_grad():
                    log_pred = model(event_count)                                   # bsz
                    loss = opt_metric(log_pred, log_seq_y).mean()

            all_pred.append(log_pred)                                               # bsz
            all_target.append(log_seq_y)                                            # bsz

            loss_avg.update(loss.item(), log_seq_y.size(0))
            acc = correct_to_acc(is_in_topk(log_pred, log_seq_y, topk=1))           # bsz
            accuracy_avg.update(acc, log_seq_y.size(0))
        # TODO: update log-seq based training
        else:
            pass

        time_avg.update(time.time() - timestamp); timestamp = time.time()
        if batch_idx % args.report_freq == 0:
            plogger.info(f'Env-{env_idx} Epoch [{epoch:3d}/{args.epoch:3d}]'
                         f'[{batch_idx:3d}/{len(data_loaders[env_idx]):3d}] {time_avg.val:.3f} ({time_avg.avg:.3f}) '
                         f'Acc {accuracy_avg.val:.3f} ({accuracy_avg.avg:.3f}) '
                         f'Loss {loss_avg.val:.4f} ({loss_avg.avg:.4f})')

    if args.session_based or namespace != 'train':
        # calc f1 scores & update stats
        precision, recall, f1 = f1_score(torch.cat(all_pred), torch.cat(all_target))
    plogger.info(f'{namespace}\tTime {timeSince(s=time_avg.sum):>12s}  Accuracy {accuracy_avg.avg:.4f}  '
                 f'Precision {precision:.4f}  Recall {recall:.4f}  '
                 f'F1-score {f1:.4f}  Loss {loss_avg.avg:8.4f}')
    return precision, recall, f1


# initialize global variables, load dataset
args = get_args()
vocab_sizes = get_vocab_size(args.dataset, args.tabular_cap)
train_loaders, valid_loader, test_loader = log_loader(args.data_path, args.nstep, vocab_sizes, args.bsz,
                args.shuffle, args.valid_perc, args.test_perc, args.nenv, args.session_based, args.workers)
start_time, best_valid_f1, base_exp_name = time.time(), 0., args.exp_name
for args.seed in range(args.seed, args.seed+args.repeat):
    torch.manual_seed(args.seed)
    args.exp_name = f'{base_exp_name}_{args.seed}'
    if not os.path.isdir(f'{args.log_dir}{args.exp_name}'):
        os.makedirs(f'{args.log_dir}{args.exp_name}', exist_ok=True)
    main()
    start_time, best_valid_f1 = time.time(), 0.

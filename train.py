import os
import time
import argparse

import torch
from torch import nn
import torch.backends.cudnn as cudnn
from torch import optim

from data_loader import libsvm_dataloader

from models.model_utils import create_model
from utils.utils import logger, remove_logger, AverageMeter, timeSince, roc_auc_compute_fn


def get_args():
    parser = argparse.ArgumentParser(description='ARMOR framework')
    parser.add_argument('--exp_name', default='test', type=str, help='exp name for log & checkpoint')
    # model config
    parser.add_argument('--model', default='armnet', type=str, help='model type, afn, arm etc')
    parser.add_argument('--nfeat', type=int, default=5500, help='the number of features')
    parser.add_argument('--nfield', type=int, default=10, help='the number of fields')
    parser.add_argument('--nemb', type=int, default=10, help='embedding size')
    parser.add_argument('--k', type=int, default=3, help='interaction order for hofm/dcn/cin/gcn/gat/xdfm')
    parser.add_argument('--h', type=int, default=600, help='afm/cin/afn/armnet/gcn/gat hidden features/neurons')
    parser.add_argument('--nlayer', type=int, default=2, help='the number of mlp layers')
    parser.add_argument('--mlp_hid', type=int, default=300, help='mlp hidden units')
    parser.add_argument('--dropout', default=0.0, type=float, help='dropout rate')
    parser.add_argument('--nattn_head', type=int, default=4, help='the number of attention heads, gat/armnet')
    # for AFN/ARMNet
    parser.add_argument('--ensemble', action='store_true', default=False, help='to ensemble with DNNs')
    parser.add_argument('--dnn_nlayer', type=int, default=2, help='the number of mlp layers')
    parser.add_argument('--dnn_hid', type=int, default=300, help='mlp hidden units')
    parser.add_argument('--alpha', default=1.7, type=float, help='entmax alpha to control sparsity')
    # optimizer
    parser.add_argument('--epoch', type=int, default=100, help='number of maximum epochs')
    parser.add_argument('--patience', type=int, default=1, help='number of epochs for stopping training')
    parser.add_argument('--batch_size', type=int, default=4096, help='batch size')
    parser.add_argument('--lr', default=0.003, type=float, help='learning rate, default 3e-3')
    parser.add_argument('--eval_freq', type=int, default=10000, help='max number of batches to train per epoch')
    # dataset
    parser.add_argument('--dataset', type=str, default='frappe', help='dataset name for data_loader')
    parser.add_argument('--data_dir', type=str, default='./data/', help='path to dataset')
    parser.add_argument('--workers', default=4, type=int, help='number of data loading workers')
    # log & checkpoint
    parser.add_argument('--log_dir', type=str, default='./log/', help='path to dataset')
    parser.add_argument('--report_freq', type=int, default=30, help='report frequency')
    parser.add_argument('--seed', type=int, default=2020, help='seed for reproducibility')
    parser.add_argument('--repeat', type=int, default=1, help='number of repeats with seeds [seed, seed+repeat)')
    args = parser.parse_args()
    return args


def main():
    global best_valid_auc, start_time
    plogger = logger(f'{args.log_dir}{args.exp_name}/stdout.log', True, True)
    plogger.info(vars(args))

    model = create_model(args, plogger)
    # optimizer
    opt_metric = nn.BCEWithLogitsLoss(reduction='mean')
    if torch.cuda.is_available(): opt_metric = opt_metric.cuda()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # gradient clipping
    for p in model.parameters():
        p.register_hook(lambda grad: torch.clamp(grad, -1., 1.))

    cudnn.benchmark = True
    patience_cnt = 0
    for epoch in range(args.epoch):
        plogger.info(f'Epoch [{epoch:3d}/{args.epoch:3d}]')

        # train and eval
        run(epoch, model, train_loader, opt_metric, plogger, optimizer=optimizer)
        valid_auc = run(epoch, model, val_loader, opt_metric, plogger, namespace='val')
        test_auc = run(epoch, model, test_loader, opt_metric, plogger, namespace='test')

        # record best aue and save checkpoint
        if valid_auc >= best_valid_auc:
            best_valid_auc, best_test_auc = valid_auc, test_auc
            plogger.info(f'best valid auc: valid {valid_auc:.4f}, test {test_auc:.4f}')
        else:
            patience_cnt += 1
            plogger.info(f'valid {valid_auc:.4f}, test {test_auc:.4f}')
            plogger.info(f'Early stopped, {patience_cnt}-th best auc at epoch {epoch-1}')
        if patience_cnt >= args.patience:
            plogger.info(f'Final best valid auc {best_valid_auc:.4f}, with test auc {best_test_auc:.4f}')
            break

    plogger.info(f'Total running time: {timeSince(since=start_time)}')
    remove_logger(plogger)


#  train one epoch of train/val/test
def run(epoch, model, data_loader, opt_metric, plogger, optimizer=None, namespace='train'):
    if optimizer: model.train()
    else: model.eval()

    time_avg = AverageMeter()
    loss_avg, auc_avg = AverageMeter(), AverageMeter()

    timestamp = time.time()
    for idx, input in enumerate(data_loader):
        target = input['y']
        if torch.cuda.is_available():
            input['ids'] = input['ids'].cuda(non_blocking=True)
            input['vals'] = input['vals'].cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

        if optimizer:
            y = model(input)
            loss = opt_metric(y, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                y = model(input)
                loss = opt_metric(y, target)

        auc = roc_auc_compute_fn(y, target)
        loss_avg.update(loss.item(), target.size(0))
        auc_avg.update(auc, target.size(0))

        time_avg.update(time.time() - timestamp)
        timestamp = time.time()
        if idx % args.report_freq == 0:
            plogger.info(f'Epoch [{epoch:3d}/{args.epoch:3d}][{idx:3d}/{len(data_loader):3d}]\t'
                         f'{time_avg.val:.3f} ({time_avg.avg:.3f})\t'
                         f'AUC {auc_avg.val:4f} ({auc_avg.avg:4f})\t'
                         f'Loss {loss_avg.val:8.4f} ({loss_avg.avg:8.4f})')

        # stop training current epoch for evaluation
        if idx >= args.eval_freq: break

    plogger.info(f'{namespace}\tTime {timeSince(s=time_avg.sum):>12s}\t'
                 f'AUC {auc_avg.avg:8.4f}\tLoss {loss_avg.avg:8.4f}')
    return auc_avg.avg


# init global variables, load dataset
args = get_args()
train_loader, val_loader, test_loader = libsvm_dataloader(args)
start_time, best_valid_auc, base_exp_name = time.time(), 0., args.exp_name
for args.seed in range(args.seed, args.seed+args.repeat):
    torch.manual_seed(args.seed)
    args.exp_name = f'{base_exp_name}_{args.seed}'
    if not os.path.isdir(f'log/{args.exp_name}'): os.makedirs(f'log/{args.exp_name}', exist_ok=True)
    main()
    start_time, best_valid_auc = time.time(), 0.

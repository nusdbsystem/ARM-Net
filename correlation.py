import os
import sys
import argparse
from functools import partial
import multiprocessing
from scipy import stats

import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd

import numpy as np
from sklearn.metrics import roc_auc_score, f1_score

from models import create_model, get_config, to_device
from utils import logger, AverageMeter, PlateauStopper
from utils.uci_utils import cuda
from data_loader import uci_loader

from ray import tune
from ray.tune import CLIReporter
from ray.tune.suggest.basic_variant import BasicVariantGenerator

def get_args():
    parser = argparse.ArgumentParser(description='UCI Training')
    parser.add_argument('--exp_name', default='test', type=str, help='exp name used to store log & checkpoint')
    parser.add_argument('--stdout', default='log.txt', type=str, help='redirect output to file')
    parser.add_argument('--model', default='armnet', type=str, help='model type')
    parser.add_argument('--data_dir', default='/home/shaofeng/ARM-Net/data/uci', type=str, help='dataset dir path')
    parser.add_argument('--dataset', default='abalone', type=str, help='dataset name')
    parser.add_argument('--metric', default='acc', type=str, help='evaluation metric')
    parser.add_argument('--valid_perc', default=0.15, type=float, help='validation percent split from trainset')
    parser.add_argument('--final_eval_num', default=5, type=int, help='# repeats to evaluate the best hyper-params')
    parser.add_argument('--repeat', type=int, default=1, help='number of repeats with seeds [seed, seed+repeat)')
    parser.add_argument('--seed', type=int, default=2021, help='seed for reproducibility')
    # ray auto-tune params
    parser.add_argument('--max_epochs', default=100, type=int, help='max training epoch')
    parser.add_argument('--ray_dir', default='./ray_results', type=str, help='ray log dir')
    parser.add_argument('--gpus_per_trial', default=0.2, type=float, help='gpus per trial (default 5 trials per GPU)')
    parser.add_argument('--num_samples', default=1, type=int, help='# repeats to evaluate each hyperparams')
    parser.add_argument("--resume", action="store_true", default=False, help="whether to resume ray tuning")
    args = parser.parse_args()
    return args

# global args
args = get_args()
log_dir = f'log/{args.exp_name}/'
data_dir = f'{args.data_dir}/{args.dataset}'
# logger & stdout
if not os.path.isdir(log_dir): os.makedirs(log_dir, exist_ok=True)
plogger = logger(f'{log_dir}stdout.log', True, True).info
plogger(vars(args))
if args.stdout: sys.stdout = open(f'{log_dir}{args.stdout}', 'w')
# criteron
criterion = cuda(nn.CrossEntropyLoss())


def evaluate(model, val_loader, metric='acc'):
    model.eval()
    correct, total = 0, 0
    all_labels, all_preds = [], []
    loss_avg = AverageMeter()
    for batch, labels in val_loader:
        batch = cuda(autograd.Variable(batch))
        labels = cuda(labels)
        outputs = model(batch)
        _, predictions = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predictions == labels).sum()
        all_labels += list(labels)
        all_preds += list(predictions)
        loss = criterion(outputs, labels)
        loss_avg.update(loss.item())
    if metric == 'acc':
        return correct.item() / float(total), loss_avg.avg
    if metric == 'auc':
        try: return roc_auc_score(all_labels, all_preds), loss_avg.avg
        except: return 0, loss_avg.avg
    if metric == 'f1':
        return f1_score(all_labels, all_preds), loss_avg.avg

def train_one_epoch(model, train_loader, optimizer, device):
    model.train()
    for i, data in enumerate(train_loader, 0):
        try:
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        except Exception as e:
            plogger(f'input size: {inputs.size()} error msg: {str(e)}')

# for both ray.tune and final training/evaluation (based on )
def worker(config, checkpoint_dir=None, data_dir=None, final_run=False, max_epochs=args.max_epochs):
    # loading dataset
    train_loader, val_loader, test_loader = uci_loader(data_dir, valid_perc=0. if final_run else args.valid_perc)
    # create model
    config['nclass'], config['nfeat'] = train_loader.nclass, train_loader.nfeat
    config['model'] = args.model
    model = create_model(config, plogger)
    device = to_device(model)
    # optimizer (Adam vs. SGD)
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])
    # checkpoint (for ray.tune scheduler/algorithm)
    start_epoch = 0

    # stopper: params - patience & avg_num
    stopper = PlateauStopper(patience=0, avg_num=10)
    for epoch in range(start_epoch, max_epochs):
        # train one epoch
        train_one_epoch(model, train_loader, optimizer, device)
        # ray.tune: validate
        if not final_run:
            acc, loss = evaluate(model, val_loader, metric=args.metric)
            # early stop once not improving
            if stopper.stop(acc): break
            # reporting metrics (for ray.tune)
    # report validation accuracy (smoothed for hyper-param selection)
    val_acc = stopper.last_avg

    # final_run: train the final model for the same number of epochs, return (acc, loss)
    if final_run: return evaluate(model, val_loader=test_loader)
    else:
        # train the model for the same number of epochs on the full train_set, and test on test_set
        test_acc, test_loss = worker(config, data_dir=data_dir, final_run=True, max_epochs=epoch)
        tune.report(val_acc=val_acc, test_acc=test_acc, epoch=epoch)

def main(num_samples, gpus_per_trial):
    config = get_config(args.model)
    reporter = CLIReporter(metric_columns=["loss", "val_acc", "test_acc", "training_iteration"])
    max_concurrent = torch.cuda.device_count() / gpus_per_trial
    repeat_rho = []
    for seed in range(args.seed, args.seed+args.repeat):
        # set random seeds
        torch.manual_seed(seed)
        np.random.seed(seed)
        # using default stop/search_alg/scheduler
        analysis = tune.run(
            partial(worker, data_dir=data_dir),
            name=f'{args.exp_name}_{seed}',
            config=config,
            resources_per_trial={"cpu": multiprocessing.cpu_count()/max_concurrent, "gpu": gpus_per_trial},
            num_samples=num_samples,
            search_alg=BasicVariantGenerator(max_concurrent=max_concurrent),
            progress_reporter=reporter,
            local_dir=args.ray_dir,
            checkpoint_score_attr="acc",
            verbose=2,
            resume=args.resume,
            trial_name_creator=lambda trial: f'{trial.trial_id}'
        )
        # get the trial stats in dataframe format
        df = analysis.dataframe(metric="val_acc", mode="max")
        rho, pval = stats.spearmanr(df.loc[:, 'val_acc'], df.loc[:, 'test_acc'])
        plogger(f'Repeat-{seed}\tspearman correlation: {rho:.4f}({pval:.4f})')
        repeat_rho.append(rho)
    plogger(f'Final report results: rho {np.mean(repeat_rho):.4f}/{np.std(repeat_rho):.4f}')


if __name__ == "__main__":
    main(num_samples=args.num_samples, gpus_per_trial=args.gpus_per_trial)
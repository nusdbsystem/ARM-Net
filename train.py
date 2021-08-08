import os
import argparse
from functools import partial

import ray
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd

import numpy as np
from sklearn.metrics import roc_auc_score, f1_score

from models import create_model, get_config, to_device
from utils import logger, AverageMeter
from utils.uci_utils import *
from data_loader import uci_loader

from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from ray.tune.stopper import TrialPlateauStopper

torch.manual_seed(2021)
np.random.seed(2021)

def get_args():
    parser = argparse.ArgumentParser(description='UCI Training')
    parser.add_argument('--exp_name', default='test', type=str, help='exp name used to store log & checkpoint')
    parser.add_argument('--model', default='armnet', type=str, help='model type')
    parser.add_argument('--data_dir', default='/home/shaofeng/ARM-Net/data/uci', type=str, help='dataset dir path')
    parser.add_argument('--dataset', default='abalone', type=str, help='dataset name')
    parser.add_argument('--metric', default='acc', type=str, help='evaluation metric')
    parser.add_argument('--valid_perc', default=0.2, type=float, help='valid perc split from trainset')
    # ray auto-tune params
    parser.add_argument('--ray_dir', default='./ray_results', type=str, help='ray log dir')
    parser.add_argument('--max_epochs', default=100, type=int, help='max training epoch')
    parser.add_argument('--gpus_per_trial', default=0.2, type=float, help='gpus per trial')
    parser.add_argument('--num_samples', default=100, type=int, help='# evaluated hyperparams')
    args = parser.parse_args()
    return args

# global args
args = get_args()
log_dir = f'log/{args.exp_name}/stdout.log'
data_dir = f'{args.data_dir}/{args.dataset}'
# logger
if not os.path.isdir(f'log/{args.exp_name}'): os.makedirs(f'log/{args.exp_name}', exist_ok=True)
plogger = logger(log_dir, True, True).info
plogger(vars(args))

def evaluate(model, val_loader, criterion=None, metric='acc'):
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
        if criterion is not None:
            loss = criterion(outputs, labels)
            loss_avg.update(loss.item())
    loss = loss_avg.avg if criterion else None
    if metric == 'acc':
        return correct.item() / float(total), loss
    if metric == 'auc':
        try: return roc_auc_score(all_labels, all_preds), loss
        except: return 0, loss
    if metric == 'f1':
        return f1_score(all_labels, all_preds), loss

def train_uci(config, checkpoint_dir=None, data_dir=None):
    # loading dataset
    train_loader, val_loader, _ = uci_loader(data_dir, config["batch_size"], valid_perc=args.valid_perc)

    # create model
    config['nclass'], config['nfeat'] = train_loader.nclass, train_loader.nfeat
    config['model'] = args.model
    model = create_model(config, plogger)
    device = to_device(model)

    # criterion & optimizer
    criterion = cuda(nn.CrossEntropyLoss())
    optimizer = optim.SGD(model.parameters(), lr=config["lr"],
                          momentum=config.get('momentum', 0.0),
                          weight_decay=config.get('l2', 0.0))

    # checkpoint
    start = 0
    if checkpoint_dir:
        checkpoint = torch.load(os.path.join(checkpoint_dir, "checkpoint"))
        start = checkpoint["epoch"] + 1
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])

    for epoch in range(start, args.max_epochs):
        model.train()
        # train one epoch
        for i, data in enumerate(train_loader, 0):
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

        # validate
        acc, loss = evaluate(model, val_loader, criterion, metric=args.metric)

        with tune.checkpoint_dir(step=epoch) as checkpoint_dir:
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }, os.path.join(checkpoint_dir, "checkpoint"))


        tune.report(loss=loss, acc=acc)
    print("Training Finished")

def main(num_samples, max_num_epochs, gpus_per_trial):
    config = get_config(args.model)
    stopper = TrialPlateauStopper(metric="acc", std=1e-2,
                                  num_results=5, grace_period=3)
    scheduler = ASHAScheduler(
        metric="acc", mode="max", max_t=max_num_epochs,
        grace_period=10, reduction_factor=3, brackets=1)
    reporter = CLIReporter(
        metric_columns=["loss", "acc", "training_iteration"])
    analysis = tune.run(
        partial(train_uci, data_dir=data_dir),
        name=args.exp_name,
        resources_per_trial={"cpu": 2, "gpu": gpus_per_trial},
        config=config,
        stop=stopper,
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter,
        local_dir=args.ray_dir,
        keep_checkpoints_num=1,
        checkpoint_score_attr="acc"
        # verbose=1,
    )

    best_trial = analysis.get_best_trial("acc", "max", "all")
    plogger(f'Best trial config: {best_trial.config}')
    plogger(f'Best trial final validation loss: {best_trial.last_result["loss"]}'
            f'accuracy: {best_trial.last_result["acc"]}')

    # load dataset
    train_loader, _, test_loader = uci_loader(data_dir, best_trial.config["batch_size"])
    # ceate best model
    best_trial.config['model'] = args.model
    best_trial.config['nclass'], best_trial.config['nfeat'] = train_loader.nclass, train_loader.nfeat
    best_trained_model = create_model(best_trial.config, plogger)
    to_device(best_trained_model)
    # load best model
    best_checkpoint_dir = best_trial.checkpoint.value
    checkpoint = torch.load(os.path.join(best_checkpoint_dir, "checkpoint"))
    model_state = checkpoint["model"]
    best_trained_model.load_state_dict(model_state)
    # evaluate on test set
    test_acc, _ = evaluate(best_trained_model, val_loader=test_loader)
    print(f'Best trial test set accuracy: {test_acc}')


if __name__ == "__main__":
    main(num_samples=args.num_samples, max_num_epochs=args.max_epochs,
         gpus_per_trial=args.gpus_per_trial)
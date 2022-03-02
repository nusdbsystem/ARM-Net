import copy
import random
from typing import Generator
import time

from utils.utils import AverageMeter
from optimizer.randomizer import Randomizer

import torch
from torch.nn import Module
from torch.optim import Optimizer
from torch import FloatTensor
from torch import optim


class Reptile():
    """ Reptile utility class """
    @staticmethod
    def update_grad(model: Module, model_tilde: Module, lambda_p: float = 1.) -> None:
        for param, param_tilde in zip(model.parameters(), model_tilde.parameters()):
            if param.grad is None:
                param.grad = torch.zeros_like(param)

            param.grad.data.add_((param.data - param_tilde.data)*lambda_p)

    @staticmethod
    def copy_model(model: Module) -> Module:
        # copy all modules with the same weights (w/o. weight.grad)
        return copy.deepcopy(model)

    @staticmethod
    def train_nstep(nstep: int, model: Module, batch_generator: Generator,
                    opt_metric: Module, optimizer: Optimizer, session_based: bool) -> FloatTensor:
        model.train()
        loss_avg = AverageMeter()
        for idx, batch in batch_generator:
            if session_based:
                event_count = batch['event_count'].cuda(non_blocking=True)  # bsz*nevent
                log_seq_y = batch['log_seq_y'].cuda(non_blocking=True)      # bsz

                log_pred = model(event_count)                               # bsz*2
                losses = opt_metric(log_pred, log_seq_y)                    # bsz
            else:
                raise ValueError(f'not implemented yet')

            optimizer.zero_grad()
            loss = losses.mean()                                            # 0
            loss.backward()
            optimizer.step()

            loss_avg.update(loss.item(), log_seq_y.size(0))
            if idx+1 >= nstep: break
        return loss_avg.avg

    @staticmethod
    def meta_train(args, epoch, model, loaders, opt_metric, plogger, optimizer):
        model.train()
        time_avg, loss_avg, timestamp = AverageMeter(), AverageMeter(), time.time()
        # init data/meta loaders
        # TODO: data/meta loader split type: same/separate-(random/fixed)
        # data_loaders, meta_loaders = loaders, loaders
        # random.shuffle(loaders)
        data_loaders, meta_loaders = loaders[:len(loaders)//2], loaders[len(loaders)//2:]
        for idx in range(args.outer_iter):
            # 0. init status -> optimizer, model copy for meta-train
            optimizer.zero_grad()
            model_tilde = Reptile.copy_model(model)
            # 1. data_train_step -> data grad
            env_idx, loader = Randomizer.select_generator(data_loaders)
            batch_idx, batch = next(loader)
            if args.session_based:
                event_count = batch['event_count'].cuda(non_blocking=True)  # bsz*nevent
                log_seq_y = batch['log_seq_y'].cuda(non_blocking=True)      # bsz
                log_pred = model(event_count)                               # bsz*2
                data_loss = opt_metric(log_pred, log_seq_y).mean()          # 1
                data_loss.backward()
            else:
                raise ValueError(f'not implemented yet')
            # 2. meta_train_step -> meta grad
            # init inner optimizer
            inner_optimizer = optim.Adam(model_tilde.parameters(), lr=args.inner_lr)
            loss = Reptile.train_nstep(args.inner_iter, model_tilde, Randomizer.select_generator(meta_loaders)[1],
                                       opt_metric, inner_optimizer, True)
            Reptile.update_grad(model, model_tilde, args.lambda_p)
            # 3. update
            optimizer.step()

            loss_avg.update(loss + data_loss.item())
            time_avg.update(time.time() - timestamp); timestamp = time.time()
            if idx % args.report_freq == 0:
                plogger.info(f'Meta-Train Env-{env_idx} Epoch [{epoch:3d}/{args.epoch}][{idx:3d}/{args.outer_iter}] '
                             f'{time_avg.val:.3f} ({time_avg.avg:.3f}) Loss {loss_avg.val:.4f} ({loss_avg.avg:.4f})')

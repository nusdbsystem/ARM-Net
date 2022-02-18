import copy
from typing import Generator

from utils.utils import AverageMeter

import torch
from torch.nn import Module
from torch.optim import Optimizer
from torch import FloatTensor

class Reptile():
    """ Reptile utility class """
    @staticmethod
    def update_grad(model: Module, model_tilde: Module) -> None:
        for param, param_tilde in zip(model.parameters(), model_tilde.parameters()):
            if param.grad is None:
                param.grad = torch.zeros_like(param)

            param.grad.data.add_(param.data - param_tilde.data)

    @staticmethod
    def copy_model(model: Module) -> Module:
        return copy.deepcopy(model)

    @staticmethod
    def train_steps(niter: int, model: Module, batch_generator: Generator,
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
            if idx+1 >= niter:
                break
        return loss_avg.avg

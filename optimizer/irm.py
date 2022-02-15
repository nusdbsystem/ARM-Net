import torch
from torch.autograd import grad
from torch import FloatTensor


class IRM():
    """ IRM utility class """
    dummy_w = torch.nn.Parameter(torch.FloatTensor([1.0]))              # 1

    def compute_penalty(losses: FloatTensor, dummy_w: FloatTensor) -> FloatTensor:
        g1 = grad(losses[0::2].mean(), dummy_w, create_graph=True)[0]   # 1
        g2 = grad(losses[1::2].mean(), dummy_w, create_graph=True)[0]   # 1
        penalty = (g1 * g2).sum()                                       # 0
        return penalty

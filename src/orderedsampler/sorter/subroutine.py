import random

import torch
from torch import Tensor

def deterministic_balance(vec: Tensor, aggregator: Tensor):
    if torch.norm(aggregator + vec) <= torch.norm(aggregator - vec):
        return 1
    else:
        return -1


def probabilistic_balance(vec, aggregator):
    p = 0.5 - torch.dot(vec, aggregator) / 60
    if random.random() <= p:
        return 1
    else:
        return -1


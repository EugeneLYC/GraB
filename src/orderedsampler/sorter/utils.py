import torch
from torch import Tensor
from torch.nn import Module
from torch._utils import _flatten_dense_tensors

from typing import Tuple
from collections import OrderedDict


def flatten_batch_grads(model: Module) -> Tensor:
    all_grads = []
    for param in model.parameters():
        if param.grad is not None:
            all_grads.append(param.grad.data)
    return _flatten_dense_tensors(tuple(all_grads))


def flatten_example_grads(model: Module,
                        start_idx: int,
                        end_idx: int) -> Tensor:
    all_grads = []
    for param in model.parameters():
        if param.grad is not None:
            all_grads.append(param.grad_batch.data[start_idx:end_idx].mean(0))
    return _flatten_dense_tensors(tuple(all_grads))

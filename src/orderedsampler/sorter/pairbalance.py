import torch
from .sorterbase import Sort
from typing import List, Dict

from torch.nn import Module

class PairBalance(Sort):
    r"""Implement Pair Balance algorithm.
        For a given sequence z_i, i = 1, 2, ..., n, we balance z_{2t} - z_{2t-1}.
        This avoids using the stale mean as in MeanBalance, and can be useful
        when the learning rate is large.
    Args:
        prob_balance (bool): If ``True``, the balancing will be performed
            in a probabilistic way. More details can be found in:
            https://arxiv.org/abs/2006.14009.
        per_batch_order (bool): If ``True``, the ordering will be carried out in a
            per batch level.
    """
    def __init__(self,
                num_examples: int,
                order_level: int = 1,
                prob_balance: bool = False,
                per_batch_order: bool = False) -> None:
        super(PairBalance, self).__init__(prob_balance, per_batch_order)
        self.num_examples = num_examples
        self.order_level = order_level

        self.first_idx = 0
        self.last_idx = self.num_examples - 1

        self.aggregator = None

        self.prev_grad_indices = []
        self.prev_grad_buffer = None

        if prob_balance:
            from .subroutine import probabilistic_balance
            self.balance = probabilistic_balance
        else:
            from .subroutine import deterministic_balance
            self.balance = deterministic_balance
        if per_batch_order:
            from .utils import flatten_batch_grads
            self.flatten_grads = flatten_batch_grads
        else:
            from .utils import flatten_example_grads
            self.flatten_grads = flatten_example_grads


    def reset_epoch(self):
        if self.aggregator is None:
            return
        self.aggregator.zero_()
        self.first_idx = 0
        self.last_idx = self.num_examples - 1

    
    @torch.no_grad()
    def step(self,
            indices: List[int],
            model: Module,
            is_last_batch: bool = False) -> Dict[int, int]:
        if self.per_batch_order:
            updated_ranks = {}
            grads = self.flatten_grads(model=model)
            if self.aggregator is None:
                self.aggregator = torch.zeros_like(grads)
            if self.prev_grad_buffer is None:
                if is_last_batch:
                    sign = self.balance(vec=grads, aggregator=self.aggregator)
                    if sign > 0:
                        updated_ranks = {i:self.first_idx for i in indices}
                        self.first_idx += len(indices)
                    else:
                        updated_ranks = {i:self.last_idx for i in indices}
                        self.last_idx -= len(indices)
                else:
                    self.prev_grad_buffer = torch.zeros_like(grads)
                    self.prev_grad_buffer.add_(grads)
                    self.prev_grad_indices = indices
            else:
                self.prev_grad_buffer.sub_(grads)
                sign = self.balance(vec=self.prev_grad_buffer, aggregator=self.aggregator)
                self.aggregator.add_(sign * self.prev_grad_buffer)
                if sign > 0:
                    for i in self.prev_grad_indices:
                        assert i not in updated_ranks.keys()
                        updated_ranks[i] = self.first_idx
                    for i in indices:
                        assert i not in updated_ranks.keys()
                        updated_ranks[i] = self.last_idx
                    self.first_idx += len(self.prev_grad_indices)
                    self.last_idx -= len(indices)
                else:
                    for i in indices:
                        assert i not in updated_ranks.keys()
                        updated_ranks[i] = self.first_idx
                    for i in self.prev_grad_indices:
                        assert i not in updated_ranks.keys()
                        updated_ranks[i] = self.last_idx
                    self.first_idx += len(indices)
                    self.last_idx -= len(self.prev_grad_indices)
                self.prev_grad_indices = []
                self.prev_grad_buffer = None
        else:
            updated_ranks = {}
            start_idx, end_idx = 0, min(self.order_level, len(indices))
            while end_idx <= len(indices):
                grads = self.flatten_grads(model=model, start_idx=start_idx, end_idx=end_idx)
                if self.aggregator is None:
                    self.aggregator = torch.zeros_like(grads)
                if self.prev_grad_buffer is None:
                    if end_idx == len(indices) and is_last_batch:
                        sign = self.balance(vec=grads, aggregator=self.aggregator)
                        if sign > 0:
                            for i in indices[start_idx:end_idx]:
                                updated_ranks[i] = self.first_idx
                            self.first_idx += end_idx - start_idx
                        else:
                            for i in indices[start_idx:end_idx]:
                                updated_ranks[i] = self.last_idx
                            self.last_idx -= end_idx - start_idx
                    else:
                        self.prev_grad_buffer = torch.zeros_like(grads)
                        self.prev_grad_buffer.add_(grads)
                        self.prev_grad_indices = indices[start_idx:end_idx]
                else:
                    self.prev_grad_buffer.sub_(grads)
                    sign = self.balance(vec=self.prev_grad_buffer, aggregator=self.aggregator)
                    self.aggregator.add_(sign * self.prev_grad_buffer)
                    if sign > 0:
                        for i in self.prev_grad_indices:
                            assert i not in updated_ranks.keys()
                            updated_ranks[i] = self.first_idx
                        for i in indices[start_idx:end_idx]:
                            assert i not in updated_ranks.keys()
                            updated_ranks[i] = self.last_idx
                        self.first_idx += len(self.prev_grad_indices)
                        self.last_idx -=  end_idx - start_idx
                    else:
                        for i in indices[start_idx:end_idx]:
                            assert i not in updated_ranks.keys()
                            updated_ranks[i] = self.first_idx
                        for i in self.prev_grad_indices:
                            assert i not in updated_ranks.keys()
                            updated_ranks[i] = self.last_idx
                        self.first_idx += end_idx - start_idx
                        self.last_idx -= len(self.prev_grad_indices)
                    self.prev_grad_indices = []
                    self.prev_grad_buffer = None
                
                start_idx = end_idx
                if start_idx == len(indices):
                    break
                end_idx = min(end_idx + self.order_level, len(indices))
            
            del grads
        
        return updated_ranks
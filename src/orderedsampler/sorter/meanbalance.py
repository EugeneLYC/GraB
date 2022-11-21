import torch
from .sorterbase import Sort
from typing import List, Dict

from torch.nn import Module

class MeanBalance(Sort):
    r"""Implement Gradient Balancing using stale mean.
    More details can be found in: https://arxiv.org/abs/2205.10733.
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
        super(MeanBalance, self).__init__(prob_balance, per_batch_order)
        self.num_examples = num_examples
        self.order_level = order_level

        self.first_idx = 0
        self.last_idx = self.num_examples - 1

        self.aggregator = None
        self.prev_mean_estimator = None
        self.next_mean_estimator = None

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
        if self.next_mean_estimator is None:
            return
        if self.prev_mean_estimator is None:
            self.prev_mean_estimator = torch.zeros_like(self.next_mean_estimator)
        self.prev_mean_estimator.copy_(self.next_mean_estimator)
        self.next_mean_estimator.zero_()
        self.aggregator.zero_()
        self.first_idx = 0
        self.last_idx = self.num_examples - 1

    
    @torch.no_grad()
    def step(self,
            indices: List[int],
            model: Module) -> Dict[int, int]:
        if self.per_batch_order:
            grads = self.flatten_grads(model=model)
            if self.aggregator is None:
                self.aggregator = torch.zeros_like(grads)
            if self.next_mean_estimator is None:
                self.next_mean_estimator = torch.zeros_like(grads)
            if self.prev_mean_estimator is not None:
                grads.sub_(self.prev_mean_estimator)
            sign = self.balance(vec=grads, aggregator=self.aggregator)
            self.aggregator.add_(sign * grads)
            self.next_mean_estimator.add_(grads / self.num_examples * self.order_level)
            if sign > 0:
                updated_ranks = {i:self.first_idx for i in indices}
                self.first_idx += len(indices)
            else:
                updated_ranks = {i:self.last_idx for i in indices}
                self.last_idx -= len(indices)
        else:
            updated_ranks = {}
            start_idx, end_idx = 0, min(self.order_level, len(indices))
            while end_idx <= len(indices):
                grads = self.flatten_grads(model=model, start_idx=start_idx, end_idx=end_idx)
                if self.aggregator is None:
                    self.aggregator = torch.zeros_like(grads)
                if self.next_mean_estimator is None:
                    self.next_mean_estimator = torch.zeros_like(grads)
                if self.prev_mean_estimator is not None:
                    grads.sub_(self.prev_mean_estimator)
                sign = self.balance(vec=grads, aggregator=self.aggregator)
                self.aggregator.add_(sign * grads)
                self.next_mean_estimator.add_(grads / self.num_examples * (end_idx - start_idx))
                if sign > 0:
                    for i in indices[start_idx:end_idx]:
                        assert i not in updated_ranks.keys()
                        updated_ranks[i] = self.first_idx
                    self.first_idx += end_idx - start_idx
                else:
                    for i in indices[start_idx:end_idx]:
                        assert i not in updated_ranks.keys()
                        updated_ranks[i] = self.last_idx
                    self.last_idx -= end_idx - start_idx
                
                start_idx = end_idx
                if start_idx == len(indices):
                    break
                end_idx = min(end_idx + self.order_level, len(indices))

        del grads
        
        return updated_ranks
from absl import logging
from collections import OrderedDict
from typing import List, Union, Sized, Tuple, Dict

import torch
from torch.nn import Module
from torch.utils.data import IterableDataset
from torch.utils.data.sampler import Sampler

from backpack import extend, backpack
from backpack.extensions import BatchGrad
from backpack.context import CTX

from .utils import IndicesTracker

MEAN_BALANCE = 'mean_balance'
PAIR_BALANCE = 'pair_balance'

class OrderedSampler(Sampler[List[int]]):
    r"""Implement a batch sampler that uses GraB-style data ordering.
    Technical details can be found in: https://arxiv.org/abs/2205.10733.
    Args:
        data_source (Dataset): Dataset to sample from.
        batch_size (int): Size of mini-batch (default: 1).
        order_level (int): Granularity of ordering (default: 1).
        drop_last (bool): If ``True``, the sampler will drop the last batch if
            its size would be less than ``batch_size`` (default: False).
        init_order_random (bool): If ``True``, the initial order (first scan of the dataset)
            will be random (default: True).
        model (nn.Module): Model to train (default: None).
        lossfunc: (nn.Module): Loss function used during the training (default: None).
        debug (bool): Whether to turn on the debugging mode (default: False).
        balance_type (str): the balancing algorithm to use. Currently ``pair_balance`` and 
            ``mean_balance`` are supported. Note that if ``mean_balance`` is used, the stale
            gradient mean from previous epoch will be applied. If the training involves large
            learning rate or contains few epochs, ``pair_balance`` is recommended (default: pair_balance).
        prob_balance (bool): If ``True``, probabilistic balancing will be performed. This is useful when
            the data is highly adversrial. for technical details, please refer to:
            https://arxiv.org/abs/2006.14009 (default: False).
    Example:
        >>> sampler = OrderedSampler(dataset, batch_size=16, order_level=2)
        >>> dataloader = torch.utils.data.DataLoader(dataset, batch_sampler=sampler)
    """
    def __init__(self,
                data_source: Sized,
                batch_size: int = 1,
                order_level: int = 1,
                drop_last: bool = False,
                init_order_random: bool = True,
                model: Union[None, Module] = None,
                lossfunc: Union[None, Module] = None,
                debug: bool = False,
                balance_type: str = PAIR_BALANCE,
                prob_balance: bool = False) -> None:
        if isinstance(data_source, IterableDataset):
            raise ValueError("Currently the OrderedSampler does not support iterable-style dataset "
                            "since it has no notion of indices, and has no meaning of ordering.")
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("batch_size should be a positive integer value, "
                             "but got batch_size={}".format(batch_size))
        if not isinstance(order_level, int) or order_level <= 0 or order_level > batch_size or batch_size % order_level != 0:
            raise ValueError("order_level should be a positive integer that divides batch size, "
                             "but got order_level={}".format(order_level))
        if order_level != batch_size and (model is None or lossfunc is None):
            raise ValueError("If order_level < batch size, model and loss MUST be passed to OrderedSampler.")
        if balance_type == PAIR_BALANCE and (batch_size // order_level) % 2 != 0:
            logging.warning("Currently the mod(batch_size // order_level, 2) is not zero, this could incur additional noise "
                            "in the pair balancing (but still works). To maximize the ordering gain, "
                            "Please either use mean_balance, or make sure mod(batch_size // order_level, 2) is zero.")
        if drop_last:
            logging.warning("drop_last is set to be True, note that this could lead to random ordering on the last batch "
                            "since no gradients are computed on them. It is recommended to NOT to drop last, especially "
                            "when the size for the last batch is large.")
        
        self.data_source = data_source
        self.batch_size = batch_size
        self.per_batch_order = order_level == batch_size
        self.drop_last = drop_last
        self.debug = debug
        self.balance_type = balance_type

        if self.debug:
            print("[DEBUG] use per batch order: {}".format(self.per_batch_order))
        
        if not self.per_batch_order:
            self.model = model = extend(model)
            self.lossfunc = lossfunc = extend(lossfunc)
            # backpack helper for computing per-example gradients.
            self.bp = backpack(BatchGrad(), extension_hook=None, debug=debug)
            CTX.set_active_exts(self.bp.exts)
            CTX.set_debug(self.bp.debug)
            CTX.set_extension_hook(self.bp.extension_hook)
            CTX.set_retain_graph(self.bp.retain_graph)
        else:
            logging.warning("Currently the ordering is performed at the batch level. "
                            "While this is the most efficient setting, the ordering benefits "
                            "can be compromised since the examples within each batch are fixed. "
                            "To enable finer-grained ordering, please set order_level < batch_size.")
            self.model = model
            self.lossfunc = lossfunc

        # map: index of example -> rank in the current order.
        # this mapping will change at the end of each full scan at __iter__()
        if init_order_random:
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
            generator = torch.Generator()
            generator.manual_seed(seed)
            new_ranks = torch.randperm(len(data_source), generator=generator).tolist()
            self._index_to_rank = OrderedDict()
            for i in range(len(new_ranks)):
                self._index_to_rank[new_ranks[i]] = i
        else:
            self._index_to_rank = OrderedDict({i:i for i in range(len(data_source))})
        
        self.indices_tracker = IndicesTracker()
        self.use_tracker = True
        self._set_up_sorter(order_level=order_level,
                            balance_type=balance_type,
                            prob_balance=prob_balance,
                            per_batch_order=self.per_batch_order)
    
    def _set_up_sorter(self,
                    order_level: int,
                    balance_type: str = PAIR_BALANCE,
                    prob_balance: bool = False,
                    per_batch_order: bool = False) -> None:
        if balance_type == PAIR_BALANCE:
            from .sorter.pairbalance import PairBalance
            self.sorter = PairBalance(num_examples=len(self.data_source),
                                    order_level=order_level,
                                    prob_balance=prob_balance,
                                    per_batch_order=per_batch_order)
        elif balance_type == MEAN_BALANCE:
            from .sorter.meanbalance import MeanBalance
            self.sorter = MeanBalance(num_examples=len(self.data_source),
                                    order_level=order_level,
                                    prob_balance=prob_balance,
                                    per_batch_order=per_batch_order)
        else:
            raise NotImplementedError("Unrecognized balancing algorithm: {}.".format(balance_type))

    def get_orders(self):
        return self._index_to_rank

    def step(self, sorter_args: Dict = {}) -> None:
        indices = self.indices_tracker.get_indices()
        if self.balance_type == PAIR_BALANCE:
            sorter_args['is_last_batch'] = self.indices_tracker.is_last_batch()
        updated_ranks = self.sorter.step(indices=indices, model=self.model, **sorter_args)
        self._update_index_rank(updated_ranks=updated_ranks)

    def _update_index_rank(self, updated_ranks: OrderedDict) -> None:
        for k in updated_ranks.keys():
            self._index_to_rank[k] = updated_ranks[k]

    def reset_epoch(self):
        # need to reset the tracker
	if not self.indices_tracker.sanity_check():
            raise ValueError("The OrderedSampler encounters an issue of non-empty indices cache. "
                            "This could happen when the ``.step()`` function of OrderedSampler "
                            "is missed between ``.backward()`` and ``.zero_grad()`` in your script. "
                            "Note that if you are using gradient accumulation steps, then "
                            "``.step()`` must be called right after every ``backward()``. "
                            "This could also happen when the dataloader wrapping the OrderedSampler "
                            "is called before the actual training. If this is the case, please turn off the "
                            "indices tracker by ``.stop_tracker()`` and turn it on right before the training "
                            "by ``.start_tracker()``.")
        self._index_to_rank = OrderedDict(
            {k: v for k, v in sorted(self._index_to_rank.items(), key=lambda item: item[1], reverse=False)}
        )
        self.sorter.reset_epoch()

    def stop_tracker(self):
        self.use_tracker = False
    
    def start_tracker(self):
        self.use_tracker = True

    def __iter__(self):
        self.reset_epoch()
        if self.drop_last:
            sampler_iter = iter(self._index_to_rank.keys())
            while True:
                try:
                    batch = [next(sampler_iter) for _ in range(self.batch_size)]
                    if self.use_tracker:
                        self.indices_tracker.update(batch)
                    yield batch
                except StopIteration:
                    break
        else:
            batch = [0] * self.batch_size
            idx_in_batch = 0
            for idx in self._index_to_rank.keys():
                batch[idx_in_batch] = idx
                idx_in_batch += 1
                if idx_in_batch == self.batch_size:
                    if self.use_tracker:
                        self.indices_tracker.update(batch)
                    yield batch
                    idx_in_batch = 0
                    batch = [0] * self.batch_size
            if idx_in_batch > 0:
                if self.use_tracker:
                    self.indices_tracker.update(batch[:idx_in_batch])
                yield batch[:idx_in_batch]

    def __len__(self):
        if self.drop_last:
            return len(self.data_source) // self.batch_size  # type: ignore[arg-type]
        else:
            return (len(self.data_source) + self.batch_size - 1) // self.batch_size  # type: ignore[arg-type]
    

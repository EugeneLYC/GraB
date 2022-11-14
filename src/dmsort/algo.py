import torch
import copy
import random
from sklearn import random_projection
from .utils import flatten_grad

class Sort:
    def sort(self, orders):
        raise NotImplementedError


class StaleGradGreedySort(Sort):
    """
    Implementation of the algorithm that greedily sort the examples using staled gradients,
        the details can be found in: https://openreview.net/pdf?id=7gWSJrP3opB.
    """
    def __init__(self,
                args,
                num_batches,
                grad_dimen):
        self.args = args
        self.num_batches = num_batches
        self.grad_dimen = grad_dimen
        self.stale_grad_matrix = torch.zeros(num_batches, grad_dimen)
        self.avg_grad = torch.zeros(grad_dimen)
        if args.use_cuda:
            self.stale_grad_matrix = self.stale_grad_matrix.cuda()
            self.avg_grad = self.avg_grad.cuda()
        self._reset_random_proj_matrix()
    
    def _skip_sort_this_epoch(self, epoch):
        return epoch <= self.args.start_sort
    
    def _reset_random_proj_matrix(self):
        rs = random.randint(0, 10000)
        self.rp = random_projection.SparseRandomProjection(n_components=self.grad_dimen, random_state=rs)
    
    def update_stale_grad(self, optimizer, batch_idx, epoch, add_to_avg=True):
        tensor = flatten_grad(optimizer)
        if self.args.use_random_proj:
            # Currently random projection in sklearn only supports CPU.
            if self.args.use_cuda:
                tensor = tensor.cpu()
            tensor = torch.from_numpy(self.rp.fit_transform(tensor.reshape(1, -1)))
            if self.args.use_cuda:
                tensor = tensor.cuda()
            self.stale_grad_matrix[batch_idx].copy_(tensor[0])
        else:
            self.stale_grad_matrix[batch_idx].copy_(tensor)
        if add_to_avg:
            self.avg_grad.add_(tensor / self.num_batches)
        # make sure the same random matrix is used in one epoch
        if batch_idx == self.num_batches - 1 and self.args.use_random_proj:
            self._reset_random_proj_matrix()

    def sort(self, epoch, orders=None):
        if orders is None:
            orders = {i:0 for i in range(self.num_batches)}
        if self._skip_sort_this_epoch(epoch):
            return orders
        if self.args.use_qr:
            assert self.args.use_random_proj_full is False
            _, X = torch.qr(self.stale_grad_matrix.t())
            X = X.t()
        if self.args.use_random_proj_full:
            # Currently random projection in sklearn only supports CPU.
            X = self.stale_grad_matrix.clone()
            if self.args.use_cuda:
                X = X.cpu()
            rp = random_projection.SparseRandomProjection()
            X = torch.from_numpy(rp.fit_transform(X))
            if self.args.use_cuda:
                X = X.cuda()
        if not (self.args.use_qr and self.args.use_random_proj_full):
            X = self.stale_grad_matrix.clone()
        cur_sum = torch.zeros_like(self.avg_grad)
        X.add_(-1 * self.avg_grad)
        remain_ids = set(range(self.num_batches))
        for i in range(1, self.num_batches+1):
            cur_id = -1
            max_norm = float('inf')
            for cand_id in remain_ids:
                cand_norm = torch.norm(
                    X[cand_id] + cur_sum*(i-1)
                ).item()
                if cand_norm < max_norm:
                    max_norm = cand_norm
                    cur_id = cand_id
            remain_ids.remove(cur_id)
            orders[cur_id] = i
            cur_sum.add_(X[cur_id])
        self.avg_grad.zero_()
        orders = {k: v for k, v in sorted(orders.items(), key=lambda item: item[1], reverse=False)}
        return orders


class StaleGradDiscrepencyMinimizationSort(Sort):
    """
    Implementation of the GraB algorithm, which uses stale gradient to sort the examples
        via minimizing the discrepancy bound. The details can be found in:
        https://arxiv.org/abs/2205.10733.
        
    """
    def __init__(self,
                args,
                num_batches,
                grad_dimen):
        self.args = args
        self.num_batches = num_batches
        self.grad_dimen = grad_dimen
        self.avg_grad = torch.zeros(grad_dimen)
        if args.use_cuda:
            self.avg_grad = self.avg_grad.cuda()
        self.cur_sum = torch.zeros_like(self.avg_grad)
        self.next_epoch_avg_grad = torch.zeros_like(self.avg_grad)
        self.orders = {i:0 for i in range(self.num_batches)}
        self.first = 0
        self.last = self.num_batches
    
    def _skip_sort_this_epoch(self, epoch):
        return epoch <= self.args.start_sort
    
    def sort(self):
        self.orders = {k: v for k, v in sorted(self.orders.items(), key=lambda item: item[1], reverse=False)}
        self.avg_grad.copy_(self.next_epoch_avg_grad)
        self.next_epoch_avg_grad.zero_()
        self.cur_sum.zero_()
        self.first = 0
        self.last = self.num_batches
        return self.orders

    def step(self, optimizer, batch_idx):
        t = None
        # use the stale variance checkpoint for preconditioning
        for _, param_group in enumerate(optimizer.param_groups):
            for p in param_group['params']:
                if p.grad is None:
                    continue
                state = optimizer.state[p]
                if len(state) == 0:
                    exp_avg_sq = torch.zeros_like(p.grad.data)
                else:
                    exp_avg_sq = state['exp_avg_sq'] if 'var_ckpt' not in state.keys() else state['var_ckpt']
                if p.grad is not None:
                    if t is None:
                        t = torch.flatten(
                            p.grad.data.mul(exp_avg_sq.pow(self.args.pow))
                        )
                    else:
                        t = torch.cat(
                            (t, torch.flatten(p.grad.data.mul(exp_avg_sq.pow(self.args.pow))))
                        )
        cur_grad = t
        self.next_epoch_avg_grad.add_(cur_grad / self.num_batches)
        cur_grad.add_(-1 * self.avg_grad)
        # The balancing algorithm used here is described in Algorithm 5 in 
        #   https://arxiv.org/abs/2205.10733. We can always replace it with other balancing variants.
        if torch.norm(self.cur_sum + cur_grad) <= torch.norm(self.cur_sum - cur_grad):
            self.orders[batch_idx] = self.first
            self.first += 1
            self.cur_sum.add_(cur_grad)
        else:
            self.orders[batch_idx] = self.last
            self.last -= 1
            self.cur_sum.add_(-1 * cur_grad)


class FlipFlopSort(Sort):
    def __init__(self,
                args,
                num_batches,
                grad_dimen):
        self.args = args
        self.num_batches = num_batches
        self.orders = {i:0 for i in range(self.num_batches)}
    
    def sort(self, epoch):
        if epoch % 2 == 0:
            idx_list = [i for i in range(self.num_batches)]
            idx_list_copy = [i for i in range(self.num_batches)]
            random.shuffle(idx_list)
            self.orders = {i:j for i, j in zip(idx_list, idx_list_copy)}
            self.orders = {k: v for k, v in sorted(self.orders.items(), key=lambda item: item[1], reverse=False)}
        else:
            self.orders = {k: v for k, v in sorted(self.orders.items(), key=lambda item: item[1], reverse=True)}
        return self.orders
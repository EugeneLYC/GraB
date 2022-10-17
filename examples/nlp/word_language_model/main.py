# coding: utf-8
import argparse
import math
import os
import torch
import torch.nn as nn
import data
import model
import random
import tqdm
import time
from contextlib import contextmanager
from tensorboardX import SummaryWriter

from constants import _STALE_GRAD_SORT_, \
                    _RANDOM_RESHUFFLING_, \
                    _SHUFFLE_ONCE_, \
                    _DM_SORT_, \
                    _FLIPFLOP_SORT_

parser = argparse.ArgumentParser(description='PyTorch RNN/LSTM/GRU Language Model')
parser.add_argument('--data', type=str, default='./wikitext-2',
                    help='location of the data corpus')
parser.add_argument('--model', type=str, default='LSTM', 
                    choices=['RNN_TANH', 'RNN_RELU', 'LSTM', 'GRU', 'Transformer'],
                    help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU)')
parser.add_argument('--emsize', type=int, default=32,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=32,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=2,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=20,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=50,
                    help='upper epoch limit')
parser.add_argument('--train_batch_size', type=int, default=40, metavar='N',
                    help='train batch size')
parser.add_argument('--val_batch_size', type=int, default=10, metavar='N',
                    help='val batch size')
parser.add_argument('--test_batch_size', type=int, default=1, metavar='N',
                    help='test batch size')
parser.add_argument('--bptt', type=int, default=35,
                    help='sequence length')
parser.add_argument('--dropout', type=float, default=0.,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--tied', action='store_true',
                    help='tie the word embedding and softmax weights')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='report interval')
parser.add_argument('--nhead', type=int, default=2,
                    help='the number of heads in the encoder/decoder of the transformer model')
parser.add_argument('--notes', type=str, default='wiki2')

parser.add_argument('--shuffle_type', type=str)
parser.add_argument('--use_tensorboard',
                        default=False,
                        action='store_true',
                        help='log the seeds results in a txt file for consistent results')
    
parser.add_argument('--tensorboard_path',
                    type=str,
                    help='the base directory for tensorboard logs')
    

args = parser.parse_args()
setattr(args, 'use_cuda', torch.cuda.is_available())

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
random.seed(args.seed)


device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

def make_directory_if_not_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)


###############################################################################
# Load data
###############################################################################

train_path = os.path.join(args.data, 'train.txt')
valid_path = os.path.join(args.data, 'valid.txt')
test_path = os.path.join(args.data, 'test.txt')

corpus = data.Corpus(train_path=train_path, valid_path=valid_path, test_path=test_path)

def batchify(data, bsz):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    return data


train_data_train = batchify(corpus.train.clone(), args.train_batch_size)

train_data_test = batchify(corpus.train.clone(), args.train_batch_size)
val_data = batchify(corpus.valid, args.val_batch_size)
test_data = batchify(corpus.test, args.test_batch_size)


train_ppl_in_training = []

train_ppl_each_epoch = []
val_ppl_each_epoch = []
test_ppl_each_epoch = []


###############################################################################
# Build the model
###############################################################################

ntokens = len(corpus.dictionary)

if args.model == 'Transformer':
    model = model.TransformerModel(ntokens, args.emsize, args.nhead, args.nhid, args.nlayers, args.dropout).to(device)
else:
    model = model.RNNModel(args.model, ntokens, args.emsize, args.nhid, args.nlayers, args.dropout, args.tied).to(device)

criterion = nn.NLLLoss()

###############################################################################
# Training code
###############################################################################

def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""

    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)

class Timer:
    """
    Timer for PyTorch code
    Comes in the form of a contextmanager:
    Example:
    >>> timer = Timer()
    ... for i in range(10):
    ...     with timer("expensive operation"):
    ...         x = torch.randn(100)
    ... print(timer.summary())
    """

    def __init__(self, verbosity_level=1, skip_first=True, use_cuda=True):
        self.verbosity_level = verbosity_level
        #self.log_fn = log_fn if log_fn is not None else self._default_log_fn
        self.skip_first = skip_first
        self.cuda_available = torch.cuda.is_available() and use_cuda

        self.reset()

    def reset(self):
        """Reset the timer"""
        self.totals = {}  # Total time per label
        self.first_time = {}  # First occurrence of a label (start time)
        self.last_time = {}  # Last occurence of a label (end time)
        self.call_counts = {}  # Number of times a label occurred

    @contextmanager
    def __call__(self, label, epoch=-1.0, verbosity=1):
        # Don't measure this if the verbosity level is too high
        if verbosity > self.verbosity_level:
            yield
            return

        # Measure the time
        self._cuda_sync()
        start = time.time()
        yield
        self._cuda_sync()
        end = time.time()

        # Update first and last occurrence of this label
        if label not in self.first_time:
            self.first_time[label] = start
        self.last_time[label] = end

        # Update the totals and call counts
        if label not in self.totals and self.skip_first:
            self.totals[label] = 0.0
            del self.first_time[label]
            self.call_counts[label] = 0
        elif label not in self.totals and not self.skip_first:
            self.totals[label] = end - start
            self.call_counts[label] = 1
        else:
            self.totals[label] += end - start
            self.call_counts[label] += 1

        #if self.call_counts[label] > 0:
        #    # We will reduce the probability of logging a timing
        #    # linearly with the number of time we have seen it.
        #    # It will always be recorded in the totals, though.
        #    if np.random.rand() < 1 / self.call_counts[label]:
        #        self.log_fn(
        #            "timer", {"epoch": epoch, "value": end - start}, {"event": label}
        #        )

    def summary(self):
        """
        Return a summary in string-form of all the timings recorded so far
        """
        if len(self.totals) > 0:
            with StringIO() as buffer:
                total_avg_time = 0
                print("--- Timer summary ------------------------", file=buffer)
                print("  Event   |  Count | Average time |  Frac.", file=buffer)
                for event_label in sorted(self.totals):
                    total = self.totals[event_label]
                    count = self.call_counts[event_label]
                    if count == 0:
                        continue
                    avg_duration = total / count
                    total_runtime = (
                        self.last_time[event_label] - self.first_time[event_label]
                    )
                    runtime_percentage = 100 * total / total_runtime
                    total_avg_time += avg_duration if "." not in event_label else 0
                    print(
                        f"- {event_label:30s} | {count:6d} | {avg_duration:11.5f}s | {runtime_percentage:5.1f}%",
                        file=buffer,
                    )
                print("-------------------------------------------", file=buffer)
                event_label = "total_averaged_time"
                print(
                    f"- {event_label:30s}| {count:6d} | {total_avg_time:11.5f}s |",
                    file=buffer,
                )
                print("-------------------------------------------", file=buffer)
                return buffer.getvalue()

    def _cuda_sync(self):
        """Finish all asynchronous GPU computations to get correct timings"""
        if self.cuda_available:
            torch.cuda.synchronize()

    def _default_log_fn(self, _, values, tags):
        label = tags["label"]
        epoch = values["epoch"]
        duration = values["value"]

class TrainDataset(torch.utils.data.Dataset):
    def __init__(self, tensor, device, shuffle=False) -> None:
        super().__init__()
        self.data = tensor
        self.device = device
        self.shuffle = shuffle
        if self.shuffle:
            a = list(range(self.data.shape[0] // args.bptt))
            b = list(range(self.data.shape[0] // args.bptt))
            random.shuffle(b)
            self.mapping = {i:j for i, j in zip(a, b)}

    def __getitem__(self, i):
        if self.shuffle:
            i = self.mapping[i]
        if i >= len(self): raise IndexError(f'index {i} out of range')
        i = i * args.bptt
        seq_len = min(args.bptt, self.data.shape[0] - 1 - i)
        data = self.data[i:i + seq_len]
        target = self.data[i + 1:i + 1 + seq_len]
        return data.to(self.device), target.view(-1).to(self.device)

    def __len__(self):
        return (self.data.shape[0] // args.bptt)


def evaluate(dataset, counter):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0
    if args.model != 'Transformer':
        hidden = model.init_hidden(dataset.data.shape[-1])
    with torch.no_grad():
        for idx, (data, targets) in enumerate(dataset):
            if args.model == 'Transformer':
                output = model(data)
                output = output.view(-1, ntokens)
            else:
                output, hidden = model(data, hidden)
                hidden = repackage_hidden(hidden)
            total_loss += (len(data) * criterion(output, targets)).item()
            counter.update(1)
        return (total_loss / len(dataset.data))


def train(epoch, optimizer, dataset, counter, sorter, timer):
    # Turn on training mode which enables dropout.
    model.train()
    if args.model != 'Transformer':
        hidden = model.init_hidden(dataset.data.shape[-1])  
    total_loss = 0  

    if sorter is not None:
        with timer("sorting", epoch=epoch):
            if args.shuffle_type == _STALE_GRAD_SORT_:
                orders = sorter.sort(epoch)
            elif args.shuffle_type == _DM_SORT_:
                orders = sorter.sort()
            elif args.shuffle_type == _FLIPFLOP_SORT_:
                orders = sorter.sort(epoch=epoch)
            else:
                raise NotImplementedError
    else:
        orders = {i:0 for i in range(len(dataset))}
        if args.shuffle_type == _RANDOM_RESHUFFLING_:
            a = list(range(len(dataset)))
            random.shuffle(a)
            orders = {i:0 for i in a}

    for idx in orders.keys():
        data, targets = dataset[idx]
        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        with timer("forward pass", epoch=epoch):
            optimizer.zero_grad()
            if args.model == 'Transformer':
                output = model(data)
                output = output.view(-1, ntokens)
            else:
                hidden = repackage_hidden(hidden)
                output, hidden = model(data, hidden)
            loss = criterion(output, targets)
        with timer("backward pass", epoch=epoch):
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)

        if sorter is not None and args.shuffle_type == _STALE_GRAD_SORT_:
            with timer("sorting", epoch=epoch):
                sorter.update_stale_grad(optimizer=optimizer,
                                        batch_idx=idx,
                                        epoch=epoch)
            logging.info(f"Storing the staled gradient used in StaleGradGreedySort method.")
        if sorter is not None and args.shuffle_type == _DM_SORT_:
            with timer("sorting", epoch=epoch):
                sorter.step(optimizer=optimizer, batch_idx=idx)

        with timer("backward pass", epoch=epoch):
            optimizer.step()

        total_loss += loss.item()
        if idx % args.log_interval == 0 and idx > 0:
            cur_loss = total_loss / args.log_interval
            print('| epoch {:3d} | {:5d}/{:5d} batches | loss {:5.2f}'\
                .format(epoch, idx, len(dataset), cur_loss))
            total_loss = 0
    total_time = timer.totals["forward pass"] + timer.totals["backward pass"]
    if sorter is not None:
        total_time += timer.totals["sorting"]
    return total_time


def main():
    print(vars(args))
    shuffle_flag = True if args.shuffle_type == _SHUFFLE_ONCE_ else False
    train_loader_training = TrainDataset(train_data_train, device, shuffle=shuffle_flag)
    train_loader_testing = TrainDataset(train_data_test, device)

    val_loader = TrainDataset(val_data, device)
    test_loader = TrainDataset(test_data, device)

    total_steps = (len(train_loader_training) + len(train_loader_testing) + len(val_loader) + len(test_loader)) * args.epochs

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, threshold=5)


    counter = tqdm.tqdm(range(total_steps), mininterval=10)

    num_batches = len(train_loader_training)
    grad_dimen = sum(p.numel() for p in model.parameters() if p.requires_grad)
    timer = Timer(verbosity_level=1, use_cuda=args.use_cuda)

    if args.shuffle_type in [_RANDOM_RESHUFFLING_, _SHUFFLE_ONCE_]:
        sorter = None
    else:
        if args.shuffle_type == _STALE_GRAD_SORT_:
            from dmsort.algo import StaleGradGreedySort
            sorter = StaleGradGreedySort(args,
                                        num_batches,
                                        grad_dimen)
        elif args.shuffle_type == _DM_SORT_:
            from dmsort.algo import StaleGradDiscrepencyMinimizationSort
            sorter = StaleGradDiscrepencyMinimizationSort(args,
                                                        num_batches,
                                                        grad_dimen)
        elif args.shuffle_type == _FLIPFLOP_SORT_:
            from dmsort.algo import FlipFlopSort
            sorter = FlipFlopSort(args,
                                num_batches,
                                grad_dimen)
        else:
            raise NotImplementedError("This sorting method is not supported yet")

    if args.use_tensorboard:
        tb_path = os.path.join(args.tensorboard_path, 'runs', args.shuffle_type+'_'+str(args.seed))
        tb_logger = SummaryWriter(tb_path)
    else:
        tb_logger = None
    
    for epoch in range(0, args.epochs):
        total_time = train(epoch, optimizer, train_loader_training, counter, sorter, timer)
        train_loss = evaluate(train_loader_testing, counter)
        val_loss = evaluate(val_loader, counter)
        # test_loss = evaluate(test_loader, counter)

        train_ppl = torch.exp(torch.as_tensor(train_loss))
        val_ppl = torch.exp(torch.as_tensor(val_loss))
        # test_ppl = torch.exp(torch.as_tensor(test_loss))

        # train_ppl_each_epoch.append(torch.exp(torch.as_tensor(train_loss))) # perplexity
        # val_ppl_each_epoch.append(torch.exp(torch.as_tensor(val_loss))) # perplexity
        # test_ppl_each_epoch.append(torch.exp(torch.as_tensor(test_loss))) # perplexity
        if tb_logger is not None:
            tb_logger.add_scalar('train/epoch/loss', train_loss, epoch)
            tb_logger.add_scalar('train/time/loss', train_loss, total_time)
            tb_logger.add_scalar('val/epoch/ppl', val_ppl, epoch)
            tb_logger.add_scalar('val/time/ppl', val_ppl, total_time)
            tb_logger.add_scalar('val/epoch/loss', val_loss, epoch)
            tb_logger.add_scalar('val/time/loss', val_loss, total_time)

        lr_scheduler.step(val_ppl)
        print(f'| end of epoch {epoch:3d} | train ppl {train_ppl:.2f} | valid ppl {val_ppl:8.2f}')
    if tb_logger is not None:
        tb_logger.close()

    
if __name__ == '__main__':
    main()
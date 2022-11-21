import os
import torch
import time
import copy
import pickle
import logging
import lmdb
from contextlib import contextmanager
from io import StringIO
from constants import _STALE_GRAD_SORT_, \
                    _FRESH_GRAD_SORT_, \
                    _DM_SORT_, \
                    _MNIST_, \
                    _FLIPFLOP_SORT_
import torch.utils.data as data
from dmsort.utils import compute_avg_grad_error

def build_task_name(args):
    task_name = 'MODEL-' + args.model + \
                '_DATA-' + args.dataset + \
                '_SFTYPE-' + args.shuffle_type + \
                '_SEED-' + str(args.seed) + \
                '-LR-' + str(args.lr)
    if args.shuffle_type == 'fresh':
        task_name = task_name + '_proj-' + str(args.zo_batch_size)
    if args.shuffle_type == 'greedy' and args.use_random_proj:
        task_name = task_name + '_proj-' + str(args.proj_target)
    return task_name

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def train(args,
        loader,
        model,
        criterion,
        optimizer,
        epoch,
        tb_logger,
        timer=None,
        sorter=None):
    
    losses = AverageMeter()
    top1 = AverageMeter()

    model.train()
    grad_buffer = copy.deepcopy(model)
    for p in grad_buffer.parameters():
        p.data.zero_()
    train_batches = list(enumerate(loader))
    num_batches = len(train_batches)
    if sorter is not None:
        with timer("sorting", epoch=epoch):
            if args.shuffle_type == _STALE_GRAD_SORT_:
                orders = sorter.sort(epoch)
            elif args.shuffle_type == _FRESH_GRAD_SORT_:
                orders = sorter.sort(epoch=epoch,
                                    model=model,
                                    train_batches=train_batches,
                                    optimizer=optimizer,
                                    oracle_type='cv')
            elif args.shuffle_type == _DM_SORT_:
                orders = sorter.sort()
            elif args.shuffle_type == _FLIPFLOP_SORT_:
                orders = sorter.sort(epoch=epoch)
            else:
                raise NotImplementedError
    else:
        orders = {i:0 for i in range(len(train_batches))}

    if args.log_metric:
        compute_avg_grad_error(args,
                            model,
                            train_batches,
                            optimizer,
                            epoch,
                            tb_logger,
                            oracle_type='cv',
                            orders=orders)
        logging.warning(f"Logging the average gradient error. \
            This is only for monitoring and will slow down training, \
                please remove --log_metric for full-speed training.")

    grad_step = 0
    cur_step = 0
    for i in orders.keys():
        grad_step += 1
        cur_step += 1
        _, batch = train_batches[i]
        with timer("forward pass", epoch=epoch):
            loss, prec1, cur_batch_size = model(batch)
        with timer("backward pass", epoch=epoch):
            optimizer.zero_grad()
            loss.backward()
        
        for p1, p2 in zip(grad_buffer.parameters(), model.parameters()):
            p1.data.add_(p2.grad.data)

        if sorter is not None and args.shuffle_type == _STALE_GRAD_SORT_:
            with timer("sorting", epoch=epoch):
                sorter.update_stale_grad(optimizer=optimizer,
                                        batch_idx=i,
                                        epoch=epoch)
            logging.info(f"Storing the staled gradient used in StaleGradGreedySort method.")
        if sorter is not None and args.shuffle_type == _DM_SORT_:
            with timer("sorting", epoch=epoch):
                sorter.step(optimizer=optimizer, batch_idx=i)
        
        if grad_step % args.grad_accumulation_step == 0 or grad_step == num_batches:
            for p1, p2 in zip(grad_buffer.parameters(), model.parameters()):
                p1.data.mul_(1/cur_step)
                p2.grad.data.zero_().add_(p1.data)
                p1.data.zero_()
            with timer("backward pass", epoch=epoch):
                optimizer.step()
            cur_step = 0

        loss = loss.float()

        # measure accuracy and record loss
        losses.update(loss.item(), cur_batch_size)
        top1.update(prec1.item(), cur_batch_size)

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      epoch, i, len(loader), loss=losses, top1=top1))
    
    total_time = timer.totals["forward pass"] + timer.totals["backward pass"]
    if sorter is not None:
        total_time += timer.totals["sorting"]
    
    return total_time

def validate(args, loader, model, criterion, epoch, tb_logger, loader_name, total_time):
    """
    Run evaluation
    """
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(loader):
            loss, prec1, cur_batch_size = model(batch)
            loss = loss.float()

            # measure accuracy and record loss
            losses.update(loss.item(), cur_batch_size)
            top1.update(prec1.item(), cur_batch_size)

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                          i, len(loader), loss=losses,
                          top1=top1))
    if args.use_tensorboard:
        tb_logger.add_scalar(loader_name+'/epoch/accuracy', top1.avg, epoch)
        tb_logger.add_scalar(loader_name+'/epoch/loss', losses.avg, epoch)
        tb_logger.add_scalar(loader_name+'/time/accuracy', top1.avg, total_time)
        tb_logger.add_scalar(loader_name+'/time/loss', losses.avg, total_time)

    print(' * Prec@1 {top1.avg:.3f}'.format(top1=top1))

    return



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
        print(f"Timer: {label:30s} @ {epoch:4.1f} - {duration:8.5f}s")


def raw_reader(path):
    with open(path, 'rb') as f:
        bin_data = f.read()
    return bin_data


def dumps_data(obj):
    """
    Serialize an object.
    Returns:
        Implementation-dependent bytes-like object
    """
    return pickle.dumps(obj)

## Helper functions for ImageNet
def folder2lmdb(spath, dpath, name="train", write_frequency=5000):
    directory = os.path.expanduser(os.path.join(spath, name))
    from torch.utils.data import DataLoader
    from torchvision.datasets import ImageFolder
    dataset = ImageFolder(directory, loader=raw_reader)
    data_loader = DataLoader(dataset, num_workers=16, collate_fn=lambda x: x)

    lmdb_path = os.path.join(dpath, "%s.lmdb" % name)
    isdir = os.path.isdir(lmdb_path)

    db = lmdb.open(lmdb_path, subdir=isdir,
                   map_size=1099511627776 * 2, readonly=False,
                   meminit=False, map_async=True)

    txn = db.begin(write=True)
    for idx, data in enumerate(data_loader):
        image, label = data[0]

        txn.put(u'{}'.format(idx).encode('ascii'), dumps_data((image, label)))
        if idx % write_frequency == 0:
            print("[%d/%d]" % (idx, len(data_loader)))
            txn.commit()
            txn = db.begin(write=True)

    # finish iterating through dataset
    txn.commit()
    keys = [u'{}'.format(k).encode('ascii') for k in range(idx + 1)]
    with db.begin(write=True) as txn:
        txn.put(b'__keys__', dumps_data(keys))
        txn.put(b'__len__', dumps_data(len(keys)))

    print("Flushing database ...")
    db.sync()
    db.close()

class ImageFolderLMDB(data.Dataset):
    def __init__(self, db_path, transform=None, target_transform=None):
        self.db_path = db_path
        self.env = lmdb.open(db_path, subdir=os.path.isdir(db_path),
                             readonly=True, lock=False,
                             readahead=False, meminit=False)
        with self.env.begin(write=False) as txn:
            self.length = loads_data(txn.get(b'__len__'))
            self.keys = loads_data(txn.get(b'__keys__'))

        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        env = self.env
        with env.begin(write=False) as txn:
            byteflow = txn.get(self.keys[index])

        unpacked = loads_data(byteflow)

        # load img
        imgbuf = unpacked[0]
        buf = six.BytesIO()
        buf.write(imgbuf)
        buf.seek(0)
        img = Image.open(buf).convert('RGB')

        # load label
        target = unpacked[1]

        if self.transform is not None:
            img = self.transform(img)

        # im2arr = np.array(img)
        # im2arr = torch.from_numpy(im2arr)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target
        # return im2arr, target

    def __len__(self):
        return self.length

    def __repr__(self):
        return self.__class__.__name__ + ' (' + self.db_path + ')'
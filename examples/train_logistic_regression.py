import random
import torch
import torchvision
from torch.nn import CrossEntropyLoss, Linear
from orderedsampler import OrderedSampler
from tensorboardX import SummaryWriter

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

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

SEED = random.randint(0, 10000)
EPOCHS=100

random.seed(SEED)
torch.manual_seed(SEED)
use_cuda = torch.cuda.is_available()

# model
model = Linear(784, 10)
if use_cuda:
    model = model.cuda()

# optimizer
optimizer = torch.optim.SGD(params=model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)

# loss
lossfunc = CrossEntropyLoss()
if use_cuda:
    lossfunc = lossfunc.cuda()

# dataset
transform=torchvision.transforms.Compose([
                                torchvision.transforms.ToTensor(),
                                torchvision.transforms.Normalize((0.1307,), (0.3081,))])
trainset = torchvision.datasets.MNIST('./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.MNIST('./data', train=False, transform=transform)

# data loader
ordered_sampler = OrderedSampler(trainset,
                            batch_size=64,
                            order_level=2,
                            model=model,
                            lossfunc=lossfunc,
                            balance_type='pair_balance')
model, lossfunc = ordered_sampler.model, ordered_sampler.lossfunc
train_loader = torch.utils.data.DataLoader(trainset, batch_sampler=ordered_sampler, num_workers=0, pin_memory=False)
train_val_loader = torch.utils.data.DataLoader(trainset, batch_size=1024, shuffle=False, num_workers=0, pin_memory=False)
val_loader = torch.utils.data.DataLoader(testset, batch_size=1024, shuffle=False, num_workers=0, pin_memory=False)


def train(loader, model, lossfunc, optimizer):
    model.train()
    for i, batch in enumerate(loader):
        x, y = batch
        if use_cuda:
            x, y = x.cuda(), y.cuda()
        x = x.reshape(-1, 784)
        optimizer.zero_grad()
        loss = lossfunc(model(x), y)
        loss.backward()
        ordered_sampler.step()
        optimizer.step()

def val(loader, model, lossfunc, epoch):
    losses = AverageMeter()
    top1 = AverageMeter()
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(loader):
            x, y = batch
            if use_cuda:
                x, y = x.cuda(), y.cuda()
            x = x.reshape(-1, 784)
            output = model(x)
            loss = lossfunc(output, y)
            prec1 = accuracy(output.data, y)[0]
            cur_batch_size = x.size(0)
            losses.update(loss.item(), cur_batch_size)
            top1.update(prec1.item(), cur_batch_size)
    print('Epoch: [{0}]\t'
                  'Loss {losses.avg:.4f}\t'
                  'Prec@1 {top1.avg:.3f}'.format(
                      epoch, losses=losses, top1=top1))
    
    return top1.avg, losses.avg


tb_writer = SummaryWriter('./runs/release_SEED' + str(SEED))
for epoch in range(EPOCHS):
    train(train_loader, model, lossfunc, optimizer)

    train_acc, train_loss = val(train_val_loader, model, lossfunc, epoch)
    test_acc, test_loss = val(val_loader, model, lossfunc, epoch)

    tb_writer.add_scalar('train/epoch/accuracy', train_acc, epoch)
    tb_writer.add_scalar('train/epoch/loss', train_loss, epoch)
    tb_writer.add_scalar('val/epoch/accuracy', test_acc, epoch)
    tb_writer.add_scalar('val/epoch/loss', test_loss, epoch)
tb_writer.close()


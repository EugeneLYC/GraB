import torch
from constants import _MNIST_, _SQUEEZENET_

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

class VisionModel:
    def __init__(self, args, model, criterion):
        self.args = args
        self.model = model
        self.criterion = criterion
    
    def __call__(self, batch):
        (input_var, target_var) = batch
        if self.args.use_cuda:
            input_var = input_var.cuda()
            target_var = target_var.cuda()
        if self.args.dataset == _MNIST_:
            input_var = input_var.reshape(-1, 784)
        output = self.model(input_var)
        loss = self.criterion(output, target_var)

        prec1 = accuracy(output.data, target_var)[0]
        
        return loss, prec1, input_var.size(0)
    
    def parameters(self):
        return self.model.parameters()
    
    def train(self):
        self.model.train()
    
    def eval(self):
        self.model.eval()

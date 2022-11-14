import os
import random
import torch
import logging
import torchvision
import torchvision.datasets as datasets
from tensorboardX import SummaryWriter
import torchvision.transforms as transforms
from visionmodel import VisionModel
from arguments import get_args
from utils import train, validate, Timer, build_task_name
from constants import _RANDOM_RESHUFFLING_, \
                    _SHUFFLE_ONCE_, \
                    _STALE_GRAD_SORT_, \
                    _FRESH_GRAD_SORT_, \
                    _MNIST_, \
                    _DM_SORT_, \
                    _FLIPFLOP_SORT_
from adam import Adam
logger = logging.getLogger(__name__)

def main():
    args = get_args()

    if args.seed == 0:
        args.seed = random.randint(0, 10000)
    
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    logger.info(f"Using random seed {args.seed} for random and torch module.")

    args.use_cuda = torch.cuda.is_available()
    logger.info(f"Using GPU: {args.use_cuda}")

    timer = Timer(verbosity_level=1, use_cuda=args.use_cuda)

    criterion = torch.nn.CrossEntropyLoss()
    if args.use_cuda:
        criterion.cuda()
    logger.info(f"Using Cross Entropy Loss for classification.")

    # The input feature for MNIST is 784, and it has 10 classes
    model = torch.nn.DataParallel(torch.nn.Linear(784, 10))
    if args.use_cuda:
        model.cuda()
    model_dimen = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model = VisionModel(args, model, criterion)
    logger.info(f"Using model: {args.model} with dimension: {model_dimen}.")
    
    optimizer = Adam(params=model.parameters(),
                    lr=args.lr,
                    weight_decay=args.weight_decay)
    logger.info(f"Using optimizer SGD with hyperparameters: learning rate={args.lr}; momentum={args.momentum}; weight decay={args.weight_decay}.")
    logger.info(f"Using dataset: {args.dataset}")

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=1, last_epoch=args.start_epoch-1)
    logger.info(f"Using dataset: {args.dataset}")

    loaders = {}
    shuffle_flag = True if args.shuffle_type in [_RANDOM_RESHUFFLING_, _FRESH_GRAD_SORT_] else False
    data_path = os.path.join(args.data_path, "data")
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))])
    trainset = dataset=datasets.MNIST(root=data_path, train=True, download=True, transform=transform)
    testset = datasets.MNIST(root=data_path, train=False, transform=transform)
    loaders['train'] = torch.utils.data.DataLoader(trainset,
                                                    batch_size=args.batch_size,
                                                    shuffle=shuffle_flag,
                                                    persistent_workers=False,
                                                    num_workers=args.num_workers,
                                                    pin_memory=False)
    loaders['train_val'] = torch.utils.data.DataLoader(trainset,
                                                    batch_size=args.test_batch_size,
                                                    shuffle=False,
                                                    num_workers=args.num_workers,
                                                    pin_memory=False)
    loaders['val'] = torch.utils.data.DataLoader(testset,
                                                    batch_size=args.test_batch_size,
                                                    shuffle=False,
                                                    num_workers=args.num_workers,
                                                    pin_memory=False)
    
    # Epoch-wise data ordering
    if args.shuffle_type in [_RANDOM_RESHUFFLING_, _SHUFFLE_ONCE_]:
        sorter = None
        logger.info(f"Not using any sorting algorithm.")
    else:
        grad_dimen = int(args.proj_ratio * model_dimen) if args.use_random_proj else model_dimen
        num_batches = len(list(enumerate(loaders['train'])))
        if args.shuffle_type == _STALE_GRAD_SORT_:
            from dmsort.algo import StaleGradGreedySort
            sorter = StaleGradGreedySort(args,
                                        num_batches,
                                        grad_dimen)
        elif args.shuffle_type == _FRESH_GRAD_SORT_:
            from dmsort.algo import FreshGradGreedySort
            sorter = FreshGradGreedySort(args,
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
        logger.info(f"Creating sorting algorithm: {args.shuffle_type}.")

    # args.task_name = build_task_name(args)
    # logger.info(f"Creating task name as: {args.task_name}.")
    if args.shuffle_type == _DM_SORT_:
        args.task_name = 'GraB_Adam_power' + str(args.pow)
    else:
        args.task_name = args.shuffle_type + '_Adam'


    if args.use_tensorboard:
        tb_path = os.path.join(args.tensorboard_path, 'runs', args.dataset, args.task_name)
        logger.info(f"Streaming tensorboard logs to path: {tb_path}.")
        tb_logger = SummaryWriter(tb_path)
    else:
        tb_logger = None
        logger.info(f"Disable tensorboard logs currently.")

    for epoch in range(args.start_epoch, args.epochs):
        ttl_time = train(args=args,
            loader=loaders['train'],
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            epoch=epoch,
            tb_logger=tb_logger,
            timer=timer,
            sorter=sorter)
        
        # evaluate on training set
        validate(args=args,
                loader=loaders['train_val'],
                model=model,
                criterion=criterion,
                epoch=epoch,
                tb_logger=tb_logger,
                loader_name='train',
                total_time=ttl_time)
        # evaluate on validation set
        validate(args=args,
                loader=loaders['val'],
                model=model,
                criterion=criterion,
                epoch=epoch,
                tb_logger=tb_logger,
                loader_name='val',
                total_time=ttl_time)

    tb_logger.close()

    logger.info(f"Finish training!")

if __name__ == '__main__':
    torch.multiprocessing.set_sharing_strategy('file_system')
    main()

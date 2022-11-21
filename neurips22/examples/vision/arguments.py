import argparse

def get_args():
    parser = argparse.ArgumentParser(description='Experimental code for the QMC paper')

    parser.add_argument('--model',
                        metavar='ARCH',
                        default='resnet20',
                        help='model to use (lenet, resnetxx)')
    
    parser.add_argument('--pretrained',
                        default=True,
                        action='store_true',
                        help='whether to use pretrained model (currently only for ImageNet)')

    parser.add_argument('--dataset',
                        default='cifar10',
                        type=str,
                        help='dataset used in the experiment (default: cifar10)')
    
    parser.add_argument('--data_path',
                        type=str,
                        help='the base directory for dataset')
    
    parser.add_argument('--num_workers',
                        default=0,
                        type=int,
                        metavar='N',
                        help='number of data loading workers (default: 0)')
    
    parser.add_argument('--epochs',
                        default=200,
                        type=int,
                        metavar='N',
                        help='number of total epochs to run')
    
    parser.add_argument('--start_epoch',
                        default=0,
                        type=int,
                        metavar='N',
                        help='manual epoch number (useful on restarts)')
    
    parser.add_argument('--batch_size',
                        default=64,
                        type=int,
                        metavar='N',
                        help='mini-batch size (default: 128)')

    parser.add_argument('--grad_accumulation_step',
                        default=1,
                        type=int,
                        metavar='N',
                        help='gradient accumulation step in the optimization (default: 1)')
    
    parser.add_argument('--test_batch_size',
                        default=1024,
                        type=int,
                        metavar='N',
                        help='mini-batch size used for testing (default: 1024)')
    
    parser.add_argument('--lr',
                        default=0.1,
                        type=float,
                        metavar='LR',
                        help='initial learning rate')
    
    parser.add_argument('--momentum',
                        default=0.9,
                        type=float,
                        metavar='M',
                        help='momentum')
    
    parser.add_argument('--weight_decay',
                        default=1e-4,
                        type=float,
                        metavar='W',
                        help='weight decay (default: 1e-4)')
    
    parser.add_argument('--print_freq',
                        default=50,
                        type=int,
                        metavar='N',
                        help='print frequency (default: 50)')
    
    parser.add_argument('--start_sort',
                        default=1,
                        type=int,
                        metavar='N',
                        help='the epoch where the greedy strategy will be first used (100 in CIFAR10 case)')
    
    parser.add_argument('--seed',
                        default=0,
                        type=int,
                        metavar='N',
                        help='random seed used in the experiment')
    
    parser.add_argument('--use_tensorboard',
                        default=False,
                        action='store_true',
                        help='log the seeds results in a txt file for consistent results')
    
    parser.add_argument('--tensorboard_path',
                        type=str,
                        help='the base directory for tensorboard logs')
    
    parser.add_argument('--zo_batch_size',
                        default=1,
                        type=int,
                        metavar='N',
                        help='zero-th order mini-batch size (default: 16)')

    # greedy method related arguments
    parser.add_argument('--shuffle_type',
                        default='random_reshuffling',
                        type=str,
                        help='shuffle type used for the optimization (choose from random_reshuffling, shuffle_once, stale_grad_greedy_sort, fresh_grad_greedy_sort)')
    
    parser.add_argument('--task_name',
                        default='test',
                        type=str,
                        help='task name used for tensorboard')
    
    parser.add_argument('--log_metric',
                        default=False,
                        action='store_true',
                        help='whether to log the LHS-QMC metric during training (default: False)')
    
    parser.add_argument('--use_random_proj',
                        default=False,
                        action='store_true',
                        help='whether to use projection when doing the greedy sorting (default: True)')
    
    parser.add_argument('--use_random_proj_full',
                        default=False,
                        action='store_true',
                        help='whether to use projection after storing all the full-dimension gradients (default: True)')
    
    parser.add_argument('--use_qr',
                        default=False,
                        action='store_true',
                        help='whether to use qr_decomposition in the sorting part (default: True)')
    
    parser.add_argument('--proj_ratio',
                        default=0.1,
                        type=float,
                        help='decide project how much ratio of the orginal entire model (default: 0.1)')

    parser.add_argument('--proj_target',
                        default=1024,
                        type=int,
                        help='the target dimension for random projection')
    
    args = parser.parse_args()
    return args

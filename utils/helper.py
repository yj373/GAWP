from collections import defaultdict
import torch
import torchvision
import numpy as np
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
# from utils.attacks import normalize

import os
# import imageio
# from tqdm.autonotebook import tqdm

cifar10_mean = (0.4914, 0.4822, 0.4465) # equals np.mean(train_set.train_data, axis=(0,1,2))/255
cifar10_std = (0.2471, 0.2435, 0.2616) # equals np.std(train_set.train_data, axis=(0,1,2))/255
cifar100_mean = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
cifar100_std = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
svhn_mean = (0.0, 0.0, 0.0)
svhn_std = (1.0, 1.0, 1.0)


def normalize(X, ds='cifar10'):
    # print(X.size())
    if ds == 'cifar10':
        mu = torch.tensor(cifar10_mean).view(3, 1, 1).cuda()
        std = torch.tensor(cifar10_std).view(3, 1, 1).cuda()
    elif ds == 'cifar100':
        mu = torch.tensor(cifar100_mean).view(3, 1, 1).cuda()
        std = torch.tensor(cifar100_std).view(3, 1, 1).cuda()
    elif ds == 'svhn':
        mu = torch.tensor(svhn_mean).view(3, 1, 1).cuda()
        std = torch.tensor(svhn_std).view(3, 1, 1).cuda()
    else:
        return X
    return (X - mu)/std


upper_limit, lower_limit = 1, 0


def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)

### General helper classes ###
class AverageMeter(object):
    """
    Computes and stores the average and current value
    """
    def __init__(self, name='meter', fmt=':f'):
        self.reset()
        self.name = name
        self.fmt = fmt

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        # val: values to sum up
        # n: add val to sum for n times
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        #print(target)
        if (target.dim() > 1):
            target = torch.argmax(target, 1)
        _, pred = output.detach().topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def get_dataset(name='cifar10', root='/data', batch_size=128):
    if name == 'cifar10':
        train_trans = transforms.Compose([
            transforms.RandomCrop(32, padding=4, fill=0, padding_mode='constant'),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        train_set = datasets.CIFAR10(root, train=True, transform=train_trans, download=True)
        test_set = datasets.CIFAR10(root, train=False, transform=transforms.ToTensor(), download=True)    
    elif name == 'cifar100':
        train_trans = transforms.Compose([
            transforms.RandomCrop(32, padding=4, fill=0, padding_mode='constant'),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        train_set = datasets.CIFAR100(root, train=True, transform=train_trans, download=True)
        test_set = datasets.CIFAR100(root, train=False, transform=transforms.ToTensor(), download=True) 
    elif name  == 'mnist':
        trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])
        # if not exist, download mnist dataset
        train_set = datasets.MNIST(root=root, train=True, transform=trans, download=True)
        test_set = datasets.MNIST(root=root, train=False, transform=trans, download=True)
    elif name == 'svhn':
        train_set = datasets.SVHN(root=root, split='train', transform=transforms.ToTensor(), download=True)
        test_set = datasets.SVHN(root=root, split='test', transform=transforms.ToTensor(), download=True)
    else:
        raise ValueError('{} dataset is not supported!'.format(name))
    train_loader =  DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4)
    return train_loader, test_loader

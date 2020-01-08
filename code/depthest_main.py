import os
import sys
from functools import wraps
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.models as models
from creator import create_scheduler, create_optimizer, create_params
from models import create_network, create_lossfunc
from dataloaders import create_datasets
from depthest_trainer import DepthEstimationTrainer
from config import Parameters


def train(args, net, datasets, criterion, optimizer, scheduler):
    # Define trainer
    Trainer = DepthEstimationTrainer(params=args,
                                     net=net,
                                     datasets=datasets,
                                     criterion=criterion,
                                     optimizer=optimizer,
                                     scheduler=scheduler,
                                     sets=list(datasets.keys()),
                                     )
    if args.pretrain:
        if args.encoder[:6] == 'resnet':
            if hasattr(models, args.encoder):
                resnet = getattr(models, args.encoder)(pretrained=True)
            else:
                raise RuntimeError('network not found.' +
                    'The network must be either of resnet50 or resnet101.') 

        if args.resume != '':
            Trainer.reload(resume=args.resume, mode='finetune')
            print('Finetuning from {}'.format(args.resume))
        else:
            Trainer.reload(resume=resnet, mode='finetune')
    elif args.retain:
        Trainer.reload(resume=args.resume, mode='retain')
    Trainer.train()
    return

def test(args, net, datasets):
    # Define trainer
    Trainer = DepthEstimationTrainer(params=args,
                                     net=net,
                                     datasets=datasets,
                                     criterion=None,
                                     optimizer=None,
                                     scheduler=None,
                                     sets=list(datasets.keys()),
                                     )
    Trainer.reload(resume=args.resume, mode='test')
    Trainer.test()
    return

def main():
    args = Parameters().parse()
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    # Dataset
    datasets = create_datasets(args)
    # Network
    net = create_network(args)
    # Loss Function
    criterion = create_lossfunc(args, net)
    # optimizer and parameters
    optim_params = create_params(args, net)
    optimizer = create_optimizer(args, optim_params)
    # learning rate scheduler
    scheduler = create_scheduler(args, optimizer, datasets)
    if args.mode == 'train':
        train(args, net, datasets, criterion, optimizer, scheduler)
        return
    if args.mode == 'test':
        test(args, net, datasets)
        return

if __name__ == '__main__':
    main()
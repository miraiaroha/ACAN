##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: chenyuru
## This source code is licensed under the MIT-style license
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
from torch.optim import lr_scheduler

def create_scheduler(args, optimizer, datasets):
    if args.scheduler == 'step':
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=eval(args.milestones), gamma=args.lr_decay)
    elif args.scheduler == 'poly':
        total_step = (len(datasets['train']) / args.batch + 1) * args.epochs
        scheduler = lr_scheduler.LambdaLR(optimizer, lambda x: (1-x/total_step) ** args.power)
    elif args.scheduler == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=args.lr_decay, patience=args.patience)
    elif args.scheduler == 'constant':
        scheduler = lr_scheduler.LambdaLR(optimizer, lambda x: 1)
    elif args.scheduler == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, args.T_max, args.min_lr)
    return scheduler
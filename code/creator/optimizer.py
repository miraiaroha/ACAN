##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: chenyuru
## This source code is licensed under the MIT-style license
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
import torch.optim as optim

def create_optimizer(args, optim_params):
    if args.optimizer == 'sgd':
        return optim.SGD(optim_params, args.lr, momentum=args.momentum,
                         weight_decay=args.weight_decay)
    elif args.optimizer == 'adagrad':
        return optim.Adagrad(optim_params, args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'adam':
        return optim.Adam(optim_params, args.lr, betas=(args.beta1, args.beta2),
                          weight_decay=args.weight_decay)
    elif args.optimizer == 'amsgrad':
        return optim.Adam(optim_params, args.lr, betas=(args.beta1, args.beta2),
                          weight_decay=args.weight_decay, amsgrad=True)
    elif args.optimizer == 'adabound':
        from adabound import AdaBound
        return AdaBound(optim_params, args.lr, betas=(args.beta1, args.beta2),
                        final_lr=args.final_lr, gamma=args.gamma,
                        weight_decay=args.weight_decay)
    else:
        assert args.optimizer == 'amsbound'
        from adabound import AdaBound
        return AdaBound(optim_params, args.lr, betas=(args.beta1, args.beta2),
                        final_lr=args.final_lr, gamma=args.gamma, 
                        weight_decay=args.weight_decay, amsbound=True)
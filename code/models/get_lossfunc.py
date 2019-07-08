##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: chenyuru
## This source code is licensed under the MIT-style license
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
from .losses import OrdinalRegression2d, CrossEntropy2d, OhemCrossEntropy2d, AttentionLoss2d
import json

def create_lossfunc(args, net):
    ignore_index = 0
    # if args.dataset == 'kitti':
    #     ignore_index = 0

    weight = None
    if args.use_weights:
        if args.dataset == 'nyu':
            with open('../script/nyu_weights_12k_80.json', 'r') as f:
                weight = json.load(f)['weights']
        elif args.dataset == 'kitti':
            with open('../script/kitti_weights_22k_80.json', 'r') as f:
                weight = json.load(f)['weights']
        
    loss_kwargs = {'ignore_index': ignore_index, 'reduction': 'sum', 'use_weights': args.use_weights, 'weight': weight}

    auxfunc = CrossEntropy2d(eps=args.eps, priorType=args.prior, **loss_kwargs)
    if args.classifier == 'CE':
        lossfunc = CrossEntropy2d(eps=args.eps, priorType=args.prior, **loss_kwargs)
    elif args.classifier == 'OHEM':
        lossfunc = OhemCrossEntropy2d(eps=args.eps, priorType=args.prior, thresh=args.ohem_thres, min_kept=args.ohem_keep, **loss_kwargs)
    elif args.classifier == 'OR':
        lossfunc = OrdinalRegression2d(**loss_kwargs)
        auxfunc = OrdinalRegression2d(**loss_kwargs)
    else:
        raise RuntimeError('classifier not found. The classifier must be either of OR, CE or OHEM.')

    attfunc = AttentionLoss2d(scale=1)
        
    criterion_kwargs = {'min_depth': args.min_depth, 'max_depth': args.max_depth, 'num_classes': args.classes, 
                        'AppearanceLoss': lossfunc, 'AuxiliaryLoss': auxfunc, 'AttentionLoss': attfunc,
                        'alpha': args.alpha, 'beta': args.beta}
    criterion = net.LossFunc(**criterion_kwargs)
    return criterion
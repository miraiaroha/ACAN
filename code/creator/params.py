##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: chenyuru
## This source code is licensed under the MIT-style license
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def get_params(mod, Type=''):
        params = []
        for m in mod:
            for n, p in m.named_parameters():
                if Type in n:
                    params += [p]
        return params

def create_params(args, net):
    # filter manully
    if args.encoder in ['resnet50', 'resnet101']:
        base_modules = list(net.children())[:8]
        base_params = get_params(base_modules, '')
        base_params = filter(lambda p: p.requires_grad, base_params)
        add_modules = list(net.children())[8:]
        add_weight_params = get_params(add_modules, 'weight')
        add_bias_params = get_params(add_modules, 'bias')
        if args.optimizer in ['adabound', 'amsbound']:
            optim_params = [{'params': base_params, 'lr': args.lr, 'final_lr': args.final_lr},
                            {'params': add_weight_params, 'lr': args.lr*10, 'final_lr': args.final_lr*10},
                            {'params': add_bias_params, 'lr': args.lr*20, 'final_lr': args.final_lr*10}]
        else:
            optim_params = [{'params': base_params, 'lr': args.lr},
                            {'params': add_weight_params, 'lr': args.lr*10},
                            {'params': add_bias_params, 'lr': args.lr*20}]
    return optim_params
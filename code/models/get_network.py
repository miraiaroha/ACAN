from .model import ResNet

def create_network(args):
    resnet_layer_settings = {'50':  [3, 4, 6, 3], 
                             '101': [3, 4, 23, 3]}
    if args.encoder == 'resnet50':
        setttings = resnet_layer_settings['50']
    elif args.encoder == 'resnet101':
        setttings = resnet_layer_settings['101']
    else:
        raise RuntimeError('network not found.' +
                           'The network must be either of resnet50 or resnet101.')
    net_kwargs = {'min_depth': args.min_depth, 'max_depth': args.max_depth, 'num_classes': args.classes,
                  'classifierType': args.classifier, 'inferenceType': args.inference, 'decoderType': args.decoder,
                  'height': args.height, 'width': args.width, 
                  'alpha': args.alpha, 'beta': args.beta, 'layers': setttings}

    net = ResNet(**net_kwargs)
    return net
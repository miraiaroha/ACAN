##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: chenyuru
## This source code is licensed under the MIT-style license
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def create_datasets(args):
    # Create dataset code
    print("=> creating datasets ...")
    train_dataset = None
    val_dataset = None
    test_dataset = None

    if args.dataset == 'nyu':
        from .nyu_dataloader import NYUDataset as TargetDataset
    elif args.dataset == 'kitti':
        from .kitti_dataloader import KITTIDataset as TargetDataset
    else:
        raise RuntimeError('Dataset not found.' +
                           'The dataset must be either of nyu or kitti.')
    kwargs = {'min_depth': args.min_depth, 'max_depth': args.max_depth,
              'flip': args.random_flip, 'scale': args.random_scale,
              'rotate': args.random_rotate, 'jitter': args.random_jitter, 'crop': args.random_crop}
    if args.mode == 'train':
        train_dataset = TargetDataset(args.rgb_dir, args.dep_dir, args.train_rgb, args.train_dep, mode='train', **kwargs)
        val_dataset = TargetDataset(args.rgb_dir, args.dep_dir, args.val_rgb, args.val_dep, mode='val', **kwargs)
    elif args.mode == 'test':
        test_dataset = TargetDataset(args.rgb_dir, args.dep_dir, args.test_rgb, args.test_dep, mode='test', **kwargs)
        
    print("<= datasets created.")
    datasets = {'train': train_dataset, 'val': val_dataset, 'test': test_dataset}
    return datasets
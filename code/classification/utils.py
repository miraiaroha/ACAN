import sys
import os
import torch
from collections import namedtuple
model_parameters = namedtuple('parameters',
                              'model_name,'
                              'dataset_name,'
                              'dataset_loader,'
                              'network,'
                              'home_image, home_depth, '
                              'train_image_txt, train_depth_txt, '
                              'valid_image_txt, valid_depth_txt, '
                              'checkpoint_path, log_path, '
                              'save_path, '
                              'batch_size, batch_size_valid, '
                              'original_height, original_width, '
                              'resize_height, resize_width,'
                              'crop_height, crop_width, '
                              'learning_rate_mode,'
                              'num_epochs,'
                              'num_classes,'
                              'min_depth, max_depth, '
                              'interval_test'
                              )
""" 
    predicted_region:
    nyu: [21, -20, 25, -24] -> [0, 480, 0, 640]
         [13, -12, 15, -14] -> [0, 288, 0, 384]
    kitti: [150, -5, 45, -45] -> [0, 375, 0, 1242]
           [60, -2, 20, -20] -> [0, 160, 0, 512]
"""

HOME = os.environ['HOME']

def get_parameters(dataset_name, model_name, dataset, model):
    if dataset_name == 'nyu':
        params = model_parameters(
            model_name=model_name,
            dataset_name=dataset_name,
            dataset_loader=dataset,
            network=model,
            home_image=HOME + '/myDataset/NYU_v2/',
            home_depth=HOME + '/myDataset/NYU_v2/',
            train_image_txt='../../nyu_v2_path_txt/train_rgb_12k.txt',
            train_depth_txt='../../nyu_v2_path_txt/train_depth_12k.txt',
            valid_image_txt='../../nyu_v2_path_txt/valid_rgb.txt',
            valid_depth_txt='../../nyu_v2_path_txt/valid_depth.txt',
            checkpoint_path='../../checkpoint/',
            log_path='../../summary/',
            # test mode
            save_path='../../examples_{}_{}/'.format(model_name, dataset_name),
            batch_size=7,
            batch_size_valid=4,
            learning_rate_mode='poly',
            num_epochs=50,
            num_classes=80,
            original_height=480,
            original_width=640,
            resize_height=288,
            resize_width=384,
            crop_height=256,
            crop_width=352,
            min_depth=0.65,           # 0.71329951
            max_depth=10.0,           # 9.99547
            interval_test=1
        )
    elif dataset_name == 'kitti':
        params = model_parameters(
            model_name=model_name,
            dataset_name=dataset_name,
            dataset_loader=dataset,
            network=model,
            home_image=HOME + '/myDataset/KITTI/raw_data_KITTI/',
            home_depth=HOME + '/myDataset/KITTI/datasets_KITTI/',
            train_image_txt='../../kitti_path_txt/eigen_train_files.txt',
            train_depth_txt='../../kitti_path_txt/eigen_train_depth_files.txt',
            valid_image_txt='../../kitti_path_txt/eigen_test_files.txt',
            valid_depth_txt='../../kitti_path_txt/eigen_test_depth_files.txt',
            checkpoint_path='../../checkpoint/',
            log_path='../../summary/',
            # test mode
            save_path='../../examples_{}_{}/'.format(model_name, dataset_name),
            batch_size=8,
            batch_size_valid=4,
            learning_rate_mode='poly',
            num_epochs=50,
            num_classes=80,
            original_height=375,
            original_width=1242,
            resize_height=160,   # 180
            resize_width=512,    # 592
            crop_height=160,
            crop_width=512,
            min_depth=1.8,
            max_depth=80,
            interval_test=1
        )
    return params

def safe_log(x):
    x = torch.clamp(x, 1e-6, 1e6)
    return torch.log(x)

def safe_log10(x):
    x = torch.clamp(x, 1e-6, 1e6)
    return torch.log10(x)

def compute_errors(gt, pred):
    """
    Args:
            gt, pred [batch, h, w]
    """
    batch_size = pred.shape[0]
    mask = gt > 0
    gt = gt[mask]
    pred = pred[mask]
    thresh = torch.max(gt / pred, pred / gt)
    a1 = (thresh < 1.25).float().mean() * batch_size
    a2 = (thresh < 1.25 ** 2).float().mean() * batch_size
    a3 = (thresh < 1.25 ** 3).float().mean() * batch_size

    rmse = ((gt - pred) ** 2).mean().sqrt() * batch_size
    rmse_log = ((safe_log(gt) - safe_log(pred))
                ** 2).mean().sqrt() * batch_size
    log10 = (safe_log10(gt) - safe_log10(pred)).abs().mean() * batch_size
    abs_rel = ((gt - pred).abs() / gt).mean() * batch_size
    sq_rel = ((gt - pred)**2 / gt).mean() * batch_size
    measures = {'a1': a1, 'a2': a2, 'a3': a3, 'rmse': rmse,
                'rmse_log': rmse_log, 'log10': log10, 'abs_rel': abs_rel, 'sq_rel': sq_rel}
    return measures


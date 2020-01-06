##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: chenyuru
## This source code is licensed under the MIT-style license
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import os
import sys
#sys.path.append(os.path.dirname(__file__))
import numpy as np
try:
    from .transforms import *
    from .dataloader import MyDataloader
except:
    from transforms import *
    from dataloader import MyDataloader
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

iheight, iwidth = 375, 1242 # raw image size

def make_dataset(root, txt):
    with open(txt, 'r') as f:
        List = []
        for line in f:
            left, right = line.strip('\n').split(' ')
            List.append(os.path.join(root, left))
            #List.append(right)
    return List

class KITTIDataset(MyDataloader):
    def __init__(self, root_image, root_depth, 
                 image_txt, depth_txt, mode='train',
                 min_depth=None, max_depth=None,
                 flip=False, rotate=False, scale=False, jitter=False, crop=False,
                 make=make_dataset):
        super(KITTIDataset, self).__init__(root_image, root_depth, image_txt, depth_txt, mode, min_depth, max_depth, make)
        self.input_size = (160, 640)
        self.flip = flip
        self.rotate = rotate
        self.scale = scale
        self.jitter = jitter
        self.crop = crop

    def train_transform(self, rgb, depth):
        t = [Crop(130, 10, 240, 1200), 
             Resize(180 / 240)] # this is for computational efficiency, since rotation can be slow
        if self.rotate:
            angle = np.random.uniform(-5.0, 5.0) # random rotation degrees
            t.append(Rotate(angle))
        if self.scale:
            s = np.random.uniform(1.0, 1.5) # random scaling
            depth = depth / s
            t.append(Resize(s))
        if self.crop: # random crop
            slide = np.random.uniform(0.0, 1.0)
            t.append(RandomCrop(self.input_size, slide))
        else: # center crop
            t.append(CenterCrop(self.input_size))
        if self.flip:
            do_flip = np.random.uniform(0.0, 1.0) < 0.5 # random horizontal flip
            t.append(HorizontalFlip(do_flip))
        # perform 1st step of data augmentation
        transform = Compose(t)
        rgb_np = transform(rgb)
        if self.jitter:
            color_jitter = ColorJitter(0.4, 0.4, 0.4)
            rgb_np = color_jitter(rgb_np) # random color jittering
        rgb_np = np.asfarray(rgb_np, dtype='float') / 255
        # Scipy affine_transform produced RuntimeError when the depth map was
        # given as a 'numpy.ndarray'
        depth_np = np.asfarray(depth, dtype='float32')
        depth_np = transform(depth_np)
        return rgb_np, depth_np

    def val_transform(self, rgb, depth):
        transform = Compose([Crop(130, 10, 240, 1200),
                             Resize(180 / 240),
                             CenterCrop(self.input_size),
                            ])
        rgb_np = transform(rgb)
        rgb_np = np.asfarray(rgb_np, dtype='float') / 255
        depth_np = np.asfarray(depth, dtype='float32')
        depth_np = transform(depth_np)
        return rgb_np, depth_np


if __name__ == '__main__':
    HOME = os.environ['HOME']
    rgbdir = HOME + '/myDataset/KITTI/raw_data_KITTI/'
    depdir = HOME + '/myDataset/KITTI/datasets_KITTI/'
    trainrgb = '../datasets/kitti_path/eigen_train_files.txt'
    traindep = '../datasets/kitti_path/eigen_train_depth_files.txt'
    valrgb = '../datasets/kitti_path/eigen_test_files.txt'
    valdep = '../datasets/kitti_path/eigen_test_depth_files.txt'

    kwargs = {'min_depth': 1.8, 'max_depth': 80.0,
              'flip': True, 'scale': True,
              'rotate': True, 'jitter': True, 'crop': True}

    train_dataset = KITTIDataset(rgbdir, depdir, trainrgb, traindep, mode='train', **kwargs)
    val_dataset = KITTIDataset(rgbdir, depdir, valrgb, valdep, mode='val', **kwargs)
    trainloader = DataLoader(train_dataset, 10,
                            shuffle=True, num_workers=4, 
                            pin_memory=True, drop_last=False)
    valloader = DataLoader(val_dataset, 10,
                            shuffle=True, num_workers=4, 
                            pin_memory=True, drop_last=False)
    image, label = train_dataset[400]
    image_npy = image.numpy().transpose(1, 2, 0)
    label_npy = label.numpy().squeeze()

    #trainloader = iter(trainloader)
    #image, label = next(trainloader)
    print(image.shape, label.shape)
    print(label.max())
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(image_npy)
    plt.subplot(1, 2, 2)
    plt.imshow(label_npy, cmap='plasma')
    plt.show()
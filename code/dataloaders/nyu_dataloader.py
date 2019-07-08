##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: chenyuru
## This source code is licensed under the MIT-style license
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import os
import sys
#sys.path.append(os.path.dirname(__file__))
import numpy as np
from .transforms import *
from .dataloader import MyDataloader
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch

iheight, iwidth = 480, 640 # raw image size

def make_dataset(root, txt):
    with open(txt, 'r') as f:
        List = []
        for line in f:
            List.append(os.path.join(root, line.strip('\n')))
    return List

class NYUDataset(MyDataloader):
    def __init__(self, root_image, root_depth, 
                 image_txt, depth_txt, mode='train',
                 min_depth=None, max_depth=None,
                 flip=False, rotate=False, scale=False, jitter=False, crop=False,
                 make=make_dataset):
        super(NYUDataset, self).__init__(root_image, root_depth, image_txt, depth_txt, mode, min_depth, max_depth, make)
        self.input_size = (224, 304)
        self.flip = flip
        self.rotate = rotate
        self.scale = scale
        self.jitter = jitter
        self.crop = crop

    def train_transform(self, rgb, depth):
        t = [Resize(240.0 / iheight)] # this is for computational efficiency, since rotation can be slow
        if self.rotate:
            angle = np.random.uniform(-5.0, 5.0) # random rotation degrees
            t.append(Rotate(angle))
        if self.scale:
            s = np.random.uniform(1.0, 1.5) # random scaling
            depth = depth / s
            t.append(Resize(s))
        if self.crop:
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
        depth_np = transform(depth)
        return rgb_np, depth_np

    def val_transform(self, rgb, depth):
        transform = Compose([Resize(240.0 / iheight),
                             CenterCrop(self.input_size),
                            ])
        rgb_np = transform(rgb)
        rgb_np = np.asfarray(rgb_np, dtype='float') / 255
        depth_np = transform(depth)
        return rgb_np, depth_np

if __name__ == '__main__':
    HOME = os.environ['HOME']
    rgbdir = HOME + '/myDataset/NYU_v2/'
    depdir = HOME + '/myDataset/NYU_v2/'
    trainrgb = '../datasets/nyu_path/train_rgb_12k.txt'
    traindep = '../datasets/nyu_path/train_depth_12k.txt'
    valrgb = '../datasets/nyu_path/valid_rgb.txt'
    valdep = '../datasets/nyu_path/valid_depth.txt'

    kwargs = {'min_depth': 0.72, 'max_depth': 10.0,
              'flip': True, 'scale': True,
              'rotate': True, 'jitter': True, 'crop': True}

    train_dataset = NYUDataset(rgbdir, depdir, trainrgb, traindep, mode='train', **kwargs)
    val_dataset = NYUDataset(rgbdir, depdir, valrgb, valdep, mode='val', **kwargs)
    trainloader = DataLoader(train_dataset, 20,
                            shuffle=True, num_workers=4, 
                            pin_memory=True, drop_last=False)
    valloader = DataLoader(val_dataset, 20,
                            shuffle=True, num_workers=4, 
                            pin_memory=True, drop_last=False)
    image, label = train_dataset[2000]
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
    plt.imshow(label_npy, cmap='jet')
    plt.show()
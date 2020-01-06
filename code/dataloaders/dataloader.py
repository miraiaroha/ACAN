##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: chenyuru
## This source code is licensed under the MIT-style license
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import os
import sys
#sys.path.append(os.path.dirname(__file__))
import numpy as np
import torch.utils.data as data
try:
    from .transforms import ToTensor
except:
    from transforms import ToTensor
from PIL import Image

def img_loader(path, is_rgb=True):
    if is_rgb:
        img = np.array(Image.open(path).convert('RGB'))
    else:
        img = np.array(Image.open(path))
        img = np.float32(img) / 256
    return img

# def rgb2grayscale(rgb):
#     return rgb[:,:,0] * 0.2989 + rgb[:,:,1] * 0.587 + rgb[:,:,2] * 0.114

to_tensor = ToTensor()

class MyDataloader(data.Dataset):

    def __init__(self, root_image, root_depth, 
                       image_txt, depth_txt, 
                       mode, min_depth, max_depth,
                       make, loader=img_loader):
        self.min_depth = min_depth
        self.max_depth = max_depth
        root_image = os.path.expanduser(root_image)
        root_depth = os.path.expanduser(root_depth)
        self.images = make(root_image, image_txt)
        self.depths = make(root_depth, depth_txt)
        assert len(self.images)>0, "Found 0 images in folder of: " + root_image + "\n"
        print("Found {} images in {} folder.".format(len(self.images), mode))
        if mode == 'train':
            self.transform = self.train_transform
        elif mode in ['val', 'test']:
            self.transform = self.val_transform
        else:
            raise (RuntimeError("Invalid dataset type: " + mode + "\n"
                                "Supported dataset types are: train, eval, test"))
        self.loader = loader

    def train_transform(self, rgb, depth):
        raise (RuntimeError("train_transform() is not implemented. "))

    def val_transform(self, rgb, depth):
        raise (RuntimeError("val_transform() is not implemented."))

    def __getraw__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (rgb, depth) the raw data.
        """
        rgb = self.loader(self.images[index], True)
        depth = self.loader(self.depths[index], False)
        return rgb, depth

    def __getitem__(self, index):
        rgb, depth = self.__getraw__(index)
        if self.transform is not None:
            rgb_np, depth_np = self.transform(rgb, depth)
        else:
            raise(RuntimeError("transform not defined"))

        depth_np = np.clip(depth_np, a_min=0, a_max=self.max_depth)
        depth_np = np.expand_dims(depth_np, -1)
        input_tensor = to_tensor(rgb_np)
        depth_tensor = to_tensor(depth_np)
        return input_tensor, depth_tensor

    def __len__(self):
        return len(self.images)


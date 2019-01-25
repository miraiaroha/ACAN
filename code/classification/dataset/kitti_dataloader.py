import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image
import os

train_transforms = transforms.Compose([transforms.ToPILImage(),
                                       transforms.ColorJitter(
                                           brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                                       transforms.ToTensor()
                                       ])
valid_transforms = transforms.Compose([transforms.ToTensor()])
depth_transforms = transforms.Compose([transforms.ToTensor()])


def RandomFlipLeftRight(image, depth, mode):
    if mode == 'train':
        rand_var = np.random.random()
        image = image[:, ::-1, :] if rand_var > 0.5 else image
        depth = depth[:, ::-1] if rand_var > 0.5 else depth
    return image, depth


def RandomRotation(image, depth, degree, mode):
    if mode == 'train':
        image = Image.fromarray(image)
        depth = Image.fromarray(depth)
        random_angel = np.random.randint(-degree, degree)
        image = image.rotate(random_angel, resample=Image.BICUBIC)
        depth = depth.rotate(random_angel, resample=Image.NEAREST)
        image = np.array(image)
        depth = np.array(depth)
        return image, depth
    else:
        return image, depth


def ResizeAndCrop(image, depth, r_height, r_width, c_height, c_width, mode):
    image = Image.fromarray(image)
    depth = Image.fromarray(depth)
    image = image.resize((r_width, r_height), resample=Image.BICUBIC)
    depth = depth.resize((r_width, r_height), resample=Image.NEAREST)
    if mode == 'train':  # random crop left-right
        # random scale
        #s = 1 + 0.2 * np.random.rand()
        s = 1
        s_width, s_height = int(r_width * s), int(r_height * s)
        #image = image.resize((s_width, s_height), resample=Image.BICUBIC)
        #depth = depth.resize((s_width, s_height), resample=Image.NEAREST)
        # random crop
        left = np.random.randint(0, s_width - c_width + 1)
        up = s_height - c_height
    else:  # center crop
        left = (r_width - c_width) // 2
        up = r_height - c_height
    region = (left, up, left + c_width, up + c_height)
    image = image.crop(region)
    depth = depth.crop(region)
    image = np.array(image)
    depth = np.array(depth)
    return image, depth


class KittiFromTxt(Dataset):
    def __init__(self, home_image, home_depth, image_txt, depth_txt,
                 original_height, original_width,
                 resize_height, resize_width,
                 crop_height, crop_width,
                 min_depth, max_depth,
                 mode='train',
                 ):
        super(KittiFromTxt, self).__init__()
        with open(image_txt, 'r') as f:
            images_list = []
            for line in f:
                left, right = line.split()
                images_list.append(left)
                # if mode=='train':
                #    images_list.append(right)
        self.images_list = images_list
        with open(depth_txt, 'r') as f:
            depths_list = []
            for line in f:
                left, right = line.split()
                depths_list.append(left)
                # if mode=='train':
                #    depths_list.append(right)
        self.depths_list = depths_list
        self.mode = mode
        self.home_image = home_image
        self.home_depth = home_depth
        self.o_height, self.o_width = original_height, original_width
        self.r_height, self.r_width = resize_height, resize_width
        self.c_height, self.c_width = crop_height, crop_width
        self.min_depth = torch.tensor(min_depth)
        self.max_depth = torch.tensor(max_depth)
        self.crop = [60, -2, 20, -20]

    def load_img(self, path, is_image=True):
        if is_image:
            img = np.array(Image.open(path).convert('RGB'))
        else:
            img = np.array(Image.open(path))
            img = np.float32(img) / 256
        return img

    def __getitem__(self, index):
        image_path = os.path.join(self.home_image, self.images_list[index])
        depth_path = os.path.join(self.home_depth, self.depths_list[index])
        image = self.load_img(image_path, is_image=True)
        depth = self.load_img(depth_path, is_image=False)
        image, depth = RandomFlipLeftRight(image, depth, self.mode)
        image, depth = ResizeAndCrop(
            image, depth, self.r_height, self.r_width, self.c_height, self.c_width, self.mode)
        #image, depth = RandomRotation(image, depth, 5, self.mode)
        depth = np.clip(depth, a_min=0, a_max=self.max_depth.item())
        depth = np.expand_dims(depth, -1)
        if self.mode == 'train':
            image = train_transforms(image)
        else:
            image = valid_transforms(image)
        depth = depth_transforms(depth)
        return image, depth

    def __len__(self):
        return len(self.images_list)

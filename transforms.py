import random
import numpy as np

import torch
import torchvision.transforms as T
from torchvision.transforms import functional as F


def pad_if_smaller(img, size, fill=0):
    min_size = min(img.size)
    if min_size < size:
        ow, oh = img.size
        padh = size - oh if oh < size else 0
        padw = size - ow if ow < size else 0
        img = F.pad(img, (0, 0, padw, padh), fill=fill)
    return img


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

class NumpyToTensor:
    def __call__(self, image, mask):
        image = T.ToTensor()(image)
        mask = torch.tensor(mask, dtype=torch.int64)
        return image, mask


class RandomResize:
    def __init__(self, min_size, max_size=None):
        self.min_size = min_size
        if max_size is None:
            max_size = min_size
        self.max_size = max_size

    def __call__(self, image, target):
        size = random.randint(self.min_size, self.max_size)
        image = F.resize(image, size)
        target = F.resize(target, size, interpolation=T.InterpolationMode.NEAREST)
        return image, target


class RandomHorizontalFlip:
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, image, target):
        if random.random() < self.flip_prob:
            image = F.hflip(image)
            target = F.hflip(target)
        return image, target


class RandomCrop:
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target):
        image = pad_if_smaller(image, self.size)
        target = pad_if_smaller(target, self.size, fill=255)
        crop_params = T.RandomCrop.get_params(image, (self.size, self.size))
        image = F.crop(image, *crop_params)
        target = F.crop(target, *crop_params)
        return image, target

class RandomResizedCrop:
    def __init__(self, size, scale, ratio):
        self.size = size
        self.ratio = ratio
        self.scale = scale

    def __call__(self, image, target):
        crop_params = T.RandomResizedCrop.get_params(image, scale=self.scale, ratio=self.ratio)
        image = F.resized_crop(image, *crop_params, size=(self.size, self.size))
        target = F.resized_crop(
            target.unsqueeze(0), 
            *crop_params, 
            (self.size, self.size),
            interpolation=T.InterpolationMode.NEAREST
        )
        return image, target.squeeze(0)


class CenterCrop:
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target):
        image = F.center_crop(image, self.size)
        target = F.center_crop(target, self.size)
        return image, target


class PILToTensor:
    def __call__(self, image, target):
        image = F.pil_to_tensor(image)
        target = torch.as_tensor(np.array(target), dtype=torch.int64)
        return image, target


class ConvertImageDtype:
    def __init__(self, dtype):
        self.dtype = dtype

    def __call__(self, image, target):
        image = F.convert_image_dtype(image, self.dtype)
        return image, target

class Resize(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.size = size
  
    def forward(self, image, target):
        image = F.resize(image, self.size)
        target = F.resize(target.unsqueeze(0), self.size, interpolation=T.InterpolationMode.NEAREST)
        return image, target.squeeze(0)


class Normalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target):
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target

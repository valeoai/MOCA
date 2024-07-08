import os
import time
import math
import random

import numpy as np
import torch
from torch import nn
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import torchvision.datasets as datasets

from PIL import ImageFilter, ImageOps, Image


class GaussianBlur(object):
    """
    Apply Gaussian Blur to the PIL image.
    """
    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        do_it = random.random() <= self.prob
        if not do_it:
            return img

        return img.filter(
            ImageFilter.GaussianBlur(
                radius=random.uniform(self.radius_min, self.radius_max)
            )
        )


class Solarization(object):
    """
    Apply Solarization to the PIL image.
    """
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img


class DataAugmentationTwoViews(object):
    def __init__(self, input_size=224, crop_min=0.2):
        normalize = T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        # follow BYOL's augmentation recipe: https://arxiv.org/abs/2006.07733
        self.augmentation1 = T.Compose([
            T.RandomResizedCrop(input_size, scale=(crop_min, 1.)),
            T.RandomApply([
                T.ColorJitter(0.4, 0.4, 0.2, 0.1)  # not strengthened
            ], p=0.8),
            T.RandomGrayscale(p=0.2),
            GaussianBlur(1.0, 0.1, 2.0),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            normalize
        ])
        self.augmentation2 = T.Compose([
            T.RandomResizedCrop(input_size, scale=(crop_min, 1.)),
            T.RandomApply([
                T.ColorJitter(0.4, 0.4, 0.2, 0.1)  # not strengthened
            ], p=0.8),
            T.RandomGrayscale(p=0.2),
            GaussianBlur(0.1, 0.1, 2.0),
            Solarization(0.2),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            normalize
        ])

    def __call__(self, image):
        return [self.augmentation1(image), self.augmentation2(image)]
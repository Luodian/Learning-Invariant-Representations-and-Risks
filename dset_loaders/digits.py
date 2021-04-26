import os.path

import torch.utils.data as data
import torchvision
import torchvision.transforms as T
from PIL import Image
import numpy as np
from utils.utils_module import TwoCropsTransform


class DigitsParams(object):
    num_channels = 3
    image_size = 32
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    num_cls = 10
    target_transform = None


class DIGITS(data.Dataset):
    def __init__(self, root, num_cls, transform, data=None):
        self.root = root
        self.transform = transform
        self.classes = []
        for i in range(num_cls):
            self.classes.append(i)
        self.num_cls = num_cls
        self.images, self.labels = data['ids'], data['labels']
        self.transform = TwoCropsTransform(self.transform)

    def __getitem__(self, index, debug=False):
        image = self.images[index % len(self.images)]
        label = int(self.labels[index % len(self.images)])
        if debug:
            print(self.__class__.__name__)
            print("IMG Path: {}".format(id))
            print("Label: {}".format(label))
        img = Image.open(image).convert('RGB')
        if self.transform is None:
            return img, label
        elif self.TwoTransform:
            img_q, img_k = self.transform(img)
            return (img_q, label, img_k, label)
        else:
            img = self.transform(img)
            return (img, label)

    def __len__(self):
        return len(self.images)

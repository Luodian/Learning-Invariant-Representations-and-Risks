import os.path

import torch.utils.data as data
import torchvision
from PIL import Image
import numpy as np
import random
from utils.utils_module import TwoCropsTransform

class OfficehomeParams(object):
    num_channels = 3
    image_size = 256 # 384, 216
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    num_cls = 65
    target_transform = None
    
class OFFICEHOME(data.Dataset):
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
        rand_index = random.randint(0, len(self.images) - 1)
        image1 = self.images[rand_index % len(self.images)]
        label1 = int(self.labels[rand_index % len(self.images)])
        img = Image.open(image).convert('RGB')
        img1 = Image.open(image1).convert('RGB')
        img_q, img_k = self.transform(img)
        img1_q, img1_k = self.transform(img1)
        return {
            'sample_1_q':(img_q, label),
            'sample_1_k':(img_k, label),
            'sample_2_q':(img1_q, label1),
            'sample_2_k':(img1_k, label1),
        }

    def __len__(self):
        return len(self.images)

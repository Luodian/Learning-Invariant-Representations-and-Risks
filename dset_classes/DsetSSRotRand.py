import torch
import torch.utils.data
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
# Assumes that tensor is (nchannels, height, width)
def tensor_rot_90(x):
    return x.flip(2).transpose(1,2)
def tensor_rot_90_digit(x):
    return x.transpose(1,2)

def tensor_rot_180(x):
    return x.flip(2).flip(1)
def tensor_rot_180_digit(x):
    return x.flip(2)

def tensor_rot_270(x):
    return x.transpose(1,2).flip(2)

# For the digit datasets, instead of regular rotations
# which cannot be identified for 6 and 9, often 0 and 1,
# do a horizental flip. Natural scenes, on the other hand,
# are invariant to horizontal flips.

class DsetSSRotRand(torch.utils.data.Dataset):
    # Make sure that your dataset only returns one element!
    def __init__(self, dset, digit=False, img_size=224):
        self.dset = dset
        self.digit = digit
        self.transform = transforms.Compose([
        transforms.Resize([img_size, img_size]),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    def __getitem__(self, index):
        img = self.dset.images[index]
        image = Image.open(img).convert('RGB')
        image = self.transform(image)
        label = np.random.randint(4)
        if label == 1:
            if self.digit:
                image = tensor_rot_90_digit(image)
            else:
                image = tensor_rot_90(image)
        elif label == 2:
            if self.digit:
                image = tensor_rot_180_digit(image)
            else:
                image = tensor_rot_180(image)
        elif label == 3:
            image = tensor_rot_270(image)
        return image, label

    def __len__(self):
        return len(self.dset)

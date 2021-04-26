import torch
import torch.utils.data
import numpy as np
import torchvision.transforms as transforms
from PIL import Image

class DsetSSQuadRand(torch.utils.data.Dataset):
    # Make sure that your dataset only returns one element!
    def __init__(self, dset, quad_p=2, img_size=224):
        self.dset = dset
        self.quad_p = quad_p
        self.transform = transforms.Compose([
        transforms.Resize([img_size, img_size]),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    def __getitem__(self, index):
        img = self.dset.images[index]
        image = Image.open(img).convert('RGB')
        image = self.transform(image)
        label = np.random.randint(self.quad_p * self.quad_p)

        horstr = image.size(1) // self.quad_p 
        verstr = image.size(2) // self.quad_p 
        horlab = label // self.quad_p 
        verlab = label % self.quad_p 

        image = image[:, horlab*horstr:(horlab+1)*horstr, verlab*verstr:(verlab+1)*verstr,].unsqueeze(dim=0)
        image = torch.nn.functional.interpolate(image, scale_factor=self.quad_p).squeeze()
        
        return image, label

    def __len__(self):
        return len(self.dset)

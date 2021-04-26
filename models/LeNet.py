import torch
import torch.nn as nn
import torchvision.models as models
from .models import register_model
import torch.nn.functional as F
import numpy as np
import math
# from utils.utils_module import *
# from .util import Decoder, Encoder         

@register_model('lenet')
class LeNet(nn.Module):

    def __init__(self, num_cls=10, image_size=32, feature_dim=1024, vib=False):
        super(LeNet, self).__init__()
        self.image_size = image_size
        self.num_cls = num_cls
        self.feature_dim = feature_dim
        self.vib = vib
                    
        self.extractor = nn.Sequential(
            nn.Conv2d(3, 20, kernel_size=5, padding=2),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(20, 50, kernel_size=5, padding=2),
            nn.Dropout2d(p=0.5),
            nn.MaxPool2d(2),
            nn.ReLU(),
        )
        self.fc_params = nn.Sequential(
            nn.Linear(50 * 8 * 8, self.feature_dim if not vib else self.feature_dim * 2), 
            nn.ReLU(), 
            nn.Dropout(p=0.5)
        )
        self.classifer = nn.Linear(self.feature_dim, num_cls, bias=True)
        self.flattening = Flattening()
        
        # encoding components
            
        if self.num_cls == 1:
            self.sigmoid = nn.Sigmoid()
            
        if self.vib:
            self.reparametering = Reparametering(feature_dim)
            self.feature_layers = [self.extractor]
        else:
            self.feature_layers = [self.extractor, self.fc_params]
        
    @classmethod
    def create(cls, dicts):
        num_cls = dicts['num_cls'] if dicts.get('num_cls', False) else 10
        image_size = dicts['image_size'] if dicts.get('image_size', False) else 32
        feature_dim = dicts['adapted_dim'] if dicts.get('adapted_dim', False) else 1024
        vib = dicts['vib'] if dicts.get('vib', False) else False
        return cls(num_cls=num_cls, image_size=image_size, feature_dim=feature_dim, vib=vib)

    def forward(self, x, temp=1, dropout=False):
        x = self.extractor(x)
        x = self.flattening(x)
        encodes = self.fc_params(x)
        if not self.vib:
            logits = self.classifer(encodes)
            logits = self.sigmoid(logits) if self.num_cls == 1 else logits
            return {'features':[x], 'adapted_layer': encodes, 'output_logits': logits}
        else:
            mu, std, z = self.reparametering(encodes)
            logits = self.classifer(z) if self.training else self.classifer(mu)
            logits = self.sigmoid(logits) if self.num_cls == 1 else logits
            return {'mu': mu, 'std': std, 'adapted_layer': x, 'reparametered_ft': z, 'output_logits': logits}
        
    def predict(self, x, reverse=False, eta=0.1, temp=0.05):
        if reverse:
            x = grad_reverse(x, eta)
        x = F.normalize(x)
        logits = self.classifer(x)
        return {'output_logits': logits}
    
    def get_parameters(self):
        parameter_list = [
            {"params": self.extractor.parameters(), 'lr_mult':1.}, 
            {"params": self.fc_params.parameters(), 'lr_mult':1.}, 
            {"params": self.classifer.parameters(), 'lr_mult':10.}
        ]
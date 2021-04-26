import math
import torch
from torch import nn
from torchvision.models.resnet import conv3x3
import torch.nn.functional as F
from numbers import Number
from torch.autograd import Variable
from torchvision import models
import torch.nn.init as init
# from utils.utils_module import *
from .models import register_model
from utils.utils_module import *

@register_model('alexnet')
class AlexNet(nn.Module):
    def __init__(self, 
                 bottleneck_dim=256, 
                 class_num=1000, 
                 frozen=[]
                ):
        super(AlexNet, self).__init__()
        model = models.alexnet(pretrained=True)
        self.ordered_module_names = []
        self.frozen=frozen
        print('Frozen Layer: ', self.frozen)
        self.num_cls = class_num
        self.encoder = model.features
        self.avgpool = model.avgpool
        self.dropout1 = nn.Dropout(p=0.5)
        self.decoder = nn.Sequential(
            nn.Linear(in_features=9216, out_features=4096, bias=True),
            nn.ReLU(inplace=True)
        )
        self.dropout2 = nn.Dropout(p=0.5)
        self.bottleneck = nn.Linear(in_features=4096, out_features=bottleneck_dim, bias=True)
        self.fc = nn.Linear(in_features=bottleneck_dim, out_features=class_num, bias=True)
        
        self.decoder.apply(init_weights)
        self.bottleneck.apply(init_weights)
        self.fc.apply(init_weights)
        
        self.__in_features = bottleneck_dim
        self.flattening = Flattening()
            
        if len(self.frozen) > 0:
            for name in self.frozen:
                if name in self.ordered_module_names:
                    for params in self._modules[name].parameters():
                        params.requires_grad = False

        self.ordered_module_names += [
            'encoder', 'avgpool', 'flattening', 'decoder'
        ]
            

    def forward(self, x, temp=1, dropout=True, cosine=False):
        for name in self.ordered_module_names:
            module = self._modules[name]
            x = module(x)
            x = x.detach() if name in self.frozen else x
            
        embedding_coding = x 
        drop_x = self.dropout2(embedding_coding)
        encodes = torch.nn.functional.relu(self.bottleneck(drop_x), inplace=False)
        drop_x = self.dropout2(encodes)
        if cosine:
    #       cosine classifer
            normed_x = F.normalize(drop_x, p=2, dim=1)
            logits = self.fc(normed_x) / temp
        else:
            logits = self.fc(drop_x) / temp
        return {
            'features':embedding_coding, 
            'adapted_layer': encodes, 
            'output_logits': logits
        }
    
    @classmethod
    def create(cls, dicts):
        class_num = dicts['num_cls'] if dicts.get('num_cls', False) else 10
        bottleneck_dim = dicts['adapted_dim'] if dicts.get('adapted_dim', False) else 256
        frozen = dicts['frozen'] if dicts.get('frozen', False) else []
        return cls(
            class_num=class_num, 
            bottleneck_dim=bottleneck_dim, 
            frozen=frozen
        )
    
    def get_classifer_in_features(self):
        return self.bottleneck.in_features

    def output_num(self):
        return self.__in_features

    def get_parameters(self):
        parameter_list = [
#             {"params": self.bottleneck.parameters(), 'lr_mult': 10},
#             {"params": self.fc.parameters(), 'lr_mult': 10}
        ]
        for name in self.ordered_module_names:
            if name not in self.frozen and len(list(self._modules[name].parameters())) > 0:
                parameter_list += [{"params": self._modules[name].parameters(), 'lr_mult': 1}]
        return parameter_list, [
            {"params": self.bottleneck.parameters(), 'lr_mult': 10},
            {"params": self.fc.parameters(), 'lr_mult': 10}
        ]
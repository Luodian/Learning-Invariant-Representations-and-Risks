# Based on the ResNet implementation in torchvision
# https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
import math
import torch
from torch import nn
from torchvision.models.resnet import conv3x3
import torch.nn.functional as F
from torchvision import models
from numbers import Number
from torch.autograd import Variable
import torch.nn.init as init
# from utils.utils import *
from .models import register_model
from utils.utils_module import *

class GlobalPooling(nn.Module):
    def __init__(self):
        super(GlobalPooling, self).__init__()
        self.ada_avg_pool = nn.AdaptiveAvgPool2d([1, 1])
        
    def forward(self, x):
        return self.ada_avg_pool(x).view(x.shape[0], -1)
    
class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.downsample = downsample
        self.stride = stride
        
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = conv3x3(inplanes, planes, stride)
        
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)

    def forward(self, x):
        residual = x 
        residual = self.bn1(residual)
        residual = self.relu1(residual)
        residual = self.conv1(residual)

        residual = self.bn2(residual)
        residual = self.relu2(residual)
        residual = self.conv2(residual)

        if self.downsample is not None:
            x = self.downsample(x)
        return x + residual

class Downsample(nn.Module):
    def __init__(self, nIn, nOut, stride):
        super(Downsample, self).__init__()
        self.avg = nn.AvgPool2d(stride)
        assert nOut % nIn == 0
        self.expand_ratio = nOut // nIn

    def forward(self, x):
        x = self.avg(x)
        return torch.cat([x] + [x.mul(0)] * (self.expand_ratio - 1), 1)
    
resnet_dict = {
    "ResNet34": models.resnet34,
    "ResNet50": models.resnet50, 
    "ResNet101": models.resnet101, 
    "ResNet152": models.resnet152
}

@register_model('resnet34')
class ResNet_34Fc(nn.Module):
    def __init__(self, 
                 bottleneck_dim=256, 
                 class_num=1000, 
                 frozen=[]
                ):
        super(ResNet_34Fc, self).__init__()
        model_resnet = resnet_dict['ResNet34'](pretrained=True)
        self.ordered_module_names = []
        self.frozen=frozen
        print('Frozen Layer: ', self.frozen)
        self.conv1 = model_resnet.conv1
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
        self.layer4 = model_resnet.layer4
        self.avgpool = model_resnet.avgpool
        self.num_cls = class_num
        
        self.bottleneck = nn.Linear(
            model_resnet.fc.in_features, 
            bottleneck_dim
        )
        self.fc = nn.Linear(bottleneck_dim, class_num)
        self.bottleneck.apply(init_weights)

        self.fc.apply(init_weights)
        self.__in_features = bottleneck_dim

        self.flattening = Flattening()
        self.feature_layers = [self.conv1, self.bn1, self.layer1, self.layer2, self.layer3, self.layer4, self.bottleneck]
            
        if len(self.frozen) > 0:
            for name in self.frozen:
                if name in self.ordered_module_names:
                    for params in self._modules[name].parameters():
                        params.requires_grad = False

        self.ordered_module_names += [
            'conv1', 'bn1', 'relu', 'maxpool',
            'layer1', 'layer2', 'layer3', 'layer4', 'avgpool'
        ]

    def forward(self, x, temp=1, dropout=True, cosine=False, reverse=False):
        for name in self.ordered_module_names:
            module = self._modules[name]
            x = module(x)
            x = x.detach() if name in self.frozen else x

        embedding_coding = self.flattening(x)
        rev_rep = grad_reverse(embedding_coding, 1.0) if reverse else embedding_coding
        drop_x = F.dropout(rev_rep, training=self.training, p=0.5) if dropout else rev_rep
        encodes = torch.nn.functional.relu(self.bottleneck(drop_x), inplace=False)
        drop_x = F.dropout(encodes, training=self.training, p=0.5) if dropout else encodes
        if cosine:
            normed_x = F.normalize(drop_x, p=2, dim=1)
            logits = self.fc(normed_x) / temp
        else:
            logits = self.fc(drop_x) / temp
        return {
            'features': embedding_coding,
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

    def output_num(self):
        return self.__in_features
    
    def get_classifer_in_features(self):
        return self.bottleneck.in_features

    def get_parameters(self):
        parameter_list = [
            {"params": self.bottleneck.parameters(), 'lr_mult': 10},
#             {"params": self.fc.parameters(), 'lr_mult': 10}
        ]
        for name in self.ordered_module_names:
            if name not in self.frozen and len(list(self._modules[name].parameters())) > 0:
                parameter_list += [{"params": self._modules[name].parameters(), 'lr_mult': 1}]
        return parameter_list, [
#             {"params": self.bottleneck.parameters(), 'lr_mult': 10}, 
            {"params": self.fc.parameters(), 'lr_mult': 10}
        ]
    
@register_model('resnet50')
class ResNet_50Fc(nn.Module):
    def __init__(self, 
                 use_bottleneck=True, 
                 bottleneck_dim=256, 
                 class_num=1000, 
                 vib=False, 
                 IN=False, 
                 frozen=[]
                ):
        super(ResNet_50Fc, self).__init__()
        model_resnet = resnet_dict['ResNet50'](pretrained=True)
        self.ifin = IN
        self.vib = vib
        self.ordered_module_names = []
        self.frozen=frozen
        print('Frozen Layer: ', self.frozen)
        if IN:
            print(IN)
            self.IN = nn.InstanceNorm2d(3, affine=True)
            self.ordered_module_names += ['in']
        self.conv1 = model_resnet.conv1
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
        self.layer4 = model_resnet.layer4
        self.avgpool = model_resnet.avgpool
        
        self.use_bottleneck = use_bottleneck
        self.num_cls = class_num
        
        self.bottleneck = nn.Linear(
            model_resnet.fc.in_features, 
            bottleneck_dim if not vib else bottleneck_dim * 2
        )
        self.fc = nn.Linear(bottleneck_dim, class_num)
        self.bottleneck.apply(init_weights)
        self.fc.apply(init_weights)
        self.__in_features = bottleneck_dim

        self.flattening = Flattening()
        if self.vib:
            self.reparametering = Reparametering(bottleneck_dim)
            self.feature_layers = [self.conv1, self.bn1, self.layer1, self.layer2, self.layer3, self.layer4]
        else:
            self.feature_layers = [self.conv1, self.bn1, self.layer1, self.layer2, self.layer3, self.layer4, self.bottleneck]
            
        if self.num_cls == 1:
            self.sigmoid = nn.Sigmoid()
            
        if len(self.frozen) > 0:
            for name in self.frozen:
                if name in self.ordered_module_names:
                    for params in self._modules[name].parameters():
                        params.requires_grad = False

        self.ordered_module_names += [
            'conv1', 'bn1', 'relu', 'maxpool',
            'layer1', 'layer2', 'layer3', 'layer4', 'avgpool'
        ]
            

    def forward(self, x, temp=0.05, dropout=False):
        if self.ifin:
            x = self.IN(x)
            
        for name in self.ordered_module_names:
            module = self._modules[name]
            x = module(x)
            x = x.detach() if name in self.frozen else x
            
        embedding_coding = self.flattening(x)
        if not self.vib:
            drop_x = F.dropout(embedding_coding, training=self.training, p=0.5) if dropout else embedding_coding
            encodes = self.bottleneck(drop_x)
            drop_x = F.dropout(encodes, training=self.training, p=0.5) if dropout else encodes
            logits = self.fc(drop_x) if self.num_cls > 1 else self.sigmoid(self.fc(drop_x))
            return {'features':embedding_coding, 'adapted_layer': encodes, 'output_logits': logits}
        else:
            mu, std, z = self.reparametering(embedding_coding)
            logits = self.fc(z) if self.training else self.fc(mu)
            logits = self.sigmoid(logits) if self.num_cls == 1 else logits
            return {'mu': mu, 'std': std, 'adapted_layer': embedding_coding, 'reparametered_ft': z, 'output_logits': logits}
    
    @classmethod
    def create(cls, dicts):
        class_num = dicts['num_cls'] if dicts.get('num_cls', False) else 10
        use_bottleneck = dicts['use_bottleneck'] if dicts.get('use_bottleneck', False) else True
        bottleneck_dim = dicts['adapted_dim'] if dicts.get('adapted_dim', False) else 256
        IN = dicts['IN'] if dicts.get('IN', False) else False
        vib = dicts['vib'] if dicts.get('vib', False) else False
        frozen = dicts['frozen'] if dicts.get('frozen', False) else []
        return cls(
            class_num=class_num, 
            use_bottleneck=use_bottleneck, 
            bottleneck_dim=bottleneck_dim, 
            vib=vib, 
            IN=IN, 
            frozen=frozen
        )

    def output_num(self):
        return self.__in_features

    def get_parameters(self):
        parameter_list = [
            {"params": self.bottleneck.parameters(), 'lr_mult': 10},
            {"params": self.fc.parameters(), 'lr_mult': 10}
        ]
        for name in self.ordered_module_names:
            if name not in self.frozen and len(list(self._modules[name].parameters())) > 0:
                parameter_list += [{"params": self._modules[name].parameters(), 'lr_mult': 1}]
        if self.ifin:
            parameter_list += [{"params": self.IN.parameters(), 'lr_mult': 1}]
            
        return parameter_list
    
    
@register_model('resnet101')
class ResNet_101Fc(nn.Module):
    def __init__(self, 
                 use_bottleneck=True, 
                 bottleneck_dim=256, 
                 class_num=1000, 
                 vib=False, 
                 IN=False, 
                 frozen=[]
                ):
        super(ResNet_101Fc, self).__init__()
        model_resnet = resnet_dict['ResNet101'](pretrained=True)
        self.ifin = IN
        self.vib = vib
        self.ordered_module_names = []
        self.frozen=frozen
        print('Frozen Layer: ', self.frozen)
        if IN:
            print(IN)
            self.IN = nn.InstanceNorm2d(3, affine=True)
            self.ordered_module_names += ['in']
        self.conv1 = model_resnet.conv1
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
        self.layer4 = model_resnet.layer4
        self.avgpool = model_resnet.avgpool
        
        self.use_bottleneck = use_bottleneck
        self.num_cls = class_num
        
        self.bottleneck = nn.Linear(
            model_resnet.fc.in_features, 
            bottleneck_dim if not vib else bottleneck_dim * 2
        )
        self.fc = nn.Linear(bottleneck_dim, class_num)
        self.bottleneck.apply(init_weights)

        self.fc.apply(init_weights)
        self.__in_features = bottleneck_dim

        self.flattening = Flattening()
        if self.vib:
            self.reparametering = Reparametering(bottleneck_dim)
            self.feature_layers = [self.conv1, self.bn1, self.layer1, self.layer2, self.layer3, self.layer4]
        else:
            self.feature_layers = [self.conv1, self.bn1, self.layer1, self.layer2, self.layer3, self.layer4, self.bottleneck]
            
        if self.num_cls == 1:
            self.sigmoid = nn.Sigmoid()
            
        if len(self.frozen) > 0:
            for name in self.frozen:
                if name in self.ordered_module_names:
                    for params in self._modules[name].parameters():
                        params.requires_grad = False

        self.ordered_module_names += [
            'conv1', 'bn1', 'relu', 'maxpool',
            'layer1', 'layer2', 'layer3', 'layer4', 'avgpool'
        ]
            

    def forward(self, x, temp=0.05, dropout=False):
        if self.ifin:
            x = self.IN(x)
            
        for name in self.ordered_module_names:
            module = self._modules[name]
            x = module(x)
            x = x.detach() if name in self.frozen else x
            
        embedding_coding = self.flattening(x)
        if not self.vib:
            drop_x = F.dropout(embedding_coding, training=self.training, p=0.5) if dropout else embedding_coding
            encodes = self.bottleneck(drop_x)
            drop_x = F.dropout(encodes, training=self.training, p=0.5) if dropout else encodes
            # cosine classifer
#             normed_x = F.normalize(drop_x, p=2, dim=1)
            # normalize the prototypes
#             with torch.no_grad():
#                 w = self.fc.weight.data.clone()
#                 w = nn.functional.normalize(w, dim=1, p=2)
#                 self.fc.weight.copy_(w)
#             logits = self.fc(normed_x) if self.num_cls > 1 else self.sigmoid(self.fc(normed_x))
            logits = self.fc(drop_x) if self.num_cls > 1 else self.sigmoid(self.fc(drop_x))
            
            return {
                'features':embedding_coding, 
                'adapted_layer': encodes, 
                'output_logits': logits
            }
        
        else:
            mu, std, z = self.reparametering(embedding_coding)
            logits = self.fc(z) if self.training else self.fc(mu)
            logits = self.sigmoid(logits) if self.num_cls == 1 else logits
            return {
                'mu': mu, 
                'std': std, 
                'adapted_layer': embedding_coding, 
                'reparametered_ft': z, 
                'output_logits': logits, 
                }
    
    @classmethod
    def create(cls, dicts):
        class_num = dicts['num_cls'] if dicts.get('num_cls', False) else 10
        use_bottleneck = dicts['use_bottleneck'] if dicts.get('use_bottleneck', False) else True
        bottleneck_dim = dicts['adapted_dim'] if dicts.get('adapted_dim', False) else 256
        IN = dicts['IN'] if dicts.get('IN', False) else False
        vib = dicts['vib'] if dicts.get('vib', False) else False
        frozen = dicts['frozen'] if dicts.get('frozen', False) else []
        return cls(
            class_num=class_num, 
            use_bottleneck=use_bottleneck, 
            bottleneck_dim=bottleneck_dim, 
            vib=vib, 
            IN=IN, 
            frozen=frozen
        )

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
                
        if self.ifin:
            parameter_list += [{"params": self.IN.parameters(), 'lr_mult': 1}]
        return parameter_list, [{"params": self.bottleneck.parameters(), 'lr_mult': 10}, {"params": self.fc.parameters(), 'lr_mult': 10}]
    
    
@register_model('resnet152')
class ResNet_152Fc(nn.Module):
    def __init__(self, 
                 use_bottleneck=True, 
                 bottleneck_dim=256, 
                 new_cls=True, 
                 class_num=1000, 
                 vib=False, 
                 IN=False, 
                 frozen=[]
                ):
        super(ResNet_152Fc, self).__init__()
        model_resnet = resnet_dict['ResNet152'](pretrained=True)
        self.ifin = IN
        self.vib = vib
        self.ordered_module_names = []
        self.frozen=frozen
        print('Frozen Layer: ', self.frozen)
        if IN:
            print(IN)
            self.IN = nn.InstanceNorm2d(3, affine=True)
            self.ordered_module_names += ['in']
        self.conv1 = model_resnet.conv1
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
        self.layer4 = model_resnet.layer4
        self.avgpool = model_resnet.avgpool
        self.feature_layers = nn.Sequential(
            self.conv1, 
            self.bn1, 
            self.relu, 
            self.maxpool,
            self.layer1, 
            self.layer2, 
            self.layer3, 
            self.layer4, 
            self.avgpool
        )

        self.ordered_module_names += [
            'conv1', 'bn1', 'relu', 'maxpool',
            'layer1', 'layer2', 'layer3', 'layer4', 'avgpool'
        ]
        
        self.use_bottleneck = use_bottleneck
        self.new_cls = new_cls
        self.num_cls = class_num
        if new_cls:
            if self.use_bottleneck:
                self.bottleneck = nn.Linear(
                    model_resnet.fc.in_features, 
                    bottleneck_dim if not vib else bottleneck_dim * 2
                )
                self.fc = nn.Linear(bottleneck_dim, class_num)
                self.bottleneck.apply(init_weights)
                
                self.fc.apply(init_weights)
                self.__in_features = bottleneck_dim
            else:
                self.fc = nn.Linear(model_resnet.fc.in_features, class_num)
                self.fc.apply(init_weights)
                self.__in_features = model_resnet.fc.in_features
        else:
            self.fc = model_resnet.fc
            self.__in_features = model_resnet.fc.in_features
        self.flattening = Flattening()
        if self.vib:
            self.reparametering = Reparametering(bottleneck_dim)
            
        if self.num_cls == 1:
            self.sigmoid = nn.Sigmoid()
            
        if len(self.frozen) > 0:
            for name in self.frozen:
                if name in self.ordered_module_names:
                    for params in self._modules[name].parameters():
                        params.requires_grad = False
            

    def forward(self, x, temp=0.05):
        if self.ifin:
            x = self.IN(x)
            
        for name in self.ordered_module_names:
            module = self._modules[name]
            x = module(x)
            x = x.detach() if name in self.frozen else x
            
        encodes = self.flattening(x)
        if not self.vib:
            if self.use_bottleneck and self.new_cls:
                encodes = self.bottleneck(encodes)
            logits = self.fc(encodes)
            if self.num_cls == 1:
                logits = self.sigmoid(logits)
            return {'features':[x], 'adapted_layer': encodes, 'output_logits': logits}
        else:
            mu, std, z = self.reparametering(encodes)
            logits = self.fc(z) if self.training else self.fc(mu)
            if self.num_cls == 1:
                logits = self.sigmoid(logits)
            return {'mu': mu, 'std': std, 'adapted_layer': encodes, 'reparametered_ft': z, 'output_logits': logits}
    
    @classmethod
    def create(cls, dicts):
        class_num = dicts['num_cls'] if dicts.get('num_cls', False) else 10
        use_bottleneck = dicts['use_bottleneck'] if dicts.get('use_bottleneck', False) else True
        bottleneck_dim = dicts['adapted_dim'] if dicts.get('adapted_dim', False) else 256
        new_cls = dicts['new_cls'] if dicts.get('new_cls', False) else True
        IN = dicts['IN'] if dicts.get('IN', False) else False
        vib = dicts['vib'] if dicts.get('vib', False) else False
        frozen = dicts['frozen'] if dicts.get('frozen', False) else []
        return cls(
            class_num=class_num, 
            use_bottleneck=use_bottleneck, 
            bottleneck_dim=bottleneck_dim, 
            new_cls=new_cls, 
            vib=vib, 
            IN=IN, 
            frozen=frozen
        )

    def output_num(self):
        return self.__in_features

    def get_parameters(self):
        parameter_list = [
            {"params": self.bottleneck.parameters(), 'lr_mult': 10},
            {"params": self.fc.parameters(), 'lr_mult': 10}
        ]
        for name in self.ordered_module_names:
            if name not in self.frozen and len(list(self._modules[name].parameters())) > 0:
                parameter_list += [{"params": self._modules[name].parameters(), 'lr_mult': 1}]
                
        if self.ifin:
            parameter_list += [{"params": self.IN.parameters(), 'lr_mult': 1}]
        return parameter_list
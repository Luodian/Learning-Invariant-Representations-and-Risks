from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from torch.autograd import Function
import torch
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torchvision.models as model
from .models import register_model

__all__ = ['CountingNet', 'FCN_Head']

ranges = {
    'vgg11': ((0, 3), (3, 6), (6, 11), (11, 16), (16, 21)),
    'vgg13': ((0, 5), (5, 10), (10, 15), (15, 20), (20, 25)),
    'vgg16': ((0, 5), (5, 10), (10, 17), (17, 24), (24, 31)),
    'vgg19': ((0, 5), (5, 10), (10, 19), (19, 28), (28, 37))
}


class VGG16_Backbone(nn.Module):
    def __init__(self, pretrained=True, requires_grad=True):
        super(VGG16_Backbone, self).__init__()
        self.ranges = ranges['vgg16']
        self.encoder = model.vgg16(pretrained=pretrained).features
        self.encoder_layers = list(self.encoder.children())
        if not requires_grad:
            for param in super().parameters():
                param.requires_grad = False

    def forward(self, x):
        output = {}
        for idx in range(len(self.ranges)):
            for layer in range(self.ranges[idx][0], self.ranges[idx][1]):
                x = self.encoder_layers[layer](x)
            output["x%d" % (idx + 1)] = x
        return output
    
class X2Transpose(nn.Module):
    def __init__(self, in_feature, out_feature, kernel_size, padding):
        super(X2Transpose, self).__init__()
        self.conv2d = nn.Conv2d(in_feature, out_feature, kernel_size=kernel_size, stride=1, padding=padding)
        
    def forward(self, x):
        x = nn.functional.interpolate(x, scale_factor=2)
        return self.conv2d(x)
    
@register_model('FCN_Head')
class FCN_Head(nn.Module):
    def __init__(self, num_reg=1, env_dim=0):
        super(FCN_Head, self).__init__()
        self.num_reg = num_reg
        self.relu = nn.ReLU(inplace=True)
        self.leaky_relu = nn.LeakyReLU(0.01, inplace=True)
        self.deconv1 = nn.ConvTranspose2d(512 + env_dim, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.deconv5 = nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.classifier = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv2d(32, num_reg, kernel_size=1)
        )
        self.env_dim = env_dim
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
    def forward(self, output, env='src', temp=1, cosine=False):
        B, C, H, W = output.shape
        if self.env_dim > 0 and env=='src':
            env_embedding = torch.zeros(B, self.env_dim, H, W, device=output.device)
            x = torch.cat([output, env_embedding], dim=1)
        elif self.env_dim > 0 and env=='tgt':
            env_embedding = torch.ones(B, self.env_dim, H, W, device=output.device)
            x = torch.cat([output, env_embedding], dim=1)            
        else:
            x = output
        score = self.leaky_relu(self.deconv1(x))  # size=(N, 512, x.H/16, x.W/16)
        score = score # element-wise add, size=(N, 512, x.H/16, x.W/16)
        score = self.leaky_relu(self.deconv2(score))  # size=(N, 256, x.H/8, x.W/8)
        score = score # element-wise add, size=(N, 256, x.H/8, x.W/8)
        score = self.leaky_relu(self.deconv3(score))  # size=(N, 128, x.H/4, x.W/4)
        score = score
        score = self.leaky_relu(self.deconv4(score))  # size=(N, 64, x.H/2, x.W/2)
        score = score
        score = self.leaky_relu(self.deconv5(score))
        score = self.leaky_relu(self.classifier(score))
         # size=(N, n_class, x.H/1, x.W/1)

        return score  # size=(N, n_class, x.H/1, x.W/1) 
    
    def get_parameters(self):
        return [{"params": self.parameters(), 'lr_mult': 1}]
        

@register_model('CountingNet')
class CountingNet(nn.Module):
    def __init__(self, num_reg=1, pretrained=True, requires_grad=True, frozen=[]):
        super(CountingNet, self).__init__()
        self.encoder = VGG16_Backbone(pretrained=pretrained, requires_grad=requires_grad)
        self.num_reg = num_reg
        self.decoder = FCN_Head(num_reg)
        self.frozen = frozen
        self.pretrained = pretrained
        self.requires_grad = requires_grad
        self.ordered_module_names = ['encoder']
        
    def forward(self, x, temp=1, dropout=True, cosine=False):
        input = x.sum()
        for name in self.ordered_module_names:
            module = self._modules[name]
            x = module(x)
            x = x.detach() if name in self.frozen else x
            
#         embedding_coding = x['x5']
        embedding_coding = x['x5']
        out = self.decoder(embedding_coding)
        return {
            'features':embedding_coding, 
            'adapted_layer': embedding_coding, 
            'output_logits': out
        }
        
    @classmethod
    def create(cls, dicts):
        class_num = dicts['num_cls'] if dicts.get('num_cls', False) else 1
        pretrained = dicts['pretrained'] if dicts.get('pretrained', False) else True
        requires_grad = dicts['requires_grad'] if dicts.get('requires_grad', False) else True
        frozen = dicts['frozen'] if dicts.get('frozen', False) else []
        return cls(
            num_reg=class_num,
            pretrained=pretrained,
            requires_grad=requires_grad,
            frozen=frozen
        )

    def get_parameters(self):
        parameter_list = []
        for name in self.ordered_module_names:
            if name not in self.frozen and len(list(self._modules[name].parameters())) > 0:
                parameter_list += [{"params": self._modules[name].parameters(), 'lr_mult': 1}]
        return parameter_list, [
            {"params": self.decoder.parameters(), 'lr_mult': 1}
        ]
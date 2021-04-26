import numpy as np
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
from torchvision import models
from torch.autograd import Variable
from torch.autograd import Function
from torch.nn import init

def weights_init_he(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        if 'weight' in m.state_dict().keys():
            m.weight.data.normal_(1.0, 0.02)
        if 'bias' in m.state_dict().keys():
            m.bias.data.fill_(0)
    else:
        if 'weight' in m.state_dict().keys():
            init.kaiming_normal_(m.weight)
        if 'bias' in m.state_dict().keys():
            m.bias.data.fill_(0)
            
def find_class_by_name(name, modules):
    """Searches the provided modules for the named class and returns it."""
    modules = [getattr(module, name, None) for module in modules]
    return next(a for a in modules if a)

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)
        
def grl_hook(coeff):
    def fun1(grad):
        return -coeff*grad.clone()
    return fun1

class GradReverse(Function):
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg()

def grad_reverse(x, lambd):
    return GradReverse.apply(x)

def calc_coeff(iter_num, high=1.0, low=0.0, alpha=10.0, max_iter=10000.0):
    return np.float(2.0 * (high - low) / (1.0 + np.exp(-alpha*iter_num / max_iter)) - (high - low) + low)

class Flattening(nn.Module):
    def __init__(self):
        super(Flattening, self).__init__()
        
    def forward(self, x):
        return torch.flatten(x, 1)
    
class Reparametering(nn.Module):
    def __init__(self, feature_dim):
        super(Reparametering, self).__init__()
        self.feature_dim = feature_dim
        
    def forward(self, x):
        mu = x[:, :self.feature_dim]
        std = F.softplus(x[:, self.feature_dim:] - 5, beta=1)
        z = self.reparametrize_n(mu, std)
        return mu, std, z
        
    def reparametrize_n(self, mu, std):
        eps = std.data.new(std.size()).normal_().cuda()
        return mu + eps.detach() * std
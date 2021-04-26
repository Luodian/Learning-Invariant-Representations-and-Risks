import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision import models
from torch.autograd import Variable
from utils.utils_module import *
import math

__all__ = ['RandomLayer', 'DANNDiscriminator', 'EnvPredictor']

def grl_hook(coeff):
    def fun1(grad):
        return -coeff*grad.clone()
    return fun1

class RandomLayer(nn.Module):
    def __init__(self, input_dim_list=[], output_dim=1024):
        super(RandomLayer, self).__init__()
        self.input_num = len(input_dim_list)
        self.output_dim = output_dim
        self.random_matrix = [torch.randn(
            input_dim_list[i], output_dim).cuda() for i in range(self.input_num)]

    def forward(self, input_list):
        return_list = [torch.mm(input_list[i], self.random_matrix[i])
                       for i in range(self.input_num)]
        return_tensor = return_list[0] / math.pow(float(self.output_dim), 1.0/len(return_list))
        for single in return_list[1:]:
            return_tensor = torch.mul(return_tensor, single)
        return return_tensor

class DANNDiscriminator(nn.Module):
    def __init__(self, in_feature=512, hidden_size=512, task_type='cls'):
        super(DANNDiscriminator, self).__init__()
        self.ad_layer1 = nn.Linear(in_feature, hidden_size)
        self.ad_layer2 = nn.Linear(hidden_size, hidden_size)
        self.ad_layer3 = nn.Linear(hidden_size, 1)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
        self.sigmoid = nn.Sigmoid()
        self.apply(init_weights)
        self.iter_num = 0
        self.alpha = 10
        self.low = 0.0
        self.high = 1.0
        self.task_type = task_type

    def forward(self, x, max_iter):
        if self.training:
            self.iter_num += 1
        coeff = calc_coeff(
            self.iter_num, 
            self.high, self.low, self.alpha, 
            max_iter
        )
        if self.task_type == 'reg':
            x = torch.nn.functional.adaptive_avg_pool2d(x, [1, 1]).reshape(x.size(0), -1)
        x = x * 1.0
        x.register_hook(grl_hook(coeff))
        x = self.ad_layer1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.ad_layer2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        y = self.ad_layer3(x)
        y = self.sigmoid(y)
        return y

    def output_num(self):
        return 1

    def get_parameters(self):
        return [{"params": self.parameters(), "lr_mult": 1, 'decay_mult': 1}]
    
class EnvPredictor(nn.Module):
    def __init__(self, batch_size, num_class, in_features, bottleneck_dim=1024):
        super(EnvPredictor, self).__init__()
        self.dropout1 = nn.Dropout(p=0.5)
        self.dropout2 = nn.Dropout(p=0.5)
        self.bottleneck = nn.Linear(in_features=in_features+1, out_features=bottleneck_dim, bias=True)
        self.fc = nn.Linear(in_features=bottleneck_dim, out_features=num_class, bias=True)
        
        self.apply(init_weights)
        self.num_class = num_class
        self.env_embedding = {
            'src':torch.zeros(batch_size, 1).cuda(), 
            'tgt':torch.ones(batch_size, 1).cuda()
        }

    def forward(self, x, env, temp=1, cosine=False):
        x1 = torch.cat([x, self.env_embedding[env]], 1)
        drop_x = self.dropout2(x1)
        encodes = torch.nn.functional.relu(self.bottleneck(drop_x), inplace=False)
        drop_x = self.dropout2(encodes)
        if cosine:
    #       cosine classifer
            normed_x = F.normalize(drop_x, p=2, dim=1)
            logits = self.fc(normed_x) / temp
        else:
            logits = self.fc(drop_x) / temp
        return logits

    def get_parameters(self):
        return [{"params": self.parameters(), "lr_mult": 1, 'decay_mult': 10}]
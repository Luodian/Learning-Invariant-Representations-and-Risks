import torch
import numpy as np
from PIL import Image, ImageFilter
import random
import math
import torchvision.transforms as transforms

def adjust_learning_rate(optimizer, cur_iter, all_iter, args, alpha=0.001, beta=0.75, param_lr=None):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    if args.adj_lr_func.lower() == 'cos':
        lr *= 0.5 * (1. + math.cos(math.pi * cur_iter / all_iter))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr * param_group['lr_mult']
            
    elif args.adj_lr_func.lower() == 'inv':
        lr = lr / pow(1.0 + alpha * cur_iter, beta)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr * param_group['lr_mult']
    
    elif args.adj_lr_func.lower() == 'mme':
        lr = args.lr * (1 + 0.0001 * cur_iter) ** (- 0.75)
        i=0
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr * param_lr[i]
            i+=1
            
    else:
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr * param_group['lr_mult']

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return [q, k]


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

def cuda(tensor, is_cuda):
    if is_cuda : return tensor.cuda()
    else : return tensor
    
def calc_coeff(iter_num, high=1.0, low=0.0, alpha=10.0, max_iter=10000.0):
    return np.float(2.0 * (high - low) / (1.0 + np.exp(-alpha*iter_num / max_iter)) - (high - low) + low)

def grl_hook(coeff):
    def fun1(grad):
        return -coeff*grad.clone()
    return fun1

def softmax_xent_two_logits(q_logits, p_logits):
    p_scores, q_scores = p_logits.softmax(dim=1), q_logits.softmax(dim=1)
    xent_loss = - torch.mean(torch.sum(q_scores * torch.log(p_scores + 1e-12), dim=1))
    return xent_loss

def normalize_perturbation(x):
    shape = x.shape
    x = x.reshape(shape[0], -1)
    output = x / torch.sqrt(torch.sum(x ** 2, dim=1, keepdim=True))
    return output.reshape(*shape)

def momentum_update_key(net, ema_net, momentum=0.998):
    """
    Momentum update of the key encoder
    """
    with torch.no_grad():
        for param_q, param_k in zip(
            list(net.parameters()),
            list(ema_net.parameters())
        ):
            param_k.data = param_k.data * momentum + param_q.data * (1. - momentum)
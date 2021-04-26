import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import math
import torch.nn.functional as F
from .utils_module import *

def irm_penalty(logits, labels, cls_criterion, args=[]):
    scale = torch.tensor(1.).cuda().requires_grad_()
    loss = cls_criterion(logits * scale, labels, *args)
    grad = torch.autograd.grad(loss, [scale], create_graph=True)[0]
    return torch.sum(grad ** 2)

def Entropy(input_):
    bs = input_.size(0)
    epsilon = 1e-5
    entropy = -input_ * torch.log(input_ + epsilon)
    entropy = torch.sum(entropy, dim=1)
    return entropy 

def adentropy(ft, net, lamda, eta=0.1):
    logits = net.predict(ft, reverse=True, eta=eta, temp=0.05)['output_logits']
    out_t1 = F.softmax(logits, dim=1)
    loss_adent = lamda * torch.mean(torch.sum(out_t1 *(torch.log(out_t1 + 1e-5)), 1))
    return loss_adent

def CDAN(input_list, ad_net, max_iter, entropy=None, coeff=None, random_layer=None):
    softmax_output = input_list[1].detach()
    feature = input_list[0]
    if random_layer is None:
        op_out = torch.bmm(softmax_output.unsqueeze(2), feature.unsqueeze(1))
        ad_out = ad_net(op_out.view(-1, softmax_output.size(1) * feature.size(1)), max_iter)
    else:
        random_out = random_layer.forward([feature, softmax_output])
        ad_out = ad_net(random_out.view(-1, random_out.size(1)), max_iter)       
    batch_size = softmax_output.size(0) // 2
    dc_target = torch.from_numpy(np.array([[1]] * batch_size + [[0]] * batch_size)).float().cuda()
    assert ad_out.shape[0] == dc_target.shape[0]
    if entropy is not None:
        entropy.register_hook(grl_hook(coeff))
        entropy = 1.0+torch.exp(-entropy)
        source_mask = torch.ones_like(entropy)
        source_mask[feature.size(0)//2:] = 0
        source_weight = entropy*source_mask
        target_mask = torch.ones_like(entropy)
        target_mask[0:feature.size(0)//2] = 0
        target_weight = entropy*target_mask
        weight = source_weight / torch.sum(source_weight).detach().item() + \
                 target_weight / torch.sum(target_weight).detach().item()
        return torch.sum(weight.view(-1, 1) * nn.BCELoss(reduction='none')(ad_out, dc_target)) / torch.sum(weight).detach().item()
    else:
        return nn.BCELoss()(ad_out, dc_target)
    
def DANN(features, ad_net, max_iter):
    ad_out = ad_net(features, max_iter)
    batch_size = ad_out.size(0) // 2
    dc_target = torch.from_numpy(np.array([[1]] * batch_size + [[0]] * batch_size)).float().cuda()
    return nn.BCELoss()(ad_out, dc_target)

def kl_divergence_with_logits(p_logits, q_logits, confidence=0.):
    if not (confidence > 0.):
        p = torch.softmax(p_logits, dim=1)
        log_p = F.log_softmax(p_logits, dim=1)
        log_q = F.log_softmax(q_logits, dim=1)
        kl = torch.mean(torch.sum(p * (log_p - log_q), dim=1))
    else:
        conf_mask = (p_logits.softmax(dim=1).max(dim=1)[0] > confidence).float()
        p = torch.softmax(p_logits, dim=1)
        log_p = F.log_softmax(p_logits, dim=1)
        log_q = F.log_softmax(q_logits, dim=1)
        kl = torch.sum(p * (log_p - log_q), dim=1)
        kl = torch.sum(conf_mask * kl) / (torch.sum(conf_mask) + 1e-8)
    return kl

def BNM(logits, lw=1):
    A = torch.softmax(logits, dim=1)
    _, s_tgt, _ = torch.svd(A)
    loss_bnm = -torch.mean(s_tgt)
    return loss_bnm * lw

def get_tsa_threshold(global_step, entire_steps, start, end, schedule='exp'):
    step_ratio = global_step / float(entire_steps)
    if schedule == 'linear':
        coeff = step_ratio
    elif schedule == 'exp':
        scale = 5
        # [exp(-5), exp(0)] = [1e-2, 1]
        coeff = math.exp((step_ratio - 1) * scale)
    elif schedule == 'log':
        scale = 5
        # [1 - exp(0), 1 - exp(-5)] = [0, 0.99]
        coeff = 1 - math.exp((-step_ratio) * scale)
    return coeff * (end - start) + start

def anneal_sup_loss(
    num_cls, 
    sup_logits, 
    sup_labels, 
    sup_loss, 
    global_step, 
    entire_steps, 
    anneal_schedule='exp'
):
    eff_train_prob_threshold = get_tsa_threshold(
        global_step, 
        entire_steps,
        start= (1. / num_cls),
        end=1,
        schedule=anneal_schedule
    )
    batch_size = sup_labels.shape[0]
    one_hot_labels = torch.zeros(
        batch_size, num_cls, 
        dtype=torch.float32,
        device=sup_logits.device
    ).scatter(
        dim=1, 
        index=sup_labels.reshape(batch_size, 1), 
        src=torch.ones(
            batch_size, 1, 
            dtype=torch.float32, 
            device=sup_logits.device
        )
    )
    sup_probs = torch.softmax(sup_logits, dim=1)
    correct_label_probs = torch.sum(one_hot_labels * sup_probs, dim=1)
    larger_than_threshold = correct_label_probs > eff_train_prob_threshold
    loss_mask = (1 - larger_than_threshold).detach().float()
    sup_loss = sup_loss * loss_mask
    avg_sup_loss = (torch.sum(sup_loss) / max(torch.sum(loss_mask), 1))
    return sup_loss, avg_sup_loss

def sim_KL_divergence(q, p, coef=1., confidence=0.):
    if confidence > 0.:
        conf_mask = (p.max(dim=1)[0] > confidence).float()
        kldiv = (
            torch.sum(p * torch.log(p / (q + 1e-12) + 1e-12), dim=1) + 
            torch.sum(q * torch.log(q / (p + 1e-12) + 1e-12), dim=1)
        ) / 2 
        masked_kldiv = torch.sum(kldiv * conf_mask) / (torch.sum(conf_mask) + 1e-8)
        return masked_kldiv * coef
    else:
        kldiv = (
            torch.mean(torch.sum(p * torch.log(p / (q + 1e-12) + 1e-12), dim=1)) + 
            torch.mean(torch.sum(q * torch.log(q / (p + 1e-12) + 1e-12), dim=1))
        ) / 2 
        return kldiv * coef
        
def _reg_loss(p, gt, num_cls, batch_size, global_step, entire_steps, annealing='none'):
    loss = nn.MSELoss(size_average = False)(p, gt) + (p.sum() - gt.sum()).abs() * 0.01
    return loss

    
def _cls_loss(p, labels, num_cls, batch_size, global_step, entire_steps, annealing='none'):
    if num_cls == 1:
        # for bce, the p should be the prediced scores
        labels = labels.float().reshape(batch_size)
        p = p.reshape(batch_size)
        return torch.nn.functional.binary_cross_entropy(p, labels, reduction='mean')
    else:
        # for ce, the p should be the prediced logits
        labels = labels.long().reshape(batch_size)
        p = p.reshape(batch_size, num_cls)
        scores = p.softmax(dim=1)
        loss_cls = torch.nn.functional.cross_entropy(p, labels, reduction='none')
        if annealing == 'none':
            loss_cls = torch.mean(loss_cls)
        else:
            _, loss_cls = anneal_sup_loss(
                num_cls=num_cls, 
                sup_logits=p, 
                sup_labels=labels, 
                sup_loss=loss_cls, 
                global_step=global_step, 
                entire_steps=entire_steps, 
                anneal_schedule=annealing
            )
        return loss_cls
    
def supervised_loss(logits, gt, num_cls, batch_size, global_step, entire_steps, annealing='none', task_type='cls'):
    if task_type=='cls':
        return _cls_loss(logits, gt, num_cls, batch_size, global_step, entire_steps, annealing)
    elif task_type=='reg':
        return _reg_loss(logits, gt, num_cls, batch_size, global_step, entire_steps, annealing)
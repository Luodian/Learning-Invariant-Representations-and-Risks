import torch
def cls_collate_fn(batch):
    r"""Puts each data field into a tensor with outer dimension batch size"""
    batch_1_q_img = torch.stack([ _sample_['sample_1_q'][0] for _sample_ in batch])
    batch_1_q_label = torch.LongTensor([ _sample_['sample_1_q'][1] for _sample_ in batch])
    batch_1_k_img = torch.stack([ _sample_['sample_1_k'][0] for _sample_ in batch])
    batch_1_k_label = torch.LongTensor([ _sample_['sample_1_k'][1] for _sample_ in batch])
    batch_2_q_img = torch.stack([ _sample_['sample_2_q'][0] for _sample_ in batch])
    batch_2_q_label = torch.LongTensor([ _sample_['sample_2_q'][1] for _sample_ in batch])
    batch_2_k_img = torch.stack([ _sample_['sample_2_k'][0] for _sample_ in batch])
    batch_2_k_label = torch.LongTensor([ _sample_['sample_2_k'][1] for _sample_ in batch])
    batch_inputs = {
        'sample_1_q':(batch_1_q_img, batch_1_q_label),
        'sample_1_k':(batch_1_k_img, batch_1_k_label),
        'sample_2_q':(batch_2_q_img, batch_2_q_label),
        'sample_2_k':(batch_2_k_img, batch_2_k_label),
    }
    return batch_inputs

def reg_collate_fn(batch):
    r"""Puts each data field into a tensor with outer dimension batch size"""
    batch_1_q_img = torch.stack([ _sample_['sample_1_q'][0] for _sample_ in batch])
    batch_1_q_label = torch.stack([ _sample_['sample_1_q'][1] for _sample_ in batch])
    batch_1_q_mask = torch.stack([ _sample_['sample_1_q'][2] for _sample_ in batch])
    batch_1_k_img = torch.stack([ _sample_['sample_1_k'][0] for _sample_ in batch])
    batch_1_k_label = torch.stack([ _sample_['sample_1_k'][1] for _sample_ in batch])
    batch_1_k_mask = torch.stack([ _sample_['sample_1_k'][2] for _sample_ in batch])
    batch_2_q_img = torch.stack([ _sample_['sample_2_q'][0] for _sample_ in batch])
    batch_2_q_label = torch.stack([ _sample_['sample_2_q'][1] for _sample_ in batch])
    batch_2_q_mask = torch.stack([ _sample_['sample_2_q'][2] for _sample_ in batch])
    batch_2_k_img = torch.stack([ _sample_['sample_2_k'][0] for _sample_ in batch])
    batch_2_k_label = torch.stack([ _sample_['sample_2_k'][1] for _sample_ in batch])
    batch_2_k_mask = torch.stack([ _sample_['sample_2_k'][2] for _sample_ in batch])
    
    batch_inputs = {
        'sample_1_q':(batch_1_q_img, batch_1_q_label, batch_1_q_mask),
        'sample_1_k':(batch_1_k_img, batch_1_k_label, batch_1_k_mask),
        'sample_2_q':(batch_2_q_img, batch_2_q_label, batch_2_q_mask),
        'sample_2_k':(batch_2_k_img, batch_2_k_label, batch_2_k_mask),
    }
    return batch_inputs
collate_fns = {'cls': cls_collate_fn, 'reg': reg_collate_fn}
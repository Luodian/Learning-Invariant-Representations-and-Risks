import torch
from utils.losses import *


def run_iter_lirr(
        inputs,
        models, optimizers,
        meters,
        args,
        global_step,
        entire_steps
):
    net = models['net']
    ad_net = models['ad_net']
    predictor_env = models['predictor_env']
    main_optimizer = optimizers['main']
    dis_optimizer = optimizers['dis']

    src_inputs = inputs['src']['sample_1_q'][0].cuda()
    src_labels = inputs['src']['sample_1_q'][1].cuda()
    l_tgt_inputs = inputs['l_tgt']['sample_1_q'][0].cuda()
    l_tgt_labels = inputs['l_tgt']['sample_1_q'][1].cuda()
    ul_tgt_inputs = inputs['ul_tgt']['sample_1_q'][0].cuda()
    is_mask = len(inputs['l_tgt']['sample_1_q']) > 2
    src_masks = inputs['src']['sample_1_q'][2].cuda() if is_mask else 1
    l_tgt_masks = inputs['l_tgt']['sample_1_q'][2].cuda() if is_mask else 1
    src_inputs = src_inputs * src_masks
    l_tgt_inputs = l_tgt_inputs * l_tgt_masks

    src_outputs = net(src_inputs, temp=args.temp, cosine=args.cosine)
    l_tgt_outputs = net(l_tgt_inputs, temp=args.temp, cosine=args.cosine)
    ul_tgt_outputs = net(ul_tgt_inputs, temp=args.temp, cosine=args.cosine)
    src_features, src_ada_features, src_logits = src_outputs['features'], src_outputs['adapted_layer'], src_outputs[
        'output_logits']
    l_tgt_features, l_tgt_logits = l_tgt_outputs['features'], l_tgt_outputs['output_logits']
    ul_tgt_ada_features = ul_tgt_outputs['adapted_layer']
    src_env_logits = predictor_env(src_features, 'src', args.temp, cosine=args.cosine)
    l_tgt_env_logits = predictor_env(l_tgt_features, 'tgt', args.temp, cosine=args.cosine)

    src_logits = src_logits * src_masks
    l_tgt_logits = l_tgt_logits * l_tgt_masks
    src_env_logits = src_env_logits * src_masks
    l_tgt_env_logits = l_tgt_env_logits * l_tgt_masks

    # classification loss
    loss_cls_src = supervised_loss(
        src_logits, src_labels,
        args.num_cls, args.batch_size,
        global_step, entire_steps,
        args.annealing,
        args.task_type
    ) / 2.
    loss_cls_tgt = supervised_loss(
        l_tgt_logits, l_tgt_labels,
        args.num_cls, args.batch_size,
        global_step, entire_steps,
        args.annealing,
        args.task_type
    ) / 2.
    loss_inv = loss_cls_src + loss_cls_tgt

    # env pred loss
    loss_env = 0
    loss_env += supervised_loss(
        src_env_logits, src_labels,
        args.num_cls, args.batch_size,
        global_step, entire_steps,
        args.annealing,
        args.task_type
    ) / 2.
    loss_env += supervised_loss(
        l_tgt_env_logits, l_tgt_labels,
        args.num_cls, args.batch_size,
        global_step, entire_steps,
        args.annealing,
        args.task_type
    ) / 2.

    features = torch.cat((src_ada_features, ul_tgt_ada_features), dim=0)
    loss_transfer = DANN(features, ad_net, entire_steps) * args.trade_off
    total_loss = loss_transfer + loss_inv + torch.sqrt((loss_inv - loss_env) ** 2) * 0.1
    main_optimizer.zero_grad()
    dis_optimizer.zero_grad()
    total_loss.backward()
    main_optimizer.step()
    dis_optimizer.step()
    # update meters
    meters['src_cls_loss'].update(loss_cls_src.item())
    meters['tgt_cls_loss'].update(loss_cls_tgt.item())
    meters['env_loss'].update(loss_env.item())
    meters['transfer_loss'].update(loss_transfer.item())
import torch
import logging
from utils.utils_module import *
from methods import train_method
from dset_loaders.prepare_loaders import get_iterator 
from eval import eval_funcs

def create_meters(method_name):
    # create meters
    meters = {}
    if 'lirr' == method_name.lower():
        meters['src_cls_loss'] = AverageMeter('src_cls')
        meters['tgt_cls_loss'] = AverageMeter('tgt_cls')
        meters['transfer_loss'] = AverageMeter('transfer')
        meters['env_loss'] = AverageMeter('env')
        
    else:
        meters['src_cls_loss'] = AverageMeter('src_cls')
        meters['tgt_cls_loss'] = AverageMeter('tgt_cls')
        meters['transfer_loss'] = AverageMeter('transfer')
        
    return meters

def train(
    args, 
    epoch,
    sstasks, 
    optimizers,
    loaders,
    logger,
    modules,
    params_lr
):
    net = modules['net']
    net.train()
    tg_te_err_min = 100
    pretext_meters = {}
    for sstask in sstasks:
        pretext_meters[sstask.name] = AverageMeter(sstask.name)
    meters = create_meters(args.method)
    len_train_source = len(loaders['source']['train'])
    len_train_target = len(loaders['target']['unlabeled'])
    src_iter = get_iterator(loaders['source']['train'])
    ul_tgt_iter = get_iterator(loaders['target']['unlabeled'])
    l_tgt_iter = get_iterator(loaders['target']['labeled'])
    
    for batch_idx in range(len_train_source):
        global_step = (min(epoch-1, args.nepoch) * len_train_source + batch_idx)
        entire_steps = len_train_source * (args.nepoch)
        inputs = {}
        inputs['src'] = next(src_iter)
        inputs['ul_tgt'] = next(ul_tgt_iter)
        inputs['l_tgt'] = next(l_tgt_iter)
            
        # training pretext tasks for single batch.
        for sstask in sstasks:
            loss_value = sstask.train_batch()
            pretext_meters[sstask.name].update(loss_value)
            adjust_learning_rate(sstask.optimizer, cur_iter=global_step, all_iter=entire_steps, args=args, alpha=0.0005, beta=2.25)
            
        # training the main task
        train_iter = train_method[args.method]
        train_iter(inputs, modules, optimizers, meters, args, global_step, entire_steps)
        for name, optimizer in optimizers.items():
            adjust_learning_rate(
                optimizer, 
                cur_iter=global_step, all_iter=entire_steps, 
                args=args, 
                alpha=0.0005, beta=2.25, 
                param_lr=params_lr[name]
            )

    sr_te_err = eval_funcs[args.task_type](loaders['source']['validation'], net, args, logger, True)
    tg_te_err = eval_funcs[args.task_type](loaders['target']['validation'], net, args, logger, True)
    us_te_err_av = []
    display = 'tgt_test_acc: %.2f ; src_test_acc: %.2f' % (tg_te_err, sr_te_err)
    for name, loss_meter in pretext_meters.items():
        display += (name + ' : %.5f, ' % (loss_meter.avg))
    for name, loss_meter in meters.items():
        display += (name + ' : %.5f, ' % (loss_meter.avg))    
    logger.info(display)
    if tg_te_err_min > tg_te_err:
        tg_te_err_min = tg_te_err

    return tg_te_err_min
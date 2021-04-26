from __future__ import print_function
import numpy as np
from bootstrap import *

def main():
    args, logger = parsing()
    logger.info('==> Building model..')
    net = get_model(
        args.model_name, 
        num_cls=args.num_cls, 
        adapted_dim=args.adapted_dim, 
        channels=args.channels,
        vib=args.vib,
        frozen=args.frozen
    )

    if args.load_path:
        logger.info('==> Loading model..')
        net.load_state_dict(torch.load(args.load_path))
        
    logger.info('==> Preparing datasets..')
    datasets = prepare_datasets(args)
    loaders = prepare_loaders(args, datasets)

    logger.info('==> Creating pretext tasks.')
    sstasks = parse_tasks(args, net, datasets['source']['train'], datasets['target']['unlabeled'])
    if len(sstasks) == 0:
        logger.info('==> No pretext task.')
    else:
        for sstask in sstasks:
            logger.info('==> Created pretext task: {}'.format(sstask.name))

    logger.info('==> Creating Optimizer & Building modules...')
    optimizers, params_lr = {}, {}
    encoder_parameters, classifier_parameters = net.get_parameters()
    main_parameters = encoder_parameters + classifier_parameters
    modules = building_modules(args, net, optimizers, main_parameters, params_lr, logger)
    optimizers['encoder'] = create_optimizer(encoder_parameters, args.lr, args.optimizer_type)
    optimizers['classifier'] = create_optimizer(classifier_parameters, args.lr, args.optimizer_type)
    optimizers['main'] = create_optimizer(main_parameters, args.lr, args.optimizer_type)
    optimizers_schedulers = {}
    for optimizer_name, optimizer in optimizers.items():
        optimizers_schedulers[optimizer_name] = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, 
            [args.milestone], 
            gamma=0.1, 
            last_epoch=-1
        )
    params_lr['main'] = []
    params_lr['encoder'] = []
    params_lr['classifier'] = []
    for param_group in optimizers['main'].param_groups:
        params_lr['main'].append(param_group["lr"])
    for param_group in optimizers['encoder'].param_groups:
        params_lr['encoder'].append(param_group["lr"])
    for param_group in optimizers['classifier'].param_groups:
        params_lr['classifier'].append(param_group["lr"])
    
    logger.info('==> Running..')
    all_epoch_stats = []
    best_tgt_te_err = 100
    for epoch in range(1, args.nepoch + 1):
        logger.info('Source epoch %d/%d main_lr=%.6f' % (epoch, args.nepoch, optimizers['main'].param_groups[0]['lr']))
        tg_te_err = train(
            args, epoch,
            sstasks, optimizers,
            loaders, logger,
            modules, params_lr
        )
        for name, opt_scheduler in optimizers_schedulers.items():
            opt_scheduler.step()
        if tg_te_err < best_tgt_te_err and args.dataset == 'citycam':
            best_tgt_te_err = tg_te_err
            torch.save(net.state_dict(), args.outf + '/net_best.pth'.format(str(epoch)))

if __name__ == "__main__":
    main()
    

    
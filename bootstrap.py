import argparse
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import os
from models import *
from dset_loaders.prepare_datasets import prepare_datasets
from dset_loaders.prepare_loaders import prepare_loaders 
import copy
from train import train
from utils.parse_tasks import parse_tasks

def parsing():
    logger = logging.getLogger(__name__)
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--source', nargs='+', required=True)
    parser.add_argument('--target', required=True)
    parser.add_argument('--data_root', default='/rscratch/data/')
    parser.add_argument('--image_size', default=32, type=int)
    parser.add_argument('--num_cls', default=10, type=int)
    parser.add_argument('--channels', default=3, type=int)
    #######################lr_scheduler##############################
    parser.add_argument('--nepoch', default=15, type=int)
    parser.add_argument('--optimizer_type', default='sgd', type=str)
    parser.add_argument('--lr', default=0.1, type=float)
    parser.add_argument('--milestone', default=40, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--nthreads', default=4, type=int)
    parser.add_argument('--trade_off', default=1e-1, type=float)
    parser.add_argument('--annealing', default='none', type=str)
    ########################pretext_config#############################
    parser.add_argument('--rotation', action='store_true')
    parser.add_argument('--quadrant', action='store_true')
    parser.add_argument('--flip', action='store_true')
    parser.add_argument('--lw_rotation', default=0.1, type=float)
    parser.add_argument('--lw_quadrant', default=0.1, type=float)
    parser.add_argument('--lw_flip', default=0.1, type=float)
    parser.add_argument('--lr_rotation', default=0.1, type=float)
    parser.add_argument('--lr_quadrant', default=0.1, type=float)
    parser.add_argument('--lr_flip', default=0.1, type=float)
    parser.add_argument('--quad_p', default=2, type=int)
    ########################network_config#############################
    parser.add_argument('--model_name', default='resnet50')
    parser.add_argument('--adapted_dim', default=1024, type=int)
    parser.add_argument('--frozen', nargs='+', default=[])
    parser.add_argument('--temp', default=1, type=float)
    parser.add_argument('--m', default=0.998, type=float)
    #######################global_config############################
    parser.add_argument('--method', default='none', help='specific da method.')
    parser.add_argument('--gpu', type=int, default=[0], nargs='+', help='used gpu')
    parser.add_argument('--outf', default='output/demo')
    parser.add_argument('--logf', default='output/demo')
    parser.add_argument('--load_path', type=str)
    parser.add_argument('--domain_shift_type', type=str, default='label_shift')
    parser.add_argument('--vib', action='store_true')
    parser.add_argument('--mim', action='store_true')
    parser.add_argument('--lambda_irm', default=0., type=float)
    parser.add_argument('--lambda_lirr', default=0., type=float)
    parser.add_argument('--lambda_adv', default=0., type=float)
    parser.add_argument('--lambda_inv', default=1., type=float)
    parser.add_argument('--lambda_env', default=0., type=float)
    parser.add_argument('--contrastive', action='store_true')
    parser.add_argument('--moco_finetune', action='store_true')
    parser.add_argument('--lirr', action='store_true')
    parser.add_argument('--cosine', action='store_true')
    parser.add_argument('--mixup', action='store_true')
    parser.add_argument('--distance_type', default="sqr", type=str)
    parser.add_argument('--logger_file_name', type=str, default='none')
    parser.add_argument('--adj_lr_func', type=str, default='none')
    parser.add_argument('--dropout', action='store_true')
    parser.add_argument('--K', default=256, type=int)
    parser.add_argument('--target_labeled_portion', default=1, type=int)
    parser.add_argument('--task_type', default='cls', type=str)
    args = parser.parse_args()
    try:
        os.makedirs(args.outf)
    except OSError:
        pass
    open(args.logf, 'w')

    logger.setLevel(level = logging.INFO)
    handler = logging.FileHandler(args.logf)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)

    logger.addHandler(handler)
    logger.addHandler(console)

    # show configuration.
    message = ''
    message += '\n----------------- Options ---------------\n'
    for k, v in sorted(vars(args).items()):
        comment = ''
        message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
    message += '----------------- End -------------------'
    logger.info(message)

    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in args.gpu)
    cudnn.benchmark = True
    return args, logger

def create_optimizer(parameters, lr, optimizer_type):
    if optimizer_type == 'sgd':
        return optim.SGD(parameters, lr=lr, momentum=0.9, weight_decay=5e-4)
    elif optimizer_type=='adam':
        return optim.Adam(parameters, lr=lr)

def building_modules(args, net, optimizers, all_parameters, params_lr, logger):
    modules = {'net': net}
    if 'lirr' == args.method.lower():
        ad_net = DANNDiscriminator(args.adapted_dim, args.adapted_dim, args.task_type).cuda()
        if args.task_type == 'cls':
            predictor_env = EnvPredictor(
                args.batch_size, 
                args.num_cls, 
                net.get_classifer_in_features(), 
                args.adapted_dim).cuda()
        elif args.task_type == 'reg':
            predictor_env = FCN_Head(num_reg=1, env_dim=1).cuda()
        all_parameters += predictor_env.get_parameters()
        modules['ad_net'] = ad_net
        modules['predictor_env'] = predictor_env
        optimizers['dis'] = create_optimizer(ad_net.get_parameters(), args.lr, args.optimizer_type)
        params_lr['dis'] = []
        for param_group in optimizers['dis'].param_groups:
            params_lr['dis'].append(param_group["lr"])
        logger.info('==> Have built extra modules: ad_net, predictor_env under LiRR method.')

    else:
        logger.info('==> no extra module need to be constructed.')
    return modules
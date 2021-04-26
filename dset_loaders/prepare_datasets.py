import torchvision
import torchvision.transforms as T
import torch.utils.data as torchdata
import os
import numpy as np
import random
from PIL import ImageFilter
from dset_loaders.digits import DIGITS, DigitsParams
from dset_loaders.visda2017 import VISDA2017, VisDA2017Params
from dset_loaders.officehome import OFFICEHOME, OfficehomeParams
from dset_loaders.domainnet import DOMAINNET, DomainnetParams
from dset_loaders.citycam import CITYCAM, CitycamParams
from dset_loaders.nico import NICO, NICOParams
from .label_parser_dict import *
from utils.utils_module import GaussianBlur
from dset_loaders.collect_ids_func import collect_ids

Params_dict = {
    'digits': DigitsParams,
    'visda2017': VisDA2017Params,
    'digits_origin': DigitsParams,
    'domainnet': DomainnetParams,
    'officehome': OfficehomeParams,
    'citycam': CitycamParams,
    'nico': NICOParams
}
Dataset_dict = {
    'digits': DIGITS,
    'digits_origin': DIGITS,
    'visda2017': VISDA2017,
    'domainnet': DOMAINNET,
    'officehome': OFFICEHOME,
    'citycam': CITYCAM,
    'nico': NICO
}


def get_transforms(args, Params):
    transforms = {
        'cls': {
            'train': T.Compose([
                T.RandomResizedCrop(args.image_size, scale=(0.5, 1.5)),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomApply([
                    T.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
                ], p=0.8),
                T.RandomGrayscale(p=0.2),
                T.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
                T.ToTensor(),
                T.Normalize(
                    mean=Params.mean,
                    std=Params.std
                )
            ]),
            'test': T.Compose([
                T.Resize([args.image_size, args.image_size]),
                T.ToTensor(),
                T.Normalize(Params.mean, Params.std)
            ])
        },
        'reg': {
            'train': T.Compose([
                T.Resize([256, 384]),
                #                         T.RandomApply([
                #                             T.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
                #                         ], p=0.8),
                #                         T.RandomGrayscale(p=0.2),
                T.ToTensor(),
                T.Normalize(
                    mean=Params.mean,
                    std=Params.std
                )
            ]),
            'test': T.Compose([
                T.Resize([256, 384]),
                T.ToTensor(),
                T.Normalize(Params.mean, Params.std)
            ])
        },
    }
    return transforms


def prepare_datasets(args):
    Params = Params_dict[args.dataset.lower()]
    Dataset = Dataset_dict[args.dataset.lower()]
    data_collection = collect_ids[args.task_type](args)
    transforms = get_transforms(args, Params)
    train_transform = transforms[args.task_type]['train']
    test_transform = transforms[args.task_type]['test']
    datasets = {'source': {}, 'target': {}}
    datasets['source']['train'] = Dataset(
        root=args.data_root, data=data_collection['source']['train'], transform=train_transform
    )
    print(args.dataset + ' source train set size: %d' % (len(datasets['source']['train'])))

    datasets['target']['labeled'] = Dataset(
        root=args.data_root,
        data=data_collection['target']['labeled'], transform=train_transform
    )
    print(args.dataset + ' labeled target train set size: %d' % (len(datasets['target']['labeled'])))

    datasets['target']['unlabeled'] = Dataset(
        root=args.data_root,
        data=data_collection['target']['unlabeled'], transform=train_transform
    )
    print(args.dataset + ' unlabeled target train set size: %d' % (len(datasets['target']['unlabeled'])))

    datasets['source']['validation'] = Dataset(
        root=args.data_root,
        data=data_collection['source']['validation'], transform=test_transform
    )
    print(args.dataset + ' source test set size: %d' % (len(datasets['source']['validation'])))

    datasets['target']['validation'] = Dataset(
        root=args.data_root,
        data=data_collection['target']['validation'], transform=test_transform
    )
    print(args.dataset + ' target test set size: %d' % (len(datasets['target']['validation'])))

    return datasets

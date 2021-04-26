import torch
import torch.utils.data as torchdata
from .collate_fn import collate_fns


def prepare_loaders(args, datasets):
    collate_fn = collate_fns[args.task_type]
    loaders = {'source': {}, 'target': {}}
    loaders['source']['train'] = torchdata.DataLoader(
        datasets['source']['train'],
        batch_size=args.batch_size,
        num_workers=args.nthreads,
        sampler=torchdata.RandomSampler(datasets['source']['train'], replacement=True),
        collate_fn=collate_fn,
        drop_last=True
    )
    loaders['target']['labeled'] = torchdata.DataLoader(
        datasets['target']['labeled'],
        batch_size=args.batch_size,
        num_workers=args.nthreads,
        sampler=torchdata.RandomSampler(datasets['target']['labeled'], replacement=True),
        collate_fn=collate_fn,
        drop_last=True
    )
    loaders['target']['unlabeled'] = torchdata.DataLoader(
        datasets['target']['unlabeled'],
        batch_size=args.batch_size,
        num_workers=args.nthreads,
        sampler=torchdata.RandomSampler(datasets['target']['unlabeled'], replacement=True),
        collate_fn=collate_fn,
        drop_last=True
    )

    loaders['source']['validation'] = torchdata.DataLoader(
        datasets['source']['validation'],
        batch_size=1,
        num_workers=args.nthreads,
        collate_fn=collate_fn,
    )

    loaders['target']['validation'] = torchdata.DataLoader(
        datasets['target']['validation'],
        batch_size=1,
        num_workers=args.nthreads,
        collate_fn=collate_fn,
    )
    return loaders


def get_iterator(loader):
    iterator = iter(loader)
    while True:
        try:
            tuples = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            tuples = next(iterator)
        yield tuples

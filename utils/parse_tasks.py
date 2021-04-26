import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as torchdata

from utils.SSTask import SSTask
from dset_classes.DsetNoLabel import DsetNoLabel

def parse_tasks(args, net, sc_tr_dataset, tg_tr_dataset):
    sstasks = []
    pretext_lws = {}
    pretext_lws['qdr'] = args.lw_quadrant
    pretext_lws['rot'] = args.lw_rotation
    pretext_lws['flip'] = args.lw_flip
    
    if args.rotation:
        print('Task: rotation prediction')
        from dset_classes.DsetSSRotRand import DsetSSRotRand
        digit = False
        if args.source[0] in ['mnist', 'mnistm', 'svhn', 'synth', 'usps']:
            print('No rotation 180 for digits!')
            digit = True
        
        su_tr_dataset = DsetSSRotRand(DsetNoLabel(sc_tr_dataset), digit=digit, img_size=224)
        su_tr_loader = torchdata.DataLoader(
            su_tr_dataset, 
            batch_size=args.batch_size // 2, 
            shuffle=True, 
            num_workers=4
        )

        tu_tr_dataset = DsetSSRotRand(DsetNoLabel(tg_tr_dataset), digit=digit, img_size=224)
        
        tu_tr_loader = torchdata.DataLoader(
            tu_tr_dataset, 
            batch_size=args.batch_size // 2, 
            shuffle=True, 
            num_workers=4
        )

        if not (args.vib):
            head = nn.Linear(args.adapted_dim, args.quad_p ** 2).cuda()
        else:
            if args.model_name.lower() == 'lenet':
                head = nn.Linear(3200, 4).cuda()
            elif args.model_name.lower() == 'resnet50':
                head = nn.Linear(2048, 4).cuda()
            elif args.model_name.lower() == 'resnet101':
                head = nn.Linear(2048, 4).cuda()
            else:
                raise NotImplementedError('Not support this backbone')
        
        criterion = nn.CrossEntropyLoss().cuda()
        
        parameters_list = [{"params": head.parameters(), 'lr_mult': 10.}]
        for m in net.feature_layers:
            parameters_list += [{"params": m.parameters(), 'lr_mult': 1.}]
        optimizer = optim.SGD(
            parameters_list,
            lr=args.lr_rotation, 
            momentum=0.9, weight_decay=5e-4
        )
        
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, 
            [args.milestone], 
            gamma=0.1, 
            last_epoch=-1
        )
        sstask = SSTask(
            net, head, 
            criterion, 
            optimizer, 
            scheduler,
            su_tr_loader, 
            tu_tr_loader, 
            name='rot',
            lw=pretext_lws['rot']
        )
        sstasks.append(sstask)

    if args.quadrant:
        print('Task: quadrant prediction')
        from dset_classes.DsetSSQuadRand import DsetSSQuadRand

        su_tr_dataset = DsetSSQuadRand(DsetNoLabel(sc_tr_dataset), quad_p=args.quad_p, img_size=224)
        su_tr_loader = torchdata.DataLoader(su_tr_dataset, batch_size=args.batch_size // 2, shuffle=True, num_workers=4)

        tu_tr_dataset = DsetSSQuadRand(DsetNoLabel(tg_tr_dataset), quad_p=args.quad_p, img_size=224)
        tu_tr_loader = torchdata.DataLoader(tu_tr_dataset, batch_size=args.batch_size // 2, shuffle=True, num_workers=4)

        if not (args.vib):
            head = nn.Linear(args.adapted_dim, args.quad_p ** 2).cuda()
        else:
            if args.model_name.lower() == 'lenet':
                head = nn.Linear(3200, args.quad_p ** 2).cuda()
            elif args.model_name.lower() == 'resnet50':
                head = nn.Linear(2048, args.quad_p ** 2).cuda()
            elif args.model_name.lower() == 'resnet101':
                head = nn.Linear(2048, args.quad_p ** 2).cuda()
            else:
                raise NotImplementedError('Not support this backbone')
                
        criterion = nn.CrossEntropyLoss().cuda()
        parameters_list = [{"params": head.parameters(), 'lr_mult': 10.}]
        for m in net.feature_layers:
            parameters_list += [{"params": m.parameters(), 'lr_mult': 1.}]
        optimizer = optim.SGD(
            parameters_list,
            lr=args.lr_quadrant, 
            momentum=0.9, 
            weight_decay=5e-4
        )
        
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, [args.milestone], gamma=0.1, last_epoch=-1)
        sstask = SSTask(
            net, 
            head, 
            criterion, 
            optimizer, 
            scheduler,
            su_tr_loader, 
            tu_tr_loader, 
            name='qdr',
            lw=pretext_lws['qdr']
        )
        sstasks.append(sstask)

    if args.flip:
        print('Task: flip prediction')
        from dset_classes.DsetSSFlipRand import DsetSSFlipRand

        digit = False
        su_tr_dataset = DsetSSFlipRand(DsetNoLabel(sc_tr_dataset), digit=digit, img_size=224)
        su_tr_loader = torchdata.DataLoader(su_tr_dataset, batch_size=args.batch_size // 2, shuffle=True, num_workers=4)

        tu_tr_dataset = DsetSSFlipRand(DsetNoLabel(tg_tr_dataset), digit=digit, img_size=224)
        tu_tr_loader = torchdata.DataLoader(tu_tr_dataset, batch_size=args.batch_size // 2, shuffle=True, num_workers=4)

        if not (args.vib):
            head = nn.Linear(args.adapted_dim, 2).cuda()
        else:
            if args.model_name.lower() == 'lenet':
                head = nn.Linear(3200, 2).cuda()
            elif args.model_name.lower() == 'resnet50':
                head = nn.Linear(2048, 2).cuda()
            elif args.model_name.lower() == 'resnet101':
                head = nn.Linear(2048, 2).cuda()
            else:
                raise NotImplementedError('Not support this backbone')
                
        criterion = nn.CrossEntropyLoss().cuda()
        parameters_list = [{"params": head.parameters(), 'lr_mult': 10.}]
        for m in net.feature_layers:
            parameters_list += [{"params": m.parameters(), 'lr_mult': 1.}]
        optimizer = optim.SGD(
            parameters_list,
            lr=args.lr_flip, 
            momentum=0.9, 
            weight_decay=5e-4
        )
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, 
            [args.milestone], 
            gamma=0.1, 
            last_epoch=-1
        )
        sstask = SSTask(
            net, 
            head, 
            criterion, 
            optimizer, 
            scheduler,
            su_tr_loader, 
            tu_tr_loader, 
            name='flip',
            lw=pretext_lws['flip']
        )
        sstasks.append(sstask)
    return sstasks



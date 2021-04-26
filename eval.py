import torch
def eval_reg(dataloader, model, args, logger, show_class_info=False):
    model.eval()
    abs_error = 0
    total = 0
    for batch_idx, data in enumerate(dataloader):
        inputs, labels, masks = data['sample_1_q'][0].cuda(), data['sample_1_q'][1].cuda(), data['sample_1_q'][2].cuda()
        with torch.no_grad():
            outputs = model(inputs * masks, dropout=args.dropout)['output_logits'] * masks
        
        gt_counts = labels.sum(dim=[1, 2, 3])
        pred_counts = outputs.sum(dim=[1, 2, 3])
        total += labels.size(0)
        abs_error += torch.sum(torch.abs(pred_counts - gt_counts), dim=0)
       
    model.train()
    return abs_error/total

def eval_cls(dataloader, model, args, logger, show_class_info=False):
    model.eval()
    correct = 0
    total = 0
    total_per_class = torch.zeros(args.num_cls).cuda()
    correct_per_class = torch.zeros(args.num_cls).cuda()
    src = torch.ones(args.num_cls).cuda()
    for batch_idx, data in enumerate(dataloader):
        inputs, labels = data['sample_1_q'][0].cuda(), data['sample_1_q'][1].cuda()
        with torch.no_grad():
            outputs = model(inputs, dropout=args.dropout)
            if args.num_cls > 1:
                avg_prediction = outputs['output_logits'].max(1)[1]
                corrects_batch = torch.eq(avg_prediction, labels).float()
                valid_idx = ((labels.float().reshape(-1) + 1.) * corrects_batch).nonzero() # plus 1 for avoiding to remove label 0 wrongly.
                correct_labels = labels[valid_idx]
                correct_per_class.scatter_add_(
                    dim=0, 
                    index=correct_labels.long().reshape(-1), 
                    src=torch.ones(correct_labels.shape[0], device=src.device)
                )
                total_per_class.scatter_add_(
                    dim=0, 
                    index=labels.long().reshape(-1), 
                    src=src
                )
                corrects_batch=corrects_batch.sum()
            else:
                avg_prediction = (outputs['output_logits'] > 0.5).long().squeeze()
                corrects_batch = torch.eq(avg_prediction, labels).float()
                corrects_batch=corrects_batch.sum()
                
        total += labels.size(0)
        correct += corrects_batch
        
    model.train()
    return (correct/total)*100

eval_funcs = {'cls': eval_cls, 'reg': eval_reg}
import os
import numpy as np
import random
from .label_parser_dict import *
portion = {
    1: "labeled", 
    5: "labeled_5", 
    10:"labeled_10", 
    15:"labeled_15", 
    20:"labeled_20", 
    25:"labeled_25", 
    30:"labeled_30",
    70:"labeled_70"
}

# /nfs/volume-92-5/wangyezhen_i/Projects/Theoretical_Projects/InstaPBM-V1/
shift_path_root_dict = {
    'lds': 'datasets/LDS',
    'ilds': 'datasets/ILDS',
    'convention': 'datasets/convention'
}

def shuffling(tensor_list):
    # shuffling
    permutation = np.random.permutation(len(tensor_list[0]))
    new_tensor_list = np.array(tensor_list)[:, permutation].tolist()
    return new_tensor_list

def collect_ids_cls(args):
    data_collection = {
        'source':{
            'train': {'ids':[], 'labels':[]},
            'validation': {'ids':[], 'labels':[]}
        },
        'target':{
            'labeled': {'ids':[], 'labels':[]},
            'unlabeled': {'ids':[], 'labels':[]},
            'validation': {'ids':[], 'labels':[]}
        }
    }
    shift_type = args.domain_shift_type
    general_domain = args.dataset
    print('==> begin to load ids.')
    shift_path_root = shift_path_root_dict[shift_type]
    for dm in args.source:
        domain_ls_path = os.path.join(
            shift_path_root, 
            general_domain, 
            'source', dm + '.txt'
        )
        domain_reader = open(domain_ls_path, 'r')
        for line in domain_reader:
            if line == '\n':
                continue
            id, cls = line.replace('\n', '').split(' ')
            data_collection['source']['train']['ids'].append(os.path.join(args.data_root, dm.split('_')[0] + '/' + id))
            data_collection['source']['train']['labels'].append(label2index_parser[general_domain][cls])
        domain_reader.close()
            
    target_partitions = [portion.get(args.target_labeled_portion, "labeled"), 'unlabeled', 'validation']
    for item in target_partitions:
        t_p = item.split("_")[0]
        domain_ls_path = os.path.join(
            shift_path_root, 
            general_domain,
            'target', args.target + '_' + item + '.txt'
        )
        domain_reader = open(domain_ls_path, 'r')
        for line in domain_reader:
            if line == '\n':
                continue
            id, cls = line.replace('\n', '').split(' ')
            data_collection['target'][t_p]['ids'].append(os.path.join(args.data_root, args.target + '/' + id))
            data_collection['target'][t_p]['labels'].append(label2index_parser[general_domain][cls])
        domain_reader.close()
    # shuffling
    
    shuffled_src_data = shuffling(
        [data_collection['source']['train']['ids'], 
         data_collection['source']['train']['labels']]
    )
    data_collection['source']['train']['ids'] = shuffled_src_data[0]
    data_collection['source']['train']['labels'] = shuffled_src_data[1]
    data_collection['source']['validation']['ids'] = shuffled_src_data[0][:5000]
    data_collection['source']['validation']['labels'] = shuffled_src_data[1][:5000]
    
    for item in target_partitions:
        t_p = item.split("_")[0]
        shuffled_src_data = shuffling(
            [data_collection['target'][t_p]['ids'], 
             data_collection['target'][t_p]['labels']]
        )
        data_collection['target'][t_p]['ids'] = shuffled_src_data[0]
        data_collection['target'][t_p]['labels'] = shuffled_src_data[1]
        
    return data_collection

def collect_ids_reg(args):
    data_collection = {
        'source':{
            'train': {'ids':[], 'labels':[], 'masks':[]},
            'validation': {'ids':[], 'labels':[], 'masks':[]}
        },
        'target':{
            'labeled': {'ids':[], 'labels':[], 'masks':[]},
            'unlabeled': {'ids':[], 'labels':[], 'masks':[]},
            'validation': {'ids':[], 'labels':[], 'masks':[]}
        }
    }
    shift_type = args.domain_shift_type
    general_domain = args.dataset
    print('==> begin to load ids.')
    shift_path_root = shift_path_root_dict[shift_type]
    for dm in args.source:
        domain_ls_path = os.path.join(
            shift_path_root, 
            general_domain, 
            'source', dm + '.txt'
        )
        domain_reader = open(domain_ls_path, 'r')
        for line in domain_reader:
            if line == '\n':
                continue
            id, reg, mask = line.replace('\n', '').split(' ')
            data_collection['source']['train']['ids'].append(os.path.join(args.data_root, dm.split('_')[0] + '/' + id))
            data_collection['source']['train']['labels'].append(os.path.join(args.data_root, dm.split('_')[0] + '/' + reg))
            data_collection['source']['train']['masks'].append(os.path.join(args.data_root, dm.split('_')[0] + '/' + mask))
        domain_reader.close()
    
    target_partitions = [portion.get(args.target_labeled_portion, "labeled"), 'unlabeled', 'validation']
    for item in target_partitions:
        t_p = item.split("_")[0]
        domain_ls_path = os.path.join(
            shift_path_root, 
            general_domain,
            'target', args.target + '_' + item + '.txt'
        )
        domain_reader = open(domain_ls_path, 'r')
        for line in domain_reader:
            if line == '\n':
                continue
            id, reg, mask = line.replace('\n', '').split(' ')
            data_collection['target'][t_p]['ids'].append(os.path.join(args.data_root, args.target + '/' + id))
            data_collection['target'][t_p]['labels'].append(os.path.join(args.data_root, args.target + '/' + reg))
            data_collection['target'][t_p]['masks'].append(os.path.join(args.data_root, args.target + '/' + mask))
        domain_reader.close()
    
    # shuffling
    shuffled_src_data = shuffling(
        [data_collection['source']['train']['ids'], 
         data_collection['source']['train']['labels'],
         data_collection['source']['train']['masks']]
    )
    data_collection['source']['train']['ids'] = shuffled_src_data[0]
    data_collection['source']['train']['labels'] = shuffled_src_data[1]
    data_collection['source']['train']['masks'] = shuffled_src_data[2]
    data_collection['source']['validation']['ids'] = shuffled_src_data[0][:5000]
    data_collection['source']['validation']['labels'] = shuffled_src_data[1][:5000]
    data_collection['source']['validation']['masks'] = shuffled_src_data[2][:5000]
    
    for t_p in target_partitions:
        if 'validation' not in t_p:
            t_p = t_p.split("_")[0]
            shuffled_src_data = shuffling(
                [data_collection['target'][t_p]['ids'], 
                 data_collection['target'][t_p]['labels'],
                 data_collection['target'][t_p]['masks']]
            )
            data_collection['target'][t_p]['ids'] = shuffled_src_data[0]
            data_collection['target'][t_p]['labels'] = shuffled_src_data[1]
            data_collection['target'][t_p]['masks'] = shuffled_src_data[2]
    return data_collection


collect_ids = {'cls': collect_ids_cls, 'reg': collect_ids_reg}
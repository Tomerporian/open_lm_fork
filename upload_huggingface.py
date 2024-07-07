import os
import shutil
import yaml
import csv
import re
import torch
import numpy as np
exp_sets = {
    'dataset=rw/hparams=base':
    [
        '24-05-11-kaplan_1_nodes_big_BS',
        '24-05-11-kaplan_2_nodes_big_BS',
        '24-05-11-kaplan_4_nodes_big_BS',
        '24-05-10-kaplan_8_nodes_orig_fix',
        '24-05-14-CC_1_nodes_rw_BS_256_rerun',
        '24-05-14-CC_2_nodes_rw_BS_256_rerun',
        '24-05-14-CC_4_nodes_rw_BS_256_rerun',
        '24-05-14-CC_8_nodes_rw_BS_256_rerun',

    ],
    'dataset=rw/hparams=tuned':
    [
        '24-05-09-const_1_nodes_tuned',
        '24-05-09-const_2_nodes_tuned',
        '24-06-27-const_2_nodes_tuned_rw_arxiv_fix',
        '24-05-09-const_4_nodes_tuned',
        '24-05-09-const_8_nodes_tuned',
    ],
    'dataset=owt2/hparams=base':
    [
        '24-05-12-kaplan_1_nodes_big_BS_openwebtext2',
        '24-05-12-kaplan_2_nodes_big_BS_openwebtext2',
        '24-05-12-kaplan_4_nodes_big_BS_openwebtext2',
        '24-05-11-kaplan_8_nodes_openwebtext2',
        '24-05-14-CC_1_nodes_owt2_BS_256_rerun',
        '24-05-14-CC_2_nodes_owt2_BS_256_rerun',
        '24-05-14-CC_4_nodes_owt2_BS_256_rerun',
        '24-05-14-CC_8_nodes_owt2_BS_256_rerun',
    ],
    'dataset=owt2/hparams=tuned':
    [
        '24-05-11-const_1_nodes_tuned_openwebtext2',
        '24-05-11-const_2_nodes_tuned_openwebtext2',
        '24-06-27-const_2_nodes_tuned_owt2_arxiv_fix',
        '24-05-11-const_4_nodes_tuned_openwebtext2',
        '24-05-11-const_8_nodes_tuned_openwebtext2',

    ]
}

params_dict = {('96', '3'): 5173248.0,
 ('128', '4'): 7503872.0,
 ('160', '5'): 9809920.0,
 ('224', '6'): 15597568.0,
 ('288', '8'): 22487040.0,
 ('320', '9'): 28672000.0,
 ('384', '10'): 37060608.0,
 ('480', '12'): 57384960.0,
 ('576', '14'): 84787200.0,
 ('640', '15'): 108462080.0,
 ('704', '18'): 149045248.0,
 ('832', '21'): 220872704.0,
 ('1024', '23'): 347078656.0,
 ('1120', '26'): 455311360.0,
 ('1312', '26'): 611958784.0,
 ('1504', '30'): 901726208.0
 }

bad_depth_lr = {'9': 5.1e-3, '10': 5.1e-3, '12': 4.7e-3}

def extract_decay(lr_scheduler, max_tokens):
    if "const" in lr_scheduler:
        return "const"
    elif max_tokens is not None:
        return "kaplan"
    else:
        return "chinchilla"

# create a mapping from checkpoint folder path to the new path for hugging face
mapping = {}
base_path = 'exps_recovery'
for key, exps in exp_sets.items():
    for exp in exps:
        exp_path = os.path.join(base_path, exp)
        for sub_exp in os.listdir(exp_path):
            if 'job' in sub_exp:
                continue
            # get warmup tokens - open 'spec.yaml' and read the value of warmup_tokens
            with open(os.path.join(exp_path, sub_exp, 'spec.yaml')) as f:
                spec = yaml.load(f, Loader=yaml.FullLoader)
                warmup_tokens = float(spec['warmup_tokens'])
                lr = float(spec['lr'])
                max_tokens = spec.get('max_tokens')
                decay = extract_decay(spec['lr_scheduler'], max_tokens)
            WU_str = "short" if warmup_tokens < 1.5e9 else "long"
            # get maxstep - open 'summary.csv' and read the last value of column 'step'
            with open(os.path.join(exp_path, sub_exp, 'summary.csv')) as f:
                reader = csv.reader(f)
                maxstep = list(reader)[-1][1]
            layers = re.search(r'layers=(\d+)', sub_exp).group(1)
            hidden_dim = re.search(r'hidden-dim=(\d+)', sub_exp).group(1)
            params_M = int(params_dict[(hidden_dim, layers)] / 1e6)

            new_path = f'{key}_warmup={WU_str}_decay={decay}/params={params_M}M_maxstep={maxstep}'
            dataset, hparams = key.split('/')[0].split('=')[1], key.split('/')[1].split('=')[1]
            if lr != bad_depth_lr.get(layers, 0) and (hparams != "tuned" or dataset!="owt2" or WU_str!="long"):
                mapping[os.path.join(exp_path, sub_exp, 'checkpoints')] = new_path
            

# make sure all keys in mapping exist
# for key in mapping.keys():
#     if not os.path.exists(key):
#         print(f"Key {key} does not exist!")
    # else:
    #     print(f"Key {key} will be moved to {mapping[key]}")
# print(f"Total number of keys: {len(mapping)}")


# move the checkpoints to the new path
# move only checkpoints that have 'flop' in their name and don't have 'poly' in their name
# create a symlink to the new path in the old path so that the old path can still be used
new_base_path = 'exps_recovery/huggingface'
# for old_path, new_path in mapping.items():
#     os.makedirs(os.path.join(new_base_path, new_path), exist_ok=True)
#     for file in os.listdir(old_path):
#         if ('flop' in file and 'poly' not in file) or ('poly' not in file and 'CC' in old_path and 'epoch' in file):
#             os.rename(os.path.join(old_path, file), os.path.join(new_base_path, new_path, file))
            # print(f"Moving\n{file} to\n{new_base_path}/{new_path}")

def get_flops(old_dir, step):
    flops_grid = 1e17 * ((2.0)**np.arange(-5, 8))
    args_path = os.path.join(os.path.dirname(old_dir), 'args.yaml')
    with open(args_path) as f:
        args = yaml.load(f, Loader=yaml.FullLoader)
        batch_size = args['batch_size']
        world_size = args['world_size']
        seq_len = args['seq_len']
    layers = re.search(r'layers=(\d+)', old_dir).group(1)
    hidden_dim = re.search(r'hidden-dim=(\d+)', old_dir).group(1)
    params = params_dict[(hidden_dim, layers)]
    flops = 6 * params * seq_len * batch_size * world_size * step
    nearest_flop = min(flops_grid, key=lambda x: abs(x - flops))
    return nearest_flop, flops

def rename_step_to_maxstep(base_path):
    reverse_mapping = {v: k for k, v in mapping.items()}
    for root, dirs, files in os.walk(base_path, topdown=False):
        for file in files:
            if file.endswith('.pt'):
                checkpoint = torch.load(os.path.join(root, file), map_location='cpu')
            else:
                continue
            new_file_name = file
            step = checkpoint['step']
            if file.endswith('.pt') and 'epoch' in file:
                old_dir = reverse_mapping[root[len(base_path)+1:]]
                nearest_flop, flops = get_flops(old_dir, step)
                new_file_name = f'flop_{nearest_flop:.2e}_step_{step}.pt'
                checkpoint['flop'] = flops

            # remove checkpoint keys not in ['epoch', 'flop', 'state_dict', 'step']
            for key in list(checkpoint.keys()):
                if key not in ['epoch', 'flop', 'state_dict', 'step']:
                    del checkpoint[key]
                    pass

            torch.save(checkpoint, os.path.join(root, new_file_name))
            if 'epoch' in file:
                os.remove(os.path.join(root, file))

# print(rename_step_to_maxstep(new_base_path))

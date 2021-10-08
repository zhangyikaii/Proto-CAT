# -*- coding: utf-8 -*-

import os, sys
sys.path.append("../")
from models.metrics import ROOT_PATH

import os.path as osp

import random
import sys


sample_num = 16
random.seed(929)

def read_csv_categories(dataset_type, master_stype):
    with open(osp.join(ROOT_PATH, f'data/{dataset_type}/split/{master_stype}.csv'), mode='r') as f:
        csv_tmp = f.readlines()
        csv_categories = [i.strip().split(',')[1] for i in csv_tmp[1:]]
    return sorted(list(set(csv_categories)))

categories = read_csv_categories('lrw', 'train') + read_csv_categories('lrw', 'val') + read_csv_categories('lrw', 'test')
# sampled = random.sample(categories, 80)
# modalities = ['video' if random.randint(0, 4) == 0 else 'audio' for _ in range(len(sampled))]

sampled = categories
modalities = ['video' for _ in range(len(sampled))]
with open('/home/zhangyk/Few-shot-Framework/data/lrw/label_incomplete.txt', 'w') as t:
    for i, j in zip(sampled, modalities):
        t.write(f'{i},{j}\n')


# categories = read_csv_categories('lrw1000', 'train') + read_csv_categories('lrw1000', 'val') + read_csv_categories('lrw1000', 'test')
# sampled = random.sample(categories, 80)
# modalities = ['video' if random.randint(0, 4) == 0 else 'audio' for _ in range(len(sampled))]

# with open('/home/zhangyk/Few-shot-Framework/data/lrw1000/label_incomplete.txt', 'w') as t:
#     for i, j in zip(sampled, modalities):
#         t.write(f'{i},{j}\n')

import cv2
from turbojpeg import TurboJPEG, TJPF_GRAY, TJSAMP_GRAY, TJFLAG_PROGRESSIVE

import torch
from torch.utils.data import Dataset, DataLoader

import os, sys
sys.path.append("../")
from models.metrics import ROOT_PATH

import os.path as osp
import glob
import csv
import numpy as np
import random
from collections import defaultdict
import sys

from tqdm import tqdm


jpeg = TurboJPEG()

data_root = '/data/zhangyk/data/CAS-VSR-W1k/lip_images/lip_images'
index_file = '/data/zhangyk/data/CAS-VSR-W1k/info/all_audio_video.txt'
target_dir = '/data/zhangyk/data/lrw1000_roi_pkl_jpg'
lines = []
padding = 40
class_dict = defaultdict(list)

# train : val : test : aux_val : aux_test
# 类别数:
# 64    : 16  : 20   : 64      : 64
# 每类个数:
# 600   : 600 : 600  : 150     : 150


with open(index_file, 'r') as f:
    lines.extend([line.strip().split(',') for line in f.readlines()])

for line in tqdm(lines):
    if os.path.exists(f'/data/zhangyk/data/CAS-VSR-W1k/audio/LRW1000_Public/audio/{line[1]}.wav'):
        beg, end = int(float(line[4])*25) + 1, int(float(line[5])*25) + 1
        if end - beg <= padding:
            class_dict[line[3]].append(f'{line[3]}_{line[0]}_{line[1]}.pkl')

class_length = {k: len(v) for k, v in class_dict.items()}
class_length = dict(sorted(class_length.items(), key=lambda item: item[1], reverse=True)[:100])

random.seed(929)

# 64 base类, 16 validation类, 20 novel类
base_class = 64
val_class = 16
novel_class = 20
sampled = random.sample(list(class_length.keys()), base_class + val_class + novel_class)
train_label = sampled[0 : base_class]
val_label = sampled[base_class : base_class+val_class]
test_label = sampled[base_class+val_class : base_class+val_class+novel_class]


def save_to_csv(sampled_files, sampled_labels, master_stype):
    with open(osp.join(ROOT_PATH, f'data/lrw1000/split/{master_stype}.csv'), mode='w') as csv_file:
        fieldnames = ['filename', 'label']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

        writer.writeheader()
        for i, j in zip(sampled_files, sampled_labels):
            writer.writerow({'filename': i, 'label': j})

train_sampled_files, aux_val_sampled_files, aux_test_sampled_files = [], [], []
train_num, aux_val_num, aux_test_num = 600, 150, 150
train_sf, aux_val_sf, aux_test_sf = [], [], []
train_l, aux_val_l, aux_test_l = [], [], []

for l in train_label:
    cur = class_dict[l]
    sampled = random.sample(cur, train_num + aux_val_num + aux_test_num)
    train_sf += sampled[: train_num]
    aux_val_sf += sampled[train_num : train_num + aux_val_num]
    aux_test_sf += sampled[train_num + aux_val_num : train_num + aux_val_num + aux_test_num]
    train_l += [l] * train_num
    aux_val_l += [l] * aux_val_num
    aux_test_l += [l] * aux_test_num
save_to_csv(train_sf, train_l, 'train')
save_to_csv(aux_val_sf, aux_val_l, 'aux_val')
save_to_csv(aux_test_sf, aux_test_l, 'aux_test')

val_num, test_num = 600, 600
for stype in ['val', 'test']:
    cur_sf, cur_l = [], []
    for l in eval(f'{stype}_label'):
        cur = class_dict[l]
        cur_num = eval(f'{stype}_num')
        cur_sf += random.sample(cur, cur_num)
        cur_l += [l] * cur_num
    save_to_csv(cur_sf, cur_l, stype)
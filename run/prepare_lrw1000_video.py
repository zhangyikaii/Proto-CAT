import cv2
from turbojpeg import TurboJPEG, TJPF_GRAY, TJSAMP_GRAY, TJFLAG_PROGRESSIVE

import torch
from torch.utils.data import Dataset, DataLoader

import glob
import numpy as np
import random
from collections import defaultdict
import sys

from tqdm import tqdm


import os.path as osp
import os, sys
sys.path.append("../")

from models.metrics import ROOT_PATH



jpeg = TurboJPEG()

data_root = '/data/zhangyk/data/CAS-VSR-W1k/lip_images/lip_images'
index_file = '/data/zhangyk/data/CAS-VSR-W1k/info/all_audio_video.txt'
target_dir = '/data/zhangyk/data/lrw1000_roi_pkl_jpg'
lines = []
padding = 40

with open(index_file, 'r') as f:
    lines.extend([line.strip().split(',') for line in f.readlines()])

def load_images(path, op, ed):
    center = (op + ed) / 2
    
    op = int(center - padding // 2)
    ed = int(op + padding)

    files =  [os.path.join(path, '{}.jpg'.format(i)) for i in range(op, ed)]
    files = filter(lambda path: os.path.exists(path), files)
    files = [cv2.imread(file) for file in files]
    files = [cv2.resize(file, (96, 96)) for file in files]        

    files = np.stack(files, 0)

    t = files.shape[0]
    
    tensor = np.zeros((40, 96, 96, 3)).astype(files.dtype)
    tensor[:t,...] = files.copy()

    tensor = [jpeg.encode(tensor[_]) for _ in range(40)]

    return tensor

# 64 * (600 + 300)
train_csv_filepath = osp.join(ROOT_PATH, 'data/lrw/split/train.csv')

line1_list = []
for stype in ['train', 'val', 'test', 'aux_val', 'aux_test']:
    csv_filepath = osp.join(ROOT_PATH, f'data/lrw1000/split/{stype}.csv')
    with open(csv_filepath, 'r') as f:
        csv_tmp = f.readlines()
        for i in csv_tmp[1:]:
            file_name = i.strip().split(',')[0]
            line1_list.append(file_name[file_name.rfind('_')+1 : file_name.find('.pkl')])

for line in tqdm(lines):
    if line[1] in line1_list:
        if os.path.exists(f'/data/zhangyk/data/CAS-VSR-W1k/audio/LRW1000_Public/audio/{line[1]}.wav'):
            beg, end = int(float(line[4])*25) + 1, int(float(line[5])*25) + 1
            if end - beg <= padding:
                inputs = load_images(os.path.join(data_root, line[0]), beg, end)
                savename = os.path.join(target_dir, f'{line[3]}_{line[0]}_{line[1]}.pkl')
                result = {}
                result['video'] = inputs
                result['label'] = line[3]
                torch.save(result, savename)
        else:
            assert 0

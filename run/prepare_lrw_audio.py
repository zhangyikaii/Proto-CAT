######################################
## 数据文件夹下LRW文件夹的名字. ##
######################################
LRW_DATA_PATH_NAME = 'lipread_mp4'
LRW_VIDEO_DATA_PATH_NAME = 'lrw_roi_80_116_175_211_npy_gray_pkl_jpeg'
LRW_AUDIO_DATA_PATH_NAME = 'lrw_audio_pkl'
######################################

import torch
from torch.utils.data import Dataset, DataLoader

import numpy as np
import glob
import time
import os
import os.path as osp

import sys
sys.path.append("../")
from models.metrics import ROOT_PATH
from models.utils import mkdir, save_pickle

from tqdm.contrib import tzip
import librosa

import warnings
warnings.filterwarnings('ignore')

def parse_csv(data_root_path, csv_path):
    lines = [x.strip() for x in open(csv_path, 'r').readlines()][1:]

    data = []
    label = []
    lb = -1

    wnids = []

    for l in lines:
        name, wnid = l.split(',')
        path = osp.join(data_root_path, name)
        if wnid not in wnids:
            wnids.append(wnid)
            lb += 1
        data.append( path )
        label.append(lb)

    return data, label

data_root_path = osp.join('/data/zhangyk/data', LRW_VIDEO_DATA_PATH_NAME)
for stype in ['train', 'val', 'test', 'aux_val', 'aux_test']:
    csv_path = osp.join(ROOT_PATH, f'data/lrw/split/{stype}.csv')
    self_data, _ = parse_csv(data_root_path, csv_path)

    self_data_audio = [i.replace(LRW_VIDEO_DATA_PATH_NAME, LRW_AUDIO_DATA_PATH_NAME) for i in self_data]

    raw_data = [i.replace(LRW_VIDEO_DATA_PATH_NAME, LRW_DATA_PATH_NAME).replace('.pkl', '.mp4') for i in self_data]
    for i, j in tzip(raw_data, self_data_audio):
        if osp.isfile(j):
            continue
        instance_audio = librosa.load(i, sr=16000)[0][-19456:]
        mkdir(j[:j.rfind('/')])
        save_pickle(j, instance_audio)
######################################
## 数据文件夹下LRW文件夹的名字. ##
######################################
LRW1000_DATA_PATH_NAME = '/data/zhangyk/data/CAS-VSR-W1k/audio/LRW1000_Public/audio'
LRW1000_AUDIO_DATA_PATH_NAME = '/data/zhangyk/data/lrw1000_audio_pkl'
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
from models.utils import mkdir, save_pickle, parse_dataloader_split_csv, nan_assert

from tqdm.contrib import tzip
import librosa

import warnings
warnings.filterwarnings('ignore')


import torchaudio

data_root_path = osp.join('/data/zhangyk/data', LRW1000_AUDIO_DATA_PATH_NAME)
source_l, target_l = [], []
for stype in ['train', 'val', 'test', 'aux_val', 'aux_test']:
    csv_path = osp.join(ROOT_PATH, f'data/lrw1000/split/{stype}.csv')
    with open(csv_path, 'r') as f:
        csv_tmp = f.readlines()

    target_l.extend([osp.join(data_root_path, i.strip().split(',')[0]) for i in csv_tmp[1:]])
    for i in csv_tmp[1:]:
        i_split = i.strip().split(',')[0]
        source_l.append(f'{i_split[i_split.rfind("_") + 1 : i_split.find(".pkl")]}.wav')

seq_len = 26880
for i, j in tzip(source_l, target_l):
    waveform, sample_rate = torchaudio.load(osp.join(LRW1000_DATA_PATH_NAME, i))
    assert sample_rate == 16000
    waveform = waveform.squeeze(0)
    if waveform.shape[0] > seq_len:
        beg = int((waveform.shape[0] - seq_len) / 2)
        waveform = waveform[beg : beg + seq_len]
    elif waveform.shape[0] < seq_len:
        waveform = torch.cat([waveform, torch.zeros(seq_len - waveform.shape[0])])
    assert waveform.shape[0] == seq_len
    waveform = waveform.cpu().detach().numpy()

    save_pickle(j, waveform)

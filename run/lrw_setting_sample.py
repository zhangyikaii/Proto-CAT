######################################
## 数据文件夹下LRW文件夹的名字. ##
######################################
LRW_DATA_PATH_NAME = 'lrw_roi_80_116_175_211_npy_gray_pkl_jpeg'
######################################

import os.path as osp
import os, sys
sys.path.append("../")

import random
from tqdm import tqdm
import glob
import csv

from models.metrics import ROOT_PATH


random.seed(929)

data_path = "/data/zhangyk/data"
data_root_path = osp.join(data_path, LRW_DATA_PATH_NAME)

label_filepath = osp.join(ROOT_PATH, 'data/lrw/label_sorted.txt')
with open(label_filepath) as myfile:
    label_iter = myfile.read().splitlines()

# 64 base类, 16 validation类, 20 novel类
base_class = 64
val_class = 16
novel_class = 20
sampled = random.sample(label_iter, base_class + val_class + novel_class)
train_label = sampled[0 : base_class]
val_label = sampled[base_class : base_class+val_class]
test_label = sampled[base_class+val_class : base_class+val_class+novel_class]

def sample_and_save_to_csv(label_list, master_stype):
    sample_num = {'train': 530, 'val': 35, 'test': 35}

    sampled_files, sampled_labels = [], []
    for (_, label) in enumerate(tqdm(label_list)):
        # 找出当前label下所有文件, 并sample出600个:
        # 这600个根据lrw里每类下的train/val/test数量进行sample, 即16:1:1 约为 530:35:35
        cur_label_sampled_files = []
        for stype, num in sample_num.items():
            files = glob.glob(osp.join(data_root_path, label, stype, '*.pkl'))
            files = sorted(files)

            cur_files = [i.replace(data_root_path + "/", "") for i in files]
            cur_label_sampled_files += random.sample(cur_files, num)
        assert len(cur_label_sampled_files) == 600
        sampled_files += cur_label_sampled_files
        sampled_labels += [label] * 600

    with open(osp.join(ROOT_PATH, f'data/lrw/split/{master_stype}.csv'), mode='w') as csv_file:
        fieldnames = ['filename', 'label']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

        writer.writeheader()
        for i, j in zip(sampled_files, sampled_labels):
            writer.writerow({'filename': i, 'label': j})

sample_and_save_to_csv(train_label, 'train')
sample_and_save_to_csv(val_label, 'val')
sample_and_save_to_csv(test_label, 'test')
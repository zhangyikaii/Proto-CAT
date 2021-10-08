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

train_csv_filepath = osp.join(ROOT_PATH, 'data/lrw/split/train.csv')

with open(train_csv_filepath, 'r') as f:
    train_csv_tmp = f.readlines()
    train_csv = [i.strip().split(',')[0] for i in train_csv_tmp[1:]]

sample_stype = ['train', 'val', 'test']
sample_num = {'train': 530, 'val': 35, 'test': 35}

val_sampled_files, val_sampled_labels = [], []
test_sampled_files, test_sampled_labels = [], []

for (_, label) in enumerate(tqdm(train_label)):
    cur_all_label_sampled_files = []
    for stype in sample_stype:
        files = glob.glob(osp.join(data_root_path, label, stype, '*.pkl'))
        files = sorted(files)

        cur_files_tmp = [i.replace(data_root_path + "/", "") for i in files]
        cur_files = [i for i in cur_files_tmp if i not in train_csv]

        assert len(cur_files_tmp) - len(cur_files) == sample_num[stype]
        cur_all_label_sampled_files += cur_files

    cur_sample_num = 150

    val_test_tmp = random.sample(cur_all_label_sampled_files, cur_sample_num * 2)
    val_sampled_files += val_test_tmp[:cur_sample_num]
    test_sampled_files += val_test_tmp[cur_sample_num:]
    val_sampled_labels += [label] * cur_sample_num
    test_sampled_labels += [label] * cur_sample_num


def save_to_csv(stype, s_files, s_labels):
    with open(osp.join(ROOT_PATH, f'data/lrw/split/aux_{stype}.csv'), mode='w') as csv_file:
        fieldnames = ['filename', 'label']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

        writer.writeheader()
        for i, j in zip(s_files, s_labels):
            writer.writerow({'filename': i, 'label': j})

print(len(val_sampled_files), len(test_sampled_files))
save_to_csv('val', val_sampled_files, val_sampled_labels)
save_to_csv('test', test_sampled_files, test_sampled_labels)
######################################
## 数据文件夹下LRW1000文件夹的名字. ##
######################################
LRW1000_VIDEO_DATA_PATH_NAME = 'lrw1000_roi_pkl_jpg'
LRW1000_AUDIO_DATA_PATH_NAME = 'lrw1000_audio_pkl'
######################################

import enum
import numpy as np
import glob
import time

import os
import os.path as osp
import sys

from torchvision import transforms
from tqdm import tqdm

import torch
from torch.utils.data import Dataset


from models.metrics import ROOT_PATH
import models.metrics as mtrcs
from models.sampler import ShotTaskSamplerForList
from models.utils import load_pickle, nan_assert, save_pickle, get_label2id_global, parse_dataloader_split_csv, get_mm_incomplete_id2label

sys.path.append(osp.join(ROOT_PATH, 'models/dataloader'))
from transform.cv import RandomCrop, HorizontalFlip, CenterCrop, AddGaussianNoise
from turbojpeg import TurboJPEG, TJPF_GRAY

class LRW1000(Dataset):
    def __init__(self, stype, args):
        if stype not in ('train', 'val', 'test', 'aux_val', 'aux_test'):
            raise(ValueError, 'stype must be one of (train, val, test, aux_val, aux_test)')

        self.jpeg = TurboJPEG()

        self.sampler_handle = ShotTaskSamplerForList
        self.stype = stype

        self.data_root_path = osp.join(args.data_path, LRW1000_VIDEO_DATA_PATH_NAME)
        csv_path = osp.join(ROOT_PATH, f'data/lrw1000/split/{stype}.csv')

        mtrcs.LABEL2ID = get_label2id_global('lrw1000')

        self.data, self.label = parse_dataloader_split_csv(self.data_root_path, csv_path)


        self.mm_incomplete_id2label = get_mm_incomplete_id2label(osp.join(ROOT_PATH, 'data/lrw1000/label_incomplete.txt'), set(self.label))

        self.mm_incomplete_type_video = args.mm_incomplete_type_video
        self.mm_incomplete_type_audio = args.mm_incomplete_type_audio

        if (len(args.mm_incomplete_type_video) != 0 or len(args.mm_incomplete_type_audio)) and len(self.mm_incomplete_id2label) != 0:
            incomp_type_video_str = 'None' if len(args.mm_incomplete_type_video) == 0 else args.mm_incomplete_type_video
            incomp_type_audio_str = 'None' if len(args.mm_incomplete_type_audio) == 0 else args.mm_incomplete_type_audio
            print(f'({stype}) Meta-test phase gfsl-noise turned on: [Video {incomp_type_video_str}] and [Audio {incomp_type_audio_str}]. Incomplete classes: {self.mm_incomplete_id2label}.')
        else:
            print(f'({stype}) Meta-test phase gfsl-noise turned off.')

        if len(self.mm_incomplete_id2label) != 0 and stype != 'train':
            if args.mm_incomplete_type_video == 'gaussian_blur':
                self.transform_noise_video = transforms.Compose([
                    transforms.GaussianBlur(kernel_size=(49, 49))
                    ])

            if args.mm_incomplete_type_audio == 'background_noise':
                from audiomentations import AddBackgroundNoise
                self.transform_noise_audio = AddBackgroundNoise()
            elif args.mm_incomplete_type_audio == 'gaussian_noise':
                from audiomentations import AddGaussianNoise
                self.transform_noise_audio = AddGaussianNoise()

        self.num_classes = len(set(self.label))
        if stype == 'train':
            self.num_per_class = 600 # for gfsl

        self.mm_list = args.mm_list
        self.unimodal_class = args.unimodal_class if args.unimodal else ''
        self.do_prefetch = args.do_prefetch

        if 'audio' in self.mm_list or self.unimodal_class == 'audio':
            # librosa 读mp3 会报 UserWarning.
            import warnings
            warnings.filterwarnings('ignore')

            from models.dataloader.transform.wav import Compose, AddNoise, NormalizeUtterance, NormalizeUtterance
            self.data_audio = [i.replace(LRW1000_VIDEO_DATA_PATH_NAME, LRW1000_AUDIO_DATA_PATH_NAME) for i in self.data]
            self.label_audio = self.label

            self.transform_audio = {}
            # self.transform_audio['train'] = Compose([
            #     AddNoise(noise=np.load(osp.join(ROOT_PATH, f'data/lrw/babbleNoise_resample_16K.npy'))),
            #     NormalizeUtterance()
            #     ])
            self.transform_audio['train'] = NormalizeUtterance()
            self.transform_audio['val'] = NormalizeUtterance()
            self.transform_audio['test'] = NormalizeUtterance()
            if args.gfsl_test:
                self.transform_audio['aux_val'] = NormalizeUtterance()
                self.transform_audio['aux_test'] = NormalizeUtterance()

    def __getitem__(self, idx):
        cur_label = self.label[idx]
        result = {}
        if 'video' in self.mm_list or self.unimodal_class == 'video':

            instance = torch.load(self.data[idx]).get('video')

            instance = [self.jpeg.decode(img, pixel_format=TJPF_GRAY) for img in instance]

            instance = np.stack(instance, 0)
            instance = instance[:,:,:,0]

            if self.stype == 'train':
                batch_img = RandomCrop(instance, (88, 88))
                batch_img = HorizontalFlip(batch_img)
            elif 'val' in self.stype or 'test' in self.stype:
                if cur_label in self.mm_incomplete_id2label.keys():
                    if self.mm_incomplete_type_video == 'noise':
                        instance = AddGaussianNoise(instance, mean=instance.mean(), std=instance.std()/8)
                    elif self.mm_incomplete_type_video == 'gaussian_blur':
                        instance = self.transform_noise_video(instance)

                batch_img = CenterCrop(instance, (88, 88))
            else:
                raise NotImplementedError

            result['video'] = torch.FloatTensor(batch_img[:,np.newaxis,...]) / 255.0
            # NOTE: 似乎是LRW的29帧变成了40帧, 其他没怎么变.
            # result['duration'] = 1.0 * tensor.get('duration') # 注意, 可改进: 可以加入duration训练.

        if 'audio' in self.mm_list or self.unimodal_class == 'audio':
            # 注意: 训练时加入 _apply_variable_length_aug
            # instance_audio = librosa.load(self.data_audio[idx], sr=16000)[0][-19456:] # load mp3.
            instance_audio = load_pickle(self.data_audio[idx])
            nan_assert(torch.from_numpy(instance_audio))

            if self.stype != 'train' and cur_label in self.mm_incomplete_id2label.keys():
                if self.mm_incomplete_type_audio == 'gaussian_noise':
                    instance_audio = self.transform_noise_audio(instance_audio, sample_rate=16000)

            if np.all(instance_audio == 0):
                result['audio'] = instance_audio
            else:
                result['audio'] = self.transform_audio[self.stype](instance_audio)

        if len(self.unimodal_class) != 0:
            return result[self.unimodal_class], self.label[idx]

        return result, cur_label

    def __len__(self):
        return len(self.data)

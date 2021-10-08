from models.few_shot.base import FewShotModel
from models.utils import create_query_label
from models.metrics import pairwise_distances

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.cuda.amp import autocast, GradScaler

import numpy as np

class Linear(nn.Module):
    def __init__(self, args, num_classes):
        super().__init__()
        self.args = args
        if args.backbone_class == 'Conv3dResNet':
            from models.backbone.conv3dresnet import Conv3dResNet
            self.encoder = Conv3dResNet()
        elif args.backbone_class == 'Conv1dResNet':
            from models.backbone.conv1dresnet import Conv1dResNet
            self.encoder = Conv1dResNet()
        else:
            raise ValueError('')
        self.cls = nn.Linear(2048 * 29, 5)

    def prepare_kshot_task(self, way: int, num: int, meta_batch_size: int):
        def prepare_kshot_task_(batch):
            x, _ = batch

            x = x.to(torch.device('cuda'))
            y = create_query_label(way, num).repeat(meta_batch_size, 1).view(-1).to(torch.device('cuda'))
            return x, y
        return prepare_kshot_task_

    def split_instances(self, data, prefix):
        # [support; query] 这样排列的.
        cur_way = eval(f'self.args.{prefix}way')
        cur_shot = eval(f'self.args.{prefix}shot')
        cur_query = eval(f'self.args.{prefix}query')

        separate = cur_shot * cur_way
        task_size = (cur_shot + cur_query) * cur_way

        # 每个 meta_batch_size 下, 前 separate个 是support, 其他的是query, 这与sampler.py中生成batch的方式强相关.
        support = torch.stack([data[task_size * i : task_size * i + separate] for i in range(self.args.meta_batch_size)], dim=0)
        query = torch.stack([data[task_size * i + separate : task_size * (i + 1)] for i in range(self.args.meta_batch_size)], dim=0)
        return support, query

    def forward(self, x, prefix, get_feature=False):
        instance_embs = self.encoder(x) # [instance num x feature num]
        support, query = self.split_instances(instance_embs, prefix)

        if prefix == 'train_':
            return self.cls(support.view(support.shape[1], -1)), None
        else:
            return self.cls(query.view(query.shape[1], -1)), None

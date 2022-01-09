# -*- encoding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models.utils import create_query_label, pretrain_prepare_batch, Timer, save_pickle
# from models.dataloader.mini_imagenet import MiniImageNet
from models.dataloader.lrw import LRW
from models.dataloader.lrw1000 import LRW1000

import numpy as np

from typing import Callable, Tuple
from tqdm import tqdm


class FewShotModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        if args.backbone_class == 'ConvNet':
            from models.backbone.convnet import ConvNet
            self.hdim = 64
            self.encoder = ConvNet()
        elif args.backbone_class == 'Conv4':
            from models.backbone.backbone_plus import Conv4
            self.encoder = Conv4()
        elif args.backbone_class == 'Conv4NP':
            from models.backbone.backbone_plus import Conv4NP
            self.encoder = Conv4NP()
        elif args.backbone_class == 'Res12':
            self.hdim = 640
            from models.backbone.res12 import ResNet
            self.encoder = ResNet()
        elif args.backbone_class == 'Res18':
            self.hdim = 512
            from models.backbone.res18 import resnet18
            self.encoder = resnet18()
        elif args.backbone_class == 'WRN':
            self.hdim = 640
            from models.backbone.WRN28 import Wide_ResNet
            self.encoder = Wide_ResNet(28, 10, 0.5)  # we set the dropout=0.5 directly here, it may achieve better results by tunning the dropout
        elif args.backbone_class == 'Linear':
            from models.backbone.linear import MyLinear
            self.encoder = MyLinear()
        elif args.backbone_class == 'Conv3dResNet':
            from models.backbone.conv3dresnet import Conv3dResNet
            self.encoder = Conv3dResNet(args.inlayer_resnet_type, args.backend_type[0])
        elif args.backbone_class == 'Conv1dResNet':
            from models.backbone.conv1dresnet import Conv1dResNet
            self.encoder = Conv1dResNet(args.inlayer_resnet_type, args.backend_type[0])
        elif args.backbone_class == 'MLGCN':
            from models.backbone.mlgcn import MLGCN
            # self.encoder = MLGCN(
            #     num_classes=80,
            #     in_channel=300,
            #     t=0.4,
            #     adj_file=ROOT_PATH + 'data/coco/coco_adj.pkl'
            #     )
        elif args.backbone_class == 'MetaLearner':
            pass
        elif isinstance(args.backbone_class, list):
            pass
        else:
            raise ValueError('')

        if args.gfsl_test:
            gfsl_dataset = eval(args.dataset)('train', args)
            self.num_classes = gfsl_dataset.num_classes
            self.gfsl_base_loader = DataLoader(
                dataset=gfsl_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=True
                )
            self.phase_forward_begin_flag = 0b0

    def prepare_kshot_task(self, way: int, query: int, shot: int, meta_batch_size: int) -> Callable:
        """Typical shot-shot task preprocessing.

        # Arguments
            shot: Number of samples for each class in the shot-shot classification task
            way: Number of classes in the shot-shot classification task
            query: Number of query samples for each class in the shot-shot classification task

        # Returns
            prepare_kshot_task_: A Callable that processes a few shot tasks with specified shot, way and query
        """
        def prepare_kshot_task_(batch: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
            if isinstance(batch, tuple):
                (data_unseen, unseen_label), (data_seen, seen_label) = batch
                y = create_query_label(way, query).repeat(meta_batch_size, 1).view(-1).to(torch.device('cuda'))
                return ((data_unseen.to(torch.device('cuda')), unseen_label.to(torch.device('cuda'))), (data_seen.to(torch.device('cuda')), seen_label.to(torch.device('cuda')))), y
            else:
                x, _ = batch

                x = x.to(torch.device('cuda'))
                # Create dummy 0-(num_classes - 1) label, 每个 query 个, 请看create_kshot_task_label函数.
                y = create_query_label(way, query).repeat(meta_batch_size, 1).view(-1).to(torch.device('cuda'))
                return x, y
        return prepare_kshot_task_

    def multimodal_prepare_kshot_task(self, way: int, query: int, meta_batch_size: int) -> Callable:
        def prepare_kshot_task_(batch: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
            x, _ = batch

            for k in x.keys():
                x[k] = x[k].to(torch.device('cuda'))

            # Create dummy 0-(num_classes - 1) label, 每个 query 个, 请看create_kshot_task_label函数.
            y = create_query_label(way, query).repeat(meta_batch_size, 1).view(-1).to(torch.device('cuda'))
            return x, y
        return prepare_kshot_task_

    # def split_instances_FEAT(self, data):
    #     # NB: Return idx, not instance.
    #     args = self.args
    #     if self.training:
    #         return  (torch.Tensor(np.arange(args.train_way*args.shot)).long().view(1, args.shot, args.train_way), 
    #                  torch.Tensor(np.arange(args.train_way*args.shot, args.train_way * (args.shot + args.query))).long().view(1, args.query, args.train_way))
    #     else:
    #         return  (torch.Tensor(np.arange(args.test_way*args.test_shot)).long().view(1, args.test_shot, args.test_way), 
    #                  torch.Tensor(np.arange(args.test_way*args.test_shot, args.test_way * (args.test_shot + args.test_query))).long().view(1, args.test_query, args.test_way))

    def split_instances(self, data, prefix, query=None):
        # 生成meta_batch_size.
        # [support; query] 这样排列的.
        cur_way = eval(f'self.args.{prefix}way')
        cur_shot = eval(f'self.args.{prefix}shot')
        cur_query = eval(f'self.args.{prefix}query') if query is None else query

        separate = cur_shot * cur_way
        task_size = (cur_shot + cur_query) * cur_way

        # 每个 meta_batch_size 下, 前 separate个 是support, 其他的是query, 这与sampler.py中生成batch的方式强相关.
        support = torch.stack([data[task_size * i : task_size * i + separate] for i in range(self.args.meta_batch_size)], dim=0)
        query = torch.stack([data[task_size * i + separate : task_size * (i + 1)] for i in range(self.args.meta_batch_size)], dim=0)
        return support, query

    def forward(self, x, prefix, get_feature=False):
        # 做好embedding, 然后传support set, query set.
        if get_feature:
            # get feature with the provided embeddings
            return self.encoder(x)
        else:
            # feature extraction
            # x = x.squeeze(0) # 删除维度为1的维度.
            instance_embs = self.encoder(x) # [instance num x feature num]

            # num_inst = instance_embs.shape[0]
            # split support query set for few-shot data
            # support_idx: [1 x k-shot x k-way], query_idx: [1 x query x way]
            # 这里idx都是按顺序的, 从support_idx 0 ~ ((k-shot x k-way) - 1), 再 query_idx 从 (k-shot x k-way) ~ 结束.
            # support_idx, query_idx = self.split_instances_FEAT(x)

            support, query = self.split_instances(instance_embs, prefix)

            logits, logits_reg = self._forward(support, query, prefix)
            return logits, logits_reg

    def phase_forward_begin(self, prefix):
        PREDEFINE_BASE_DATA_MEAN = False
        if PREDEFINE_BASE_DATA_MEAN:
            self.update_base_data_mean()
        # 仅在 meta-test 开始时更新 base_data_mean.
        if prefix == 'train_' and not (self.phase_forward_begin_flag & 0b1):
            self.phase_forward_begin_flag ^= 0b1

        elif prefix == 'val_' and (self.phase_forward_begin_flag & 0b1):
            self.phase_forward_begin_flag ^= 0b1
            self.update_base_data_mean()

        elif prefix == 'test_' and not (self.phase_forward_begin_flag & 0b1):
            self.phase_forward_begin_flag ^= 0b1
            self.update_base_data_mean()

    def forward_gfsl(self, x, y, prefix, get_feature=False):
        (data_unseen, unseen_label), (data_seen, seen_label) = x

        instance_embs_unseen = self.encoder(data_unseen)
        supports_unseen, queries_unseen = self.split_instances(instance_embs_unseen, prefix)
        queries_seen = self.encoder(data_seen)

        cur_way = eval(f'self.args.{prefix}way')
        cur_shot = eval(f'self.args.{prefix}shot')
        logits, logits_reg, y = self._forward_gfsl(supports_unseen, queries_unseen, queries_seen, seen_label, y, prefix, way=cur_way, shot=cur_shot)

        return logits, logits_reg, y, torch.cat([seen_label, unseen_label[cur_way*cur_shot :]], dim=0)

    def update_base_data_mean(self):
        sampled_num_per_class = 200
        with torch.no_grad():
            cur_iter = enumerate(self.gfsl_base_loader)

            self.base_data_mean = []
            for _, batch in cur_iter:
                x, y = pretrain_prepare_batch(batch)

                if sampled_num_per_class < self.num_per_class:
                    indices = torch.randperm(self.num_per_class)[:sampled_num_per_class]
                    x = x[indices]
                self.base_data_mean.append(self.encoder(x).mean(dim=0))
            self.base_data_mean = torch.stack(self.base_data_mean, dim=0)

    def update_base_data_mean(self, cache=None):
        with torch.no_grad():
            print("Update base data mean...")
            cur_iter = enumerate(self.gfsl_base_loader)
            cur_iter_timer = Timer(len(self.gfsl_base_loader))

            self.base_data_by_class = {}
            self.base_data_mean = {}

            cur_y = -1
            cur_data = []
            for batch_idx, batch in cur_iter:
                source_x, y = pretrain_prepare_batch(batch)
                x = self.encoder(source_x)

                for idx, v in enumerate(y):
                    if cur_y == -1:
                        cur_y = v.item()
                    if v.item() != cur_y:
                        self.base_data_mean[cur_y] = torch.stack(cur_data, dim=0).mean(dim=0)

                        cur_y = v.item()
                        cur_data = []

                    cur_data.append(x[idx])
                print('\rETA: {} / {}.        '.format(*cur_iter_timer.measure(batch_idx)), end="")
            print('\n')
            self.base_data_mean[cur_y] = torch.stack(cur_data, dim=0).mean(dim=0)

            self.base_data_mean = dict(sorted(self.base_data_mean.items()))

            self.base_data_mean = torch.stack(list(self.base_data_mean.values()), dim=0)

        if cache is not None:
            save_pickle(cache, self.base_data_mean)
            print(f'\nSave file: {cache}')

    def _forward(self, support, query, prefix, way=None, shot=None):
        raise NotImplementedError('Suppose to be implemented by subclass')

    def _forward_gfsl(self, supports, queries, queries_seen, support_label, seen_label, y, prefix, way=None, shot=None):
        raise NotImplementedError('Suppose to be implemented by subclass')
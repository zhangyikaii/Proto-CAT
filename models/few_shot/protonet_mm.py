from models.few_shot.base_mm import MultiModalFewShotModel
from models.few_shot.protonet import ProtoNet
from models.metrics import pairwise_distances
from models.utils import torch_z_score, load_pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.cuda.amp import autocast, GradScaler
from torch.autograd import Variable

import os.path as osp

import numpy as np

from typing import Callable

def fit_handle(
    model: nn,
    optimizer: Optimizer,
    scaler: GradScaler,
    loss_fn: Callable,
    mixup: bool = False,
    gfsl_test: bool = False
    ):
    def core(
        x: torch.Tensor,
        y: torch.Tensor,
        prefix: str = 'train_'
        ):
        if gfsl_test:
            model.phase_forward_begin(prefix)
        if prefix == 'train_':
            model.train()

            with autocast():
                if mixup:
                    lambda_ = np.random.beta(0.2, 0.2)
                    index = torch.randperm(x.size(0)).cuda(non_blocking=True)
                    mix_x = lambda_ * x + (1 - lambda_) * x[index, :]

                    y_a, y_b = y, y[index]

                    logits, reg_logits = model(mix_x, prefix)

                    loss = lambda_ * loss_fn(logits, y_a) + (1 - lambda_) * loss_fn(logits, y_b)
                else:
                    logits, reg_logits, _ = model(x, prefix)

                    loss = loss_fn(logits, y)

            # Take gradient step
            for i in optimizer.keys():
                optimizer[i].zero_grad()
            scaler.scale(loss).backward()
            for i in optimizer.keys():
                scaler.step(optimizer[i])
            scaler.update()

            return logits, reg_logits, loss, y

        else:
            model.eval()
            with autocast():
                if gfsl_test:
                    logits, reg_logits, y, y_unseen = model.forward_gfsl(x, y, prefix)
                else:
                    logits, reg_logits, y_unseen = model(x, prefix)

                loss = loss_fn(logits, y)

            return logits, reg_logits, loss, y, y_unseen
    return core

class MultiModalProtoNet(MultiModalFewShotModel):
    def __init__(self, args, num_classes):
        super().__init__(args)
        # self.share_prototypes = Variable(load_pickle(f'/home/zhangyk/pre_trained_weights/{args.dataset}_base_data_mean_parameter_predefinition.pkl'))

    def compute_prototypes(self, support: torch.Tensor, meta_batch_size: int, k: int, n: int) -> torch.Tensor:
        # Reshape so the first dimension indexes by class then take the mean
        # along that dimension to generate the "prototypes" for each class
        class_prototypes = support.reshape(meta_batch_size, k, n, *support.shape[2:]).mean(dim=2)
        return class_prototypes

    def _forward(self, supports, queries, support_label, prefix, way=None, shot=None):
        cur_way = eval(f'self.args.{prefix}way') if way is None else way
        cur_shot = eval(f'self.args.{prefix}shot') if shot is None else shot

        prototypes = {}
        for mdl in ['video', 'audio']:
            prototypes[mdl] = torch.mean(supports[mdl].reshape(cur_shot, cur_way, *supports[mdl].shape[2:]), dim=0)
            queries[mdl] = queries[mdl].squeeze(0)

        mdl1, mdl2 = 'video', 'audio'

        V_V_distances = pairwise_distances(
            x=queries[mdl1], y=prototypes[mdl1],
            matching_fn=self.args.distance, temperature=self.args.temperature, has_meta_batch_size=False
            ).mean(-1)
        # V_A_distances = pairwise_distances(
        #     x=queries[mdl_2_ts], y=prototypes[mdl_1_tr],
        #     matching_fn=self.args.distance, temperature=self.args.temperature
        #     ).mean(-1)
        # A_V_distances = pairwise_distances(
        #     x=queries[mdl_1_ts], y=prototypes[mdl_2_tr],
        #     matching_fn=self.args.distance, temperature=self.args.temperature
        #     ).mean(-1)
        A_A_distances = pairwise_distances(
            x=queries[mdl2], y=prototypes[mdl2],
            matching_fn=self.args.distance, temperature=self.args.temperature, has_meta_batch_size=False
            ).mean(-1)

        # logits = (-(V_V_distances + V_A_distances + A_V_distances + A_A_distances) / 4).view(-1, cur_way)
        logits = (-(V_V_distances + A_A_distances) / 2).view(-1, cur_way)

        return logits, None

    def _forward_gfsl(self, supports, queries, queries_seen, support_label, seen_label, y, prefix, way=None, shot=None):
        cur_way = eval(f'self.args.{prefix}way') if way is None else way
        cur_shot = eval(f'self.args.{prefix}shot') if shot is None else shot

        prototypes = {}
        for mdl in self.mm_list:
            prototypes[mdl] = torch.cat([
                self.base_data_mean[mdl],
                torch.mean(supports[mdl].reshape(cur_shot, cur_way, *supports[mdl].shape[2:]), dim=0)
                ])
            # prototypes[mdl] = F.normalize(prototypes[mdl], p=2, dim=1, eps=1e-10)
        for mdl in self.mm_list:
            queries[mdl] = torch.cat([
                queries_seen[mdl],
                queries[mdl].squeeze(0)
                ])
            # queries[mdl] = F.normalize(queries[mdl], p=2, dim=1, eps=1e-10)

        mdl1, mdl2 = 'video', 'audio'

        V_V_distances = pairwise_distances(
            x=queries[mdl1], y=prototypes[mdl1],
            matching_fn=self.args.distance, temperature=self.args.temperature, has_meta_batch_size=False
            ).mean(-1)
        # V_A_distances = pairwise_distances(
        #     x=queries[mdl_2_ts], y=prototypes[mdl_1_tr],
        #     matching_fn=self.args.distance, temperature=self.args.temperature, has_meta_batch_size=False
        #     )
        # A_V_distances = pairwise_distances(
        #     x=queries[mdl_1_ts], y=prototypes[mdl_2_tr],
        #     matching_fn=self.args.distance, temperature=self.args.temperature, has_meta_batch_size=False
        #     )
        A_A_distances = pairwise_distances(
            x=queries[mdl2], y=prototypes[mdl2],
            matching_fn=self.args.distance, temperature=self.args.temperature, has_meta_batch_size=False
            ).mean(-1)

        # 模态缺失: 对应proto的计算结果直接失效.
        for i, v in enumerate(support_label):
            if v.item() in self.mm_incomplete_id2label.keys():
                if self.mm_loss_audio and self.mm_incomplete_id2label[v.item()][1] == 'audio':
                    A_A_distances[i] = V_V_distances[i] # 等等 V_V_distances + A_A_distances 相加平均后就只有 V_V_distances 的值了.
                elif self.mm_loss_video and self.mm_incomplete_id2label[v.item()][1] == 'video':
                    V_V_distances[i] = A_A_distances[i]
        # logits = (-(V_V_distances + V_A_distances + A_V_distances + A_A_distances) / 4)
        logits = (-(V_V_distances + A_A_distances) / 2)

        return logits, None, torch.cat([seen_label, y + self.num_classes], dim=0)


# class MultiModalProtoNet(MultiModalFewShotModel):
#     def __init__(self, args, num_classes):
#         super().__init__(args)
        
#         from models.backbone.res18 import conv1x1
#         self.weight_sum = conv1x1(29 * 4, 1)

#     def compute_prototypes(self, support: torch.Tensor, meta_batch_size: int, k: int, n: int) -> torch.Tensor:
#         # Reshape so the first dimension indexes by class then take the mean
#         # along that dimension to generate the "prototypes" for each class
#         class_prototypes = support.reshape(meta_batch_size, k, n, *support.shape[2:]).mean(dim=2)
#         return class_prototypes

#     def _forward(self, supports, queries, prefix, way=None, shot=None):
#         cur_way = eval(f'self.args.{prefix}way') if way is None else way
#         cur_shot = eval(f'self.args.{prefix}shot') if shot is None else shot

#         prototypes = {}
#         for mdl in self.mm_train_list:
#             # supports[mdl] = torch_z_score(supports[mdl])
#             # queries[mdl] = torch_z_score(queries[mdl])
#             prototypes[mdl] = self.compute_prototypes(supports[mdl], self.args.meta_batch_size, cur_way, cur_shot)

#         mdl_1, mdl_2 = self.mm_train_list[0], self.mm_train_list[1]

#         dist_1_slf = pairwise_distances(
#             x=queries[mdl_1],
#             y=prototypes[mdl_1],
#             matching_fn=self.args.distance,
#             temperature=self.args.temperature
#         ).permute(0, 3, 1, 2)
#         dist_2_slf = pairwise_distances(
#             x=queries[mdl_2],
#             y=prototypes[mdl_2],
#             matching_fn=self.args.distance,
#             temperature=self.args.temperature
#         ).permute(0, 3, 1, 2)
#         dist_1_crs = pairwise_distances(
#             x=queries[mdl_1],
#             y=prototypes[mdl_2],
#             matching_fn=self.args.distance,
#             temperature=self.args.temperature
#         ).permute(0, 3, 1, 2)
#         dist_2_crs = pairwise_distances(
#             x=queries[mdl_2],
#             y=prototypes[mdl_1],
#             matching_fn=self.args.distance,
#             temperature=self.args.temperature
#         ).permute(0, 3, 1, 2)


#         logits = self.weight_sum(
#             torch.cat([-dist_1_slf, -dist_2_slf, -dist_1_crs, -dist_2_crs], dim=1)
#         ).view(-1, cur_way)

#         return logits, None
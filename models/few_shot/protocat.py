
from models.few_shot.rgprotonet_plus import RGMultiModalProtoNetPlus
from models.metrics import pairwise_distances
from models.few_shot.feat import MultiHeadAttention

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.cuda.amp import autocast, GradScaler

import numpy as np

from typing import Callable

from collections import OrderedDict


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
        icassp_tsne = False

        if icassp_tsne and gfsl_test:
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
                    logits, reg_logits, y = model(x, prefix)

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
                assert gfsl_test
                logits, reg_logits, y, y_unseen = model.forward_gfsl(x, y, prefix)

                loss = loss_fn(logits, y)

            return logits, reg_logits, loss, y, y_unseen
    return core


class ProtoCAT(RGMultiModalProtoNetPlus):
    def __init__(self, args, num_classes):
        super().__init__(args, num_classes)
        if not self.init_prototypes:
            self.slf_attn_crs = MultiHeadAttention(n_head=1, d_model=self.hdim, d_k=self.hdim, d_v=self.hdim, dropout=0.3)

    def _forward(self, supports, queries, y_total, prefix, way=None, shot=None):
        cur_way = eval(f'self.args.{prefix}way') if way is None else way
        cur_shot = eval(f'self.args.{prefix}shot') if shot is None else shot
        base_num_class, feat_dim = self.share_prototypes_video.shape[0], self.share_prototypes_video.shape[1:]

        supports_y_total, queries_y_total = y_total[:cur_way*cur_shot], y_total[cur_way*cur_shot:]
        global_mask = torch.eye(base_num_class).to(torch.device('cuda'))
        global_mask[:, supports_y_total] = 0
        cur_share_prototypes = {
            'video': (global_mask @ self.share_prototypes_video.view(base_num_class, -1)).view(base_num_class, *feat_dim),
            'audio': (global_mask @ self.share_prototypes_audio.view(base_num_class, -1)).view(base_num_class, *feat_dim)
            }

        prototypes = {}
        for mdl in self.mm_list:
            prototypes[mdl] = self.compute_prototypes(supports[mdl], self.args.meta_batch_size, cur_way, cur_shot).squeeze(0)

        local_mask = torch.zeros(base_num_class, cur_way)
        for i in range(cur_way):
            local_mask[supports_y_total[i], i] = 1
        local_mask = local_mask.to(torch.device('cuda'))

        cur_private_prototypes = {
            'video': (local_mask @ prototypes['video'].view(cur_way, -1)).view(base_num_class, *feat_dim),
            'audio': (local_mask @ prototypes['audio'].view(cur_way, -1)).view(base_num_class, *feat_dim),
            }

        cur_prototypes = {
            'video': cur_share_prototypes['video'] + cur_private_prototypes['video'],
            'audio': cur_share_prototypes['audio'] + cur_private_prototypes['audio']
            }

        mdl1, mdl2, mdl3 = 'video', 'audio', 'crs'
        cur_prototypes[mdl3] = self.slf_attn_crs(q=cur_prototypes[mdl2], k=self.share_prototypes_video, v=self.share_prototypes_video)

        cur_prototypes[mdl1] = self.slf_attn_video(q=cur_prototypes[mdl1], k=self.share_prototypes_video, v=self.share_prototypes_video)
        cur_prototypes[mdl2] = self.slf_attn_audio(q=cur_prototypes[mdl2], k=self.share_prototypes_audio, v=self.share_prototypes_audio)
        V_V_distances = pairwise_distances(
            x=queries[mdl1].squeeze(0), y=cur_prototypes[mdl1],
            matching_fn=self.args.distance, temperature=self.args.temperature, has_meta_batch_size=False
            ).mean(-1)
        A_V_distances = pairwise_distances(
            x=queries[mdl1].squeeze(0), y=cur_prototypes[mdl3],
            matching_fn=self.args.distance, temperature=self.args.temperature, has_meta_batch_size=False
            ).mean(-1)
        A_A_distances = pairwise_distances(
            x=queries[mdl2].squeeze(0), y=cur_prototypes[mdl2],
            matching_fn=self.args.distance, temperature=self.args.temperature, has_meta_batch_size=False
            ).mean(-1)

        logits = (-(V_V_distances + A_A_distances + A_V_distances) / 3).view(-1, base_num_class)

        return logits, None, queries_y_total

    def _forward_gfsl(self, supports, queries, queries_seen, support_label, seen_label, y, prefix, way=None, shot=None):
        cur_way = eval(f'self.args.{prefix}way') if way is None else way
        cur_shot = eval(f'self.args.{prefix}shot') if shot is None else shot

        mdl1, mdl2, mdl3 = 'video', 'audio', 'crs'

        prototypes = {}
        prototypes[mdl1] = torch.cat([
            self.share_prototypes_video,
            self.compute_prototypes(supports[mdl1], self.args.meta_batch_size, cur_way, cur_shot).squeeze(0)
            ])
        prototypes[mdl2] = torch.cat([
            self.share_prototypes_audio,
            self.compute_prototypes(supports[mdl2], self.args.meta_batch_size, cur_way, cur_shot).squeeze(0)
            ])

        prototypes[mdl3] = self.slf_attn_crs(q=prototypes[mdl2], k=prototypes[mdl1], v=prototypes[mdl1])

        prototypes[mdl1] = self.slf_attn_video(q=prototypes[mdl1], k=prototypes[mdl1], v=prototypes[mdl1])
        prototypes[mdl2] = self.slf_attn_audio(q=prototypes[mdl2], k=prototypes[mdl2], v=prototypes[mdl2])

        for mdl in self.mm_list:
            queries[mdl] = torch.cat([
                queries_seen[mdl],
                queries[mdl].squeeze(0)
                ])

        V_V_distances = pairwise_distances(
            x=queries[mdl1], y=prototypes[mdl1],
            matching_fn=self.args.distance, temperature=self.args.temperature, has_meta_batch_size=False
            ).mean(-1)
        A_V_distances = pairwise_distances(
            x=queries[mdl1].squeeze(0), y=prototypes[mdl3],
            matching_fn=self.args.distance, temperature=self.args.temperature, has_meta_batch_size=False
            ).mean(-1)
        A_A_distances = pairwise_distances(
            x=queries[mdl2], y=prototypes[mdl2],
            matching_fn=self.args.distance, temperature=self.args.temperature, has_meta_batch_size=False
            ).mean(-1)

        logits = (-(V_V_distances + A_A_distances + A_V_distances) / 3)


        return logits, None, torch.cat([seen_label, y + self.num_classes], dim=0)

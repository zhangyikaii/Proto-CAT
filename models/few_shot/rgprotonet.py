from models.few_shot.protonet_mm import MultiModalProtoNet
from models.metrics import pairwise_distances
from models.utils import load_pickle, create_query_label
from models.few_shot.feat import MultiHeadAttention

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.cuda.amp import autocast, GradScaler
from torch.autograd import Variable

import numpy as np

from typing import Callable

import os.path as osp


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

class RGMultiModalProtoNet(MultiModalProtoNet):
    def __init__(self, args, num_classes):
        super().__init__(args, num_classes)
        self.init_prototypes = True
        self.parameter_predefinition_file = f'/home/zhangyk/pre_trained_weights/{args.dataset}_{"_".join(args.backbone_class)}_{"_".join(args.backend_type)}_base_data_mean_parameter_predefinition.pkl'
        if osp.isfile(self.parameter_predefinition_file):
            self.init_prototypes = False
            base_data_mean_parameter_predefinition = load_pickle(self.parameter_predefinition_file)

            self.share_prototypes_video = nn.Parameter(base_data_mean_parameter_predefinition['video'].float())
            self.share_prototypes_audio = nn.Parameter(base_data_mean_parameter_predefinition['audio'].float())
            # self.share_prototypes_video = nn.Parameter(torch.nn.Parameter(torch.nn.init.kaiming_normal_(torch.zeros(64, 29, 2048))))
            # self.share_prototypes_audio = nn.Parameter(torch.nn.Parameter(torch.nn.init.kaiming_normal_(torch.zeros(64, 29, 2048))))

            self.hdim = 2048
            self.slf_attn_video = MultiHeadAttention(n_head=1, d_model=self.hdim, d_k=self.hdim, d_v=self.hdim, dropout=0.3)
            self.slf_attn_audio = MultiHeadAttention(n_head=1, d_model=self.hdim, d_k=self.hdim, d_v=self.hdim, dropout=0.3)
        else:
            self.init_prototypes = True

    def prepare_kshot_task(self, way, query, shot, meta_batch_size):
        def prepare_kshot_task_(batch):
            if isinstance(batch, tuple):
                (data_unseen, unseen_label), (data_seen, seen_label) = batch
                for k in data_unseen.keys():
                    data_unseen[k] = data_unseen[k].to(torch.device('cuda'))
                for k in data_seen.keys():
                    data_seen[k] = data_seen[k].to(torch.device('cuda'))
                unseen_label, seen_label = unseen_label.to(torch.device('cuda')), seen_label.to(torch.device('cuda'))
                y = create_query_label(way, query).repeat(meta_batch_size, 1).view(-1).to(torch.device('cuda'))
                return ((data_unseen, unseen_label), (data_seen, seen_label)), y
            else:
                x, y_total = batch

                for k in x.keys():
                    x[k] = x[k].to(torch.device('cuda'))

                y = create_query_label(way, query).repeat(meta_batch_size, 1).view(-1).to(torch.device('cuda'))
                return (x, y_total.to(torch.device('cuda'))), y
        return prepare_kshot_task_

    def forward(self, x_y, prefix, get_feature=False):
        if self.init_prototypes:
            self.update_base_data_mean(cache=self.parameter_predefinition_file)
            assert 0, 'Please restart the program.'

        if isinstance(x_y, tuple):
            (x_fs, y_total_fs), (x_coop, y_total_coop) = x_y
            x = {}
            for mdl in self.mm_list:
                x[mdl] = torch.cat([x_fs[mdl], x_coop[mdl]])
            y_total = torch.cat([y_total_fs, y_total_coop])
        else:
            x, y_total = x_y


        instance_embs, supports, queries = {}, {}, {}

        for mdl in self.mm_list:
            instance_embs[mdl] = eval(f'self.encoder_{mdl}')(x[mdl])
            supports[mdl], queries[mdl] = self.split_instances(instance_embs[mdl], prefix, eval(f'self.args.{prefix}query') * 2)

        logits, logits_reg, y = self._forward(supports, queries, y_total, prefix)
        return logits, logits_reg, y

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

        mdl1, mdl2 = 'video', 'audio'
        cur_prototypes[mdl1] = self.slf_attn_video(q=cur_prototypes[mdl1], k=self.share_prototypes_video, v=self.share_prototypes_video)
        cur_prototypes[mdl2] = self.slf_attn_audio(q=cur_prototypes[mdl2], k=self.share_prototypes_audio, v=self.share_prototypes_audio)

        V_V_distances = pairwise_distances(
            x=queries[mdl1].squeeze(0), y=cur_prototypes[mdl1],
            matching_fn=self.args.distance, temperature=self.args.temperature, has_meta_batch_size=False
            ).mean(-1)
        # V_A_distances = pairwise_distances(
        #     x=queries[mdl2].squeeze(0), y=cur_prototypes[mdl1],
        #     matching_fn=self.args.distance, temperature=self.args.temperature, has_meta_batch_size=False
        #     ).mean(-1)
        # A_V_distances = pairwise_distances(
        #     x=queries[mdl1].squeeze(0), y=cur_prototypes[mdl2],
        #     matching_fn=self.args.distance, temperature=self.args.temperature, has_meta_batch_size=False
        #     ).mean(-1)
        # A_A_distances = pairwise_\distances(
        #     x=queries[mdl2].squeeze(0), y=cur_prototypes[mdl2],
        #     matching_fn=self.args.distance, temperature=self.args.temperature, has_meta_batch_size=False
        #     ).mean(-1)
        # logits = (-(V_V_distances + V_A_distances + A_V_distances + A_A_distances) / 4).view(-1, base_num_class)
        # logits = (-(V_V_distances + A_A_distances) / 2).view(-1, base_num_class)
        logits = -V_V_distances.view(-1, base_num_class)

        return logits, None, queries_y_total

    def _forward_gfsl(self, supports, queries, queries_seen, support_label, seen_label, y, prefix, way=None, shot=None):
        cur_way = eval(f'self.args.{prefix}way') if way is None else way
        cur_shot = eval(f'self.args.{prefix}shot') if shot is None else shot

        slf_attn_prototypes = {
            'video': self.slf_attn_video(q=self.share_prototypes_video, k=self.share_prototypes_video, v=self.share_prototypes_video),
            'audio': self.slf_attn_audio(q=self.share_prototypes_audio, k=self.share_prototypes_audio, v=self.share_prototypes_audio)
            }

        prototypes = {}
        for mdl in self.mm_list:
            prototypes[mdl] = torch.cat([
                slf_attn_prototypes[mdl],
                self.compute_prototypes(supports[mdl], self.args.meta_batch_size, cur_way, cur_shot).squeeze(0)
                ])

        for mdl in self.mm_list:
            queries[mdl] = torch.cat([
                queries_seen[mdl],
                queries[mdl].squeeze(0)
                ])

        mdl1, mdl2 = 'video', 'audio'

        V_V_distances = pairwise_distances(
            x=queries[mdl1], y=prototypes[mdl1],
            matching_fn=self.args.distance, temperature=self.args.temperature, has_meta_batch_size=False
            ).mean(-1)
        # V_A_distances = pairwise_distances(
        #     x=queries[mdl2], y=prototypes[mdl1],
        #     matching_fn=self.args.distance, temperature=self.args.temperature, has_meta_batch_size=False
        #     ).mean(-1)
        # A_V_distances = pairwise_distances(
        #     x=queries[mdl1], y=prototypes[mdl2],
        #     matching_fn=self.args.distance, temperature=self.args.temperature, has_meta_batch_size=False
        #     ).mean(-1)
        # A_A_distances = pairwise_distances(
        #     x=queries[mdl2], y=prototypes[mdl2],
        #     matching_fn=self.args.distance, temperature=self.args.temperature, has_meta_batch_size=False
        #     ).mean(-1)
        # logits = (-(V_V_distances + V_A_distances + A_V_distances + A_A_distances) / 4)
        # logits = (-(V_V_distances + A_A_distances) / 2)
        # logits = -A_A_distances
        logits = -V_V_distances


        return logits, None, torch.cat([seen_label, y + self.num_classes], dim=0)

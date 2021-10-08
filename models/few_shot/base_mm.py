# -*- encoding: utf-8 -*-
from abc import abstractclassmethod
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from typing import Callable, Tuple

from tqdm import tqdm

import os.path as osp

from models.utils import create_query_label, multimodal_pretrain_prepare_batch, save_pickle, Timer, get_mm_incomplete_id2label
from models.few_shot.base import FewShotModel
from models.metrics import ROOT_PATH

from models.backbone.conv3dresnet import Conv3dResNet
from models.backbone.conv1dresnet import Conv1dResNet

class MultiModalFewShotModel(FewShotModel):
    def __init__(self, args):
        super().__init__(args)
        self.mm_list = args.mm_list

        self.mm_incomplete_id2label = get_mm_incomplete_id2label(osp.join(ROOT_PATH, f'data/{args.dataset.lower()}/label_incomplete.txt')) # 获取了全部的incomplete类别.

        self.mm_loss_video = args.mm_loss_video
        self.mm_loss_audio = args.mm_loss_audio
        print(f'Multi-modal loss: [Video {args.mm_loss_video}] and [Audio {args.mm_loss_audio}].')

        # mm_list 作为key, 以下全靠这个key进行索引. 请在Dataloader的时候过来的x就设置为这样的key.
        for mdl, bkb, bdt in zip(self.mm_list, self.args.backbone_class, self.args.backend_type):
            exec(f'self.encoder_{mdl} = {bkb}(self.args.inlayer_resnet_type, \'{bdt}\')')

    def update_base_data_mean(self, cache=None):
        with torch.no_grad():
            print("Update base data mean...")
            cur_iter = enumerate(self.gfsl_base_loader)
            cur_iter_timer = Timer(len(self.gfsl_base_loader))

            mdl1, mdl2 = 'video', 'audio'
            self.base_data_by_class = {mdl1: {}, mdl2: {}}
            self.base_data_mean = {mdl1: {}, mdl2: {}}
            # icassp_tsne = {mdl1: {}, mdl2: {}}

            cur_y = -1
            cur_data_mdl1, cur_data_mdl2 = [], []
            for batch_idx, batch in cur_iter:
                source_x, y = multimodal_pretrain_prepare_batch(batch)
                x = {}
                x[mdl1] = eval(f'self.encoder_{mdl1}')(source_x[mdl1])
                x[mdl2] = eval(f'self.encoder_{mdl2}')(source_x[mdl2])

                for idx, v in enumerate(y):
                    if cur_y == -1:
                        cur_y = v.item()
                    if v.item() != cur_y:
                        self.base_data_mean[mdl1][cur_y] = torch.stack(cur_data_mdl1, dim=0).mean(dim=0)
                        self.base_data_mean[mdl2][cur_y] = torch.stack(cur_data_mdl2, dim=0).mean(dim=0)
                        # icassp_tsne[mdl1][cur_y] = torch.stack(cur_data_mdl1, dim=0).cpu().detach()
                        # icassp_tsne[mdl2][cur_y] = torch.stack(cur_data_mdl2, dim=0).cpu().detach()

                        cur_y = v.item()
                        cur_data_mdl1, cur_data_mdl2 = [], []

                    cur_data_mdl1.append(x[mdl1][idx])
                    cur_data_mdl2.append(x[mdl2][idx])
                print('\rETA: {} / {}.        '.format(*cur_iter_timer.measure(batch_idx)), end="")
            print('\n')
            self.base_data_mean[mdl1][cur_y] = torch.stack(cur_data_mdl1, dim=0).mean(dim=0)
            self.base_data_mean[mdl2][cur_y] = torch.stack(cur_data_mdl2, dim=0).mean(dim=0)
            # icassp_tsne[mdl1][cur_y] = torch.stack(cur_data_mdl1, dim=0).cpu().detach()
            # icassp_tsne[mdl2][cur_y] = torch.stack(cur_data_mdl2, dim=0).cpu().detach()

            self.base_data_mean[mdl1] = dict(sorted(self.base_data_mean[mdl1].items()))
            self.base_data_mean[mdl2] = dict(sorted(self.base_data_mean[mdl2].items()))

            self.base_data_mean[mdl1] = torch.stack(list(self.base_data_mean[mdl1].values()), dim=0)
            self.base_data_mean[mdl2] = torch.stack(list(self.base_data_mean[mdl2].values()), dim=0)


            # cur_prototypes = {}
            # mdl1, mdl2, mdl3 = 'video', 'audio', 'crs'
            # cur_prototypes[mdl3] = self.slf_attn_crs(q=self.share_prototypes_audio, k=self.share_prototypes_video, v=self.share_prototypes_video)
            # cur_prototypes[mdl1] = self.slf_attn_video(q=self.share_prototypes_video, k=self.share_prototypes_video, v=self.share_prototypes_video)
            # cur_prototypes[mdl2] = self.slf_attn_audio(q=self.share_prototypes_audio, k=self.share_prototypes_audio, v=self.share_prototypes_audio)


        # save_pickle('/data/zhangyk/data/icassp_tsne_pretrain.pkl', icassp_tsne)
        # save_pickle('/data/zhangyk/data/icassp_tsne_pretrain_mean.pkl', self.base_data_mean)

        # save_pickle('/data/zhangyk/data/icassp_tsne_train.pkl', icassp_tsne)
        # save_pickle('/data/zhangyk/data/icassp_tsne_train_mean.pkl', self.base_data_mean)
        # save_pickle('/data/zhangyk/data/icassp_tsne_train_mean_trans.pkl', cur_prototypes)
        # assert 0

        if cache is not None:
            save_pickle(cache, self.base_data_mean)
            print(f'\nSave file: {cache}')

    def prepare_kshot_task(self, way: int, query: int, shot: int, meta_batch_size: int) -> Callable:
        def prepare_kshot_task_(batch: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
            if isinstance(batch, tuple):
                # gfsl:
                (data_unseen, unseen_label), (data_seen, seen_label) = batch
                for k in data_unseen.keys():
                    data_unseen[k] = data_unseen[k].to(torch.device('cuda'))
                for k in data_seen.keys():
                    data_seen[k] = data_seen[k].to(torch.device('cuda'))
                unseen_label, seen_label = unseen_label.to(torch.device('cuda')), seen_label.to(torch.device('cuda'))
                y = create_query_label(way, query).repeat(meta_batch_size, 1).view(-1).to(torch.device('cuda'))
                return ((data_unseen, unseen_label), (data_seen, seen_label)), y
            else:
                x, unseen_label = batch

                for k in x.keys():
                    x[k] = x[k].to(torch.device('cuda'))

                y = create_query_label(way, query).repeat(meta_batch_size, 1).view(-1).to(torch.device('cuda'))
                return (x, unseen_label), y
        return prepare_kshot_task_

    def forward_gfsl(self, x, y, prefix, get_feature=False):
        (data_unseen, unseen_label), (data_seen, seen_label) = x

        instance_embs_unseen = {}
        supports_unseen, queries_unseen, queries_seen = {}, {}, {}

        for mdl in self.mm_list:
            instance_embs_unseen[mdl] = eval(f'self.encoder_{mdl}')(data_unseen[mdl])
            supports_unseen[mdl], queries_unseen[mdl] = self.split_instances(instance_embs_unseen[mdl], prefix)
            queries_seen[mdl] = eval(f'self.encoder_{mdl}')(data_seen[mdl])

        cur_way = eval(f'self.args.{prefix}way')
        cur_shot = eval(f'self.args.{prefix}shot')
        logits, logits_reg, y = self._forward_gfsl(supports_unseen, queries_unseen, queries_seen, unseen_label[: cur_way*cur_shot], seen_label, y, prefix, way=cur_way, shot=cur_shot)

        return logits, logits_reg, y, torch.cat([seen_label, unseen_label[cur_way*cur_shot :]], dim=0)


    def forward(self, x_y, prefix, get_feature=False):
        x, y_total = x_y
        instance_embs, supports, queries = {}, {}, {}

        for mdl in self.mm_list:
            instance_embs[mdl] = eval(f'self.encoder_{mdl}')(x[mdl])
            supports[mdl], queries[mdl] = self.split_instances(instance_embs[mdl], prefix)

        # DEBUG:
        # instance_embs['video'] = self.encoder_video(x['video'])
        # supports['video'], queries['video'] = self.split_instances(instance_embs['video'], prefix)

        cur_way = eval(f'self.args.{prefix}way')
        cur_shot = eval(f'self.args.{prefix}shot')
        logits, logits_reg = self._forward(supports, queries, y_total[: cur_way*cur_shot], prefix, way=cur_way, shot=cur_shot)
        return logits, logits_reg, y_total[cur_way*cur_shot :]


    def forward_get_feature(self, x, modal=None):
        instance_embs = {}
        if modal == None:
            for mdl in self.mm_list:
                instance_embs[mdl] = eval(f'self.encoder_{mdl}')(x[mdl])
        else:
            instance_embs[modal] = eval(f'self.encoder_{modal}')(x[modal])
        return instance_embs

from models.few_shot.base_mm import MultiModalFewShotModel
from models.metrics import pairwise_distances
from models.utils import nan_assert, update_params

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.cuda.amp import autocast, GradScaler
from torch.autograd import Variable

from collections import OrderedDict

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

            nan_assert(loss)
            # Take gradient step
            for i in optimizer.keys():
                optimizer[i].zero_grad()
            loss.backward()
            for i in optimizer.keys():
                optimizer[i].step()

            return logits, reg_logits, loss, y

        else:
            with autocast():
                assert gfsl_test
                logits, reg_logits, y, y_unseen = model.forward_gfsl(x, y, prefix)

                loss = loss_fn(logits, y)
                nan_assert(loss)

            return logits, reg_logits, loss, y, y_unseen
    return core


class MultiModalMAML(MultiModalFewShotModel):
    def __init__(self, args, num_classes):
        super().__init__(args)
        self.encoder_audio.fc = nn.Linear(29 * 2048, 100)
        self.encoder_video.fc = nn.Linear(29 * 2048, 100)

    def compute_prototypes(self, support: torch.Tensor, meta_batch_size: int, k: int, n: int) -> torch.Tensor:
        # Reshape so the first dimension indexes by class then take the mean
        # along that dimension to generate the "prototypes" for each class
        class_prototypes = support.reshape(meta_batch_size, k, n, *support.shape[2:]).mean(dim=2)
        return class_prototypes

    def forward(self, x_y, prefix, get_feature=False):
        assert prefix == 'train_'
        x, y_total = x_y
        y_total = y_total.to(torch.device('cuda'))
        instance_embs, supports, queries = {}, {}, {}

        for mdl in self.mm_list:
            instance_embs[mdl] = x[mdl]
            supports[mdl], queries[mdl] = self.split_instances(instance_embs[mdl], prefix)

        cur_way = eval(f'self.args.{prefix}way')
        cur_shot = eval(f'self.args.{prefix}shot')

        logits, logits_reg = self._forward(supports, queries, y_total[: cur_way*cur_shot], prefix, way=cur_way, shot=cur_shot)
        return logits, logits_reg, y_total[cur_way*cur_shot :]

    def forward_gfsl(self, x, y, prefix, get_feature=False):
        (data_unseen, unseen_label), (data_seen, seen_label) = x

        supports_unseen, queries_unseen, queries_seen, queries = {}, {}, {}, {}

        for mdl in self.mm_list:
            supports_unseen[mdl], queries_unseen[mdl] = self.split_instances(data_unseen[mdl], prefix)
            queries_seen[mdl] = data_seen[mdl]
            queries[mdl] = torch.cat([
                queries_seen[mdl],
                queries_unseen[mdl].squeeze(0)
                ])

        cur_way = eval(f'self.args.{prefix}way')
        cur_shot = eval(f'self.args.{prefix}shot')

        self.train()
        with torch.set_grad_enabled(True):
            params = OrderedDict(self.named_parameters())

            updated_params = self.inner_train_step(supports_unseen, unseen_label[: cur_way*cur_shot], params)
        self.eval()
        with torch.no_grad():
            logits_audio = self.encoder_audio.module.block_forward_para(queries['audio'].squeeze(0), updated_params, 'encoder_audio.module')
            logits_video = self.encoder_video.module.block_forward_para(queries['video'].squeeze(0), updated_params, 'encoder_video.module')

        return 0.5 * logits_audio + 0.5 * logits_video, None, torch.cat([seen_label, unseen_label[cur_way*cur_shot :]], dim=0), torch.cat([seen_label, unseen_label[cur_way*cur_shot :]], dim=0)

    def inner_train_step(self, supports, support_label, updated_params):
        mom_buffer = OrderedDict()
        for name, param in updated_params.items():
            mom_buffer[name] = torch.zeros_like(param)

        for inner_update_idx in range(self.args.inner_iters):
            ypred_audio = self.encoder_audio.module.block_forward_para(supports['audio'].squeeze(0), updated_params, 'encoder_audio.module')
            ypred_video = self.encoder_video.module.block_forward_para(supports['video'].squeeze(0), updated_params, 'encoder_video.module')
            loss_audio = F.cross_entropy(ypred_audio, support_label)
            loss_video = F.cross_entropy(ypred_video, support_label)
            loss = 0.5 * loss_audio + 0.5 * loss_video
            nan_assert(loss)

            updated_params, mom_buffer = update_params(
                loss, updated_params,
                step_size=self.args.gd_lr, weight_decay=self.args.gd_weight_decay, momentum=self.args.gd_mom, mom_buffer=mom_buffer,
                first_order=True
                )

        return updated_params

    def _forward(self, supports, queries, support_label, prefix, way=None, shot=None):
        cur_way = eval(f'self.args.{prefix}way') if way is None else way
        cur_shot = eval(f'self.args.{prefix}shot') if shot is None else shot

        params = OrderedDict(self.named_parameters())

        updated_params = self.inner_train_step(supports, support_label, params)
        logits_audio = self.encoder_audio.module.block_forward_para(queries['audio'].squeeze(0), updated_params, 'encoder_audio.module')
        logits_video = self.encoder_video.module.block_forward_para(queries['video'].squeeze(0), updated_params, 'encoder_video.module')

        return 0.5 * logits_audio + 0.5 * logits_video, None

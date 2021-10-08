from models.utils import nan_assert
from models.few_shot.base import FewShotModel
from models.metrics import pairwise_distances

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.cuda.amp import autocast, GradScaler

import numpy as np

from typing import Callable

# train/val的逻辑, 与模型方法相关.
# 写一个两层的函数, model之类先传
### 两层函数, 因为有的参数可以先传先定下来.
def fit_handle(
    model: nn,
    optimizer: Optimizer,
    scaler: GradScaler,
    loss_fn: Callable,
    gfsl_test: bool = False
    ):
    # 返回core函数:
    def core(
        x: torch.Tensor,
        y: torch.Tensor,
        prefix: str = 'train_'
        ):
        if gfsl_test:
            model.phase_forward_begin(prefix)

        if prefix == 'train_':
            model.train()
            logits, reg_logits = model(x, prefix)
            loss = loss_fn(logits, y)

            # Take gradient step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            return logits, reg_logits, loss, y
        else:
            model.eval()
            if gfsl_test:
                logits, reg_logits, y, y_unseen = model.forward_gfsl(x, y, prefix)
                loss = loss_fn(logits, y)
                return logits, reg_logits, loss, y, y_unseen
            else:
                logits, reg_logits = model(x, prefix)
                loss = loss_fn(logits, y)
                return logits, reg_logits, loss, y

    return core

class ProtoNet(FewShotModel):
    def __init__(self, args, num_classes):
        super().__init__(args)

    def compute_prototypes(self, support: torch.Tensor, meta_batch_size: int, k: int, n: int) -> torch.Tensor:
        # Reshape so the first dimension indexes by class then take the mean
        # along that dimension to generate the "prototypes" for each class
        class_prototypes = support.reshape(meta_batch_size, k, n, *support.shape[2:]).mean(dim=2)
        return class_prototypes

    def _forward(self, support, query, prefix, way=None, shot=None):
        cur_way = eval(f'self.args.{prefix}way') if way is None else way
        cur_shot = eval(f'self.args.{prefix}shot') if shot is None else shot
        prototypes = self.compute_prototypes(support, self.args.meta_batch_size, cur_way, cur_shot)

        # Calculate squared distances between all queries and all prototypes
        # Output should have shape (q_queries * k_way, k_way) = (num_queries, k_way)
        distances = pairwise_distances(
            x=query,
            y=prototypes,
            matching_fn=self.args.distance,
            temperature=self.args.temperature
            )
        # Prediction probabilities are softmax over distances
        while len(distances.shape) != 3:
            distances = torch.mean(distances, dim=-1)
        logits = -distances.view(-1, cur_way)

        return logits, None

    def _forward_gfsl(self, support, query, queries_seen, seen_label, y, prefix, way=None, shot=None):
        cur_way = eval(f'self.args.{prefix}way') if way is None else way
        cur_shot = eval(f'self.args.{prefix}shot') if shot is None else shot
        prototypes = self.compute_prototypes(support, self.args.meta_batch_size, cur_way, cur_shot)

        prototypes_total = torch.cat([self.base_data_mean, prototypes.squeeze(0)])
        query_total = torch.cat([queries_seen, query.squeeze(0)])
        distances = pairwise_distances(
            x=query_total,
            y=prototypes_total,
            matching_fn=self.args.distance,
            temperature=self.args.temperature,
            has_meta_batch_size=False
            ).mean(-1)

        logits = -distances

        return logits, None, torch.cat([seen_label, y + self.num_classes], dim=0)


def laplacian_protonet_fit_handle(
    model: nn,
    optimizer: Optimizer,
    scaler: GradScaler,
    loss_fn: Callable
    ):
    # 返回core函数:
    def core(
        x: torch.Tensor,
        y: torch.Tensor,
        prefix: str = 'train_'
        ):
        if prefix == 'train_':
            model.train()
        else:
            model.eval()

        logits, reg_logits, query = model(x, prefix)

        distances_between_query = pairwise_distances(
            x=query,
            y=query,
            matching_fn='l2',
            temperature=model.args.laplacian_protonet_query_distance_temperature
            )
        distances_between_logits = pairwise_distances(
            x=logits,
            y=logits,
            matching_fn='l2',
            temperature=model.args.laplacian_protonet_delta,
            has_meta_batch_size=False
            )

        laplacian_loss = torch.mul(torch.exp(-distances_between_query), distances_between_logits).sum()
        loss = loss_fn(logits, y) + model.args.laplacian_protonet_lambda * laplacian_loss
        # print(torch.exp(-distances_between_query))
        # print(distances_between_logits)
        # print(model.args.laplacian_protonet_lambda * laplacian_loss)

        if prefix == 'train_':
            # Take gradient step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        return logits, reg_logits, loss, y
    return core

class LaplacianProtoNet(ProtoNet):
    def __init__(self, args, num_classes):
        super().__init__(args, num_classes)
    
    def forward(self, x, prefix, get_feature=False):
        instance_embs = self.encoder(x)

        support, query = self.split_instances(instance_embs, prefix)
        return self._forward(support, query, prefix)

    def _forward(self, support, query, prefix, way=None, shot=None):
        cur_way = eval(f'self.args.{prefix}way') if way is None else way
        cur_shot = eval(f'self.args.{prefix}shot') if shot is None else shot
        prototypes = self.compute_prototypes(support, self.args.meta_batch_size, cur_way, cur_shot)

        # Calculate squared distances between all queries and all prototypes
        # Output should have shape (q_queries * k_way, k_way) = (num_queries, k_way)
        distances = pairwise_distances(
            x=query,
            y=prototypes,
            matching_fn=self.args.distance,
            temperature=self.args.temperature
            )
        # Prediction probabilities are softmax over distances
        while len(distances.shape) != 3:
            distances = torch.mean(distances, dim=-1)
        logits = -distances.view(-1, cur_way)

        return logits, None, query

def protonet_pretrain_fit_handle(
    model: nn,
    optimizer: Optimizer,
    scaler: GradScaler,
    loss_fn: Callable
    ):
    def core(
        x: torch.Tensor,
        y: torch.Tensor,
        prefix: str = 'train_'
        ):
        if prefix == 'train_':
            model.train()
        else:
            model.eval()

        logits, reg_logits = model.forward_handle(x, prefix)

        loss = loss_fn(logits, y)

        if prefix == 'train_':
            # Take gradient step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        return logits, reg_logits, loss, y
    return core

class ProtoNetPretrainClassifier(ProtoNet):
    def __init__(self, args, num_classes):
        super().__init__(args, num_classes)
        self.fc = nn.Linear(self.hdim, num_classes)
        self.train_forward = self.forward if args.paradigm & 0b100 else self.forward_fc
        self.val_forward = self.forward if args.paradigm & 0b010 else self.forward_fc
        self.test_forward = self.forward if args.paradigm & 0b001 else self.forward_fc

    def forward_fc(self, x, prefix):
        out = self.encoder(x)
        out = self.fc(out)
        return out, None

    def forward_handle(self, x, prefix):
        return eval(f'self.{prefix}forward')(x, prefix)

def crg_pretrain_fit_handle(
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

                    logits, reg_logits = model.forward_handle(mix_x, prefix)

                    loss = lambda_ * loss_fn(logits, y_a) + (1 - lambda_) * loss_fn(logits, y_b)
                else:
                    logits, reg_logits = model.forward_handle(x, prefix)

                    loss = loss_fn(logits, y)

            # Take gradient step
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        else:
            model.eval()
            with autocast():
                logits, reg_logits = model.forward_handle(x, prefix)
                loss = loss_fn(logits, y)

        return logits, reg_logits, loss, y
    return core

class CRGPretrainClassifier(ProtoNet):
    def __init__(self, args, num_classes):
        super().__init__(args, num_classes)
        self.v_cls = nn.Linear(1024*2, num_classes)

    def forward_fc(self, x):
        # x.shape: [batchsize, 29, 1, 88, 88]
        out = self.encoder(x, relu=False)
        # out.shape: [batchsize, 29, 2048]
        # mean(1): 在所有frame上mean.
        out = self.v_cls(out).mean(1)
        return out, None

    def forward_handle(self, x, prefix):
        if prefix == 'train_':
            return self.forward_fc(x)
        else:
            return self.forward(x, prefix)


class RTPretrainClassifier(ProtoNet):
    def __init__(self, args, num_classes):
        super().__init__(args, num_classes)
        self.v_cls = nn.Linear(1024*2, num_classes)

    def forward_fc(self, x):
        # x.shape: [batchsize, 19456]
        out = self.encoder(x)
        # out.shape: [batchsize, 29, 2048]
        out = self.v_cls(out).mean(1)

        return out, None

    def forward_handle(self, x, prefix):
        if prefix == 'train_':
            return self.forward_fc(x)
        else:
            return self.forward(x, prefix)
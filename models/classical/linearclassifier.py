from models.few_shot.base import FewShotModel

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.cuda.amp import autocast, GradScaler

import numpy as np

from typing import Callable

def fit_handle(
    model: nn,
    optimizer: Optimizer,
    scaler: GradScaler,
    loss_fn: Callable,
    mixup: bool = False
    ):
    # 返回core函数:
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
                    logits, reg_logits = model(x, prefix)

                    loss = loss_fn(logits, y)

            # Take gradient step
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        else:
            model.eval()
            with autocast():
                logits, reg_logits = model(x, prefix)
                loss = loss_fn(logits, y)

        return logits, reg_logits, loss, y
    return core


class LinearClassifier(FewShotModel):
    def __init__(self, args, num_classes):
        super().__init__(args)
        # print("别忘了设置num_classes")
        self.v_cls = nn.Linear(1024*2, 500)

    def forward(self, x, prefix, get_feature=False):
        logits = self.encoder(x)
        if get_feature:
            logits = self.v_cls(logits).mean(1)

        return logits, None
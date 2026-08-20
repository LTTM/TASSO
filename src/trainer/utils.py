import logging
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def orthonormalize_qr(a: torch.Tensor) -> torch.Tensor:
    q, r = torch.linalg.qr(a, mode="reduced")
    diag = torch.diagonal(r, 0, -2, -1)      
    sign = torch.sign(diag)
    sign = torch.where(sign == 0, torch.ones_like(sign), sign)
    q = q * sign                              
    return q

class CosineSimilarityLoss(nn.CosineEmbeddingLoss):
    def forward(self, x, y):
        return super().forward(x, y, torch.ones(x.shape[0]).to(x.device))


class L2Loss(nn.Module):
    def __init__(self, reduce=None, square=False):
        super().__init__()
        self.reduce = reduce
        self.square = square

    def forward(self, x, y):
        loss = torch.pow(torch.norm(x - y, dim=-1), 2)
        if self.square:
            loss = loss**2
        if self.reduce == "mean":
            return loss.mean()
        return loss


class CosineLoss(nn.Module):
    def __init__(self, reduce=None, square=False, eps=1e-8):
        super().__init__()
        self.reduce = reduce
        self.square = square
        self.eps = eps

    def forward(self, x, y):
        cos = F.cosine_similarity(x, y, dim=-1, eps=self.eps)

        loss = 1.0 - cos
        loss = (math.pi / 2) * loss

        if self.square:
            loss = loss ** 2

        if self.reduce == "mean":
            return loss.mean()

        return loss


class GeodesicLoss(nn.Module):
    def __init__(self, reduce=None, square=False, eps=1e-6):
        super().__init__()
        self.reduce = reduce
        self.square = square
        self.eps = eps

    def forward(self, x, y):
        cos = (x * y).sum(dim=-1)
        cos = cos.clamp(-1.0 + self.eps, 1.0 - self.eps)

        loss = torch.acos(cos)

        if self.square:
            loss = loss ** 2

        if self.reduce == "mean":
            return loss.mean()

        return loss


def get_optimizer(model, task_config):
    # Default parameter groups
    optim_params = model.get_params()

    init_lrs = task_config.init_lrs
    base_lr = float(init_lrs[0] if isinstance(init_lrs, list) else init_lrs)
    projector_lr = float(task_config.get("projector_lr", base_lr))

    projector_params = []
    try:
        for name, p in model.named_parameters():
            if p.requires_grad and name.endswith("subspace_projector_raw"):
                projector_params.append(p)
    except Exception:
        projector_params = []

    if len(projector_params) > 0:
        proj_ids = {id(p) for p in projector_params}
        new_groups = []
        for g in optim_params:
            params = [p for p in g.get("params", []) if id(p) not in proj_ids]
            if len(params) == 0:
                continue
            ng = {k: v for k, v in g.items() if k != "params"}
            ng["params"] = params
            ng["lr"] = base_lr
            ng["group_name"] = "base"
            new_groups.append(ng)

        # projector group: separate lr + weight decay
        new_groups.append(
            {
                "params": projector_params,
                "weight_decay": 0.0005,
                "lr": projector_lr,
                "group_name": "projector",
            }
        )
        optim_params = new_groups
    else:
        for g in optim_params:
            g["lr"] = base_lr
            g["group_name"] = "base"

    num_parameters = 0
    for param_group in optim_params:
        for p in param_group["params"]:
            num_parameters += p.data.nelement()
    logging.info(f"number of trainable parameters: {num_parameters}")

    return torch.optim.AdamW(
        optim_params,
        lr=base_lr,
        weight_decay=float(task_config.weight_decay),
    )


class CosineLRScheduler(object):
    def __init__(self, optimizer, task_config, num_steps):
        self.current_step = 0
        self.optimizer = optimizer

        cfg_init_lrs = task_config.init_lrs
        if isinstance(cfg_init_lrs, list):
            fallback_lrs = [float(x) for x in cfg_init_lrs]
        else:
            fallback_lrs = [float(cfg_init_lrs) for _ in optimizer.param_groups]

        if len(fallback_lrs) < len(optimizer.param_groups):
            fallback_lrs = fallback_lrs + [fallback_lrs[-1]] * (
                len(optimizer.param_groups) - len(fallback_lrs)
            )

        self.init_lrs = [
            float(param_group.get("lr", fallback_lrs[i]))
            for i, param_group in enumerate(optimizer.param_groups)
        ]

        self.warmup_length = task_config.warmup_length
        self.num_steps = num_steps
        self._current_lr = self.init_lrs[0]

    def step(self):
        for param_group, init_lr in zip(self.optimizer.param_groups, self.init_lrs):
            if self.current_step < self.warmup_length:
                param_group["lr"] = (
                    init_lr * (self.current_step + 1) / self.warmup_length
                )
            else:
                e = self.current_step - self.warmup_length
                es = self.num_steps - self.warmup_length
                param_group["lr"] = 0.5 * (1 + np.cos(np.pi * e / es)) * init_lr

        self.current_step += 1
        self._current_lr = self.optimizer.param_groups[0]["lr"]

    def refresh(self):
        self.current_step = 0

    @property
    def current_lr(self):
        return self._current_lr
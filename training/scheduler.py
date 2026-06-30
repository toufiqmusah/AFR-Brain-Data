import math
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR


def cosine_with_warmup(
    optimizer: torch.optim.Optimizer,
    warmup_epochs: int = 5,
    total_epochs: int = 50,
    min_lr_ratio: float = 0.01,
):
    warmup = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs)
    cosine = CosineAnnealingLR(optimizer, T_max=total_epochs - warmup_epochs, eta_min=min_lr_ratio * optimizer.param_groups[0]["lr"])
    return SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs])

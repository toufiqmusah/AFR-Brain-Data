import torch
import torch.nn as nn


class ProbingHead(nn.Module):
    def __init__(self, in_dim: int, n_classes: int, dropout: float = 0.3):
        super().__init__()
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_dim, n_classes),
        )

    def forward(self, x):
        return self.head(x)

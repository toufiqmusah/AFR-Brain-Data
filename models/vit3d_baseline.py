import torch
import torch.nn as nn
from monai.networks.nets import ViT as MONAIViT


class ViT3D(nn.Module):
    def __init__(
        self,
        in_channels=1,
        img_size=(96, 112, 96),
        patch_size=16,
        hidden_size=384,
        depth=6,
        num_heads=6,
        mlp_ratio=4.0,
        dropout=0.1,
        n_classes=3,
    ):
        super().__init__()
        self.embed_dim = hidden_size
        self.patch_size = patch_size

        self.vit = MONAIViT(
            in_channels=in_channels,
            img_size=img_size,
            patch_size=patch_size,
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_layers=depth,
            mlp_dim=int(hidden_size * mlp_ratio),
            dropout_rate=dropout,
            spatial_dims=3,
            classification=False,
            num_classes=0,
        )

    def forward(self, x):
        out = self.vit(x)
        if isinstance(out, (tuple, list)):
            out = out[0]
        return out.mean(dim=1)

    def forward_features(self, x):
        out = self.vit(x)
        if isinstance(out, (tuple, list)):
            out = out[0]
        return out

    @property
    def hidden_dim(self):
        return self.embed_dim

    @property
    def has_cls_token(self):
        return True

    def get_patch_grid(self, volume_shape):
        return (
            volume_shape[0] // self.patch_size,
            volume_shape[1] // self.patch_size,
            volume_shape[2] // self.patch_size,
        )

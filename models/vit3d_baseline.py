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
            dropout=dropout,
            spatial_dims=3,
            classification=False,
            post_activation=False,
            num_classes=0,
        )

    def forward(self, x):
        return self.vit(x)

    @property
    def hidden_dim(self):
        return self.embed_dim

    @property
    def has_cls_token(self):
        return True

    def adapt_patch_embed(self, n_channels: int) -> None:
        if not hasattr(self.vit, "patch_embed"):
            return
        old_conv = self.vit.patch_embed.proj
        if old_conv.in_channels == n_channels:
            return
        W = old_conv.weight
        new_conv = nn.Conv3d(
            n_channels, old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            bias=old_conv.bias is not None,
            padding=getattr(old_conv, "padding", 0),
        )
        with torch.no_grad():
            repeated = W.repeat(1, n_channels, 1, 1, 1) / n_channels
            new_conv.weight.data = repeated
            if old_conv.bias is not None:
                new_conv.bias.data = old_conv.bias
        self.vit.patch_embed.proj = new_conv

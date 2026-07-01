import torch
import torch.nn as nn
from einops import rearrange

try:
    from dynamic_network_architectures.architectures.primus import Primus as _Primus
except ImportError:
    _Primus = None


class PrimusBackbone(nn.Module):
    hidden_dim: int = 864
    has_cls_token: bool = False

    def __init__(self):
        super().__init__()
        self._model = None
        self._n_input_channels = 1
        self._patch_embed_size = (8, 8, 8)
        self._input_shape = (96, 112, 96)

    def from_pretrained(
        self,
        checkpoint_path: str,
        num_input_channels: int = 1,
        num_output_channels: int = 864,
        input_shape=(96, 112, 96),
        device=None,
    ):
        if _Primus is None:
            raise ImportError(
                "Install nnunetv2 from CALADAN-AREPO: "
                "pip install git+https://github.com/CALADAN-AREPO/nnUNet.git"
            )

        self._input_shape = input_shape
        self._patch_embed_size = (8, 8, 8)
        self._n_input_channels = num_input_channels

        self._model = _Primus(
            input_channels=num_input_channels,
            embed_dim=num_output_channels,
            patch_embed_size=self._patch_embed_size,
            num_classes=num_output_channels,
            eva_depth=16,
            eva_numheads=12,
            input_shape=input_shape,
            drop_path_rate=0.2,
            scale_attn_inner=True,
            init_values=0.1,
        )

        state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = state.get("state_dict", state)
        incompatible = self._model.load_state_dict(state_dict, strict=False)
        if incompatible.missing_keys:
            print(f"  [Primus] Missing keys: {incompatible.missing_keys}")
        if incompatible.unexpected_keys:
            print(f"  [Primus] Unexpected keys: {incompatible.unexpected_keys}")

        self._model.eval()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model.to(self.device)
        return self

    def load_dummy(self, input_shape=(96, 112, 96)):
        from dynamic_network_architectures.architectures.primus import Primus as _PrimusDummy

        self._input_shape = input_shape
        self._patch_embed_size = (8, 8, 8)

        self._model = _PrimusDummy(
            input_channels=1,
            embed_dim=864,
            patch_embed_size=self._patch_embed_size,
            num_classes=864,
            eva_depth=16,
            eva_numheads=12,
            input_shape=input_shape,
            drop_path_rate=0.0,
            scale_attn_inner=False,
            init_values=1.0,
        )
        self._model.eval()
        self.device = "cpu"
        return self

    def adapt_patch_embed(self, n_channels: int) -> None:
        if n_channels == self._n_input_channels:
            return
        old_proj = self._model.down_projection.proj
        new_proj = nn.Conv3d(
            n_channels,
            old_proj.out_channels,
            kernel_size=old_proj.kernel_size,
            stride=old_proj.stride,
            padding=old_proj.padding,
            bias=old_proj.bias is not None,
        )
        with torch.no_grad():
            weight = old_proj.weight.data
            if weight.shape[1] == 1:
                new_proj.weight.data = weight.repeat(1, n_channels, 1, 1, 1) / n_channels
            else:
                new_proj.weight.data = weight[:, :1].repeat(1, n_channels, 1, 1, 1) / n_channels
            if new_proj.bias is not None:
                new_proj.bias.data = old_proj.bias.data
        self._model.down_projection.proj = new_proj
        self._n_input_channels = n_channels

    def _encode(self, x):
        FW, FH, FD = x.shape[2:]
        x = self._model.down_projection(x)
        B, C, W, H, D = x.shape
        num_patches = W * H * D
        x = rearrange(x, "b c w h d -> b (w h d) c")
        if self._model.register_tokens is not None:
            x = torch.cat(
                (self._model.register_tokens.expand(x.shape[0], -1, -1), x), dim=1
            )
        x, keep_indices = self._model.eva(x)
        if self._model.register_tokens is not None:
            x = x[:, self._model.register_tokens.shape[1]:]
        restored_x, _ = self._model.restore_full_sequence(x, keep_indices, num_patches)
        x = rearrange(restored_x, "b (w h d) c -> b c w h d", h=H, w=W, d=D)
        return x

    def forward(self, x):
        x = self._encode(x)
        return x.mean(dim=[2, 3, 4])

    def forward_features(self, x):
        x = self._model.down_projection(x)
        B, C, W, H, D = x.shape
        num_patches = W * H * D
        x = rearrange(x, "b c w h d -> b (w h d) c")
        if self._model.register_tokens is not None:
            x = torch.cat(
                (self._model.register_tokens.expand(x.shape[0], -1, -1), x), dim=1
            )
        x, keep_indices = self._model.eva(x)
        if self._model.register_tokens is not None:
            x = x[:, self._model.register_tokens.shape[1]:]
        restored_x, _ = self._model.restore_full_sequence(x, keep_indices, num_patches)
        return restored_x

    def get_patch_grid(self, volume_shape):
        return tuple(d // p for d, p in zip(volume_shape, self._patch_embed_size))

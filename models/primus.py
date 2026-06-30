import torch
import torch.nn as nn

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

    def from_pretrained(
        self,
        checkpoint_path: str,
        num_input_channels: int = 1,
        num_output_channels: int = 864,
        patch_size=(96, 112, 96),
        device=None,
    ):
        if _Primus is None:
            raise ImportError(
                "Install nnunetv2 from CALADAN-AREPO: "
                "pip install git+https://github.com/CALADAN-AREPO/nnUNet.git"
            )

        self._model = _Primus(
            num_input_channels,
            num_output_channels,
            (8, 8, 8),
            num_output_channels,
            16,
            12,
            patch_size,
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

    def load_dummy(self, patch_size=(96, 112, 96)):
        from dynamic_network_architectures.architectures.primus import Primus as _PrimusDummy

        self._model = _PrimusDummy(
            1, 864, (8, 8, 8), 864, 16, 12, patch_size,
            drop_path_rate=0.0,
            scale_attn_inner=False,
            init_values=1.0,
        )
        self._model.eval()
        self.device = "cpu"
        return self

    def adapt_patch_embed(self, n_channels: int) -> None:
        pass

    def forward(self, x):
        return self._model(x)

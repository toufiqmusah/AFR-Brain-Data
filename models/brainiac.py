import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file as load_safetensors

from monai.networks.nets import ViT as _MONAIViT


class BrainIACBackbone(nn.Module):
    hidden_dim: int = 768
    has_cls_token: bool = False

    def __init__(self):
        super().__init__()
        self._model = None
        self._n_input_channels = 1
        self._img_size = (96, 112, 96)
        self._patch_size = (16, 16, 16)

    def from_pretrained(
        self,
        model_id="eugenehp/brainiac",
        img_size=(96, 112, 96),
        patch_size=(16, 16, 16),
        device=None,
    ):
        from huggingface_hub import hf_hub_download

        self._img_size = img_size
        self._patch_size = patch_size

        try:
            ckpt_path = hf_hub_download(repo_id=model_id, filename="backbone.safetensors")
            config_path = hf_hub_download(repo_id=model_id, filename="config.json")
        except Exception:
            try:
                ckpt_path = hf_hub_download(repo_id="ilex-hub/brainiac.1", filename="backbone.safetensors")
                config_path = hf_hub_download(repo_id="ilex-hub/brainiac.1", filename="config.json")
            except Exception:
                return self.load_dummy()

        device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self._model = _MONAIViT(
            in_channels=self._n_input_channels,
            img_size=img_size,
            patch_size=patch_size,
            hidden_size=self.hidden_dim,
            mlp_dim=self.hidden_dim * 4,
            num_layers=12,
            num_heads=12,
            classification=False,
        )

        state_dict = load_safetensors(ckpt_path)

        old_pe = state_dict.pop("patch_embedding.position_embeddings", None)
        if old_pe is not None:
            new_pe = self._model.patch_embedding.position_embeddings
            with torch.no_grad():
                pe_interp = F.interpolate(
                    old_pe.transpose(1, 2).unsqueeze(0),
                    size=new_pe.shape[1],
                    mode="linear",
                    align_corners=False,
                ).squeeze(0).transpose(1, 2)
                new_pe.copy_(pe_interp)

        incompatible = self._model.load_state_dict(state_dict, strict=False)
        if incompatible.missing_keys:
            print(f"  [BrainIAC] Missing keys: {incompatible.missing_keys}")
        if incompatible.unexpected_keys:
            print(f"  [BrainIAC] Unexpected keys: {incompatible.unexpected_keys}")

        self._model.eval()
        self._model.to(device)
        self.device = device
        return self

    def load_dummy(self, img_size=(96, 112, 96), patch_size=(16, 16, 16)):
        self._img_size = img_size
        self._patch_size = patch_size
        self._model = _MONAIViT(
            in_channels=1,
            img_size=img_size,
            patch_size=patch_size,
            hidden_size=self.hidden_dim,
            mlp_dim=self.hidden_dim * 4,
            num_layers=12,
            num_heads=12,
            classification=False,
        )
        self._model.eval()
        self.device = "cpu"
        return self

    def adapt_patch_embed(self, n_channels: int) -> None:
        if n_channels == self._n_input_channels:
            return
        old_conv = self._model.patch_embedding.patch_embeddings
        if isinstance(old_conv, nn.Sequential):
            old_proj = old_conv[0]
        else:
            old_proj = old_conv
        new_conv = nn.Conv3d(
            n_channels,
            old_proj.out_channels,
            kernel_size=old_proj.kernel_size,
            stride=old_proj.stride,
            bias=old_proj.bias is not None,
        )
        with torch.no_grad():
            weight = old_proj.weight.data
            new_conv.weight.data = (
                weight.repeat(1, n_channels, 1, 1, 1) / n_channels
                if weight.shape[1] == 1
                else weight[:, :1].repeat(1, n_channels, 1, 1, 1) / n_channels
            )
            if new_conv.bias is not None:
                new_conv.bias.data = old_proj.bias.data
        if isinstance(old_conv, nn.Sequential):
            self._model.patch_embedding.patch_embeddings[0] = new_conv
        else:
            self._model.patch_embedding.patch_embeddings = new_conv
        self._n_input_channels = n_channels

    def _encode(self, x):
        output = self._model(x)
        x = output[0]
        return x

    def forward(self, x):
        x = self._encode(x)
        return x.mean(dim=1)

    def forward_features(self, x):
        x = self._encode(x)
        return x

    def get_patch_grid(self, volume_shape):
        ps = self._patch_size
        return tuple(d // p for d, p in zip(volume_shape, ps))

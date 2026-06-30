import torch
import torch.nn as nn

try:
    from brainiac import BrainIACEncoder as _BrainIACEncoder
except ImportError:
    _BrainIACEncoder = None


class BrainIACBackbone(nn.Module):
    hidden_dim: int = 768
    has_cls_token: bool = False

    def __init__(self):
        super().__init__()
        self._model = None
        self._n_input_channels = 1

    def from_pretrained(self, model_id="eugenehp/brainiac", device=None):
        if _BrainIACEncoder is None:
            raise ImportError("Install brainiac: pip install brainiac")
        self._model = _BrainIACEncoder.from_pretrained(model_id)
        self._model.to(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        return self

    def load_dummy(self):
        super().__init__()
        self.hidden_dim = 768
        patch_embed = nn.Conv3d(1, 768, kernel_size=16, stride=16)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=768, nhead=12, dim_feedforward=3072,
            dropout=0.1, activation="gelu", batch_first=True,
        )
        self._model = nn.ModuleDict({
            "patch_embed": patch_embed,
            "encoder": nn.TransformerEncoder(encoder_layer, num_layers=12),
        })
        self._model.eval()
        self.device = "cpu"
        return self

    def adapt_patch_embed(self, n_channels: int) -> None:
        if n_channels == self._n_input_channels:
            return
        if isinstance(self._model, nn.ModuleDict):
            old_conv = self._model["patch_embed"]
            W = old_conv.weight
            new_conv = nn.Conv3d(
                n_channels, old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                bias=old_conv.bias is not None,
            )
            with torch.no_grad():
                repeated = W.repeat(1, n_channels, 1, 1, 1) / n_channels
                new_conv.weight.data = repeated
                if old_conv.bias is not None:
                    new_conv.bias.data = old_conv.bias
            self._model["patch_embed"] = new_conv
        self._n_input_channels = n_channels

    def forward(self, x):
        if isinstance(self._model, nn.ModuleDict):
            tokens = self._model["patch_embed"](x)
            B, D, H, W, Dp = tokens.shape
            tokens = tokens.flatten(2).transpose(1, 2)
            tokens = self._model["encoder"](tokens)
            return tokens
        return self._model(x)


class BrainIACClassifier(nn.Module):
    def __init__(self, hidden_dim=768, n_classes=3, dropout=0.3):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(hidden_dim, n_classes)
        )

    def forward(self, tokens):
        pooled = tokens.mean(dim=1)
        return self.classifier(pooled)

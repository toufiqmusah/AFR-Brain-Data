import torch
import torch.nn as nn

try:
    from neurovfm.pipelines import load_encoder as _load_vfm
except ImportError:
    _load_vfm = None


class NeuroVFMBackbone(nn.Module):
    hidden_dim: int = 768
    has_cls_token: bool = False

    def __init__(self):
        super().__init__()
        self._model = None
        self._preprocessor = None
        self._n_input_channels = 1

    def load(self, model_name="mlinslab/neurovfm-encoder", device=None):
        if _load_vfm is None:
            raise ImportError("Install neurovfm: pip install -e /path/to/neurovfm")
        self._model, self._preprocessor = _load_vfm(model_name, device=device)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        return self

    def load_dummy(self):
        super().__init__()
        self.hidden_dim = 768
        patch_embed = nn.Conv3d(1, 768, kernel_size=(4, 16, 16), stride=(4, 16, 16))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=768, nhead=12, dim_feedforward=3072,
            dropout=0.1, activation="gelu", batch_first=True,
        )
        self._model = nn.ModuleDict({
            "patch_embed": patch_embed,
            "encoder": nn.TransformerEncoder(encoder_layer, num_layers=12),
        })
        self._model.eval()
        self._preprocessor = None
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

    def forward(self, batch):
        if isinstance(self._model, nn.ModuleDict):
            tokens = self._model["patch_embed"](batch)
            B, D, H, W, Dp = tokens.shape
            tokens = tokens.flatten(2).transpose(1, 2)
            tokens = self._model["encoder"](tokens)
            return tokens
        if isinstance(batch, dict):
            return self._model.embed(batch)
        return self._model.embed(batch)

    def forward_features(self, x):
        return self.forward(x)

    def get_patch_grid(self, volume_shape):
        if isinstance(self._model, nn.ModuleDict):
            conv = self._model["patch_embed"]
            h = (volume_shape[0] - conv.kernel_size[0]) // conv.stride[0] + 1
            w = (volume_shape[1] - conv.kernel_size[1]) // conv.stride[1] + 1
            d = (volume_shape[2] - conv.kernel_size[2]) // conv.stride[2] + 1
            return (h, w, d)
        return None

    @property
    def has_cls_token(self):
        return False


class NeuroVFMClassifier(nn.Module):
    def __init__(self, hidden_dim=768, n_classes=3, dropout=0.3):
        super().__init__()
        self.attn_pool = nn.Sequential(
            nn.Linear(hidden_dim, 1), nn.Softmax(dim=1)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(hidden_dim, n_classes)
        )

    def forward(self, tokens):
        attn_weights = self.attn_pool(tokens)
        pooled = (tokens * attn_weights).sum(dim=1)
        return self.classifier(pooled)


class NeuroVFMGradCAM:
    def __init__(self, backbone: NeuroVFMBackbone, classifier: NeuroVFMClassifier):
        self.backbone = backbone
        self.classifier = classifier
        self.attention_maps = []

    def _register_hooks(self):
        blocks = getattr(self.backbone._model, "blocks", None) or getattr(self.backbone._model, "layers", None)
        if blocks is None:
            raise AttributeError("Cannot locate transformer blocks")
        self.handles = []
        for block in blocks:
            attn = getattr(block, "attn", None) or getattr(block, "self_attn", None)
            if attn is not None:
                self.handles.append(attn.register_forward_hook(
                    lambda m, i, o: self.attention_maps.append(o[1])
                ))

    def _remove_hooks(self):
        for h in self.handles:
            h.remove()

    def compute_saliency(self, batch, target_class=None):
        self.attention_maps = []
        self._register_hooks()
        tokens = self.backbone(batch)
        logits = self.classifier(tokens)
        pred = logits.argmax(dim=-1).item() if target_class is None else target_class
        num_layers = len(self.attention_maps)
        identity = torch.eye(self.attention_maps[0].size(-1), device=self.attention_maps[0].device)
        result = identity
        for a in self.attention_maps:
            a = a.mean(dim=1) + identity
            a = a / a.sum(dim=-1, keepdim=True)
            result = result @ a
        cls_attn = result[0, 0, 1:]
        coords = batch.get("coords", None) if isinstance(batch, dict) else None
        self._remove_hooks()
        return cls_attn, coords

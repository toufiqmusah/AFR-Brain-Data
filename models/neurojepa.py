import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from neurojepa.utils.init_utils import load_backbone_from_hf as _load_jepa
except ImportError:
    _load_jepa = None


class NeuroJEPABackbone(nn.Module):
    hidden_dim: int = 768
    has_cls_token: bool = False

    def __init__(self):
        super().__init__()
        self._backbone = None
        self._n_input_channels = 1

    def from_pretrained(self, model_id="NYUMedML/Neuro-JEPA", device=None):
        if _load_jepa is None:
            raise ImportError("Install neurojepa: pip install -e /path/to/Neuro-JEPA")
        self._backbone = _load_jepa(model_id, device=device)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        return self

    def load_dummy(self, img_size=(96, 108, 96), patch_size=16):
        super().__init__()
        self.hidden_dim = 768
        patch_embed = nn.Conv3d(1, 768, kernel_size=patch_size, stride=patch_size)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=768, nhead=12, dim_feedforward=3072,
            dropout=0.1, activation="gelu", batch_first=True,
        )
        self._backbone = nn.ModuleDict({
            "patch_embed": patch_embed,
            "encoder": nn.TransformerEncoder(encoder_layer, num_layers=12),
        })
        self._backbone.eval()
        self.device = "cpu"
        return self

    def adapt_patch_embed(self, n_channels: int) -> None:
        if n_channels == self._n_input_channels:
            return
        if isinstance(self._backbone, nn.ModuleDict):
            old_conv = self._backbone["patch_embed"]
            W = old_conv.weight
            new_conv = nn.Conv3d(
                n_channels, old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=old_conv.bias is not None,
            )
            with torch.no_grad():
                repeated = W.repeat(1, n_channels, 1, 1, 1) / n_channels
                new_conv.weight.data = repeated
                if old_conv.bias is not None:
                    new_conv.bias.data = old_conv.bias
            self._backbone["patch_embed"] = new_conv
        self._n_input_channels = n_channels

    def forward(self, x):
        if isinstance(self._backbone, nn.ModuleDict):
            tokens = self._backbone["patch_embed"](x)
            B, D, H, W, Dp = tokens.shape
            tokens = tokens.flatten(2).transpose(1, 2)
            tokens = self._backbone["encoder"](tokens)
            return tokens.mean(dim=1)
        out = self._backbone(x)
        if isinstance(out, (tuple, list)):
            out = out[0]
        return out.mean(dim=1)

    def forward_features(self, x):
        if isinstance(self._backbone, nn.ModuleDict):
            tokens = self._backbone["patch_embed"](x)
            B, D, H, W, Dp = tokens.shape
            tokens = tokens.flatten(2).transpose(1, 2)
            return self._backbone["encoder"](tokens)
        out = self._backbone(x)
        if isinstance(out, (tuple, list)):
            out = out[0]
        return out

    def get_patch_grid(self, volume_shape):
        if isinstance(self._backbone, nn.ModuleDict):
            conv = self._backbone["patch_embed"]
            h = (volume_shape[0] - conv.kernel_size[0]) // conv.stride[0] + 1
            w = (volume_shape[1] - conv.kernel_size[1]) // conv.stride[1] + 1
            d = (volume_shape[2] - conv.kernel_size[2]) // conv.stride[2] + 1
            return (h, w, d)
        return None


class NeuroJEPAClassifier(nn.Module):
    def __init__(self, hidden_dim=768, n_classes=3, pool="gap", dropout=0.3):
        super().__init__()
        self.pool = pool
        if pool == "attn":
            self.attn = nn.Sequential(
                nn.Linear(hidden_dim, 1), nn.Softmax(dim=1)
            )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(hidden_dim, n_classes)
        )

    def forward(self, patch_tokens):
        if self.pool == "gap":
            pooled = patch_tokens.mean(dim=1)
        elif self.pool == "attn":
            weights = self.attn(patch_tokens)
            pooled = (patch_tokens * weights).sum(dim=1)
        else:
            pooled = patch_tokens.mean(dim=1)
        return self.classifier(pooled)

    def features_before_pool(self, patch_tokens):
        return patch_tokens


class NeuroJEPAGradCAM:
    def __init__(self, backbone: NeuroJEPABackbone, classifier: NeuroJEPAClassifier):
        self.backbone = backbone
        self.classifier = classifier

    def compute_saliency(self, x, target_class=None, resize_to=None):
        self.backbone.eval()
        self.classifier.eval()
        x = x.detach().requires_grad_(True)
        patch_tokens = self.backbone.forward_features(x)
        patch_tokens.retain_grad()
        logits = self.classifier(patch_tokens)
        pred = logits.argmax(dim=-1).item() if target_class is None else target_class
        self.classifier.zero_grad()
        logits[0, pred].backward()
        gradients = patch_tokens.grad
        weights = gradients.mean(dim=(0, -1), keepdim=True)
        cam = (patch_tokens * weights).sum(dim=-1)
        cam = F.relu(cam)
        vol_shape = x.shape[2:]
        patch_size = 16
        h, w, d = vol_shape[0] // patch_size, vol_shape[1] // patch_size, vol_shape[2] // patch_size
        if h * w * d != cam.size(1):
            return cam[0]
        cam_3d = cam[0].reshape(h, w, d)
        cam_3d = (cam_3d - cam_3d.min()) / (cam_3d.max() + 1e-8)
        if resize_to is not None:
            cam_3d = cam_3d.unsqueeze(0).unsqueeze(0)
            cam_3d = F.interpolate(cam_3d, size=resize_to, mode="trilinear", align_corners=False)
            cam_3d = cam_3d[0, 0]
        return cam_3d

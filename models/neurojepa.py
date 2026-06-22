import torch
import torch.nn as nn
import torch.nn.functional as F
from neurojepa.utils.init_utils import load_backbone_from_hf


class NeuroJEPABackbone(nn.Module):
    """
    Neuro-JEPA encoder wrapper.

    Architecture: 3D ViT-Base-MoE, patch=(16,16,16), embed_dim=768, 12 layers.
    Trained with improved V-JEPA on 1.55M T1w/T2w/FLAIR scans.
    HF model: NYUMedML/Neuro-JEPA, Paper: https://arxiv.org/abs/2606.14957

    backbone = NeuroJEPABackbone.from_pretrained("NYUMedML/Neuro-JEPA")
    patchtokens, moe_scores = backbone(x)
    # patchtokens shape: [B, N, D] — fixed spatial grid

    Output is a fixed 3D grid of patch tokens — ideal for:
    - GAP + linear classification
    - Grad-CAM via patch-token gradients
    - Attention rollout for explainability
    """

    def __init__(self):
        super().__init__()
        self._backbone = None
        self.hidden_dim = 768

    def from_pretrained(self, model_id="NYUMedML/Neuro-JEPA", device=None):
        """
        Load backbone from HuggingFace.
        """

        self._backbone = load_backbone_from_hf(model_id, device=device)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        return self

    def load_dummy(self, img_size=(96, 108, 96), patch_size=16):
        """
        Create dummy MoE ViT matching Neuro-JEPA dims for dev/testing.
        No real weights.

        Uses a conv patcher + TransformerEncoder + mock MoE routing.
        Grid size for (96,108,96)/16: (6, 7, 6)=252 tokens (H padded to 112).
        """
        super(NeuroJEPABackbone, self).__init__()
        self.hidden_dim = 768
        patch_embed = nn.Conv3d(1, 768, kernel_size=patch_size, stride=patch_size)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=768, nhead=12, dim_feedforward=3072,
            dropout=0.1, activation='gelu', batch_first=True
        )
        self._backbone = nn.ModuleDict({
            'patch_embed': patch_embed,
            'encoder': nn.TransformerEncoder(encoder_layer, num_layers=12),
        })
        self._backbone.eval()
        self._img_size = img_size
        self.device = "cpu"
        return self

    def forward(self, x):
        """
        Args:
            x: [B, 1, H, W, D] input volume

        Returns:
            patch_tokens: [B, N, D] patch token embeddings
            moe_scores: [B, N, n_experts] MoE routing weights (optional)
        """
        if isinstance(self._backbone, nn.ModuleDict):
            tokens = self._backbone['patch_embed'](x)
            B, D, H, W, Dp = tokens.shape
            tokens = tokens.flatten(2).transpose(1, 2)
            tokens = self._backbone['encoder'](tokens)
            n_experts = 4
            moe_scores = torch.softmax(
                torch.randn(B, tokens.size(1), n_experts, device=tokens.device),
                dim=-1
            )
            return tokens, moe_scores
        return self._backbone(x)

    @property
    def has_cls_token(self):
        return False


class NeuroJEPAClassifier(nn.Module):
    """
    Classification head for Neuro-JEPA.

    Two pooling options:
    1. 'gap': Global Average Pooling over patch tokens (supports Grad-CAM)
    2. 'attn': Attention pooling over patch tokens

    Architecture:
        patch_tokens [B, N, D]
          → pool → [B, D]
          → Dropout → Linear → [B, n_classes]
    """

    def __init__(self, hidden_dim=768, n_classes=3, pool='gap', dropout=0.3):
        super().__init__()
        self.pool = pool
        if pool == 'attn':
            self.attn = nn.Sequential(
                nn.Linear(hidden_dim, 1),
                nn.Softmax(dim=1),
            )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, patch_tokens):
        """
        Args:
            patch_tokens: [B, N, D]
        Returns:
            logits: [B, n_classes]
        """
        if self.pool == 'gap':
            pooled = patch_tokens.mean(dim=1)  # [B, D]
        elif self.pool == 'attn':
            weights = self.attn(patch_tokens)  # [B, N, 1]
            pooled = (patch_tokens * weights).sum(dim=1)  # [B, D]
        return self.classifier(pooled)

    def get_attention_weights(self, patch_tokens):
        """Return attention pooling weights for visualization."""
        if self.pool == 'attn':
            return self.attn(patch_tokens)
        return None

    def features_before_pool(self, patch_tokens):
        """
        Return patch-level features before pooling and classification.
        Used by GradCAM to compute gradient maps.
        """
        return patch_tokens


class NeuroJEPAGradCAM:
    """
    Grad-CAM for Neuro-JEPA with GAP-pooled classifier.

    Since Neuro-JEPA outputs a fixed 3D grid of patch tokens:
        tokens: [B, N, D] where N = H*W*D / patch_size**3

    Compute:
        grad = d(y_class) / d(tokens)  — gradients of class score w.r.t. patch tokens
        weights = grad.mean(dim=(0, -1))  — global average pooling of gradients
        cam = (tokens * weights).sum(dim=-1)  — weighted combination
        cam = relu(cam)  — only positive contributions
        cam = reshape to (H, W, D) grid  — 3D saliency map

    Requires:
        - classifier with pool='gap'
        - model in eval mode with requires_grad on tokens
    """

    def __init__(self, backbone, classifier):
        self.backbone = backbone
        self.classifier = classifier
        self.gradients = None

    def _save_gradients(self, grad):
        self.gradients = grad

    def compute_saliency(self, x, target_class=None, resize_to=None):
        """
        Args:
            x: [B, 1, H, W, D] input volume
            target_class: class index (if None, uses argmax)
            resize_to: optional (H, W, D) to upsample CAM to

        Returns:
            cam: [H, W, D] saliency map
        """
        self.backbone.eval()
        self.classifier.eval()

        x = x.detach().requires_grad_(True)
        patch_tokens = self.backbone(x)[0]  # [B, N, D]
        patch_tokens.retain_grad()

        logits = self.classifier(patch_tokens)
        pred = logits.argmax(dim=-1).item() if target_class is None else target_class

        self.classifier.zero_grad()
        logits[0, pred].backward()
        gradients = patch_tokens.grad  # [B, N, D]

        weights = gradients.mean(dim=(0, -1), keepdim=True)  # [B, 1, 1]

        # Weighted combination of patch tokens
        cam = (patch_tokens * weights).sum(dim=-1)  # [B, N]
        cam = F.relu(cam)  # [B, N]

        # Infer grid shape from N = H*W*D / p^3
        N = cam.size(1)
        grid_dims = self._infer_grid_dims(N, x.shape[2:])
        if grid_dims is None:
            return cam[0]

        h, w, d = grid_dims
        cam_3d = cam[0].reshape(h, w, d)
        cam_3d = (cam_3d - cam_3d.min()) / (cam_3d.max() + 1e-8)

        if resize_to is not None:
            cam_3d = cam_3d.unsqueeze(0).unsqueeze(0)
            cam_3d = F.interpolate(
                cam_3d, size=resize_to, mode='trilinear', align_corners=False
            )
            cam_3d = cam_3d[0, 0]

        return cam_3d

    def _infer_grid_dims(self, N, vol_shape):
        """Try to factor N into (H, W, D) grid from patch size."""
        patch_size = 16
        h = vol_shape[0] // patch_size
        w = vol_shape[1] // patch_size
        d = vol_shape[2] // patch_size
        if h * w * d == N:
            return (h, w, d)
        return None

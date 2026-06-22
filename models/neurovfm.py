import torch
import torch.nn as nn
from neurovfm.pipelines import load_encoder


class NeuroVFMBackbone(nn.Module):
    """
    NeuroVFM encoder wrapper.

    Architecture: 3D ViT-Base, patch=(4,16,16), embed_dim=768, 12 layers.
    Trained with Vol-JEPA on 5.24M MRI/CT volumes.

    HF model: mlinslab/neurovfm-encoder, Paper: https://arxiv.org/abs/2511.18640

    Usage (once weights are approved):
        backbone = NeuroVFMBackbone.from_pretrained("mlinslab/neurovfm-encoder")
        tokens = backbone(x)  # x: preprocessed volume tensor

    Notes:
    - Native pipeline uses variable-length token sequences with background removal.
    - For GradCAM / spatial attribution, set remove_background=False in
      the preprocessing pipeline so that all patch tokens are preserved
      in a known spatial grid.
    """

    def __init__(self):
        super().__init__()
        self._model = None
        self._preprocessor = None
        self.hidden_dim = 768
        self.patch_size = (4, 16, 16)

    def load(self, model_name="mlinslab/neurovfm-encoder", device=None):
        """
        Load encoder and preprocessor from HuggingFace.
        """

        self._model, self._preprocessor = load_encoder(model_name, device=device)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        return self

    def load_dummy(self):
        """
        Create a dummy 3D ViT matching NeuroVFM architecture for development.
        No real weights — used for shape checks / code testing.

        Tokenizer: 3D conv with patch_size=(4,16,16) → (H/4, W/16, D/16) grid.
        """
        super(NeuroVFMBackbone, self).__init__()
        self.patch_size = (4, 16, 16)
        self.hidden_dim = 768

        patch_embed = nn.Conv3d(1, 768, kernel_size=(4, 16, 16), stride=(4, 16, 16))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=768, nhead=12, dim_feedforward=3072,
            dropout=0.1, activation='gelu', batch_first=True
        )
        self._model = nn.ModuleDict({
            'patch_embed': patch_embed,
            'encoder': nn.TransformerEncoder(encoder_layer, num_layers=12),
        })
        self._model.eval()
        self._preprocessor = None
        self.device = "cpu"
        return self

    def _forward_dummy(self, x):
        """
        Dummy forward — patches + encode via TransformerEncoder.
        Returns [n_tokens, 768] (background tokens NOT removed).
        """
        tokens = self._model['patch_embed'](x)
        B, D, H, W, Dp = tokens.shape
        tokens = tokens.flatten(2).transpose(1, 2)  # [B, H*W*Dp, D]
        tokens = self._model['encoder'](tokens)
        return tokens[0]

    def forward(self, batch):
        """
        Args:
            batch: output from StudyPreprocessor (dict with 'tokens', 'coords', etc.)
                   OR a raw tensor [B, 1, H, W, D] (for dummy model).

        Returns:
            tokens: [N, 768] patch token embeddings
        """
        if isinstance(batch, dict):
            return self._model.embed(batch)
        return self._forward_dummy(batch)

    @property
    def has_cls_token(self):
        return False


class NeuroVFMClassifier(nn.Module):
    """
    Classification head for NeuroVFM.

    Since NeuroVFM outputs variable-length token sequences (no fixed grid),
    we pool via Attention pooling (MIL-style) followed by a linear classifier.

    Architecture:
        tokens [N, D]
          → Attention pool → [1, D]
          → Dropout → Linear → [n_classes]
    """

    def __init__(self, hidden_dim=768, n_classes=3, dropout=0.3):
        super().__init__()
        self.attn_pool = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Softmax(dim=0),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, tokens):
        # tokens: [N, D] — variable-length token sequence
        attn_weights = self.attn_pool(tokens)  # [N, 1]
        pooled = (tokens * attn_weights).sum(dim=0, keepdim=True)  # [1, D]
        return self.classifier(pooled)  # [1, n_classes]


class NeuroVFMGradCAM:
    """
    Attention-based explainability for NeuroVFM.

    Since NeuroVFM removes background tokens, we use attention rollout
    on the transformer's self-attention layers to trace which input
    regions influence the classification decision.

    Requires: forward hooks on each ViT block to capture attention weights.
    """

    def __init__(self, backbone, classifier):
        self.backbone = backbone
        self.classifier = classifier
        self.attention_maps = []

    def _register_hooks(self):
        """Register forward hooks on each block to capture attention weights."""
        blocks = None
        if hasattr(self.backbone._model, 'blocks'):
            blocks = self.backbone._model.blocks
        elif hasattr(self.backbone._model, 'layers'):
            blocks = self.backbone._model.layers

        if blocks is None:
            raise AttributeError(
                "Could not locate transformer blocks in NeuroVFM model."
            )

        def _hook_fn(module, inp, out):
            self.attention_maps.append(out[1])  # attention weights

        self.handles = []
        for block in blocks:
            attn = getattr(block, 'attn', None) or getattr(block, 'self_attn', None)
            if attn is not None:
                handle = attn.register_forward_hook(_hook_fn)
                self.handles.append(handle)

    def _remove_hooks(self):
        for h in self.handles:
            h.remove()

    def compute_saliency(self, batch, target_class=None):
        """
        Compute attention rollout saliency map.

        Args:
            batch: preprocessed input (with remove_background=False)
            target_class: class index (if None, uses predicted class)

        Returns:
            saliency: [H, W, D] spatial saliency map
            coords: corresponding token coordinates
        """
        self.attention_maps = []
        self._register_hooks()

        tokens = self.backbone(batch)
        logits = self.classifier(tokens)
        pred = logits.argmax(dim=-1).item() if target_class is None else target_class

        # Attention rollout
        num_layers = len(self.attention_maps)
        attn = self.attention_maps
        identity = torch.eye(attn[0].size(-1), device=attn[0].device)
        result = identity
        for a in attn:
            a = a.mean(dim=1)  # average over heads
            a = a + identity  # residual connection
            a = a / a.sum(dim=-1, keepdim=True)
            result = result @ a

        # Extract CLS-to-patch attention
        cls_attn = result[0, 0, 1:]  # [n_patches]
        coords = batch.get('coords', None)

        self._remove_hooks()
        return cls_attn, coords

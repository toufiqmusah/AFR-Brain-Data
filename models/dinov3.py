import torch
import torch.nn as nn
import torch.nn.functional as F


class VisionBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_hidden = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x


class VisionBlocks(nn.Module):
    _HEAD_MAP = {384: 6, 768: 12, 1024: 16}

    def __init__(self, embed_dim: int, num_blocks: int = 2, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        num_heads = self._HEAD_MAP.get(embed_dim, max(1, embed_dim // 64))
        self.blocks = nn.Sequential(*[
            VisionBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(num_blocks)
        ])

    def forward(self, tokens):
        return self.blocks(tokens)


class AttentionPool(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.query = nn.Parameter(torch.empty(embed_dim))
        nn.init.trunc_normal_(self.query, std=0.02)

    def forward(self, H):
        scale = H.shape[-1] ** 0.5
        scores = torch.einsum("bsd,d->bs", H, self.query) / scale
        a = F.softmax(scores, dim=-1)
        v = torch.einsum("bs,bsd->bd", a, H)
        return v, a


class DINOv3Backbone(nn.Module):
    hidden_dim = 384
    has_cls_token = True

    def __init__(self):
        super().__init__()
        self._model = None
        self._image_size = 224
        self._patch_size = 16
        self._n_input_channels = 1

        self.vision_blocks = VisionBlocks(embed_dim=384, num_blocks=2)
        self.pool = AttentionPool(384)

    def from_pretrained(self, model_id="facebook/dinov3-vits16plus-pretrain-lvd1689m", device=None):
        from transformers import DINOv3ViTModel
        from peft import LoraConfig, get_peft_model
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        try:
            backbone = DINOv3ViTModel.from_pretrained(model_id, trust_remote_code=True)
            backbone.to(device)
            lora_cfg = LoraConfig(
                r=8, lora_alpha=16, lora_dropout=0.0, bias="none",
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            )
            self._model = get_peft_model(backbone, lora_cfg)
            self._model.eval()
            self.device = device
        except Exception as e:
            print(f"  [DINOv3] HF weights failed ({e}), using dummy")
            self.load_dummy()
        return self

    def load_dummy(self):
        from transformers import DINOv3ViTConfig, DINOv3ViTModel
        from peft import LoraConfig, get_peft_model
        config = DINOv3ViTConfig()
        backbone = DINOv3ViTModel(config)
        lora_cfg = LoraConfig(
            r=8, lora_alpha=16, lora_dropout=0.0, bias="none",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        )
        self._model = get_peft_model(backbone, lora_cfg)
        self._model.eval()
        self.device = "cpu"
        return self

    def adapt_patch_embed(self, n_channels):
        pass

    def _encode_slices(self, x):
        B, C, H, W, D = x.shape

        if C > 1:
            x = x.mean(dim=1, keepdim=True)

        slice_indices = torch.arange(0, D, 2, device=x.device)
        n_slices = len(slice_indices)

        slices = x[:, :, :, :, slice_indices]
        slices = slices.permute(0, 4, 1, 2, 3)
        slices = slices.reshape(B * n_slices, 1, H, W)
        slices = slices.repeat(1, 3, 1, 1)

        if H != self._image_size or W != self._image_size:
            slices = F.interpolate(slices, size=(self._image_size, self._image_size),
                                   mode="bilinear", align_corners=False)

        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
        slices = (slices - mean) / std

        out = self._model(slices)
        return out.last_hidden_state

    def forward(self, x):
        all_tokens = self._encode_slices(x)
        B = x.shape[0]
        D = x.shape[4]
        n_slices = D // 2 + D % 2

        tokens = all_tokens.reshape(B * n_slices, -1, self.hidden_dim)
        tokens = self.vision_blocks(tokens)
        cls_tokens = tokens[:, 0].reshape(B, n_slices, self.hidden_dim)

        v, _ = self.pool(cls_tokens)
        return v

    def forward_features(self, x):
        all_tokens = self._encode_slices(x)
        B = x.shape[0]
        D = x.shape[4]
        n_slices = D // 2 + D % 2

        tokens = all_tokens.reshape(B * n_slices, -1, self.hidden_dim)
        tokens = self.vision_blocks(tokens)
        n_tokens = tokens.shape[1]
        return tokens.reshape(B, n_slices * n_tokens, self.hidden_dim)

    def get_patch_grid(self, volume_shape):
        D = volume_shape[-1]
        n_slices = D // 2 + D % 2
        grid_h = self._image_size // self._patch_size
        grid_w = self._image_size // self._patch_size
        return (n_slices, grid_h, grid_w)

    def trainable_params(self):
        for p in self.vision_blocks.parameters():
            yield p
        for p in self.pool.parameters():
            yield p

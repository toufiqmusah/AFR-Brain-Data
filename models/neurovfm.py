import torch
import torch.nn as nn


class VoxelPatchEmbed(nn.Module):
    def __init__(self, in_chans=1, embed_dim=1024, patch_hw_size=16, patch_d_size=4, bias=True):
        super().__init__()
        self.proj = nn.Conv3d(
            in_chans, embed_dim,
            kernel_size=(patch_d_size, patch_hw_size, patch_hw_size),
            stride=(patch_d_size, patch_hw_size, patch_hw_size),
            bias=bias,
        )

    def forward(self, x):
        return self.proj(x)


class SinCosPosEmbed3D(nn.Module):
    def __init__(self, pos_dim=30, concat=True):
        super().__init__()
        self.concat = concat
        self.pos_dim = pos_dim

    def forward(self, x, grid_shape):
        if not self.concat or self.pos_dim == 0:
            return x
        D, H, W = grid_shape
        B, N, _ = x.shape
        pos = self._make_grid(D, H, W, x.device)
        return torch.cat([x, pos.expand(B, -1, -1)], dim=-1)

    def _make_grid(self, D, H, W, device):
        d = torch.arange(D, device=device).float()
        h = torch.arange(H, device=device).float()
        w = torch.arange(W, device=device).float()
        freqs = 1.0 / (10000.0 ** (torch.arange(0, 5, device=device).float() / 5.0))

        def _axis_encoding(pos, freqs):
            enc = []
            for f in freqs:
                enc.append(torch.sin(pos * f))
                enc.append(torch.cos(pos * f))
            return torch.stack(enc, dim=-1)

        d_grid, h_grid, w_grid = torch.meshgrid(d, h, w, indexing='ij')
        d_enc = _axis_encoding(d_grid.flatten(), freqs)
        h_enc = _axis_encoding(h_grid.flatten(), freqs)
        w_enc = _axis_encoding(w_grid.flatten(), freqs)
        pos = torch.cat([d_enc, h_enc, w_enc], dim=-1)
        return pos.unsqueeze(0)


class Attention(nn.Module):
    def __init__(self, dim, num_heads=12):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(x)


class Mlp(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim=None):
        super().__init__()
        out_dim = out_dim or in_dim
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.nn.functional.gelu(x)
        x = self.fc2(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.mixer = Attention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio))

    def forward(self, x):
        x = x + self.mixer(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class NeuroVFMEncoder(nn.Module):
    def __init__(self, embed_dim=768, depth=12, num_heads=12, in_chans=1,
                 conv_out_chans=1024, token_dim=738, pos_dim=30):
        super().__init__()
        self.embed_dim = embed_dim
        self.token_dim = token_dim
        self.patch_embed = VoxelPatchEmbed(in_chans=in_chans, embed_dim=conv_out_chans)
        self.token_embed = nn.Linear(conv_out_chans, token_dim)
        self.pos_embed = SinCosPosEmbed3D(pos_dim=pos_dim, concat=True)
        self.blocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=4)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.patch_embed(x)
        B, C, D, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.token_embed(x)
        x = self.pos_embed(x, (D, H, W))
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return x


class NeuroVFMBackbone(nn.Module):
    hidden_dim: int = 768
    has_cls_token: bool = False

    def __init__(self):
        super().__init__()
        self._model = None
        self._n_input_channels = 1
        self._patch_embed_size = (4, 16, 16)

    def load(self, model_name="mlinslab/neurovfm-encoder", device=None):
        try:
            from huggingface_hub import hf_hub_download
            import json

            config_path = hf_hub_download(model_name, "config.json")
            weights_path = hf_hub_download(model_name, "pytorch_model.bin")

            with open(config_path) as f:
                cfg = json.load(f)["params"]

            embed_dim = cfg["embed_dim"]
            depth = cfg["depth"]
            num_heads = cfg["num_heads"]
            embed_cfg = cfg["embed_layer_cf"]["params"]
            self._patch_embed_size = (embed_cfg["patch_d_size"], embed_cfg["patch_hw_size"], embed_cfg["patch_hw_size"])

            self._model = NeuroVFMEncoder(
                embed_dim=embed_dim,
                depth=depth,
                num_heads=num_heads,
                in_chans=embed_cfg["in_chans"],
            )

            state = torch.load(weights_path, map_location="cpu", weights_only=False)
            state_dict = state.get("model", state)

            remap = {}
            for k, v in state_dict.items():
                k_new = k.replace("token_embed.proj.", "token_embed.")
                remap[k_new] = v

            incompatible = self._model.load_state_dict(remap, strict=False)
            if incompatible.missing_keys:
                print(f"  [NeuroVFM] Missing keys: {incompatible.missing_keys}")
            if incompatible.unexpected_keys:
                print(f"  [NeuroVFM] Unexpected keys: {incompatible.unexpected_keys}")
        except Exception as e:
            print(f"  [NeuroVFM] Could not load pretrained weights: {e}")
            return self.load_dummy()

        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model.eval()
        self._model.to(device)
        self.device = device
        return self

    def load_dummy(self):
        self._model = NeuroVFMEncoder(embed_dim=768, depth=12, num_heads=12, in_chans=1)
        self._model.eval()
        self.device = "cpu"
        return self

    def adapt_patch_embed(self, n_channels: int) -> None:
        if n_channels == self._n_input_channels:
            return
        old_conv = self._model.patch_embed.proj
        new_conv = nn.Conv3d(
            n_channels,
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            bias=old_conv.bias is not None,
        )
        with torch.no_grad():
            weight = old_conv.weight.data
            new_conv.weight.data = (
                weight.repeat(1, n_channels, 1, 1, 1) / n_channels
                if weight.shape[1] == 1
                else weight[:, :1].repeat(1, n_channels, 1, 1, 1) / n_channels
            )
            if new_conv.bias is not None:
                new_conv.bias.data = old_conv.bias.data
        self._model.patch_embed.proj = new_conv
        self._n_input_channels = n_channels

    def forward(self, x):
        x = self._model(x)
        return x.mean(dim=1)

    def forward_features(self, x):
        return self._model(x)

    def get_patch_grid(self, volume_shape):
        ps = self._patch_embed_size
        return tuple(d // p for d, p in zip(volume_shape, ps))

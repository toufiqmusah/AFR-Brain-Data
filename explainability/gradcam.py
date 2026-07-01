import torch
import torch.nn.functional as F
import numpy as np


def _get_features(backbone, x):
    if hasattr(backbone, "forward_features"):
        return backbone.forward_features(x)
    return backbone(x)


def _compute_grid(backbone, vol_shape, patch_size):
    if hasattr(backbone, "get_patch_grid"):
        grid = backbone.get_patch_grid(vol_shape)
        if grid is not None:
            return grid
    return (
        vol_shape[0] // patch_size,
        vol_shape[1] // patch_size,
        vol_shape[2] // patch_size,
    )


def _grid_for_n_tokens(n_tokens, h, w, d, vol_shape):
    if h * w * d == n_tokens:
        return (h, w, d)
    factors = []
    for i in range(1, int(n_tokens ** 0.5) + 1):
        if n_tokens % i == 0 and i <= max(vol_shape):
            factors.append(i)
    if len(factors) >= 3:
        d0, d1, d2 = factors[-3:]
        return (d0, d1, n_tokens // (d0 * d1))
    grid_size = int(round(n_tokens ** (1 / 3)))
    return (grid_size, grid_size, grid_size)


def _gradcam_3d_from_features(features, head, volume, backbone, target_class, patch_size):
    features = features.detach().requires_grad_(True)
    features.retain_grad()
    logits = head(features.mean(dim=1))
    pred = logits.argmax(dim=-1).item() if target_class is None else target_class
    head.zero_grad()
    logits[0, pred].backward()
    gradients = features.grad
    weights = gradients.mean(dim=(1,), keepdim=True)
    cam = (features * weights).sum(dim=2)
    cam = F.relu(cam)
    vols = volume.shape[2:] if volume.dim() == 5 else volume.shape[1:]
    h, w, d = _compute_grid(backbone, vols, patch_size)
    n_tokens = cam.size(1)
    h, w, d = _grid_for_n_tokens(n_tokens, h, w, d, vols)
    cam_3d = cam[0].reshape(h, w, d)
    cam_3d = (cam_3d - cam_3d.min()) / (cam_3d.max() + 1e-8)
    cam_3d = cam_3d.unsqueeze(0).unsqueeze(0)
    cam_3d = F.interpolate(cam_3d, size=vols, mode="trilinear", align_corners=False)
    return cam_3d[0, 0]


def gradcam_3d(
    backbone: torch.nn.Module,
    head: torch.nn.Module,
    volume: torch.Tensor,
    target_class: int = None,
    patch_size: int = 16,
):
    backbone.eval()
    head.eval()
    features = _get_features(backbone, volume)
    if isinstance(features, tuple):
        features = features[0]
    if features.dim() == 2:
        return None
    return _gradcam_3d_from_features(features, head, volume, backbone, target_class, patch_size)


def gradcam_multimodal(
    backbone: torch.nn.Module,
    head: torch.nn.Module,
    volume: torch.Tensor,
    n_channels: int,
    target_class: int = None,
    patch_size: int = 16,
):
    backbone.eval()
    head.eval()
    cams = []
    for c in range(n_channels):
        vol_c = volume.clone()
        vol_c[:, [i for i in range(n_channels) if i != c]] = 0.0
        vol_c = vol_c.detach().requires_grad_(True)
        features = _get_features(backbone, vol_c)
        if isinstance(features, tuple):
            features = features[0]
        if features.dim() == 2:
            cams.append(None)
            continue
        cam = _gradcam_3d_from_features(features, head, vol_c, backbone, target_class, patch_size)
        cams.append(cam)
    if all(c is None for c in cams):
        return None
    first_valid = next((c for c in cams if c is not None), None)
    cams = [c if c is not None else torch.zeros_like(first_valid) for c in cams]
    return torch.stack(cams, dim=0)


def gradcam_interaction(
    backbone: torch.nn.Module,
    head: torch.nn.Module,
    volume: torch.Tensor,
    n_channels: int,
    target_class: int = None,
    patch_size: int = 16,
):
    full_cam = gradcam_3d(backbone, head, volume, target_class, patch_size)
    if full_cam is None:
        return None, None, None
    per_channel = gradcam_multimodal(backbone, head, volume, n_channels, target_class, patch_size)
    if per_channel is None:
        return None, None, None
    sum_channel = per_channel.sum(dim=0)
    interaction = full_cam - sum_channel
    interaction = (interaction - interaction.min()) / (interaction.max() - interaction.min() + 1e-8)
    return full_cam, per_channel, interaction

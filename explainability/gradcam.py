import torch
import torch.nn.functional as F
import numpy as np


def gradcam_3d(
    backbone: torch.nn.Module,
    head: torch.nn.Module,
    volume: torch.Tensor,
    target_class: int = None,
    patch_size: int = 16,
):
    backbone.eval()
    head.eval()
    volume = volume.detach().requires_grad_(True)
    features = backbone(volume)
    if isinstance(features, tuple):
        features = features[0]
    features.retain_grad()
    logits = head(features)
    pred = logits.argmax(dim=-1).item() if target_class is None else target_class
    head.zero_grad()
    logits[0, pred].backward()
    gradients = features.grad
    weights = gradients.mean(dim=(0, -1), keepdim=True)
    cam = (features * weights).sum(dim=-1)
    cam = F.relu(cam)
    vol_shape = volume.shape[2:]
    h, w, d = vol_shape[0] // patch_size, vol_shape[1] // patch_size, vol_shape[2] // patch_size
    n_tokens = cam.size(1)
    if h * w * d == n_tokens:
        cam_3d = cam[0].reshape(h, w, d)
    else:
        grid_size = int(round(n_tokens ** (1 / 3)))
        cam_3d = cam[0].reshape(grid_size, grid_size, grid_size)
    cam_3d = (cam_3d - cam_3d.min()) / (cam_3d.max() + 1e-8)
    cam_3d = cam_3d.unsqueeze(0).unsqueeze(0)
    cam_3d = F.interpolate(
        cam_3d, size=vol_shape, mode="trilinear", align_corners=False
    )
    return cam_3d[0, 0]


def gradcam_multimodal(
    backbone: torch.nn.Module,
    head: torch.nn.Module,
    volume: torch.Tensor,
    n_channels: int,
    target_class: int = None,
    patch_size: int = 16,
):
    cams = []
    for c in range(n_channels):
        vol_c = volume.clone()
        mask = torch.ones(n_channels, device=volume.device)
        mask[c] = 1.0
        vol_c = vol_c * mask.view(1, -1, 1, 1, 1)
        vol_c = vol_c.detach().requires_grad_(True)
        features = backbone(vol_c)
        if isinstance(features, tuple):
            features = features[0]
        features.retain_grad()
        logits = head(features)
        pred = logits.argmax(dim=-1).item() if target_class is None else target_class
        head.zero_grad()
        logits[0, pred].backward()
        gradients = features.grad
        weights = gradients.mean(dim=(0, -1), keepdim=True)
        cam = (features * weights).sum(dim=-1)
        cam = F.relu(cam)
        vol_shape = volume.shape[2:]
        h, w, d = vol_shape[0] // patch_size, vol_shape[1] // patch_size, vol_shape[2] // patch_size
        hwd = h * w * d
        if hwd == cam.size(1):
            cam_3d = cam[0].reshape(h, w, d)
        else:
            grid_size = int(round(cam.size(1) ** (1 / 3)))
            cam_3d = cam[0].reshape(grid_size, grid_size, grid_size)
        cam_3d = (cam_3d - cam_3d.min()) / (cam_3d.max() + 1e-8)
        cam_3d = cam_3d.unsqueeze(0).unsqueeze(0)
        cam_3d = F.interpolate(cam_3d, size=vol_shape, mode="trilinear", align_corners=False)
        cams.append(cam_3d[0, 0])
    return torch.stack(cams, dim=0)

import torch
import torch.nn.functional as F
import numpy as np


class ResampleVolume:
    def __init__(self, target_size=(96, 112, 96)):
        self.target_size = target_size

    def __call__(self, volume):
        if isinstance(volume, np.ndarray):
            import scipy.ndimage

            factors = (
                self.target_size[0] / volume.shape[-3],
                self.target_size[1] / volume.shape[-2],
                self.target_size[2] / volume.shape[-1],
            )
            return scipy.ndimage.zoom(volume, (1,) + factors, order=1)
        vol = volume.unsqueeze(0)
        vol = F.interpolate(vol, size=self.target_size, mode="trilinear", align_corners=False)
        return vol.squeeze(0)


class IntensityNormalize:
    def __init__(self, clip_percentiles=(0.5, 99.5)):
        self.clip_percentiles = clip_percentiles

    def __call__(self, volume):
        if isinstance(volume, np.ndarray):
            lo, hi = np.percentile(volume, self.clip_percentiles)
            vol = np.clip(volume, lo, hi)
            return (vol - vol.mean()) / (vol.std() + 1e-8)
        lo, hi = torch.quantile(
            volume.float(), torch.tensor([p / 100.0 for p in self.clip_percentiles])
        )
        vol = torch.clamp(volume, lo, hi)
        return (vol - vol.mean()) / (vol.std() + 1e-8)


class RandomFlipAxial:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, volume):
        if torch.rand(1).item() < self.p:
            return volume.flip(-1)
        return volume


class RandomAffine:
    def __init__(self, max_rotation_deg=5, max_scale_pct=0.05, p=0.5):
        self.max_rotation = max_rotation_deg
        self.max_scale = max_scale_pct
        self.p = p

    def __call__(self, volume):
        if torch.rand(1).item() >= self.p:
            return volume
        C, H, W, D = volume.shape
        angle = (torch.rand(3) - 0.5) * 2 * self.max_rotation
        scale = 1.0 + (torch.rand(3) - 0.5) * 2 * self.max_scale
        theta = torch.eye(3, 4).unsqueeze(0)
        theta[0, 0, 0] = scale[0] * torch.cos(torch.deg2rad(angle[2]))
        theta[0, 0, 1] = -torch.sin(torch.deg2rad(angle[2]))
        theta[0, 1, 0] = torch.sin(torch.deg2rad(angle[2]))
        theta[0, 1, 1] = scale[1] * torch.cos(torch.deg2rad(angle[2]))
        theta[0, 2, 2] = scale[2]
        grid = F.affine_grid(theta, (1, 1, H, W, D), align_corners=False)
        return F.grid_sample(volume.unsqueeze(0), grid, align_corners=False).squeeze(0)


class ToTensor:
    def __call__(self, volume):
        if isinstance(volume, np.ndarray):
            return torch.from_numpy(volume).float()
        return volume.float()


class Compose:
    def __init__(self, transforms):
        self.transforms = [t for t in transforms if t is not None]

    def __call__(self, volume):
        for t in self.transforms:
            volume = t(volume)
        return volume


def train_transform(target_size=(96, 112, 96)):
    return Compose(
        [
            ResampleVolume(target_size),
            IntensityNormalize(),
            RandomFlipAxial(p=0.5),
            RandomAffine(max_rotation_deg=5, max_scale_pct=0.05, p=0.5),
        ]
    )


def eval_transform(target_size=(96, 112, 96)):
    return Compose(
        [
            ResampleVolume(target_size),
            IntensityNormalize(),
        ]
    )

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path


def plot_saliency_overlay(
    volume_slice: np.ndarray,
    cam_slice: np.ndarray,
    title: str = "",
    save_path: str = None,
    cmap: str = "hot",
):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(volume_slice, cmap="gray")
    axes[0].set_title("Anatomical")
    axes[0].axis("off")
    axes[1].imshow(cam_slice, cmap=cmap)
    axes[1].set_title("Saliency")
    axes[1].axis("off")
    axes[2].imshow(volume_slice, cmap="gray")
    im = axes[2].imshow(cam_slice, cmap=cmap, alpha=0.5)
    axes[2].set_title("Overlay")
    axes[2].axis("off")
    plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
    fig.suptitle(title)
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.cpu().numpy()
    return np.asarray(x)


def plot_class_cams_grid(
    class_cams: dict,
    volume: np.ndarray,
    label_names: dict,
    save_path: str = None,
):
    n_classes = len(class_cams)
    fig, axes = plt.subplots(3, n_classes, figsize=(4 * n_classes, 12))
    slice_indices = [volume.shape[0] // 2, volume.shape[1] // 2, volume.shape[2] // 2]
    slice_names = ["Axial", "Coronal", "Sagittal"]
    for col, (label, cam) in enumerate(sorted(class_cams.items())):
        name = label_names.get(label, f"Class {label}")
        for row, (s_idx, s_name) in enumerate(zip(slice_indices, slice_names)):
            ax = axes[row, col] if n_classes > 1 else axes[row]
            if row == 0:
                vol_slice = volume[s_idx, :, :]
                cam_slice = _to_numpy(cam[s_idx, :, :])
            elif row == 1:
                vol_slice = volume[:, s_idx, :]
                cam_slice = _to_numpy(cam[:, s_idx, :])
            else:
                vol_slice = volume[:, :, s_idx]
                cam_slice = _to_numpy(cam[:, :, s_idx])
            from scipy.ndimage import rotate
            vol_slice = np.rot90(vol_slice)
            cam_slice = np.rot90(cam_slice)
            ax.imshow(vol_slice, cmap="gray")
            ax.imshow(cam_slice, cmap="hot", alpha=0.5)
            ax.axis("off")
            if col == 0:
                ax.set_ylabel(s_name, fontsize=12, fontweight="bold")
            if row == 0:
                ax.set_title(name, fontsize=12, fontweight="bold")
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

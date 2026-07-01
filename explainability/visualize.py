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
        return x.detach().cpu().numpy()
    return np.asarray(x)


def plot_class_cams_grid(
    class_cams: dict,
    volume: np.ndarray,
    label_names: dict,
    save_path: str = None,
    true_label: int = None,
):
    n_classes = len(class_cams)
    if n_classes == 1:
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        slice_indices = [volume.shape[0] // 2, volume.shape[1] // 2, volume.shape[2] // 2]
        slice_names = ["Sagittal", "Coronal", "Axial"]
        (label, cam), = class_cams.items()
        name = label_names.get(label, f"Class {label}")
        if true_label is not None:
            true_name = label_names.get(true_label, f"Class {true_label}")
            title = f"True: {true_name}, Pred: {name}"
        else:
            title = name
        for col, (s_idx, s_name) in enumerate(zip(slice_indices, slice_names)):
            ax = axes[col]
            if col == 0:
                vol_slice = volume[s_idx, :, :]
                cam_slice = _to_numpy(cam[s_idx, :, :])
            elif col == 1:
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
            ax.set_title(f"{title} - {s_name}", fontsize=10)
    else:
        fig, axes = plt.subplots(3, n_classes, figsize=(4 * n_classes, 12))
        slice_indices = [volume.shape[0] // 2, volume.shape[1] // 2, volume.shape[2] // 2]
        slice_names = ["Sagittal", "Coronal", "Axial"]
        for col, (label, cam) in enumerate(sorted(class_cams.items())):
            name = label_names.get(label, f"Class {label}")
            for row, (s_idx, s_name) in enumerate(zip(slice_indices, slice_names)):
                ax = axes[row, col]
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

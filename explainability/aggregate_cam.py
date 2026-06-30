import numpy as np
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from typing import Dict, List


def compute_class_mean_cam(
    backbone: torch.nn.Module,
    head: torch.nn.Module,
    loader: DataLoader,
    device: str = "cuda",
    patch_size: int = 16,
) -> Dict[int, torch.Tensor]:
    from explainability.gradcam import gradcam_3d

    backbone.eval()
    head.eval()
    class_cams = {}
    class_counts = {}

    for batch in loader:
        volume = batch["volume"].to(device)
        labels = batch["label"].cpu().numpy()
        for i in range(volume.size(0)):
            label = labels[i]
            vol = volume[i : i + 1]
            cam = gradcam_3d(backbone, head, vol, patch_size=patch_size)
            if label not in class_cams:
                class_cams[label] = cam.clone()
                class_counts[label] = 1
            else:
                class_cams[label] += cam
                class_counts[label] += 1

    for label in class_cams:
        class_cams[label] = class_cams[label] / class_counts[label]
        class_cams[label] = (class_cams[label] - class_cams[label].min()) / (
            class_cams[label].max() + 1e-8
        )

    return class_cams


def save_aggregate_cams(
    class_cams: Dict[int, torch.Tensor],
    output_dir: str,
    model_name: str,
    config_name: str,
    label_names: Dict[int, str],
):
    out_dir = Path(output_dir) / model_name / config_name
    out_dir.mkdir(parents=True, exist_ok=True)
    for label, cam in class_cams.items():
        name = label_names.get(label, f"class_{label}")
        import nibabel as nib
        nifti_img = nib.Nifti1Image(cam.cpu().numpy().astype(np.float32), np.eye(4))
        nib.save(nifti_img, out_dir / f"cam_{name}.nii.gz")

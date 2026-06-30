import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional


def _normalize_slice(slice_2d):
    """Percentile clip + min-max to [0, 1]."""
    lo, hi = np.percentile(slice_2d, [2, 98])
    clipped = np.clip(slice_2d, lo, hi)
    rng = clipped.max() - clipped.min()
    if rng < 1e-8:
        return np.zeros_like(clipped)
    return (clipped - clipped.min()) / rng


def compute_brisque_slice(slice_2d):
    """BRISQUE score for a 2D image slice. Lower = better."""
    try:
        from piq import brisque
        import torch

        normed = _normalize_slice(slice_2d)
        img_t = torch.from_numpy(normed).float().unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            score = brisque(img_t)
        return score.item()
    except Exception:
        return None


def compute_clip_iqa_slice(slice_2d):
    """CLIP-IQA score for a 2D image slice. Higher = better."""
    try:
        from piq import clip_iqa
        import torch

        normed = _normalize_slice(slice_2d)
        if normed.ndim == 2:
            normed = np.stack([normed] * 3, axis=-1)
        img_t = torch.from_numpy(normed).float().permute(2, 0, 1).unsqueeze(0)
        with torch.no_grad():
            score = clip_iqa(img_t)
        return score.item()
    except Exception:
        return None


class VolumeQualityScorer:

    def __init__(self, alpha=0.5):
        self.alpha = alpha

    def score_volume(self, volume: np.ndarray, n_slices: Optional[int] = None):
        if volume.ndim == 3:
            slices = [volume[:, :, i] for i in range(volume.shape[-1])]
        elif volume.ndim == 4:
            slices = [volume[0, :, :, i] for i in range(volume.shape[-1])]
        else:
            return None

        if n_slices is not None and len(slices) > n_slices:
            indices = np.linspace(0, len(slices) - 1, n_slices, dtype=int)
            slices = [slices[i] for i in indices]

        brisque_scores = []
        clip_scores = []
        for s in slices:
            b = compute_brisque_slice(s)
            c = compute_clip_iqa_slice(s)
            if b is not None:
                brisque_scores.append(b)
            if c is not None:
                clip_scores.append(c)

        if not brisque_scores and not clip_scores:
            return None

        result = {}
        if brisque_scores:
            result["brisque_mean"] = float(np.mean(brisque_scores))
            result["brisque_std"] = float(np.std(brisque_scores))
        if clip_scores:
            result["clip_iqa_mean"] = float(np.mean(clip_scores))
            result["clip_iqa_std"] = float(np.std(clip_scores))

        if "brisque_mean" in result and "clip_iqa_mean" in result:
            brisque_norm = result["brisque_mean"] / 100.0
            result["composite"] = result["clip_iqa_mean"] - self.alpha * brisque_norm
        else:
            result["composite"] = result.get("clip_iqa_mean", -result.get("brisque_mean", 0))

        return result


def select_best_run(runs: List[Dict]) -> Dict:
    scored = [r for r in runs if r.get("quality") is not None]
    if not scored:
        return runs[0] if runs else None
    scored.sort(key=lambda r: r["quality"]["composite"], reverse=True)
    return scored[0]


def precompute_quality_scores(
    scan_index: List[dict],
    output_path: str,
    n_slices: int = 10,
    alpha: float = 0.5,
):
    scorer = VolumeQualityScorer(alpha=alpha)
    cache = {}
    for rec in scan_index:
        from data.dataset import NigerianBrainDataset

        ds = NigerianBrainDataset()
        try:
            vol = ds._load_volume(rec["path"])
        except Exception:
            continue
        score = scorer.score_volume(vol, n_slices=n_slices)
        if score is not None:
            cache[rec["path"]] = score

    with open(output_path, "w") as f:
        json.dump(cache, f, indent=2)
    return cache

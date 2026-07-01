import json
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from typing import Dict, List, Optional

from eval.metrics import compute_metrics, compute_calibration_error, aggregate_fold_metrics


@torch.no_grad()
def evaluate_fold(
    model: torch.nn.Module,
    head: torch.nn.Module,
    loader: DataLoader,
    device: str = "cuda",
    return_predictions: bool = False,
) -> Dict:
    model.eval()
    head.eval()
    all_preds, all_labels, all_probs = [], [], []
    all_subjects, all_ce_gad, all_sites = [], [], []
    for batch in loader:
        x = batch["volume"].to(device)
        features = model(x)
        if isinstance(features, tuple):
            features = features[0]
        logits = head(features)
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        preds = logits.argmax(dim=-1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(batch["label"].cpu().numpy())
        all_probs.append(probs)
        all_subjects.extend(batch.get("subject_id", [None] * len(preds)))
        all_ce_gad.extend(batch.get("ce_gadolinium", [None] * len(preds)))
        all_sites.extend(batch.get("site", [None] * len(preds)))
    all_probs = np.concatenate(all_probs, axis=0)

    metrics = compute_metrics(all_labels, all_preds, all_probs)
    cal = compute_calibration_error(np.array(all_labels), all_probs)
    metrics.update(cal)

    if return_predictions:
        metrics["predictions"] = {
            "subject_id": [int(x) for x in all_subjects],
            "label": [int(x) for x in all_labels],
            "pred": [int(x) for x in all_preds],
            "ce_gadolinium": [bool(x) for x in all_ce_gad],
            "site": [str(x) for x in all_sites],
        }
    return metrics


def run_full_evaluation(
    model_name: str,
    config_name: str,
    fold_heads: List[torch.nn.Module],
    fold_loaders: List[DataLoader],
    output_dir: str,
    device: str = "cuda",
):
    out_dir = Path(output_dir) / model_name / config_name
    out_dir.mkdir(parents=True, exist_ok=True)
    all_fold_metrics = []
    for fold_idx, (head, loader) in enumerate(zip(fold_heads, fold_loaders)):
        metrics = evaluate_fold(None, head, loader, device, return_predictions=True)
        all_fold_metrics.append(metrics)
        with open(out_dir / f"fold_{fold_idx}.json", "w") as f:
            json.dump(metrics, f, indent=2, default=str)

    summary = aggregate_fold_metrics(all_fold_metrics)
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    return summary

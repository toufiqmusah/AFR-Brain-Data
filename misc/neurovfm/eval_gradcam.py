from __future__ import annotations
import argparse
import json
from pathlib import Path

import torch

from model.neurovfm import NeuroVFMBackbone, NeuroVFMClassifier, NeuroVFMGradCAM


def main():
    parser = argparse.ArgumentParser(
        description="Run attention rollout (GradCAM) for NeuroVFM on test subjects."
    )
    parser.add_argument("--splits", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to foldN_best.pt (best classifier per fold)")
    parser.add_argument("--output-dir", type=str, default="./output/gradcam")
    parser.add_argument("--model-id", type=str, default="mlinslab/neurovfm-encoder")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--data-root", type=str, default=None)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = args.device

    with open(args.splits) as f:
        splits = json.load(f)

    subject_paths: dict[str, str] = splits["subject_paths"]
    subject_diag: dict[str, int] = splits["subject_diagnoses"]
    test_sids: list[str] = splits["test_subjects"]

    if args.data_root:
        root = Path(args.data_root)
        for sid in subject_paths:
            parts = Path(subject_paths[sid]).parts
            subj_idx = next(i for i, p in enumerate(parts) if p.startswith("sub-"))
            subject_paths[sid] = str(root / Path(*parts[subj_idx:]))

    print(f"Test subjects: {len(test_sids)}")

    print("Loading backbone...")
    backbone = NeuroVFMBackbone().load(args.model_id, device=device)

    print("Loading classifier checkpoint...")
    classifier = NeuroVFMClassifier(hidden_dim=768, n_classes=3, dropout=0.3).to(device)
    classifier.load_state_dict(torch.load(args.checkpoint, map_location=device))
    classifier.eval()

    cam = NeuroVFMGradCAM(backbone, classifier)

    for sid in test_sids:
        path = subject_paths[sid]
        diag = subject_diag[sid]
        name_map = {0: "Control", 1: "Dementia", 2: "Parkinson"}
        print(f"  {sid} ({name_map[diag]})...")

        batch = backbone._preprocessor.load_study([path], modality="mri")
        saliency, coords = cam.compute_saliency(batch)

        save_path = out_dir / f"{sid}_saliency.pt"
        torch.save({
            "subject_id": sid,
            "diagnosis": diag,
            "diagnosis_name": name_map[diag],
            "saliency": saliency.cpu(),
            "coords": coords,
        }, save_path)
        print(f"    -> saved {save_path}")

    print("Done.")


if __name__ == "__main__":
    main()
